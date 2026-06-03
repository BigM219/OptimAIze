from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import difflib
import io
import json
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.layout import LAYOUT_TO_OCR_CATEGORY
from optimaize_ocr.core.pipeline import setup_cpu_optimization

DEFAULT_IMAGE = Path("assets/IC-Basic-Document-Control-Template-Example.png")
DEFAULT_REFS = Path("outputs-falcon/ocr_results.json")
DEFAULT_CROPS_DIR = Path("outputs/vlm_eval_falcon_crops")


def normalize_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+\|>", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.casefold().split())


def compact(text: str, limit: int = 500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def similarity(reference: str, candidate: str) -> float:
    ref = normalize_text(reference)
    cand = normalize_text(candidate)
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    return difflib.SequenceMatcher(None, ref, cand).ratio()


def looks_hallucinated(text: str) -> bool:
    lower = str(text).casefold()
    markers = (
        "the image shows",
        "the image provided",
        "the text in the image is about",
        "census of canada",
        "football season",
        "### 1. introduction",
    )
    return any(marker in lower for marker in markers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VLM backend on Falcon-reference PP-DocLayout crops.")
    parser.add_argument("--model", default="dots-mocr")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quantize-mode", default="selective")
    parser.add_argument("--auto-runtime", choices=("off", "conservative", "speed", "experimental"), default="off")
    parser.add_argument("--quantize-int8", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--paddle-table-prompt", choices=("fast", "official"), default="fast")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpu-percent", type=float)
    parser.add_argument("--categories", help="Comma-separated source layout categories to include.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Warn when similarity is below this value.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--use-original-dots", action="store_true")
    parser.add_argument("--dots-fuse-mlp-swiglu", choices=("true", "false"), default="true")
    parser.add_argument("--dots-int8-lm-head", choices=("true", "false"), default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_cpu_optimization(args.threads, args.cpu_percent)
    image = Image.open(args.image).convert("RGB")
    items = json.loads(args.refs.read_text(encoding="utf-8"))
    args.crops_dir.mkdir(parents=True, exist_ok=True)

    allowed_categories = None
    if args.categories:
        allowed_categories = {c.strip().casefold() for c in args.categories.split(",") if c.strip()}

    quantize_int8 = None
    if args.quantize_int8 == "true":
        quantize_int8 = True
    elif args.quantize_int8 == "false":
        quantize_int8 = False

    backend = get_vlm_backend(
        args.model,
        device="cpu",
        quantize_int8=quantize_int8,
        use_optimized_dots=not args.use_original_dots,
        quantize_mode=args.quantize_mode,
        dots_fuse_mlp_swiglu=(args.dots_fuse_mlp_swiglu == "true"),
        dots_int8_lm_head=(args.dots_int8_lm_head == "true"),
        auto_runtime=args.auto_runtime,
        paddle_table_prompt=args.paddle_table_prompt,
    )
    if hasattr(backend, "max_new_tokens"):
        backend.max_new_tokens = args.max_new_tokens

    print(f"image={args.image} size={image.size}")
    print(f"refs={args.refs} items={len(items)}")
    print(
        f"model={args.model} quantize_mode={args.quantize_mode} auto_runtime={args.auto_runtime} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    runtime_policy = getattr(backend, "runtime_policy", None)
    if runtime_policy is not None:
        print(f"runtime_policy={runtime_policy.to_dict()}")
    print()

    rows = []
    total_time = 0.0
    for idx, item in enumerate(items):
        category = str(item["category"]).strip().lower()
        if allowed_categories and category.casefold() not in allowed_categories:
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break

        ocr_category = LAYOUT_TO_OCR_CATEGORY.get(category, "text") or "text"
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = args.crops_dir / f"crop_{idx:02d}_{category}.png"
        crop.save(crop_path)

        t0 = time.time()
        output = backend.generate_ocr(crop, category=ocr_category)
        dt = time.time() - t0
        total_time += dt

        ref = item.get("text", "")
        score = similarity(ref, output)
        hallucinated = looks_hallucinated(output)
        status = "OK"
        if hallucinated:
            status = "HALLUCINATION"
        elif score < args.threshold:
            status = "LOW_SIM"

        row = {
            "idx": idx,
            "category": category,
            "ocr_category": ocr_category,
            "bbox": [x1, y1, x2, y2],
            "crop": str(crop_path),
            "seconds": dt,
            "similarity": score,
            "status": status,
            "reference": ref,
            "output": output,
        }
        rows.append(row)

        print("=" * 90)
        print(
            f"#{idx} status={status} sim={score:.3f} category={category} "
            f"ocr_category={ocr_category} size={crop.size} t={dt:.2f}s"
        )
        print(f"crop={crop_path}")
        print(f"REF: {compact(ref)}")
        print(f"OUT: {compact(output)}")

        if args.fail_fast and status != "OK":
            break

    summary = {
        "model": args.model,
        "quantize_mode": args.quantize_mode,
        "auto_runtime": args.auto_runtime,
        "paddle_table_prompt": args.paddle_table_prompt,
        "runtime_policy": runtime_policy.to_dict() if runtime_policy is not None else None,
        "max_new_tokens": args.max_new_tokens,
        "items": len(rows),
        "total_seconds": total_time,
        "avg_seconds": total_time / len(rows) if rows else 0.0,
        "avg_similarity": sum(r["similarity"] for r in rows) / len(rows) if rows else 0.0,
        "failures": sum(1 for r in rows if r["status"] != "OK"),
        "rows": rows,
    }

    print("\n" + "=" * 90)
    print(
        f"SUMMARY items={summary['items']} failures={summary['failures']} "
        f"avg_sim={summary['avg_similarity']:.3f} total={summary['total_seconds']:.2f}s "
        f"avg={summary['avg_seconds']:.2f}s"
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote={args.output_json}")


if __name__ == "__main__":
    main()
