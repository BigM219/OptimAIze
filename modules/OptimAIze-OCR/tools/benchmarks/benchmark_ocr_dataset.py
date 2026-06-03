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
from collections import Counter
from pathlib import Path
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import setup_cpu_optimization

DEFAULT_MODELS = ("falcon-ocr", "lighton-ocr", "paddleocr-vl", "glm-ocr", "surya-ocr", "dots-mocr")
PRESETS = {
    "funsd": {
        "dataset": "nielsr/funsd",
        "split": "test",
        "image_column": "image",
        "text_column": None,
    },
    "cord": {
        "dataset": "naver-clova-ix/cord-v2",
        "split": "test",
        "image_column": "image",
        "text_column": "ground_truth",
    },
}


def normalize_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+\|>", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.casefold().split())


def similarity(reference: str, candidate: str) -> float:
    ref = normalize_text(reference)
    cand = normalize_text(candidate)
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    return difflib.SequenceMatcher(None, ref, cand).ratio()


def token_f1(reference: str, candidate: str) -> float:
    ref_tokens = normalize_text(reference).split()
    cand_tokens = normalize_text(candidate).split()
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum((ref_counts & cand_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


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


def compact(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_flatten_strings(v))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten_strings(v))
        return out
    return [str(value)]


def extract_reference(item: dict[str, Any], text_column: str | None) -> str:
    if text_column:
        raw = item.get(text_column)
        if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                pass
        return " ".join(_flatten_strings(raw))
    words = item.get("words")
    if words:
        return " ".join(map(str, words))
    tokens = item.get("tokens")
    if tokens:
        return " ".join(map(str, tokens))
    for key in ("text", "transcription", "ground_truth", "label"):
        if key in item:
            return " ".join(_flatten_strings(item[key]))
    return ""


def as_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if "path" in value:
            return Image.open(value["path"]).convert("RGB")
        if "bytes" in value:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def load_dataset_rows(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'datasets'. Install it with: pip install datasets\n"
            "Then rerun this script."
        ) from exc

    preset = PRESETS.get(args.preset) if args.preset else None
    dataset_name = args.dataset or (preset["dataset"] if preset else None)
    split = args.split or (preset["split"] if preset else "test")
    image_column = args.image_column or (preset["image_column"] if preset else "image")
    text_column = args.text_column if args.text_column is not None else (preset["text_column"] if preset else None)
    if not dataset_name:
        raise SystemExit("Pass --preset funsd/cord or --dataset <hf_dataset_name>.")

    ds = load_dataset(dataset_name, split=split, streaming=args.streaming)
    rows = []
    for idx, item in enumerate(ds):
        if args.offset and idx < args.offset:
            continue
        if args.limit is not None and len(rows) >= args.limit:
            break
        image = as_image(item[image_column])
        reference = extract_reference(item, text_column)
        rows.append({"idx": idx, "image": image, "reference": reference})
    return rows, {"dataset": dataset_name, "split": split, "image_column": image_column, "text_column": text_column}


def run_model(model: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    quantize_int8 = None
    if args.quantize_int8 == "true":
        quantize_int8 = True
    elif args.quantize_int8 == "false":
        quantize_int8 = False

    t_load = time.perf_counter()
    try:
        backend = get_vlm_backend(
            model,
            device="cpu",
            quantize_int8=quantize_int8,
            use_optimized_dots=not args.use_original_dots,
            quantize_mode=args.quantize_mode,
            dots_fuse_mlp_swiglu=(args.dots_fuse_mlp_swiglu == "true"),
            dots_int8_lm_head=(args.dots_int8_lm_head == "true"),
            auto_runtime=args.auto_runtime,
            paddle_table_prompt=args.paddle_table_prompt,
        )
    except Exception as exc:
        load_seconds = time.perf_counter() - t_load
        return {
            "model": model,
            "load_seconds": load_seconds,
            "runtime_policy": None,
            "items": 0,
            "total_seconds": 0.0,
            "avg_seconds": 0.0,
            "avg_score": 0.0,
            "avg_similarity": 0.0,
            "avg_token_f1": 0.0,
            "failures": len(rows),
            "load_error": repr(exc),
            "rows": [],
        }
    if hasattr(backend, "max_new_tokens"):
        backend.max_new_tokens = args.max_new_tokens
    load_seconds = time.perf_counter() - t_load

    model_rows = []
    total = 0.0
    for row in rows:
        t0 = time.perf_counter()
        try:
            output = backend.generate_ocr(row["image"], category=args.category)
            error = None
        except Exception as exc:
            output = ""
            error = repr(exc)
        dt = time.perf_counter() - t0
        total += dt
        seq_score = similarity(row["reference"], output)
        f1_score = token_f1(row["reference"], output)
        score = f1_score if args.metric == "token_f1" else seq_score
        hallucinated = looks_hallucinated(output)
        status = "OK"
        if error:
            status = "ERROR"
        elif hallucinated:
            status = "HALLUCINATION"
        elif score < args.threshold:
            status = "LOW_SCORE"
        model_rows.append({
            "idx": row["idx"],
            "seconds": dt,
            "score": score,
            "similarity": seq_score,
            "token_f1": f1_score,
            "status": status,
            "error": error,
            "reference": row["reference"],
            "output": output,
        })
        print(
            f"{model} idx={row['idx']} status={status} score={score:.3f} sim={seq_score:.3f} f1={f1_score:.3f} "
            f"t={dt:.2f}s ref={compact(row['reference'], 100)!r} out={compact(output, 100)!r}",
            flush=True,
        )
        if args.fail_fast and status != "OK":
            break

    return {
        "model": model,
        "load_seconds": load_seconds,
        "runtime_policy": getattr(backend, "runtime_policy", None).to_dict() if getattr(backend, "runtime_policy", None) is not None else None,
        "items": len(model_rows),
        "total_seconds": total,
        "avg_seconds": total / len(model_rows) if model_rows else 0.0,
        "avg_score": sum(r["score"] for r in model_rows) / len(model_rows) if model_rows else 0.0,
        "avg_similarity": sum(r["similarity"] for r in model_rows) / len(model_rows) if model_rows else 0.0,
        "avg_token_f1": sum(r["token_f1"] for r in model_rows) / len(model_rows) if model_rows else 0.0,
        "failures": sum(1 for r in model_rows if r["status"] != "OK"),
        "rows": model_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OCR backends on a larger HF dataset.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="funsd")
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--image-column")
    parser.add_argument("--text-column")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--category", default="plain")
    parser.add_argument("--metric", choices=("token_f1", "sequence"), default="token_f1")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--auto-runtime", choices=("off", "conservative", "speed", "experimental"), default="conservative")
    parser.add_argument("--quantize-mode", default="auto")
    parser.add_argument("--quantize-int8", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--paddle-table-prompt", choices=("fast", "official"), default="fast")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpu-percent", type=float)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/dataset_benchmark/summary.json"))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--use-original-dots", action="store_true")
    parser.add_argument("--dots-fuse-mlp-swiglu", choices=("true", "false"), default="true")
    parser.add_argument("--dots-int8-lm-head", choices=("true", "false"), default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_cpu_optimization(args.threads, args.cpu_percent)
    rows, dataset_info = load_dataset_rows(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"dataset={dataset_info} rows={len(rows)} models={models}")

    results = []
    for model in models:
        print("=" * 100)
        print(f"MODEL {model}")
        results.append(run_model(model, rows, args))

    summary = {
        "dataset": dataset_info,
        "limit": args.limit,
        "offset": args.offset,
        "category": args.category,
        "auto_runtime": args.auto_runtime,
        "quantize_mode": args.quantize_mode,
        "paddle_table_prompt": args.paddle_table_prompt,
        "metric": args.metric,
        "threshold": args.threshold,
        "models": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 100)
    for result in results:
        print(
            f"SUMMARY model={result['model']} items={result['items']} failures={result['failures']} "
            f"avg_score={result['avg_score']:.3f} avg_f1={result['avg_token_f1']:.3f} "
            f"avg_sim={result['avg_similarity']:.3f} total={result['total_seconds']:.2f}s "
            f"avg={result['avg_seconds']:.2f}s load={result['load_seconds']:.2f}s"
        )
    print(f"wrote={args.output_json}")


if __name__ == "__main__":
    main()
