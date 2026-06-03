from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Benchmark script for the OCR pipeline.

Runs the full pipeline on a test image and reports per-crop timings.
Also runs synthetic crops (random PIL images) to isolate model-only cost.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench")


def make_synthetic_crops(out_dir: Path) -> list[Path]:
    """Create a small set of synthetic crops at varied sizes/contents.

    Sizes chosen to approximate real crops from a document image:
    small (footnote), medium (paragraph), large (full-column).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    crops: list[Path] = []
    specs = [
        ("small_text.png",   320, 60,  "Page 12 — Footnote line of text."),
        ("medium_para.png",  640, 200, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
                                       "Sed do eiusmod tempor incididunt ut labore et dolore magna.\n"
                                       "Ut enim ad minim veniam, quis nostrud exercitation ullamco."),
        ("large_text.png",   800, 400, "Section 4.3 Detailed Methodology\n\n"
                                       "The proposed CPU-optimized pipeline applies dynamic INT8\n"
                                       "quantization layer by layer to keep peak RAM low, then patches\n"
                                       "every Linear layer with hand-tuned AVX2 / Numba GEMV kernels.\n"
                                       "All decoder attention paths are unchanged mathematically; the\n"
                                       "speedup comes from kernel-level scheduling and quantization.\n\n"
                                       "We measured 4-7x latency reductions on Intel Meteor Lake CPUs."),
    ]
    for fname, w, h, text in specs:
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, fill="black")
        path = out_dir / fname
        img.save(path)
        crops.append(path)
    return crops


def bench_backend_direct(backend, crops: list[Path], label: str) -> dict:
    """Call backend.generate_ocr directly on each crop, bypassing layout."""
    times: list[float] = []
    chars: list[int] = []
    for i, p in enumerate(crops):
        img = Image.open(p).convert("RGB")
        # Warm up the first call by running it twice (Numba JIT compile on first call)
        if i == 0:
            t0 = time.perf_counter()
            _ = backend.generate_ocr(img, category="text")
            warmup = time.perf_counter() - t0
            logger.info(f"[{label}] Warmup (incl. JIT compile): {warmup:.2f}s")

        t0 = time.perf_counter()
        out = backend.generate_ocr(img, category="text")
        dt = time.perf_counter() - t0
        times.append(dt)
        chars.append(len(out))
        logger.info(f"[{label}] crop={p.name:<22} size={img.size}  t={dt:.2f}s  chars={len(out)}")
        logger.info(f"         output: {out[:100]!r}")

    avg = sum(times) / max(1, len(times))
    mx = max(times) if times else 0.0
    return {"avg": avg, "max": mx, "times": times, "chars": chars}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="falcon-ocr", choices=["falcon-ocr", "lighton-ocr", "dots-mocr"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--no-quantize-int8", action="store_true")
    parser.add_argument("--use-optimized-dots", action="store_true", help="Use optimized Dots-MOCR backend")
    parser.add_argument("--synth-only", action="store_true",
                        help="Skip the full layout pipeline and benchmark only synthetic crops.")
    args = parser.parse_args()

    quant = None
    if args.quantize_int8:
        quant = True
    elif args.no_quantize_int8:
        quant = False

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    crops_dir = root / "outputs" / "synth_crops"
    crops = make_synthetic_crops(crops_dir)

    logger.info(f"=== Benchmark backend={args.model} quantize_int8={quant} threads={args.threads} ===")
    logger.info(f"Synthetic crops:  {[p.name for p in crops]}")

    from optimaize_ocr.backends import get_vlm_backend
    from optimaize_ocr.core.pipeline import setup_cpu_optimization

    setup_cpu_optimization(args.threads)

    t0 = time.perf_counter()
    if args.model == "dots-mocr" and args.use_optimized_dots:
        backend = get_vlm_backend(args.model, device="cpu", quantize_int8=quant, use_optimized_dots=True)
    else:
        backend = get_vlm_backend(args.model, device="cpu", quantize_int8=quant)
    load_dt = time.perf_counter() - t0
    logger.info(f"Backend load time: {load_dt:.2f}s")

    res = bench_backend_direct(backend, crops, label=args.model)
    logger.info("---")
    logger.info(f"AVG per-crop: {res['avg']:.2f}s")
    logger.info(f"MAX per-crop: {res['max']:.2f}s")
    logger.info(f"Times:        {[round(t, 2) for t in res['times']]}")

    # Verdict against 5s target
    target = 5.0
    if res["max"] <= target:
        logger.info(f"PASS: all crops <= {target}s")
    else:
        logger.info(f"FAIL: max {res['max']:.2f}s > {target}s — needs more optimization")


if __name__ == "__main__":
    main()
