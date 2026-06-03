# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# main.py - CLI entrypoint for flexible CPU layout-aware OCR.

import os
import json
import logging
import sys
from typing import Literal
from pathlib import Path
import tyro

OCR_MODULE_ROOT = Path(__file__).resolve().parent
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
if str(OCR_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(OCR_SRC_ROOT))

from optimaize_ocr import LayoutAwareOCRPipeline

# Set up logging format — DEBUG enabled for dots.mocr patch diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logging.getLogger("optimaize_ocr.backends.dots_mocr.patches").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def main(
    image: str,
    model: Literal["falcon-ocr", "lighton-ocr", "dots-mocr", "paddleocr-vl", "surya-ocr", "glm-ocr"] = "falcon-ocr",
    threshold: float = 0.3,
    threads: int | None = None,
    cpu_percent: float | None = None,
    output_dir: str = "./outputs",
    layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    quantize_int8: bool | None = None,
    use_optimized_dots: bool = True,
    quantize_mode: Literal["selective", "full", "fp16", "none", "lm_head", "mlp", "mlp_lm_head", "auto"] = "selective",
    auto_runtime: Literal["off", "conservative", "speed", "experimental"] = "off",
    skip_layout: bool = False,
    full_page_mode: Literal["layout", "svg"] = "layout",
    dots_fuse_mlp_swiglu: bool = True,
    dots_int8_lm_head: bool = True,
    test_all_prompts: bool = False,
):
    """Run CPU layout-aware OCR pipeline on a single image.

    Args:
        image: Path to the input document image. (Required)
        model: OCR Vision-Language Model backend to use ('falcon-ocr', 'lighton-ocr', 'dots-mocr', 'paddleocr-vl', 'surya-ocr', or 'glm-ocr').
        threshold: Confidence threshold for PP-DocLayoutV3 detection (0.0 to 1.0).
        threads: Number of CPU threads to utilize. None will default to all threads unless cpu_percent is set.
        cpu_percent: Approximate CPU thread budget as a percentage of logical CPUs. Ignored when threads is set.
        output_dir: Output folder to save markdown text, json results, and cropped regions.
        layout_model: Hugging Face repository ID for the layout detection model.
        quantize_int8: Force INT8 dynamic quantization on/off. Default per-model:
            False for Falcon-OCR and LightOn-OCR, True for Dots-MOCR (saves RAM).
        use_optimized_dots: Use fused-kernel OptimizedDotsMOCRBackend for dots-mocr
            (recommended — ~35x faster, correct results). Set False only for debugging.
        quantize_mode: Dots-MOCR weight precision. ``selective`` (default) keeps
            attention FP32 + INT8 MLP (~6 GB, accurate). ``full`` quantizes
            everything (~4 GB but hallucinates on small dense crops). ``fp16``
            runs the model in float16 (~5 GB, native bf16-trained weights map
            cleanly to fp16). ``none`` is full FP32 (~16 GB).
        auto_runtime: Automatically derive safe CPU/decode settings from model architecture and CPU capabilities.
            ``conservative`` keeps promoted correctness-gated defaults; ``speed`` and ``experimental`` may select candidates that require re-gating.
        skip_layout: Skip PP-DocLayoutV3 entirely and use the VLM's native
            full-page parsing instead. EXPERIMENTAL — for dots-mocr on
            CPU+INT8, the model loses its text-generation capability
            under quantization and emits skeleton layout JSON without
            actual text content. Full-page mode is only useful on GPU
            with bfloat16 (the vendor's native setup). The default
            per-crop path remains the recommended way to run dots-mocr
            on CPU.
        full_page_mode: Which vendor full-page prompt to use when
            ``skip_layout=True``. ``layout`` (default) uses
            ``prompt_layout_all_en`` and parses JSON. ``svg`` uses
            ``prompt_image_to_svg`` (sampling temperature 0.9 per vendor)
            and extracts text from SVG ``<text>`` elements.
        dots_fuse_mlp_swiglu: dots-mocr only. Enable fused INT8 SwiGLU MLP decode kernel.
        dots_int8_lm_head: dots-mocr only. Enable INT8 lm_head GEMV replacement.
        test_all_prompts: dots-mocr only. Runs every vendor prompt mode
            (layout_all_en, layout_only_en, ocr, scene_spotting,
            web_parsing, image_to_svg, general) on the same image and
            saves the raw output for each into ``output_dir/prompts/``.
            Use this to compare which mode actually works on the current
            CPU+quantization setup. Skips all other output paths.
    """
    logger.info("Initializing CPU Layout-Aware OCR Pipeline...")

    # Initialize pipeline
    pipeline = LayoutAwareOCRPipeline(
        model_type=model,
        layout_model=layout_model,
        num_threads=threads,
        cpu_percent=cpu_percent,
        device="cpu",
        quantize_int8=quantize_int8,
        use_optimized_dots=use_optimized_dots,
        quantize_mode=quantize_mode,
        skip_layout=skip_layout,
        full_page_mode=full_page_mode,
        dots_fuse_mlp_swiglu=dots_fuse_mlp_swiglu,
        dots_int8_lm_head=dots_int8_lm_head,
        auto_runtime=auto_runtime,
    )

    # Test-all-prompts branch: skip the regular pipeline and dump every
    # vendor prompt mode's raw output for offline comparison.
    if test_all_prompts:
        from PIL import Image
        from optimaize_ocr.backends.dots_mocr.custom_backend import (
            DOTS_PROMPT_MODES,
        )

        backend = pipeline.vlm_backend
        if not hasattr(backend, "run_prompt"):
            raise RuntimeError(
                f"--test-all-prompts requires a backend with run_prompt; "
                f"model={model!r} does not."
            )

        out_path = Path(output_dir) / "prompts"
        out_path.mkdir(parents=True, exist_ok=True)
        pil_img = Image.open(image).convert("RGB")
        # ``general`` and ``grounding_ocr`` need extra args; skip them
        # in the bulk sweep so the user can drive them manually.
        skip_modes = {"general", "grounding_ocr"}
        modes_to_run = [m for m in DOTS_PROMPT_MODES if m not in skip_modes]
        logger.info(
            "Sweeping %d prompt modes: %s",
            len(modes_to_run), ", ".join(modes_to_run),
        )
        summary: list[dict] = []
        for mode in modes_to_run:
            logger.info("=== prompt_mode=%s ===", mode)
            try:
                raw, prompt_text = backend.run_prompt(pil_img, mode)
            except Exception as e:
                logger.exception("prompt_mode=%s crashed: %s", mode, e)
                raw, prompt_text = f"<ERROR: {e}>", ""
            (out_path / f"{mode}.txt").write_text(raw, encoding="utf-8")
            summary.append({
                "mode": mode,
                "prompt_chars": len(prompt_text),
                "output_chars": len(raw),
                "first_120": raw[:120],
            })
        (out_path / "_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Wrote %d prompt outputs to %s", len(modes_to_run), out_path,
        )
        return

    # Run parsing
    logger.info(f"Running pipeline with model '{model}' and layout model '{layout_model}'...")
    save_crops_path = os.path.join(output_dir, "crops")
    markdown_content, results = pipeline.parse(
        image_path=image,
        layout_threshold=threshold,
        save_crops_dir=save_crops_path
    )
    
    # Make sure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save structured markdown content
    md_file = out_path / "parsed_document.md"
    md_file.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Structured Markdown saved to: {md_file.absolute()}")
    
    # Save full JSON details
    json_file = out_path / "ocr_results.json"
    json_results = []
    for item in results:
        json_results.append({
            "category": item["category"],
            "bbox": item["bbox"],
            "score": item["score"],
            "text": item["text"]
        })
    json_file.write_text(json.dumps(json_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Full details JSON saved to: {json_file.absolute()}")
    
    # Generate and save premium interactive HTML Viewer
    html_content = pipeline.generate_html(
        results=results,
        model_type=model,
        image_name=os.path.basename(image)
    )
    html_file = out_path / "parsed_document.html"
    html_file.write_text(html_content, encoding="utf-8")
    logger.info(f"Premium HTML Viewer saved to: {html_file.absolute()}")
    
    # Print the markdown preview (safe UTF-8 on all platforms)
    import sys
    out = sys.stdout if hasattr(sys.stdout, "reconfigure") else sys.stdout
    try:
        out.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("\n" + "=" * 60)
    print("PARSED DOCUMENT MARKDOWN PREVIEW:")
    print("=" * 60)
    print(markdown_content)
    print("=" * 60 + "\n")
    logger.info("Pipeline execution finished successfully!")

if __name__ == "__main__":
    tyro.cli(main)
