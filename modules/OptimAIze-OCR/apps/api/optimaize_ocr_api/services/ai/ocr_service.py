from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline

MODELS = ["falcon-ocr", "lighton-ocr", "paddleocr-vl", "glm-ocr", "surya-ocr", "surya-package", "dots-mocr"]


@lru_cache(maxsize=1)
def get_pipeline_cached(
    model_type: str,
    threads: int | None,
    cpu_percent: float | None,
    layout_model: str,
    quantize_int8: bool | None,
    quantize_mode: str,
    skip_layout: bool,
    full_page_mode: str,
    auto_runtime: str,
    use_optimized_dots: bool,
    dots_fuse_mlp_swiglu: bool,
    dots_int8_lm_head: bool,
    paddle_table_prompt: str,
) -> LayoutAwareOCRPipeline:
    return LayoutAwareOCRPipeline(
        model_type=model_type,
        layout_model=layout_model,
        num_threads=threads,
        cpu_percent=cpu_percent,
        quantize_int8=quantize_int8,
        quantize_mode=quantize_mode,
        skip_layout=skip_layout,
        full_page_mode=full_page_mode,
        auto_runtime=auto_runtime,
        use_optimized_dots=use_optimized_dots,
        dots_fuse_mlp_swiglu=dots_fuse_mlp_swiglu,
        dots_int8_lm_head=dots_int8_lm_head,
        paddle_table_prompt=paddle_table_prompt,
    )


def rows_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(results, start=1):
        rows.append({
            "index": idx,
            "category": row.get("category", ""),
            "bbox": row.get("bbox", ""),
            "score": row.get("score", ""),
            "text": row.get("text", ""),
        })
    return rows


class OCRService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def run_single_image(
        self,
        image_path: Path,
        image: Image.Image,
        model_type: str,
        threads: int | None,
        cpu_percent: float | None,
        layout_model: str,
        layout_threshold: float,
        skip_layout: bool,
        full_page_mode: str,
        quantize_int8: bool | None,
        quantize_mode: str,
        auto_runtime: str,
        use_optimized_dots: bool,
        dots_fuse_mlp_swiglu: bool,
        dots_int8_lm_head: bool,
        paddle_table_prompt: str,
        save_crops: bool,
    ) -> dict[str, Any]:
        del image
        if model_type not in MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        run_dir = self.output_dir / "api_runs" / str(int(time.time() * 1000))
        run_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = run_dir / "crops" if save_crops else None
        pipeline = get_pipeline_cached(
            model_type,
            threads,
            cpu_percent,
            layout_model,
            quantize_int8,
            quantize_mode,
            skip_layout,
            full_page_mode,
            auto_runtime,
            use_optimized_dots,
            dots_fuse_mlp_swiglu,
            dots_int8_lm_head,
            paddle_table_prompt,
        )
        markdown, results = pipeline.parse(str(image_path), layout_threshold=float(layout_threshold), save_crops_dir=str(crops_dir) if crops_dir else None)
        html = pipeline.generate_html(results, model_type, image_path.name)
        (run_dir / "parsed_document.md").write_text(markdown, encoding="utf-8")
        (run_dir / "parsed_document.html").write_text(html, encoding="utf-8")
        return {
            "markdown": markdown,
            "html": html,
            "regions": rows_from_results(results),
            "timings": getattr(pipeline, "last_timings", {}),
            "output_dir": str(run_dir),
            "image_name": image_path.name,
        }
