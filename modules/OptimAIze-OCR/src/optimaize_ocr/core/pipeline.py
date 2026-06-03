# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Pipeline orchestrating CPU Layout Detection and VLM OCR.

import os
import time
import logging
from PIL import Image
import torch
from tqdm import tqdm

from .layout import PPDocLayoutDetector, LAYOUT_TO_OCR_CATEGORY
from ..backends import get_vlm_backend
from ..output.markdown import assemble_markdown
from ..output.html import generate_html as render_html

logger = logging.getLogger(__name__)

_TEXT_CATEGORIES = {
    "text",
    "content",
    "abstract",
    "reference_content",
    "header",
    "footer",
    "number",
    "page-header",
    "page-footer",
}


def _threads_from_percent(cpu_percent: float | None) -> int | None:
    if cpu_percent is None:
        return None
    if cpu_percent <= 0:
        raise ValueError("cpu_percent must be > 0")
    logical = os.cpu_count() or 1
    return max(1, min(logical, round(logical * min(cpu_percent, 100.0) / 100.0)))


def setup_cpu_optimization(num_threads: int | None = None, cpu_percent: float | None = None):
    """Optimize PyTorch + OpenMP for CPU inference."""
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    effective_threads = num_threads if num_threads is not None else _threads_from_percent(cpu_percent)
    if effective_threads is not None and effective_threads > 0:
        torch.set_num_threads(effective_threads)
        try:
            torch.set_num_interop_threads(max(1, min(effective_threads, 4)))
        except RuntimeError as error:
            logger.warning("Skipping interop thread update after Torch parallel work started: %s", error)
        os.environ["OMP_NUM_THREADS"] = str(effective_threads)
        os.environ["MKL_NUM_THREADS"] = str(effective_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(effective_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(effective_threads)
        try:
            import numba
            numba.set_num_threads(effective_threads)
        except Exception:
            pass

    threads = torch.get_num_threads()
    logger.info(f"PyTorch CPU optimized. Using {threads} execution threads.")

class LayoutAwareOCRPipeline:
    """Document parsing pipeline using CPU layout detector and multiple VLM backends."""

    def __init__(
        self,
        model_type: str = "falcon-ocr",
        layout_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        num_threads: int | None = None,
        cpu_percent: float | None = None,
        device: str = "cpu",
        quantize_int8: bool | None = None,
        use_optimized_dots: bool = True,
        quantize_mode: str = "selective",
        skip_layout: bool = False,
        full_page_mode: str = "layout",
        dots_fuse_mlp_swiglu: bool = True,
        dots_int8_lm_head: bool = True,
        auto_runtime: str = "off",
        paddle_table_prompt: str = "fast",
    ):
        setup_cpu_optimization(num_threads, cpu_percent)
        self.device = device
        self.model_type = model_type
        self.skip_layout = skip_layout
        if full_page_mode not in ("layout", "svg"):
            raise ValueError(
                f"Unknown full_page_mode={full_page_mode!r} (expected 'layout' or 'svg')"
            )
        self.full_page_mode = full_page_mode

        # Load CPU layout detector (skipped for dots-mocr full-page-only mode,
        # which doesn't need pre-cropping — the VLM emits its own bboxes).
        if skip_layout:
            logger.info("Skipping PP-DocLayoutV3 load (skip_layout=True).")
            self.layout_detector = None
        else:
            self.layout_detector = PPDocLayoutDetector(model_id=layout_model, device=device)

        # Load CPU VLM OCR backend with optional INT8 quantization (cuts RAM ~4x)
        self.vlm_backend = get_vlm_backend(
            model_type, device=device, quantize_int8=quantize_int8,
            use_optimized_dots=use_optimized_dots,
            quantize_mode=quantize_mode,
            dots_fuse_mlp_swiglu=dots_fuse_mlp_swiglu,
            dots_int8_lm_head=dots_int8_lm_head,
            auto_runtime=auto_runtime,
            paddle_table_prompt=paddle_table_prompt,
        )

    def parse(
        self,
        image_path: str,
        layout_threshold: float = 0.3,
        save_crops_dir: str | None = None
    ) -> tuple[str, list[dict]]:
        """Run end-to-end layout-aware CPU OCR on a single image with detailed timing breakdown.

        1. Runs PP-DocLayoutV3.
        2. Segregates and crops layout regions containing text.
        3. Invokes VLM backend on each crop.
        4. Reassembles results into structured markdown text.

        Returns:
            markdown_content: Reassembled structured markdown document.
            results: List of dicts representing each detected region and its OCR.
        """
        overall_start = time.time()

        # 1. Image Load & Prep
        start_time = time.time()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        pil_img = Image.open(image_path).convert("RGB")
        img_w, img_h = pil_img.size
        img_prep_time = time.time() - start_time
        logger.info(f"[Time Breakdown] 1. Image loading & RGB conversion: {img_prep_time:.3f}s")

        logger.info(f"Starting pipeline on image: {pil_img.size} ({image_path})")

        # Optional full-page fast path. Vendor trained dots.mocr on FULL
        # pages with ``prompt_layout_all_en`` and that's its native mode
        # — but on a CPU + INT8 build the model loses its text-generation
        # capability and emits skeleton layout JSON without the ``text``
        # field, so we DO NOT enable this by default. ``--skip-layout``
        # forces it on for users who want to experiment.
        # ``full_page_mode`` selects between vendor-recommended modes:
        #   - "layout" → ``prompt_layout_all_en`` (JSON bbox+category+text)
        #   - "svg"    → ``prompt_image_to_svg`` (SVG with <text> elements)
        if self.skip_layout:
            if self.full_page_mode == "svg" and hasattr(self.vlm_backend, "parse_full_page_svg"):
                logger.info("Using VLM full-page SVG mode (skip_layout=True, full_page_mode=svg).")
                start_ocr = time.time()
                results = self.vlm_backend.parse_full_page_svg(pil_img)
                ocr_time_total = time.time() - start_ocr
                mode_label = "SVG"
            elif hasattr(self.vlm_backend, "parse_full_page"):
                logger.info("Using VLM full-page LAYOUT mode (skip_layout=True).")
                start_ocr = time.time()
                results = self.vlm_backend.parse_full_page(pil_img)
                ocr_time_total = time.time() - start_ocr
                mode_label = "LAYOUT"
            else:
                raise RuntimeError(
                    f"skip_layout=True requires a backend that supports "
                    f"parse_full_page; backend for model_type={self.model_type!r} does not."
                )
            logger.info(
                f"[Time Breakdown] 2-4. Full-page {mode_label} parse: "
                f"{ocr_time_total:.3f}s ({len(results)} elements)"
            )

            start_time = time.time()
            markdown_content = assemble_markdown(results)
            markdown_time = time.time() - start_time

            overall_time = time.time() - overall_start
            self.last_timings = {
                "overall_time": overall_time,
                "img_prep_time": img_prep_time,
                "layout_time": 0.0,
                "crop_time": 0.0,
                "ocr_time_total": ocr_time_total,
                "markdown_time": markdown_time,
            }
            logger.info(f"[Time Breakdown] TOTAL PIPELINE EXECUTION TIME: {overall_time:.3f}s")
            return markdown_content, results

        # 2. Run Layout Detection
        start_time = time.time()
        detections = self.layout_detector.detect(pil_img, threshold=layout_threshold)
        layout_time = time.time() - start_time
        logger.info(f"[Time Breakdown] 2. PP-DocLayoutV3 detection: {layout_time:.3f}s (found {len(detections)} raw layout boxes)")

        # 3. Filter layout categories and Crop
        start_time = time.time()
        valid_crops = []
        for det_idx, det in enumerate(detections):
            cat_key = det["category"].strip().lower()
            ocr_cat = LAYOUT_TO_OCR_CATEGORY.get(cat_key)
            if ocr_cat is None:
                # skip non-text areas like image, figure, chart, etc.
                continue
                
            x1, y1, x2, y2 = det["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w, int(x2 + 0.5)), min(img_h, int(y2 + 0.5))
            
            cw, ch = x2 - x1, y2 - y1
            if cw < 8 or ch < 8:
                # skip tiny crops
                continue
                
            crop_img = pil_img.crop((x1, y1, x2, y2))
            valid_crops.append({
                "det_idx": det_idx,
                "category": cat_key,
                "ocr_category": ocr_cat,
                "bbox": [x1, y1, x2, y2],
                "score": det["score"],
                "image": crop_img
            })
        raw_crop_count = len(valid_crops)
        crop_time = time.time() - start_time
        logger.info(f"[Time Breakdown] 3. Filtering & Cropping: {crop_time:.3f}s ({len(valid_crops)} crops kept from {raw_crop_count} layout crops)")

        # 3b. Fallback: If no valid crops, fall back to plain full-page OCR
        if not valid_crops:
            logger.info("No layout regions found. Falling back to full-page OCR.")
            start_ocr = time.time()
            text = self.vlm_backend.generate_ocr(pil_img, category="plain")
            ocr_time = time.time() - start_ocr
            logger.info(f"[Time Breakdown] Fallback Full-page VLM OCR: {ocr_time:.3f}s")
            
            fallback_res = [{
                "category": "plain",
                "bbox": [0, 0, img_w, img_h],
                "score": 1.0,
                "text": text
            }]
            
            overall_time = time.time() - overall_start
            logger.info(f"[Time Breakdown] TOTAL PIPELINE EXECUTION TIME: {overall_time:.3f}s")
            self.last_timings = {
                "overall_time": overall_time,
                "img_prep_time": img_prep_time,
                "layout_time": layout_time,
                "crop_time": crop_time,
                "ocr_time_total": ocr_time,
                "markdown_time": 0.0
            }
            return text, fallback_res

        # 4. Generate OCR text per crop region
        logger.info(f"Running VLM OCR on {len(valid_crops)} cropped text regions...")
        results = []
        
        # Save crops helper if specified
        if save_crops_dir:
            os.makedirs(save_crops_dir, exist_ok=True)

        ocr_start_total = time.time()
        crop_timings = []
        for idx, item in enumerate(tqdm(valid_crops, desc="OCR Processing")):
            crop_img = item["image"]
            
            if save_crops_dir:
                # Save crops ONLY for non-text structural layout blocks (tables and formulas)
                if item["category"] in ("table", "formula"):
                    crop_name = f"crop_{idx}_{item['category']}.png"
                    crop_img.save(os.path.join(save_crops_dir, crop_name))
                
            # Perform OCR on CPU and measure time
            start_single_ocr = time.time()
            text = self.vlm_backend.generate_ocr(crop_img, category=item["ocr_category"])
            single_ocr_time = time.time() - start_single_ocr
            
            crop_timings.append(single_ocr_time)
            logger.info(f"[Time Breakdown]   - Crop {idx} ({item['category']}): {single_ocr_time:.3f}s")
            
            results.append({
                "category": item["category"],
                "bbox": item["bbox"],
                "score": item["score"],
                "text": text
            })
            
        ocr_time_total = time.time() - ocr_start_total
        logger.info(f"[Time Breakdown] 4. VLM OCR processing (All Crops): {ocr_time_total:.3f}s (Average: {sum(crop_timings)/len(crop_timings):.3f}s per crop)")

        # 5. Format results into structured Markdown
        start_time = time.time()
        markdown_content = assemble_markdown(results)
        markdown_time = time.time() - start_time
        logger.info(f"[Time Breakdown] 5. Markdown formatting & assembly: {markdown_time:.3f}s")
        
        overall_time = time.time() - overall_start
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE TIME BREAKDOWN SUMMARY")
        logger.info("=" * 50)
        logger.info(f"1. Image Loading & Prep     : {img_prep_time:.3f}s ({img_prep_time/overall_time*100:.1f}%)")
        logger.info(f"2. Layout Detection         : {layout_time:.3f}s ({layout_time/overall_time*100:.1f}%)")
        logger.info(f"3. Filtering & Cropping     : {crop_time:.3f}s ({crop_time/overall_time*100:.1f}%)")
        logger.info(f"4. VLM OCR (All Crops)      : {ocr_time_total:.3f}s ({ocr_time_total/overall_time*100:.1f}%)")
        logger.info(f"5. Markdown Assembly        : {markdown_time:.3f}s ({markdown_time/overall_time*100:.1f}%)")
        logger.info("-" * 50)
        logger.info(f"TOTAL PIPELINE EXECUTION    : {overall_time:.3f}s (100.0%)")
        logger.info("=" * 50 + "\n")
        
        self.last_timings = {
            "overall_time": overall_time,
            "img_prep_time": img_prep_time,
            "layout_time": layout_time,
            "crop_time": crop_time,
            "ocr_time_total": ocr_time_total,
            "markdown_time": markdown_time
        }
        
        return markdown_content, results

    def generate_html(
        self,
        results: list[dict],
        model_type: str,
        image_name: str = "Document"
    ) -> str:
        """Generate a simple, clean, and readable HTML file showing OCR results with visible table borders."""
        return render_html(results, model_type, image_name)
