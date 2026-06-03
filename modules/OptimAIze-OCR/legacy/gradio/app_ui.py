import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import torch
from PIL import Image

OCR_MODULE_ROOT = Path(__file__).resolve().parents[2]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline
from optimaize_ocr.storage.history_db import DEFAULT_HISTORY_DB, HistoryDB, image_sha256, runtime_hash, safe_name
from legacy.cli.ocr_history import (
    load_document_images,
    processor_config_hash,
    tensor_cache_path,
    visual_cache_prompt_hash,
    visual_cache_runtime_config,
)

UPLOAD_DIR = Path("outputs/ui/uploads")
EXPORT_DIR = Path("outputs/ui")
PIPELINE_CACHE: dict[tuple[Any, ...], LayoutAwareOCRPipeline] = {}
BACKEND_CACHE: dict[tuple[Any, ...], Any] = {}

MODELS = [
    "falcon-ocr",
    "lighton-ocr",
    "paddleocr-vl",
    "PaddlePaddle/PaddleOCR-VL-1.5",
    "glm-ocr",
    "surya-ocr",
    "surya-package",
    "dots-mocr",
]

CATEGORIES = ["plain", "text", "table", "formula", "caption", "html"]
RUNTIME_MODES = ["off", "conservative", "speed", "experimental"]
QUANTIZE_MODES = ["auto", "none", "selective", "mlp", "mlp_lm_head", "lm_head"]

APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,750&family=Manrope:wght@400;500;650;750&display=swap');

:root {
    --ocr-bg: #f7f1e8;
    --ocr-bg-2: #ede2d3;
    --ocr-panel: rgba(255, 252, 246, 0.86);
    --ocr-panel-strong: rgba(255, 252, 246, 0.96);
    --ocr-text: #24201b;
    --ocr-muted: #70675d;
    --ocr-border: rgba(61, 48, 36, 0.14);
    --ocr-accent: #c96342;
    --ocr-accent-2: #df9b63;
    --ocr-green: #7d8f67;
    --ocr-blue: #697e9d;
    --ocr-shadow: rgba(113, 73, 38, 0.18);
    --ocr-glow: rgba(218, 132, 75, 0.28);
    --ocr-ring: rgba(201, 100, 66, 0.4);
    --ocr-cursor-x: 50%;
    --ocr-cursor-y: 10%;
}

.dark,
.dark :root,
body.dark,
.gradio-container.dark {
    --ocr-bg: #171411;
    --ocr-bg-2: #231c17;
    --ocr-panel: rgba(39, 33, 28, 0.78);
    --ocr-panel-strong: rgba(48, 40, 34, 0.94);
    --ocr-text: #f2e8da;
    --ocr-muted: #c9bba9;
    --ocr-border: rgba(245, 203, 164, 0.18);
    --ocr-accent: #e18a5a;
    --ocr-accent-2: #f1c27d;
    --ocr-green: #b8ca98;
    --ocr-blue: #aebfda;
    --ocr-shadow: rgba(0, 0, 0, 0.34);
    --ocr-glow: rgba(225, 138, 90, 0.24);
    --ocr-ring: rgba(241, 194, 125, 0.42);
}

html, body, .gradio-container {
    min-height: 100%;
    font-family: 'Manrope', ui-sans-serif, system-ui, sans-serif !important;
    color: var(--ocr-text) !important;
    background:
        radial-gradient(circle 520px at var(--ocr-cursor-x) var(--ocr-cursor-y), var(--ocr-glow), transparent 64%),
        radial-gradient(circle 760px at 12% 8%, rgba(232, 161, 95, 0.16), transparent 58%),
        linear-gradient(135deg, var(--ocr-bg), var(--ocr-bg-2)) !important;
    transition: background 180ms ease-out, color 180ms ease-out;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.28;
    background-image:
        linear-gradient(rgba(120, 83, 46, 0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120, 83, 46, 0.07) 1px, transparent 1px);
    background-size: 34px 34px;
    mask-image: radial-gradient(circle at center, black, transparent 78%);
}

.gradio-container > .main,
.gradio-container .wrap,
.gradio-container .contain {
    position: relative;
    z-index: 1;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container .markdown h1,
.gradio-container .markdown h2 {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.035em;
    color: var(--ocr-text) !important;
}

.gradio-container .markdown h1,
.gradio-container h1 {
    font-size: clamp(2.1rem, 4vw, 4.6rem) !important;
    line-height: 0.95 !important;
    margin-bottom: 0.45rem !important;
    text-shadow: 0 16px 42px var(--ocr-shadow);
}

.gradio-container .markdown p,
.gradio-container label,
.gradio-container span,
.gradio-container p {
    color: var(--ocr-muted) !important;
}

.gradio-container .tabs {
    border-radius: 28px !important;
    padding: 10px !important;
    background: linear-gradient(145deg, rgba(255,255,255,0.24), rgba(255,255,255,0.05)) !important;
    border: 1px solid var(--ocr-border) !important;
    box-shadow: 0 24px 80px var(--ocr-shadow), inset 0 1px 0 rgba(255,255,255,0.18) !important;
    backdrop-filter: blur(18px) saturate(1.15);
}

.gradio-container button[role='tab'] {
    border-radius: 18px !important;
    color: var(--ocr-muted) !important;
    font-weight: 750 !important;
    letter-spacing: -0.015em;
    transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease, color 160ms ease !important;
}

.gradio-container button[role='tab']:hover {
    transform: translateY(-1px);
    color: var(--ocr-accent) !important;
    background: rgba(225, 138, 90, 0.12) !important;
}

.gradio-container button[role='tab'][aria-selected='true'] {
    color: var(--ocr-text) !important;
    background: linear-gradient(135deg, rgba(225, 138, 90, 0.22), rgba(241, 194, 125, 0.16)) !important;
    box-shadow: 0 10px 34px rgba(201, 100, 66, 0.18), inset 0 1px 0 rgba(255,255,255,0.28) !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .dataframe,
.gradio-container .tabitem {
    border-color: var(--ocr-border) !important;
    background: var(--ocr-panel) !important;
    box-shadow: 0 18px 55px var(--ocr-shadow), inset 0 1px 0 rgba(255,255,255,0.14) !important;
    backdrop-filter: blur(16px) saturate(1.1);
}

.gradio-container .block {
    border-radius: 22px !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container .wrap.svelte-1xfsv4t,
.gradio-container .secondary-wrap {
    background: var(--ocr-panel-strong) !important;
    border-color: var(--ocr-border) !important;
    color: var(--ocr-text) !important;
    border-radius: 16px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.12) !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    outline: 2px solid var(--ocr-ring) !important;
    box-shadow: 0 0 0 5px rgba(225, 138, 90, 0.12), 0 12px 34px var(--ocr-shadow) !important;
}

.gradio-container button:not([role='tab']) {
    border-radius: 999px !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: #fff9ef !important;
    border: 1px solid rgba(255,255,255,0.24) !important;
    background: linear-gradient(135deg, var(--ocr-accent), var(--ocr-accent-2)) !important;
    box-shadow: 0 14px 36px rgba(201, 100, 66, 0.28), inset 0 1px 0 rgba(255,255,255,0.28) !important;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease !important;
}

.gradio-container button:not([role='tab']):hover {
    transform: translateY(-2px);
    filter: saturate(1.08) brightness(1.03);
    box-shadow: 0 20px 46px rgba(201, 100, 66, 0.36), 0 0 34px var(--ocr-glow) !important;
}

.gradio-container table,
.gradio-container th,
.gradio-container td {
    color: var(--ocr-text) !important;
    border-color: var(--ocr-border) !important;
}

.gradio-container th {
    background: rgba(225, 138, 90, 0.12) !important;
    font-weight: 800 !important;
}

.gradio-container code,
.gradio-container pre {
    color: var(--ocr-text) !important;
    background: rgba(225, 138, 90, 0.1) !important;
    border-radius: 18px !important;
}

.ocr-hero {
    position: relative;
    overflow: hidden;
    min-height: 390px;
    padding: clamp(30px, 6vw, 72px);
    margin-bottom: 18px;
    border: 1px solid var(--ocr-border);
    border-radius: 34px;
    background:
        linear-gradient(120deg, rgba(255,255,255,0.28), rgba(255,255,255,0.04)),
        radial-gradient(circle at 92% 8%, rgba(201, 99, 66, 0.15), transparent 35%);
    box-shadow: 0 28px 90px var(--ocr-shadow), inset 0 1px 0 rgba(255,255,255,0.14);
}

.ocr-hero::after {
    content: 'OCR / DOCQA / HISTORY';
    position: absolute;
    right: clamp(20px, 5vw, 58px);
    bottom: clamp(18px, 4vw, 44px);
    max-width: 260px;
    color: rgba(112, 103, 93, 0.42);
    font-weight: 850;
    letter-spacing: 0.16em;
    line-height: 1.45;
    text-align: right;
}

.ocr-kicker {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    color: var(--ocr-accent) !important;
    background: rgba(225, 138, 90, 0.12);
    border: 1px solid rgba(225, 138, 90, 0.24);
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.75rem;
}

.ocr-subtitle {
    max-width: 710px;
    font-size: clamp(1rem, 1.7vw, 1.22rem);
    line-height: 1.65;
    color: var(--ocr-muted) !important;
}

.ocr-hero h1 {
    max-width: 860px;
    font-size: clamp(3rem, 7.2vw, 7rem) !important;
    line-height: 0.88 !important;
    margin: 0.48rem 0 1.05rem !important;
}

.ocr-accent-text {
    color: var(--ocr-accent) !important;
}

.ocr-chip-row,
.ocr-card-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 22px;
}

.ocr-chip {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--ocr-border);
    background: var(--ocr-panel);
    color: var(--ocr-muted) !important;
    font-weight: 750;
}

.ocr-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin: 18px 0;
}

.ocr-card,
.ocr-composer-note {
    padding: 20px;
    border-radius: 26px;
    border: 1px solid var(--ocr-border);
    background: var(--ocr-panel);
    box-shadow: 0 16px 46px var(--ocr-shadow);
}

.ocr-section-title {
    margin: 0 0 10px;
    color: var(--ocr-muted) !important;
    font-weight: 850;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.76rem;
}

@media (max-width: 760px) {
    .ocr-hero { min-height: auto; }
    .ocr-hero::after { display: none; }
}

@media (prefers-reduced-motion: no-preference) {
    .gradio-container .tabitem,
    .gradio-container .block,
    .ocr-hero {
        animation: ocrRise 520ms ease both;
    }
    @keyframes ocrRise {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
}
"""

APP_JS = r"""
() => {
    const root = document.documentElement;
    const update = (event) => {
        const x = `${event.clientX}px`;
        const y = `${event.clientY}px`;
        root.style.setProperty('--ocr-cursor-x', x);
        root.style.setProperty('--ocr-cursor-y', y);
    };
    window.addEventListener('pointermove', update, { passive: true });
    return 'ok';
}
"""


def available_models() -> list[str]:
    return MODELS


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_choice(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def _safe_status_error(exc: Exception) -> str:
    return f"Error: {type(exc).__name__}: {exc}"


def save_upload_image(image: Image.Image | None, prefix: str = "upload") -> Path:
    if image is None:
        raise gr.Error("Please upload an image first.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{prefix}_{int(time.time() * 1000)}.png"
    image.convert("RGB").save(path)
    return path


def save_file_image(file_obj: Any, prefix: str = "page") -> tuple[Path, Image.Image]:
    if file_obj is None:
        raise gr.Error("Please upload one or more image files first.")
    source = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    image = Image.open(source).convert("RGB")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{prefix}_{int(time.time() * 1000)}_{safe_name(source.stem)}.png"
    image.save(path)
    return path, image


def get_pipeline_cached(
    model_type: str,
    threads: int | None,
    cpu_percent: float | None,
    quantize_mode: str,
    skip_layout: bool,
    full_page_mode: str,
    auto_runtime: str,
) -> LayoutAwareOCRPipeline:
    key = (model_type, threads, cpu_percent, quantize_mode, skip_layout, full_page_mode, auto_runtime)
    if key not in PIPELINE_CACHE:
        PIPELINE_CACHE[key] = LayoutAwareOCRPipeline(
            model_type=model_type,
            num_threads=threads,
            cpu_percent=cpu_percent,
            quantize_mode=quantize_mode,
            skip_layout=skip_layout,
            full_page_mode=full_page_mode,
            auto_runtime=auto_runtime,
        )
    return PIPELINE_CACHE[key]


def get_backend_cached(
    model_type: str,
    quantize_mode: str,
    auto_runtime: str,
    paddle_table_prompt: str,
) -> Any:
    key = (model_type, quantize_mode, auto_runtime, paddle_table_prompt)
    if key not in BACKEND_CACHE:
        BACKEND_CACHE[key] = get_vlm_backend(
            model_type,
            device="cpu",
            quantize_mode=quantize_mode,
            auto_runtime=auto_runtime,
            paddle_table_prompt=paddle_table_prompt,
        )
    return BACKEND_CACHE[key]


def rows_from_results(results: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for idx, row in enumerate(results, start=1):
        rows.append([
            idx,
            row.get("category", ""),
            row.get("bbox", ""),
            row.get("score", ""),
            row.get("text", ""),
        ])
    return rows


def run_single_image_ocr(
    image: Image.Image | None,
    model_type: str,
    threads: int | None,
    cpu_percent: float | None,
    quantize_mode: str,
    auto_runtime: str,
    layout_threshold: float,
    skip_layout: bool,
    full_page_mode: str,
    save_crops: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, list[list[Any]], str, str]:
    try:
        image_path = save_upload_image(image, "single")
        progress(0.05, desc="Loading OCR pipeline")
        pipeline = get_pipeline_cached(
            model_type,
            _optional_int(threads),
            _optional_float(cpu_percent),
            quantize_mode,
            skip_layout,
            full_page_mode,
            auto_runtime,
        )
        progress(0.2, desc="Running layout-aware OCR")
        crops_dir = str(EXPORT_DIR / "crops") if save_crops else None
        markdown, results = pipeline.parse(str(image_path), layout_threshold=float(layout_threshold), save_crops_dir=crops_dir)
        html = pipeline.generate_html(results, model_type, image_path.name)
        timings = json.dumps(getattr(pipeline, "last_timings", {}), ensure_ascii=False, indent=2)
        status = f"Done. regions={len(results)} image={image_path}"
        return markdown, html, rows_from_results(results), timings, status
    except Exception as exc:
        return "", "", [], "{}", _safe_status_error(exc)


def run_multipage_ocr_docqa(
    files: list[Any] | None,
    model_type: str,
    mode: str,
    category: str,
    question: str,
    single_call: str,
    max_new_tokens: int,
    quantize_mode: str,
    auto_runtime: str,
    paddle_table_prompt: str,
    save_history: bool,
    db_path: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str, str]:
    try:
        if not files:
            raise gr.Error("Please upload one or more page images.")
        progress(0.05, desc="Loading pages")
        page_paths = []
        images = []
        for idx, file_obj in enumerate(files, start=1):
            path, image = save_file_image(file_obj, f"multi_p{idx}")
            page_paths.append(path)
            images.append(image)
        progress(0.15, desc="Loading backend")
        backend = get_backend_cached(model_type, quantize_mode, auto_runtime, paddle_table_prompt)
        use_single = backend.supports_multi_image_single_call() if single_call == "auto" else single_call == "true"
        started = time.perf_counter()
        page_texts: list[str] = []
        answer = ""
        if mode in ("OCR pages", "OCR + DocQA"):
            progress(0.35, desc="Running page OCR")
            if use_single:
                page_texts = backend.generate_ocr_pages(images, category=category)
            else:
                page_texts = [backend.generate_ocr(image, category=category) for image in images]
        if mode in ("DocQA", "OCR + DocQA"):
            progress(0.7, desc="Running DocQA")
            if use_single:
                answer = backend.docqa_pages(images, question, max_new_tokens=int(max_new_tokens))
            else:
                if not page_texts:
                    page_texts = [backend.generate_ocr(image, category=category) for image in images]
                answer = "\n\n".join(f"Page {idx}: {text}" for idx, text in enumerate(page_texts, start=1))
        elapsed = time.perf_counter() - started
        run_id = ""
        document_id = ""
        if save_history:
            db = HistoryDB(Path(db_path))
            try:
                pages = []
                for idx, (path, image) in enumerate(zip(page_paths, images, strict=True), start=1):
                    pages.append({
                        "page_number": idx,
                        "image_sha256": image_sha256(image),
                        "image_path": str(path),
                        "width": image.width,
                        "height": image.height,
                    })
                document_id = db.upsert_document(pages, source_path="ui_upload", metadata={"source": "app_ui"})
                runtime_config = {
                    "mode": mode,
                    "single_call": use_single,
                    "category": category,
                    "auto_runtime": auto_runtime,
                    "quantize_mode": quantize_mode,
                    "paddle_table_prompt": paddle_table_prompt,
                    "max_new_tokens": max_new_tokens,
                }
                if page_texts:
                    run_id = db.add_ocr_run(document_id, model_type, str(getattr(backend, "model_id", model_type)), category, runtime_config, page_texts, elapsed)
                if answer:
                    db.add_qa_run(document_id, run_id or None, question, answer, [], {}, elapsed)
            finally:
                db.close()
        pages_text = "\n\n".join(f"Page {idx}: {text}" for idx, text in enumerate(page_texts, start=1))
        meta = json.dumps({
            "seconds": elapsed,
            "single_call": use_single,
            "pages": len(images),
            "document_id": document_id,
            "run_id": run_id,
        }, ensure_ascii=False, indent=2)
        return pages_text, answer, meta, "Done."
    except Exception as exc:
        return "", "", "{}", _safe_status_error(exc)


def history_documents(db_path: str, limit: int) -> tuple[list[list[Any]], str]:
    try:
        db = HistoryDB(Path(db_path))
        try:
            docs = db.list_documents(int(limit))
        finally:
            db.close()
        rows = [[doc["document_id"], doc["page_count"], doc["updated_at"], doc.get("source_path") or ""] for doc in docs]
        choices = [row[0] for row in rows]
        return rows, "\n".join(choices)
    except Exception as exc:
        return [], _safe_status_error(exc)


def history_document_detail(db_path: str, document_id: str) -> tuple[str, list[list[Any]]]:
    try:
        if not document_id:
            return "Select or enter a document id.", []
        db = HistoryDB(Path(db_path))
        try:
            doc = db.get_document(document_id)
        finally:
            db.close()
        if doc is None:
            return f"Document not found: {document_id}", []
        page_rows = [[p["page_number"], p["page_id"], f"{p.get('width')}x{p.get('height')}", p.get("image_path") or ""] for p in doc["pages"]]
        return json.dumps({k: v for k, v in doc.items() if k != "pages"}, ensure_ascii=False, indent=2), page_rows
    except Exception as exc:
        return _safe_status_error(exc), []


def history_runs(db_path: str, document_id: str) -> list[list[Any]]:
    try:
        if not document_id:
            return []
        db = HistoryDB(Path(db_path))
        try:
            runs = db.list_runs(document_id)
        finally:
            db.close()
        return [[run["run_id"], run["backend"], run["model_id"], run["category"], run["seconds"], run["created_at"]] for run in runs]
    except Exception:
        return []


def history_export_text(db_path: str, run_id: str) -> tuple[str, str]:
    try:
        if not run_id:
            raise gr.Error("Please enter a run id.")
        db = HistoryDB(Path(db_path))
        try:
            text = db.export_text(run_id)
        finally:
            db.close()
        return text, "Exported."
    except Exception as exc:
        return "", _safe_status_error(exc)


def history_delete_document(db_path: str, document_id: str) -> str:
    try:
        if not document_id:
            raise gr.Error("Please enter a document id.")
        db = HistoryDB(Path(db_path))
        try:
            db.delete_document(document_id)
        finally:
            db.close()
        return f"Deleted {document_id}."
    except Exception as exc:
        return _safe_status_error(exc)


def _extract_args(db_path: str, quantize_mode: str, auto_runtime: str, paddle_table_prompt: str) -> SimpleNamespace:
    return SimpleNamespace(
        db=Path(db_path),
        quantize_mode=quantize_mode,
        auto_runtime=auto_runtime,
        paddle_table_prompt=paddle_table_prompt,
    )


def extract_history_document(
    db_path: str,
    document_id: str,
    model_type: str,
    category: str,
    quantize_mode: str,
    auto_runtime: str,
    paddle_table_prompt: str,
    use_visual_cache: bool,
    save_history: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    try:
        if not document_id:
            raise gr.Error("Please enter a document id.")
        db = HistoryDB(Path(db_path))
        try:
            doc = db.get_document(document_id)
            if doc is None:
                raise gr.Error(f"Document not found: {document_id}")
            progress(0.1, desc="Loading document pages")
            images = load_document_images(doc)
            progress(0.2, desc="Loading backend")
            backend = get_backend_cached(model_type, quantize_mode, auto_runtime, paddle_table_prompt)
            backend_name = model_type
            model_id = str(getattr(backend, "model_id", model_type))
            helper_args = _extract_args(db_path, quantize_mode, auto_runtime, paddle_table_prompt)
            runtime_config = visual_cache_runtime_config(helper_args)
            runtime_config_hash = runtime_hash(runtime_config)
            proc_hash = processor_config_hash(backend)
            prompt_hash = visual_cache_prompt_hash(category)
            cache_hits = 0
            page_texts = []
            started = time.perf_counter()
            for idx, (image, page) in enumerate(zip(images, doc["pages"], strict=True), start=1):
                progress(0.25 + 0.65 * (idx - 1) / max(1, len(images)), desc=f"Extracting page {idx}")
                cache = None
                if use_visual_cache and backend.supports_visual_token_cache():
                    cached = db.find_embed_cache(page["page_id"], backend_name, model_id, proc_hash, runtime_config_hash, prompt_hash)
                    if cached is not None and Path(cached["tensor_path"]).exists():
                        cache = torch.load(cached["tensor_path"], map_location="cpu", weights_only=False)
                        cache_hits += 1
                    else:
                        cache = backend.build_visual_cache(image, category=category)
                        embed_id = db.upsert_embed_cache(
                            page["page_id"], backend_name, model_id, proc_hash, runtime_config_hash, prompt_hash,
                            "", {"kind": "processor_outputs_v1"}, "torch", "cpu"
                        )
                        path = tensor_cache_path(helper_args, backend_name, model_id, embed_id)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(cache, path)
                        db.upsert_embed_cache(
                            page["page_id"], backend_name, model_id, proc_hash, runtime_config_hash, prompt_hash,
                            str(path), {"kind": "processor_outputs_v1"}, "torch", "cpu"
                        )
                if cache is not None:
                    text = backend.generate_ocr_from_visual_cache(cache, category=category)
                else:
                    text = backend.generate_ocr(image, category=category)
                page_texts.append(text)
            elapsed = time.perf_counter() - started
            run_id = ""
            if save_history:
                run_id = db.add_ocr_run(document_id, backend_name, model_id, category, runtime_config, page_texts, elapsed)
        finally:
            db.close()
        output_text = "\n\n".join(f"Page {idx}: {text}" for idx, text in enumerate(page_texts, start=1))
        meta = json.dumps({
            "seconds": elapsed,
            "visual_cache_hits": cache_hits,
            "pages": len(page_texts),
            "run_id": run_id,
            "visual_cache_supported": bool(backend.supports_visual_token_cache()),
        }, ensure_ascii=False, indent=2)
        return output_text, meta, "Done."
    except Exception as exc:
        return "", "{}", _safe_status_error(exc)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Flexible CPU OCR") as demo:
        gr.HTML(
            """
            <section class="ocr-hero">
              <div class="ocr-kicker">Claude calm × AirCenter editorial</div>
              <h1>OCR workspace for documents.</h1>
              <p class="ocr-subtitle">
                Layout-aware OCR, multi-page DocQA, durable history, and visual-cache workflows in an independent child app. The interface keeps Claude-like calm spacing with premium editorial rhythm.
              </p>
              <div class="ocr-chip-row">
                <span class="ocr-chip">CPU-first runtime</span>
                <span class="ocr-chip">History DB</span>
                <span class="ocr-chip">Visual processor cache</span>
                <span class="ocr-chip">Independent child UI</span>
              </div>
            </section>
            """
        )

        with gr.Tab("Single Image OCR"):
            gr.HTML("""
            <div class="ocr-composer-note">
              <div class="ocr-section-title">Primary composer</div>
              <h3>Upload one document image and run layout-aware OCR.</h3>
              <p>Controls stay close to the action, while output panels remain ready for markdown, HTML, regions, and timing diagnostics.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    single_image = gr.Image(type="pil", label="Document image")
                    single_model = gr.Dropdown(available_models(), value="falcon-ocr", label="Model")
                    single_threads = gr.Number(value=None, precision=0, label="Threads")
                    single_cpu_percent = gr.Number(value=None, label="CPU percent")
                    single_quant = gr.Dropdown(QUANTIZE_MODES, value="selective", label="Quantize mode")
                    single_runtime = gr.Dropdown(RUNTIME_MODES, value="conservative", label="Auto runtime")
                    single_threshold = gr.Slider(0.05, 0.95, value=0.3, step=0.05, label="Layout threshold")
                    single_skip_layout = gr.Checkbox(value=False, label="Skip layout / full-page backend mode")
                    single_full_page = gr.Radio(["layout", "svg"], value="layout", label="Full-page mode")
                    single_save_crops = gr.Checkbox(value=False, label="Save table/formula crops")
                    single_run = gr.Button("Run OCR", variant="primary")
                with gr.Column(scale=2):
                    single_status = gr.Textbox(label="Status")
                    single_markdown = gr.Textbox(label="Markdown", lines=14)
                    single_html = gr.HTML(label="HTML preview")
                    single_regions = gr.Dataframe(headers=["#", "Category", "BBox", "Score", "Text"], label="Regions")
                    single_timings = gr.Code(label="Timings", language="json")
            single_run.click(
                run_single_image_ocr,
                inputs=[single_image, single_model, single_threads, single_cpu_percent, single_quant, single_runtime, single_threshold, single_skip_layout, single_full_page, single_save_crops],
                outputs=[single_markdown, single_html, single_regions, single_timings, single_status],
            )

        with gr.Tab("Multi-page / DocQA"):
            gr.HTML("""
            <div class="ocr-composer-note">
              <div class="ocr-section-title">Document set</div>
              <h3>Ask questions across multiple pages.</h3>
              <p>Upload a page set, choose OCR/DocQA mode, and keep history capture available for later extraction.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    multi_files = gr.Files(label="Page images", file_types=["image"])
                    multi_model = gr.Dropdown(available_models(), value="paddleocr-vl", label="Model")
                    multi_mode = gr.Radio(["OCR pages", "DocQA", "OCR + DocQA"], value="OCR + DocQA", label="Mode")
                    multi_category = gr.Dropdown(CATEGORIES, value="plain", label="OCR category")
                    multi_question = gr.Textbox(value="What are the key facts across these pages?", label="Question")
                    multi_single_call = gr.Radio(["auto", "true", "false"], value="auto", label="Single call")
                    multi_max_tokens = gr.Number(value=512, precision=0, label="Max new tokens")
                    multi_quant = gr.Dropdown(QUANTIZE_MODES, value="auto", label="Quantize mode")
                    multi_runtime = gr.Dropdown(RUNTIME_MODES, value="conservative", label="Auto runtime")
                    multi_paddle_prompt = gr.Radio(["fast", "official"], value="fast", label="Paddle table prompt")
                    multi_save_history = gr.Checkbox(value=True, label="Save to history DB")
                    multi_db = gr.Textbox(value=str(DEFAULT_HISTORY_DB), label="History DB")
                    multi_run = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    multi_status = gr.Textbox(label="Status")
                    multi_pages = gr.Textbox(label="OCR pages", lines=14)
                    multi_answer = gr.Textbox(label="DocQA answer", lines=8)
                    multi_meta = gr.Code(label="Metadata", language="json")
            multi_run.click(
                run_multipage_ocr_docqa,
                inputs=[multi_files, multi_model, multi_mode, multi_category, multi_question, multi_single_call, multi_max_tokens, multi_quant, multi_runtime, multi_paddle_prompt, multi_save_history, multi_db],
                outputs=[multi_pages, multi_answer, multi_meta, multi_status],
            )

        with gr.Tab("History"):
            gr.HTML("""
            <div class="ocr-card-grid">
              <article class="ocr-card"><div class="ocr-section-title">History</div><h3>Durable document memory</h3><p>Browse documents, pages, OCR runs, and exported text from the SQLite history database.</p></article>
              <article class="ocr-card"><div class="ocr-section-title">Separation</div><h3>History is not cache</h3><p>History stores records and outputs; visual processor cache remains a separate acceleration layer.</p></article>
            </div>
            """)
            with gr.Row():
                hist_db = gr.Textbox(value=str(DEFAULT_HISTORY_DB), label="History DB", elem_id="history-db-input")
                hist_limit = gr.Number(value=50, precision=0, label="Limit")
                hist_refresh = gr.Button("Refresh")
            hist_docs = gr.Dataframe(headers=["Document ID", "Pages", "Updated", "Source"], label="Documents")
            hist_doc_choices = gr.Textbox(label="Document IDs from refresh")
            hist_doc_id = gr.Textbox(label="Selected document id", elem_id="history-document-id")
            with gr.Row():
                hist_show = gr.Button("Show document")
                hist_list_runs = gr.Button("List runs")
                hist_delete = gr.Button("Delete document", variant="stop")
            hist_detail = gr.Code(label="Document metadata", language="json")
            hist_pages = gr.Dataframe(headers=["Page", "Page ID", "Size", "Path"], label="Pages")
            hist_runs_table = gr.Dataframe(headers=["Run ID", "Backend", "Model", "Category", "Seconds", "Created"], label="Runs")
            hist_run_id = gr.Textbox(label="Run id to export")
            hist_export = gr.Button("Export run text")
            hist_text = gr.Textbox(label="Exported text", lines=14)
            hist_status = gr.Textbox(label="Status")
            hist_refresh.click(history_documents, inputs=[hist_db, hist_limit], outputs=[hist_docs, hist_doc_choices])
            hist_show.click(history_document_detail, inputs=[hist_db, hist_doc_id], outputs=[hist_detail, hist_pages])
            hist_list_runs.click(history_runs, inputs=[hist_db, hist_doc_id], outputs=[hist_runs_table])
            hist_export.click(history_export_text, inputs=[hist_db, hist_run_id], outputs=[hist_text, hist_status])
            hist_delete.click(history_delete_document, inputs=[hist_db, hist_doc_id], outputs=[hist_status])

        with gr.Tab("Extract from History / Visual Cache"):
            gr.HTML("""
            <div class="ocr-composer-note">
              <div class="ocr-section-title">Extraction</div>
              <h3>Reuse stored documents and optional visual processor cache.</h3>
              <p>Pick a history document, choose backend settings, then extract text while reporting cache support and hits.</p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    extract_db = gr.Textbox(value=str(DEFAULT_HISTORY_DB), label="History DB")
                    extract_doc_id = gr.Textbox(label="Document id")
                    extract_model = gr.Dropdown(available_models(), value="paddleocr-vl", label="Model")
                    extract_category = gr.Dropdown(CATEGORIES, value="plain", label="Category")
                    extract_quant = gr.Dropdown(QUANTIZE_MODES, value="selective", label="Quantize mode")
                    extract_runtime = gr.Dropdown(RUNTIME_MODES, value="off", label="Auto runtime")
                    extract_paddle_prompt = gr.Radio(["fast", "official"], value="fast", label="Paddle table prompt")
                    extract_visual_cache = gr.Checkbox(value=True, label="Use visual processor-output cache")
                    extract_save_history = gr.Checkbox(value=True, label="Save OCR run to history")
                    extract_run = gr.Button("Extract", variant="primary")
                with gr.Column(scale=2):
                    extract_status = gr.Textbox(label="Status")
                    extract_text = gr.Textbox(label="Extracted text", lines=18)
                    extract_meta = gr.Code(label="Metadata", language="json")
            extract_run.click(
                extract_history_document,
                inputs=[extract_db, extract_doc_id, extract_model, extract_category, extract_quant, extract_runtime, extract_paddle_prompt, extract_visual_cache, extract_save_history],
                outputs=[extract_text, extract_meta, extract_status],
            )

        with gr.Tab("About"):
            gr.HTML("""
            <div class="ocr-card-grid">
              <article class="ocr-card"><div class="ocr-section-title">Runtime</div><h3>CPU-first OCR APIs</h3><p>This app calls the existing Python OCR APIs directly without changing backend behavior.</p></article>
              <article class="ocr-card"><div class="ocr-section-title">History</div><h3>Durable SQLite records</h3><p>History DB stores metadata, OCR runs, QA runs, page references, and exported text.</p></article>
              <article class="ocr-card"><div class="ocr-section-title">Cache</div><h3>Processor-output tensors</h3><p>Visual cache currently means PaddleOCR-VL processor-output tensor cache, not hidden KV cache.</p></article>
            </div>
            """)
            with gr.Accordion("Operational constraints", open=False):
                gr.Markdown(
                    """
                    - OCR JSON cache and history DB are separate concepts.
                    - The UI does not change resize policy, image token sizing, or crop geometry.
                    - Child OCR remains independently runnable under `OptimAIze-OCR`.
                    """
                )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Flexible CPU OCR Gradio UI.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", choices=("true", "false"), default="false")
    parser.add_argument("--inbrowser", choices=("true", "false"), default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share == "true",
        inbrowser=args.inbrowser == "true",
        theme=gr.themes.Soft(),
        css=APP_CSS,
        js=APP_JS,
    )


if __name__ == "__main__":
    main()
