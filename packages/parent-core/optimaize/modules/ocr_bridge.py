from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

def _discover_project_root() -> Path:
    """Locate the OptimAIze workspace root.

    Prefers the ``OPTIMAIZE_PROJECT_ROOT`` env var (robust to file moves), then
    falls back to walking up until a directory containing ``modules`` is found,
    and finally to the historical fixed depth. The fixed-depth form alone broke
    silently whenever this file moved, so it is now the last resort.
    """
    configured = os.getenv("OPTIMAIZE_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "modules" / "OptimAIze-OCR").is_dir():
            return parent
    return here.parents[4]


PROJECT_ROOT = _discover_project_root()
OCR_CHILD_DIR = PROJECT_ROOT / "modules" / "OptimAIze-OCR"
OCR_CHILD_SRC = OCR_CHILD_DIR / "src"
OCR_API_DIR = OCR_CHILD_DIR / "apps" / "api"
OCR_WEB_DIR = OCR_CHILD_DIR / "apps" / "web"
OCR_LEGACY_UI = OCR_CHILD_DIR / "legacy" / "gradio" / "app_ui.py"
OCR_HISTORY_CLI = OCR_CHILD_DIR / "legacy" / "cli" / "ocr_history.py"


@dataclass(frozen=True)
class OCRChildStatus:
    exists: bool
    child_path: str
    source_available: bool
    ui_available: bool
    history_available: bool
    message: str


def child_status() -> OCRChildStatus:
    source_available = (OCR_CHILD_SRC / "optimaize_ocr" / "__init__.py").exists()
    ui_available = OCR_LEGACY_UI.exists()
    history_available = OCR_HISTORY_CLI.exists()
    exists = OCR_CHILD_DIR.exists()
    if exists and source_available and ui_available:
        message = "OptimAIze-OCR is available and can run independently."
    elif exists:
        message = "OptimAIze-OCR exists, but some expected OCR files are missing."
    else:
        message = "OptimAIze-OCR child project was not found."
    return OCRChildStatus(
        exists=exists,
        child_path=str(OCR_CHILD_DIR),
        source_available=source_available,
        ui_available=ui_available,
        history_available=history_available,
        message=message,
    )


def child_status_json() -> str:
    return json.dumps(asdict(child_status()), indent=2, ensure_ascii=False)


@contextlib.contextmanager
def child_import_path() -> Iterator[None]:
    paths = [str(OCR_CHILD_SRC), str(OCR_CHILD_DIR), str(OCR_API_DIR)]
    inserted = []
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    try:
        yield
    finally:
        for path in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def load_child_ui_factory():
    if not OCR_LEGACY_UI.exists():
        raise FileNotFoundError("OptimAIze-OCR UI entrypoint is missing")
    with child_import_path():
        import importlib.util

        spec = importlib.util.spec_from_file_location("optimaize_ocr_legacy_app_ui", OCR_LEGACY_UI)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load OptimAIze-OCR legacy UI")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.create_app


def launch_child_ui(server_port: int = 7860, share: bool = False, inbrowser: bool = False) -> dict[str, object]:
    status = child_status()
    if not status.exists or not status.ui_available:
        return {"ok": False, "pid": None, "url": None, "message": status.message}

    cmd = [
        sys.executable,
        str(OCR_LEGACY_UI),
        "--server-port",
        str(server_port),
        "--share",
        "true" if share else "false",
        "--inbrowser",
        "true" if inbrowser else "false",
    ]
    process = subprocess.Popen(
        cmd,
        cwd=OCR_CHILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "ok": True,
        "pid": process.pid,
        "url": f"http://127.0.0.1:{server_port}",
        "message": f"Started OptimAIze-OCR UI on port {server_port}.",
    }


def run_single_image_ocr(
    image_path: str,
    model: str = "falcon-ocr",
    threshold: float = 0.3,
    output_dir: str | None = None,
) -> dict[str, object]:
    status = child_status()
    if not status.exists or not status.source_available:
        return {"ok": False, "markdown": "", "html": "", "results": [], "message": status.message}

    src = Path(image_path)
    if not src.exists():
        return {"ok": False, "markdown": "", "html": "", "results": [], "message": f"Image not found: {src}"}

    out_dir = Path(output_dir) if output_dir else OCR_CHILD_DIR / "outputs" / "parent_calls" / str(int(time.time()))
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    with child_import_path():
        from optimaize_ocr.core.pipeline import LayoutAwareOCRPipeline

        pipeline = LayoutAwareOCRPipeline(model_type=model, device="cpu")
        markdown, results = pipeline.parse(
            image_path=str(src),
            layout_threshold=threshold,
            save_crops_dir=str(crops_dir),
        )
        html = pipeline.generate_html(results=results, model_type=model, image_name=src.name)

    (out_dir / "parsed_document.md").write_text(markdown, encoding="utf-8")
    (out_dir / "parsed_document.html").write_text(html, encoding="utf-8")
    return {
        "ok": True,
        "markdown": markdown,
        "html": html,
        "results": results,
        "output_dir": str(out_dir),
        "message": f"OCR completed with {len(results)} regions.",
    }


def save_parent_upload(image: Image.Image | None) -> Path | None:
    if image is None:
        return None
    upload_dir = OCR_CHILD_DIR / "outputs" / "parent_calls" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"parent_upload_{int(time.time() * 1000)}.png"
    image.convert("RGB").save(path)
    return path
