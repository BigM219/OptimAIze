from __future__ import annotations

from optimaize.modules import ocr_bridge


def ocr_status() -> dict[str, object]:
    status = ocr_bridge.child_status()
    return {
        "id": "ocr",
        "name": "OptimAIze-OCR",
        "kind": "child-module",
        "available": status.exists and status.source_available,
        "source_available": status.source_available,
        "ui_available": status.ui_available,
        "api_available": (ocr_bridge.OCR_API_DIR / "optimaize_ocr_api" / "main.py").exists(),
        "web_available": (ocr_bridge.OCR_WEB_DIR / "package.json").exists(),
        "path": status.child_path,
        "message": status.message,
        "web_url": "http://127.0.0.1:5173",
        "api_url": "http://127.0.0.1:8001",
    }


def launch_legacy_ui(server_port: int, share: bool, inbrowser: bool) -> dict[str, object]:
    return ocr_bridge.launch_child_ui(server_port=server_port, share=share, inbrowser=inbrowser)
