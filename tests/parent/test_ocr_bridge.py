from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_CORE = PROJECT_ROOT / "packages" / "parent-core"
if str(PARENT_CORE) not in sys.path:
    sys.path.insert(0, str(PARENT_CORE))

from optimaize.modules import ocr_bridge
from optimaize.app_ui import create_app


def test_child_status_detects_ocr_project() -> None:
    status = ocr_bridge.child_status()
    assert status.exists
    assert status.source_available
    assert status.ui_available
    assert status.child_path.endswith("OptimAIze-OCR")


def test_parent_ui_constructs_without_loading_ocr_models() -> None:
    app = create_app()
    assert app is not None


def test_launch_child_ui_uses_child_working_directory(monkeypatch) -> None:
    calls = []

    class DummyProcess:
        pid = 12345

    def fake_popen(cmd, cwd, stdout, stderr, start_new_session):
        calls.append({
            "cmd": cmd,
            "cwd": cwd,
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": start_new_session,
        })
        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    result = ocr_bridge.launch_child_ui(server_port=7861)

    assert result["ok"] is True
    assert result["pid"] == 12345
    assert result["url"] == "http://127.0.0.1:7861"
    assert calls
    assert calls[0]["cwd"] == ocr_bridge.OCR_CHILD_DIR
    assert calls[0]["cmd"][1:] == [
        str(ocr_bridge.OCR_LEGACY_UI),
        "--server-port",
        "7861",
        "--share",
        "false",
        "--inbrowser",
        "false",
    ]
