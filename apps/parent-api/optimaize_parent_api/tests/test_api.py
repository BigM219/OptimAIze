from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = PROJECT_ROOT / "apps" / "parent-api"
PARENT_CORE = PROJECT_ROOT / "packages" / "parent-core"
for path in (BACKEND_ROOT, PARENT_CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimaize_parent_api.main import app


def test_health_and_modules() -> None:
    client = TestClient(app)

    health = client.get("/api/v1/health")
    modules = client.get("/api/v1/modules")
    ocr = client.get("/api/v1/modules/ocr/status")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert modules.status_code == 200
    assert modules.json()["modules"][0]["id"] == "ocr"
    assert ocr.status_code == 200
    assert ocr.json()["source_available"] is True


def test_launch_ocr_ui_uses_bridge(monkeypatch) -> None:
    from optimaize_parent_api.services.modules import ocr_module_service

    def fake_launch_child_ui(server_port: int, share: bool, inbrowser: bool):
        return {"ok": True, "pid": 123, "url": f"http://127.0.0.1:{server_port}", "message": "started"}

    monkeypatch.setattr(ocr_module_service.ocr_bridge, "launch_child_ui", fake_launch_child_ui)
    client = TestClient(app)
    response = client.post("/api/v1/modules/ocr/launch-ui", json={"server_port": 7862})

    assert response.status_code == 200
    assert response.json()["pid"] == 123
    assert response.json()["url"] == "http://127.0.0.1:7862"
