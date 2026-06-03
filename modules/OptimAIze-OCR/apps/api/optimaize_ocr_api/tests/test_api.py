from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

OCR_MODULE_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = OCR_MODULE_ROOT / "apps" / "api"
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for path in (BACKEND_ROOT, OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimaize_ocr.storage.history_db import HistoryDB, image_sha256
from optimaize_ocr_api.core import config
from optimaize_ocr_api.main import app
from optimaize_ocr_api.services.ai import ocr_service


def test_health_and_models(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTIMAIZE_OCR_HISTORY_DB", str(tmp_path / "history.sqlite"))
    client = TestClient(app)

    health = client.get("/api/v1/health")
    models = client.get("/api/v1/ocr/models")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert models.status_code == 200
    assert "falcon-ocr" in models.json()["models"]


def test_history_documents_endpoint(monkeypatch, tmp_path):
    db_path = tmp_path / "history.sqlite"
    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (12, 10), "white")
    image.save(image_path)
    db = HistoryDB(db_path)
    db.upsert_document([
        {
            "page_number": 1,
            "image_sha256": image_sha256(image),
            "image_path": str(image_path),
            "width": image.width,
            "height": image.height,
        }
    ], source_path="pytest", metadata={"source": "api-test"})
    db.close()

    monkeypatch.setenv("OPTIMAIZE_OCR_HISTORY_DB", str(db_path))
    client = TestClient(app)
    response = client.get("/api/v1/history/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["page_count"] == 1


def test_single_image_endpoint_uses_service_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("OPTIMAIZE_OCR_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OPTIMAIZE_OCR_OUTPUT_DIR", str(tmp_path / "outputs"))

    class DummyPipeline:
        last_timings = {"overall_time": 0.01}

        def parse(self, image_path, layout_threshold=0.3, save_crops_dir=None):
            return "# OCR", [{"category": "text", "bbox": [1, 2, 3, 4], "score": 0.9, "text": "hello"}]

        def generate_html(self, results, model_type, image_name):
            return f"<p>{model_type}:{image_name}:{len(results)}</p>"

    monkeypatch.setattr(ocr_service, "get_pipeline_cached", lambda *args: DummyPipeline())
    image_path = tmp_path / "upload.png"
    Image.new("RGB", (16, 12), "white").save(image_path)

    client = TestClient(app)
    with image_path.open("rb") as file_obj:
        response = client.post(
            "/api/v1/ocr/single-image",
            files={"image": ("upload.png", file_obj, "image/png")},
            data={"model_type": "falcon-ocr", "layout_threshold": "0.4"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"] == "# OCR"
    assert payload["regions"][0]["text"] == "hello"
    assert payload["timings"]["overall_time"] == 0.01
