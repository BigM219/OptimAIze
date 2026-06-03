import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

OCR_MODULE_ROOT = Path(__file__).resolve().parents[1]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for path in (OCR_MODULE_ROOT, OCR_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from legacy.gradio import app_ui
from optimaize_ocr.storage.history_db import HistoryDB, image_sha256


class DummyBackend:
    model_id = "dummy-model"

    def __init__(self, multi=True, visual=False):
        self.multi = multi
        self.visual = visual
        self.ocr_calls = 0
        self.visual_builds = 0
        self.visual_generations = 0

    def supports_multi_image_single_call(self):
        return self.multi

    def supports_visual_token_cache(self):
        return self.visual

    def generate_ocr(self, image, category="plain"):
        self.ocr_calls += 1
        return f"ocr-{category}-{image.size[0]}x{image.size[1]}"

    def generate_ocr_pages(self, images, category="plain"):
        return [f"multi-{idx}-{category}" for idx, _ in enumerate(images, start=1)]

    def docqa_pages(self, images, question, max_new_tokens=None):
        return f"answer:{question}:{len(images)}:{max_new_tokens}"

    def build_visual_cache(self, image, category="plain"):
        self.visual_builds += 1
        return {"category": category, "size": image.size}

    def generate_ocr_from_visual_cache(self, cache, category="plain"):
        self.visual_generations += 1
        return f"cached-{category}-{cache['size'][0]}x{cache['size'][1]}"


class DummyPipeline:
    def __init__(self):
        self.last_timings = {"overall_time": 1.25}
        self.parse_args = None

    def parse(self, image_path, layout_threshold=0.3, save_crops_dir=None):
        self.parse_args = (image_path, layout_threshold, save_crops_dir)
        return "# OCR", [{"category": "text", "bbox": [1, 2, 3, 4], "score": 0.9, "text": "hello"}]

    def generate_html(self, results, model_type, image_name):
        return f"<h1>{model_type}</h1><p>{image_name}</p>"


@pytest.fixture()
def sample_image(tmp_path):
    path = tmp_path / "page.png"
    image = Image.new("RGB", (32, 24), "white")
    image.save(path)
    return path, image


@pytest.fixture()
def history_db_with_doc(tmp_path, sample_image):
    image_path, image = sample_image
    db_path = tmp_path / "history.sqlite"
    db = HistoryDB(db_path)
    document_id = db.upsert_document([
        {
            "page_number": 1,
            "image_sha256": image_sha256(image),
            "image_path": str(image_path),
            "width": image.width,
            "height": image.height,
        }
    ], source_path="pytest", metadata={"source": "test"})
    run_id = db.add_ocr_run(document_id, "dummy", "dummy-model", "plain", {"k": "v"}, ["hello"], 0.5)
    db.close()
    return db_path, document_id, run_id


def test_rows_from_results_and_optional_parsers():
    rows = app_ui.rows_from_results([
        {"category": "table", "bbox": [1, 2, 3, 4], "score": 0.8, "text": "A"},
        {"text": "B"},
    ])
    assert rows == [
        [1, "table", [1, 2, 3, 4], 0.8, "A"],
        [2, "", "", "", "B"],
    ]
    assert app_ui._optional_int("") is None
    assert app_ui._optional_int("4") == 4
    assert app_ui._optional_float(None) is None
    assert app_ui._optional_float("12.5") == 12.5
    assert app_ui._bool_choice("auto") is None
    assert app_ui._bool_choice("true") is True
    assert app_ui._bool_choice("false") is False


def test_save_file_image_sanitizes_and_converts(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_ui, "UPLOAD_DIR", upload_dir)
    source = tmp_path / "name with spaces.png"
    Image.new("RGBA", (10, 12), "red").save(source)

    path, image = app_ui.save_file_image(SimpleNamespace(name=str(source)), "page")

    assert path.exists()
    assert path.parent == upload_dir
    assert "name_with_spaces" in path.name
    assert image.mode == "RGB"
    assert image.size == (10, 12)


def test_history_functions_round_trip(history_db_with_doc):
    db_path, document_id, run_id = history_db_with_doc

    docs, choices = app_ui.history_documents(str(db_path), 10)
    assert docs[0][0] == document_id
    assert document_id in choices

    detail, pages = app_ui.history_document_detail(str(db_path), document_id)
    parsed = json.loads(detail)
    assert parsed["document_id"] == document_id
    assert pages[0][1].endswith("_p0001")

    runs = app_ui.history_runs(str(db_path), document_id)
    assert runs[0][0] == run_id

    text, status = app_ui.history_export_text(str(db_path), run_id)
    assert "Page 1: hello" == text
    assert status == "Exported."


def test_history_delete_document(history_db_with_doc):
    db_path, document_id, _ = history_db_with_doc
    status = app_ui.history_delete_document(str(db_path), document_id)
    assert document_id in status
    docs, _ = app_ui.history_documents(str(db_path), 10)
    assert docs == []


def test_run_single_image_ocr_uses_pipeline_without_model_load(tmp_path, monkeypatch, sample_image):
    _, image = sample_image
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_ui, "UPLOAD_DIR", upload_dir)
    dummy = DummyPipeline()
    monkeypatch.setattr(app_ui, "get_pipeline_cached", lambda *args: dummy)

    markdown, html, rows, timings, status = app_ui.run_single_image_ocr(
        image, "falcon-ocr", None, None, "selective", "off", 0.4, False, "layout", False
    )

    assert markdown == "# OCR"
    assert "falcon-ocr" in html
    assert rows == [[1, "text", [1, 2, 3, 4], 0.9, "hello"]]
    assert json.loads(timings)["overall_time"] == 1.25
    assert "Done" in status
    assert dummy.parse_args[1] == 0.4


def test_run_multipage_ocr_docqa_single_call_and_history(tmp_path, monkeypatch, sample_image):
    image_path, _ = sample_image
    monkeypatch.setattr(app_ui, "UPLOAD_DIR", tmp_path / "uploads")
    backend = DummyBackend(multi=True)
    monkeypatch.setattr(app_ui, "get_backend_cached", lambda *args: backend)
    db_path = tmp_path / "history.sqlite"

    pages_text, answer, meta_json, status = app_ui.run_multipage_ocr_docqa(
        [SimpleNamespace(name=str(image_path))],
        "dummy",
        "OCR + DocQA",
        "plain",
        "question?",
        "auto",
        123,
        "none",
        "off",
        "fast",
        True,
        str(db_path),
    )

    meta = json.loads(meta_json)
    assert pages_text == "Page 1: multi-1-plain"
    assert answer == "answer:question?:1:123"
    assert meta["single_call"] is True
    assert meta["document_id"].startswith("doc_")
    assert meta["run_id"].startswith("ocr_")
    assert status == "Done."


def test_extract_history_document_builds_then_hits_visual_cache(tmp_path, monkeypatch, history_db_with_doc):
    db_path, document_id, _ = history_db_with_doc
    monkeypatch.setattr(app_ui, "get_backend_cached", lambda *args: DummyBackend(visual=True))

    text_1, meta_1_json, status_1 = app_ui.extract_history_document(
        str(db_path), document_id, "dummy", "plain", "none", "off", "fast", True, True
    )
    meta_1 = json.loads(meta_1_json)
    assert text_1.startswith("Page 1: cached-plain")
    assert meta_1["visual_cache_hits"] == 0
    assert meta_1["run_id"].startswith("ocr_")
    assert status_1 == "Done."

    text_2, meta_2_json, status_2 = app_ui.extract_history_document(
        str(db_path), document_id, "dummy", "plain", "none", "off", "fast", True, False
    )
    meta_2 = json.loads(meta_2_json)
    assert text_2 == text_1
    assert meta_2["visual_cache_hits"] == 1
    assert status_2 == "Done."


def test_create_app_constructs_blocks():
    app = app_ui.create_app()
    assert app.__class__.__name__ == "Blocks"


def test_app_ui_help_command():
    result = subprocess.run(
        [sys.executable, "legacy/gradio/app_ui.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=OCR_MODULE_ROOT,
    )
    assert "Launch the Flexible CPU OCR Gradio UI" in result.stdout
    assert "--server-port" in result.stdout
