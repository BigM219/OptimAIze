from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[1]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PIL import Image
from playwright.sync_api import sync_playwright

from optimaize_ocr.storage.history_db import HistoryDB, image_sha256

OUT = Path("outputs/ui/browser_tests")
OUT.mkdir(parents=True, exist_ok=True)
TEST_IMAGE = OUT / "browser_page.png"
TEST_DB = OUT / "browser_history.sqlite"
Image.new("RGB", (80, 60), "white").save(TEST_IMAGE)
image = Image.open(TEST_IMAGE).convert("RGB")
db = HistoryDB(TEST_DB)
document_id = db.upsert_document([
    {
        "page_number": 1,
        "image_sha256": image_sha256(image),
        "image_path": str(TEST_IMAGE),
        "width": image.width,
        "height": image.height,
    }
], source_path="browser_test", metadata={"source": "browser"})
run_id = db.add_ocr_run(document_id, "browser", "browser-model", "plain", {"source": "browser"}, ["browser text"], 0.1)
db.close()

console_messages = []
page_errors = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))

    page.goto("http://127.0.0.1:7860", wait_until="networkidle", timeout=120_000)
    page.screenshot(path=str(OUT / "01_home.png"), full_page=True)

    title = page.locator("text=Flexible CPU OCR").first
    assert title.is_visible(), "App title is not visible"

    expected_tabs = [
        "Single Image OCR",
        "Multi-page / DocQA",
        "History",
        "Extract from History / Visual Cache",
        "About",
    ]
    for tab in expected_tabs:
        assert page.get_by_role("tab", name=tab, exact=True).is_visible(), f"Missing tab: {tab}"

    page.get_by_role("tab", name="History", exact=True).click()
    page.wait_for_load_state("networkidle")
    page.screenshot(path=str(OUT / "02_history_tab.png"), full_page=True)
    assert page.locator("label", has_text="History DB").first.is_visible(), "History DB input missing"
    history_db_input = page.locator("textarea").nth(2)
    history_db_input.fill(str(TEST_DB))
    assert page.get_by_role("button", name="Refresh").is_visible(), "Refresh button missing"

    page.get_by_role("button", name="Refresh").click()
    page.wait_for_timeout(2500)
    page.screenshot(path=str(OUT / "03_history_refreshed.png"), full_page=True)
    body_text = page.locator("body").inner_text()
    assert "doc_" in body_text, "History refresh did not show any document ids"

    doc_id = next(line.strip() for line in body_text.splitlines() if line.strip().startswith("doc_"))
    doc_input = page.locator("textarea").nth(4)
    doc_input.fill(doc_id)
    page.get_by_role("button", name="Show document").click()
    page.wait_for_timeout(2000)
    page.screenshot(path=str(OUT / "04_history_detail.png"), full_page=True)
    assert doc_id in page.locator("body").inner_text(), "Document detail did not contain selected document id"

    page.get_by_role("button", name="List runs").click()
    page.wait_for_timeout(2000)
    page.screenshot(path=str(OUT / "05_history_runs.png"), full_page=True)
    runs_text = page.locator("body").inner_text()
    assert run_id in runs_text, "Runs table did not show OCR run ids"

    page.get_by_role("tab", name="Extract from History / Visual Cache", exact=True).click()
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "06_extract_tab.png"), full_page=True)
    assert page.locator("label", has_text="Use visual processor-output cache").first.is_visible(), "Visual cache toggle missing"

    page.get_by_role("tab", name="About", exact=True).click()
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "07_about_tab.png"), full_page=True)
    about_text = page.locator("body").inner_text()
    assert "processor-output tensor cache" in about_text, "About tab missing cache explanation"

    browser.close()

bad_console = [msg for msg in console_messages if msg.startswith("error:")]
(OUT / "console.log").write_text("\n".join(console_messages), encoding="utf-8")
(OUT / "page_errors.log").write_text("\n".join(page_errors), encoding="utf-8")
assert not page_errors, "Page errors found: " + "\n".join(page_errors)
assert not bad_console, "Console errors found: " + "\n".join(bad_console)
print("browser_ui_ok")
print(f"screenshots={OUT}")
