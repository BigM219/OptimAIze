from __future__ import annotations

import sys
from pathlib import Path

OCR_MODULE_ROOT = Path(__file__).resolve().parents[4]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
BACKEND_ROOT = OCR_MODULE_ROOT / "apps" / "api"

for path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_OUTPUT_DIR = OCR_MODULE_ROOT / "outputs"
DEFAULT_UPLOAD_DIR = DEFAULT_OUTPUT_DIR / "api_uploads"
DEFAULT_HISTORY_DB = DEFAULT_OUTPUT_DIR / "history" / "ocr_history.sqlite"
