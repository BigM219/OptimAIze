from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def module_root() -> Path:
    configured = os.getenv("OPTIMAIZE_OCR_MODULE_ROOT")
    if configured:
        return Path(configured)
    return package_root().parents[1]


def weights_dir() -> Path:
    configured = os.getenv("OPTIMAIZE_OCR_WEIGHTS_DIR")
    if configured:
        return Path(configured)
    return module_root() / "weights"
