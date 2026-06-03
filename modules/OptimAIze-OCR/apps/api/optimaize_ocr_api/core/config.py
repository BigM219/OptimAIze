from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import DEFAULT_HISTORY_DB, DEFAULT_OUTPUT_DIR, DEFAULT_UPLOAD_DIR


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_title: str = "OptimAIze OCR API"
    api_version: str = "0.1.0"
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5175",
        "http://127.0.0.1:5176",
        "http://localhost:5176",
        "http://127.0.0.1:5177",
        "http://localhost:5177",
        "http://127.0.0.1:7850",
        "http://localhost:7850",
    )
    output_dir: Path = DEFAULT_OUTPUT_DIR
    upload_dir: Path = DEFAULT_UPLOAD_DIR
    history_db_path: Path = DEFAULT_HISTORY_DB
    max_upload_bytes: int = 25 * 1024 * 1024


def get_settings() -> Settings:
    return Settings(
        cors_origins=tuple(_csv(os.getenv("OPTIMAIZE_OCR_CORS_ORIGINS", ",".join(Settings.cors_origins)))),
        output_dir=Path(os.getenv("OPTIMAIZE_OCR_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))),
        upload_dir=Path(os.getenv("OPTIMAIZE_OCR_UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR))),
        history_db_path=Path(os.getenv("OPTIMAIZE_OCR_HISTORY_DB", str(DEFAULT_HISTORY_DB))),
        max_upload_bytes=int(os.getenv("OPTIMAIZE_OCR_MAX_UPLOAD_BYTES", str(Settings.max_upload_bytes))),
    )
