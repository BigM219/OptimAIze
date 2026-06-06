from __future__ import annotations

import os
from dataclasses import dataclass


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_title: str = "OptimAIze Parent API"
    api_version: str = "0.1.0"
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5174", "http://localhost:5174")


def get_settings() -> Settings:
    return Settings(
        cors_origins=tuple(
            _csv(os.getenv("OPTIMAIZE_PARENT_CORS_ORIGINS", ",".join(Settings.cors_origins)))
        ),
    )
