from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    api_title: str = "OptimAIze Parent API"
    api_version: str = "0.1.0"
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5174", "http://localhost:5174")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
