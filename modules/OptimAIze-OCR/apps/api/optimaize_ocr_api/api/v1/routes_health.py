from __future__ import annotations

from fastapi import APIRouter

from optimaize_ocr_api.api.v1.schemas import HealthResponse
from optimaize_ocr_api.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def api_health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(ok=True, service=settings.api_title, version=settings.api_version, history_db=str(settings.history_db_path))
