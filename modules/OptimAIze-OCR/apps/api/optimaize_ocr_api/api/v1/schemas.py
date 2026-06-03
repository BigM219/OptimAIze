from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str
    history_db: str


class ModelListResponse(BaseModel):
    models: list[str]
    default_model: str


class RuntimeRange(BaseModel):
    default: int | float | None = None
    min: int | float
    max: int | float
    recommended: int | float | None = None


class RuntimeConfigResponse(BaseModel):
    logical_cpus: int
    threads: RuntimeRange
    cpu_percent: RuntimeRange
    labels: dict[str, str]


class OCRRegion(BaseModel):
    index: int
    category: str = ""
    bbox: Any = ""
    score: Any = ""
    text: str = ""


class SingleImageOCRResponse(BaseModel):
    markdown: str
    html: str
    regions: list[OCRRegion]
    timings: dict[str, Any] = Field(default_factory=dict)
    output_dir: str
    image_name: str


class HistoryDocument(BaseModel):
    document_id: str
    source_path: str | None = None
    image_sha256: str
    page_count: int
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoryDocumentsResponse(BaseModel):
    documents: list[HistoryDocument]


class HistoryRun(BaseModel):
    run_id: str
    document_id: str
    backend: str
    model_id: str
    category: str
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    seconds: float
    created_at: float


class HistoryRunsResponse(BaseModel):
    runs: list[HistoryRun]


class ExportTextResponse(BaseModel):
    run_id: str
    text: str
