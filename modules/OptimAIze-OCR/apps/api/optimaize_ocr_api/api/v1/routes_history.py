from __future__ import annotations

from fastapi import APIRouter

from optimaize_ocr_api.api.v1.schemas import ExportTextResponse, HistoryDocumentsResponse, HistoryRunsResponse
from optimaize_ocr_api.core.config import get_settings
from optimaize_ocr_api.core.errors import NotFoundError
from optimaize_ocr_api.services.data.history_repository import HistoryRepository

router = APIRouter(prefix="/history", tags=["history"])


def _repo() -> HistoryRepository:
    return HistoryRepository(get_settings().history_db_path)


@router.get("/documents", response_model=HistoryDocumentsResponse)
def documents(limit: int = 50) -> HistoryDocumentsResponse:
    return HistoryDocumentsResponse(documents=_repo().list_documents(limit))


@router.get("/documents/{document_id}")
def document_detail(document_id: str) -> dict:
    document = _repo().get_document(document_id)
    if document is None:
        raise NotFoundError(f"Document not found: {document_id}")
    return document


@router.get("/documents/{document_id}/runs", response_model=HistoryRunsResponse)
def document_runs(document_id: str) -> HistoryRunsResponse:
    return HistoryRunsResponse(runs=_repo().list_runs(document_id))


@router.get("/runs/{run_id}/text", response_model=ExportTextResponse)
def run_text(run_id: str) -> ExportTextResponse:
    return ExportTextResponse(run_id=run_id, text=_repo().export_text(run_id))


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    _repo().delete_document(document_id)
    return {"status": "deleted", "document_id": document_id}
