from __future__ import annotations

from pathlib import Path
from typing import Any

from optimaize_ocr.storage.history_db import HistoryDB


class HistoryRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        db = HistoryDB(self.db_path)
        try:
            return db.list_documents(limit)
        finally:
            db.close()

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        db = HistoryDB(self.db_path)
        try:
            return db.get_document(document_id)
        finally:
            db.close()

    def list_runs(self, document_id: str) -> list[dict[str, Any]]:
        db = HistoryDB(self.db_path)
        try:
            return db.list_runs(document_id)
        finally:
            db.close()

    def export_text(self, run_id: str) -> str:
        db = HistoryDB(self.db_path)
        try:
            return db.export_text(run_id)
        finally:
            db.close()

    def delete_document(self, document_id: str) -> None:
        db = HistoryDB(self.db_path)
        try:
            db.delete_document(document_id)
        finally:
            db.close()
