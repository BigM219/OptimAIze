import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from PIL import Image


SCHEMA_VERSION = 1
DEFAULT_HISTORY_DB = Path("outputs/history/ocr_history.sqlite")


def now_ts() -> float:
    return time.time()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_sha256(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    h = hashlib.sha256()
    h.update(image.mode.encode("utf-8"))
    h.update(str(image.size).encode("utf-8"))
    h.update(image.tobytes())
    return h.hexdigest()


def combined_hash(parts: list[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def runtime_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json_dumps(config).encode("utf-8")).hexdigest()


class HistoryDB:
    def __init__(self, path: Path | str = DEFAULT_HISTORY_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_path TEXT,
                image_sha256 TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                image_sha256 TEXT NOT NULL,
                image_path TEXT,
                width INTEGER,
                height INTEGER,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
                UNIQUE(document_id, page_number)
            );
            CREATE TABLE IF NOT EXISTS ocr_runs (
                run_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                backend TEXT NOT NULL,
                model_id TEXT NOT NULL,
                category TEXT NOT NULL,
                runtime_config_json TEXT NOT NULL,
                runtime_config_hash TEXT NOT NULL,
                seconds REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS ocr_pages (
                run_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                parse_error TEXT,
                PRIMARY KEY(run_id, page_number),
                FOREIGN KEY(run_id) REFERENCES ocr_runs(run_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS qa_runs (
                qa_run_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                run_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                evidence_pages_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                seconds REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id) REFERENCES ocr_runs(run_id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS embed_cache (
                embed_id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL,
                backend TEXT NOT NULL,
                model_id TEXT NOT NULL,
                processor_config_hash TEXT NOT NULL,
                runtime_config_hash TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                tensor_path TEXT NOT NULL,
                shape_json TEXT NOT NULL,
                dtype TEXT NOT NULL,
                device_saved_as TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE,
                UNIQUE(page_id, backend, model_id, processor_config_hash, runtime_config_hash, prompt_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_ocr_runs_lookup ON ocr_runs(document_id, backend, model_id, category, runtime_config_hash, created_at);
            CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id, page_number);
            CREATE INDEX IF NOT EXISTS idx_embed_lookup ON embed_cache(page_id, backend, model_id, processor_config_hash, runtime_config_hash, prompt_hash);
            """
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def upsert_document(self, pages: list[dict[str, Any]], source_path: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        page_hashes = [str(page["image_sha256"]) for page in pages]
        doc_hash = combined_hash(page_hashes)
        document_id = f"doc_{doc_hash[:20]}"
        t = now_ts()
        existing = self.conn.execute("SELECT created_at FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        created_at = float(existing["created_at"]) if existing else t
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents(document_id, source_path, image_sha256, page_count, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, source_path, doc_hash, len(pages), created_at, t, json_dumps(metadata or {})),
        )
        for page in pages:
            page_number = int(page["page_number"])
            page_id = f"{document_id}_p{page_number:04d}"
            self.conn.execute(
                """
                INSERT OR REPLACE INTO pages(page_id, document_id, page_number, image_sha256, image_path, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page_id,
                    document_id,
                    page_number,
                    str(page["image_sha256"]),
                    page.get("image_path"),
                    page.get("width"),
                    page.get("height"),
                ),
            )
        self.conn.commit()
        return document_id

    def add_ocr_run(
        self,
        document_id: str,
        backend: str,
        model_id: str,
        category: str,
        runtime_config: dict[str, Any],
        page_texts: list[str],
        seconds: float,
        parse_errors: list[str | None] | None = None,
    ) -> str:
        cfg_hash = runtime_hash(runtime_config)
        run_hash = combined_hash([document_id, backend, model_id, category, cfg_hash, str(now_ts())])
        run_id = f"ocr_{run_hash[:20]}"
        t = now_ts()
        self.conn.execute(
            """
            INSERT INTO ocr_runs(run_id, document_id, backend, model_id, category, runtime_config_json, runtime_config_hash, seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, document_id, backend, model_id, category, json_dumps(runtime_config), cfg_hash, seconds, t),
        )
        parse_errors = parse_errors or [None] * len(page_texts)
        for idx, text in enumerate(page_texts, start=1):
            self.conn.execute(
                "INSERT INTO ocr_pages(run_id, page_number, text, parse_error) VALUES (?, ?, ?, ?)",
                (run_id, idx, text, parse_errors[idx - 1] if idx - 1 < len(parse_errors) else None),
            )
        self.conn.commit()
        return run_id

    def find_ocr_run(
        self,
        document_id: str,
        backend: str,
        model_id: str,
        category: str,
        runtime_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        cfg_hash = runtime_hash(runtime_config)
        run = self.conn.execute(
            """
            SELECT * FROM ocr_runs
            WHERE document_id = ? AND backend = ? AND model_id = ? AND category = ? AND runtime_config_hash = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (document_id, backend, model_id, category, cfg_hash),
        ).fetchone()
        if run is None:
            return None
        pages = self.conn.execute(
            "SELECT page_number, text, parse_error FROM ocr_pages WHERE run_id = ? ORDER BY page_number",
            (run["run_id"],),
        ).fetchall()
        return {
            "run_id": run["run_id"],
            "document_id": run["document_id"],
            "backend": run["backend"],
            "model_id": run["model_id"],
            "category": run["category"],
            "runtime_config": json_loads(run["runtime_config_json"], {}),
            "seconds": run["seconds"],
            "pages": [page["text"] for page in pages],
            "parse_errors": [page["parse_error"] for page in pages],
        }

    def add_qa_run(
        self,
        document_id: str,
        run_id: str | None,
        question: str,
        answer: str,
        evidence_pages: list[Any],
        scores: dict[str, Any],
        seconds: float,
    ) -> str:
        qa_hash = combined_hash([document_id, run_id or "", question, answer, str(now_ts())])
        qa_run_id = f"qa_{qa_hash[:20]}"
        self.conn.execute(
            """
            INSERT INTO qa_runs(qa_run_id, document_id, run_id, question, answer, evidence_pages_json, scores_json, seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (qa_run_id, document_id, run_id, question, answer, json_dumps(evidence_pages), json_dumps(scores), seconds, now_ts()),
        )
        self.conn.commit()
        return qa_run_id

    def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) | {"metadata": json_loads(row["metadata_json"], {})} for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        doc = self.conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        if doc is None:
            return None
        pages = self.conn.execute("SELECT * FROM pages WHERE document_id = ? ORDER BY page_number", (document_id,)).fetchall()
        return dict(doc) | {"metadata": json_loads(doc["metadata_json"], {}), "pages": [dict(row) for row in pages]}

    def list_runs(self, document_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM ocr_runs WHERE document_id = ? ORDER BY created_at DESC",
            (document_id,),
        ).fetchall()
        return [dict(row) | {"runtime_config": json_loads(row["runtime_config_json"], {})} for row in rows]

    def export_text(self, run_id: str) -> str:
        rows = self.conn.execute(
            "SELECT page_number, text FROM ocr_pages WHERE run_id = ? ORDER BY page_number",
            (run_id,),
        ).fetchall()
        return "\n\n".join(f"Page {row['page_number']}: {row['text']}" for row in rows)

    def delete_document(self, document_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        self.conn.commit()

    def upsert_embed_cache(
        self,
        page_id: str,
        backend: str,
        model_id: str,
        processor_config_hash: str,
        runtime_config_hash: str,
        prompt_hash: str,
        tensor_path: str,
        shape: dict[str, Any],
        dtype: str,
        device_saved_as: str = "cpu",
    ) -> str:
        embed_hash = combined_hash([page_id, backend, model_id, processor_config_hash, runtime_config_hash, prompt_hash])
        embed_id = f"emb_{embed_hash[:20]}"
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embed_cache(embed_id, page_id, backend, model_id, processor_config_hash, runtime_config_hash, prompt_hash, tensor_path, shape_json, dtype, device_saved_as, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embed_id,
                page_id,
                backend,
                model_id,
                processor_config_hash,
                runtime_config_hash,
                prompt_hash,
                tensor_path,
                json_dumps(shape),
                dtype,
                device_saved_as,
                now_ts(),
            ),
        )
        self.conn.commit()
        return embed_id

    def find_embed_cache(
        self,
        page_id: str,
        backend: str,
        model_id: str,
        processor_config_hash: str,
        runtime_config_hash: str,
        prompt_hash: str,
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM embed_cache
            WHERE page_id = ? AND backend = ? AND model_id = ? AND processor_config_hash = ? AND runtime_config_hash = ? AND prompt_hash = ?
            LIMIT 1
            """,
            (page_id, backend, model_id, processor_config_hash, runtime_config_hash, prompt_hash),
        ).fetchone()
        return None if row is None else dict(row) | {"shape": json_loads(row["shape_json"], {})}
