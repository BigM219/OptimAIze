import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

OCR_MODULE_ROOT = Path(__file__).resolve().parents[2]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.storage.history_db import DEFAULT_HISTORY_DB, HistoryDB, image_sha256, runtime_hash, safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage OCR history DB.")
    parser.add_argument("--db", type=Path, default=DEFAULT_HISTORY_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize the history database.")

    list_docs = sub.add_parser("list-documents", help="List stored documents.")
    list_docs.add_argument("--limit", type=int, default=50)
    list_docs.add_argument("--json", action="store_true")

    show_doc = sub.add_parser("show-document", help="Show document metadata and pages.")
    show_doc.add_argument("--document-id", required=True)
    show_doc.add_argument("--json", action="store_true")

    list_runs = sub.add_parser("list-runs", help="List OCR runs for a document.")
    list_runs.add_argument("--document-id", required=True)
    list_runs.add_argument("--json", action="store_true")

    export = sub.add_parser("export-text", help="Export OCR text for a run.")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path)

    extract = sub.add_parser("extract", help="Run OCR extraction from a stored document.")
    extract.add_argument("--document-id", required=True)
    extract.add_argument("--model", default="paddleocr-vl")
    extract.add_argument("--category", default="plain")
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--quantize-mode", default="selective")
    extract.add_argument("--auto-runtime", default="off")
    extract.add_argument("--paddle-table-prompt", choices=("fast", "official"), default="fast")
    extract.add_argument("--use-visual-cache", action="store_true")
    extract.add_argument("--save-history", action="store_true")
    extract.add_argument("--output", type=Path)

    delete = sub.add_parser("delete-document", help="Delete a document and associated history records.")
    delete.add_argument("--document-id", required=True)

    return parser.parse_args()


def print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def processor_config_hash(backend: Any) -> str:
    processor = getattr(backend, "processor", None)
    image_processor = getattr(processor, "image_processor", None)
    config = {
        "processor": type(processor).__name__ if processor is not None else None,
        "image_processor": type(image_processor).__name__ if image_processor is not None else None,
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "max_pixels": getattr(image_processor, "max_pixels", None),
    }
    return runtime_hash(config)


def visual_cache_prompt_hash(category: str) -> str:
    return runtime_hash({"category": category, "cache_kind": "processor_outputs_v1"})


def visual_cache_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quantize_mode": args.quantize_mode,
        "auto_runtime": args.auto_runtime,
        "paddle_table_prompt": args.paddle_table_prompt,
        "cache_kind": "processor_outputs_v1",
    }


def tensor_cache_path(args: argparse.Namespace, backend_name: str, model_id: str, embed_id: str) -> Path:
    return args.db.parent / "embeddings" / safe_name(backend_name) / safe_name(model_id) / f"{embed_id}.pt"


def load_document_images(doc: dict[str, Any]) -> list[Image.Image]:
    images = []
    for page in doc["pages"]:
        image_path = page.get("image_path")
        if not image_path:
            raise SystemExit(f"Page {page['page_number']} has no image_path in history DB")
        path = Path(image_path)
        if not path.exists():
            raise SystemExit(f"Image not found for page {page['page_number']}: {path}")
        image = Image.open(path).convert("RGB")
        expected_hash = page.get("image_sha256")
        actual_hash = image_sha256(image)
        if expected_hash and actual_hash != expected_hash:
            raise SystemExit(f"Image hash mismatch for page {page['page_number']}: {path}")
        images.append(image)
    return images


def extract_document(args: argparse.Namespace, db: HistoryDB) -> None:
    doc = db.get_document(args.document_id)
    if doc is None:
        raise SystemExit(f"Document not found: {args.document_id}")
    images = load_document_images(doc)
    backend = get_vlm_backend(
        args.model,
        device=args.device,
        quantize_mode=args.quantize_mode,
        auto_runtime=args.auto_runtime,
        paddle_table_prompt=args.paddle_table_prompt,
    )
    backend_name = args.model
    model_id = str(getattr(backend, "model_id", args.model))
    runtime_config = visual_cache_runtime_config(args)
    runtime_config_hash = runtime_hash(runtime_config)
    proc_hash = processor_config_hash(backend)
    prompt_hash = visual_cache_prompt_hash(args.category)
    page_texts: list[str] = []
    cache_hits = 0
    started = time.perf_counter()
    for image, page in zip(images, doc["pages"], strict=True):
        cache = None
        if args.use_visual_cache and backend.supports_visual_token_cache():
            cached = db.find_embed_cache(
                page["page_id"], backend_name, model_id, proc_hash, runtime_config_hash, prompt_hash
            )
            if cached is not None and Path(cached["tensor_path"]).exists():
                cache = torch.load(cached["tensor_path"], map_location="cpu", weights_only=False)
                cache_hits += 1
            else:
                cache = backend.build_visual_cache(image, category=args.category)
                embed_id = db.upsert_embed_cache(
                    page["page_id"], backend_name, model_id, proc_hash, runtime_config_hash, prompt_hash,
                    "", {"kind": "processor_outputs_v1"}, "torch", "cpu"
                )
                path = tensor_cache_path(args, backend_name, model_id, embed_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cache, path)
                db.upsert_embed_cache(
                    page["page_id"], backend_name, model_id, proc_hash, runtime_config_hash, prompt_hash,
                    str(path), {"kind": "processor_outputs_v1"}, "torch", "cpu"
                )
        if cache is not None:
            text = backend.generate_ocr_from_visual_cache(cache, category=args.category)
        else:
            text = backend.generate_ocr(image, category=args.category)
        page_texts.append(text)
    elapsed = time.perf_counter() - started
    run_id = None
    if args.save_history:
        run_id = db.add_ocr_run(args.document_id, backend_name, model_id, args.category, runtime_config, page_texts, elapsed)
    output_text = "\n\n".join(f"Page {idx}: {text}" for idx, text in enumerate(page_texts, start=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)
    print(f"extract_seconds={elapsed:.2f} visual_cache_hits={cache_hits}/{len(images)} run_id={run_id or ''}")


def main() -> None:
    args = parse_args()
    db = HistoryDB(args.db)
    try:
        if args.command == "init":
            print(f"initialized={args.db}")
        elif args.command == "list-documents":
            docs = db.list_documents(args.limit)
            if args.json:
                print_json(docs)
            else:
                for doc in docs:
                    print(
                        f"{doc['document_id']} pages={doc['page_count']} updated={doc['updated_at']:.0f} "
                        f"source={doc.get('source_path') or ''}"
                    )
        elif args.command == "show-document":
            doc = db.get_document(args.document_id)
            if doc is None:
                raise SystemExit(f"Document not found: {args.document_id}")
            if args.json:
                print_json(doc)
            else:
                print(f"document_id={doc['document_id']} pages={doc['page_count']} hash={doc['image_sha256']}")
                print(f"source={doc.get('source_path') or ''}")
                for page in doc["pages"]:
                    print(
                        f"  page={page['page_number']} page_id={page['page_id']} "
                        f"size={page.get('width')}x{page.get('height')} path={page.get('image_path') or ''}"
                    )
        elif args.command == "list-runs":
            runs = db.list_runs(args.document_id)
            if args.json:
                print_json(runs)
            else:
                for run in runs:
                    print(
                        f"{run['run_id']} backend={run['backend']} model={run['model_id']} "
                        f"category={run['category']} seconds={run['seconds']:.2f} created={run['created_at']:.0f}"
                    )
        elif args.command == "export-text":
            text = db.export_text(args.run_id)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
                print(f"wrote={args.output}")
            else:
                print(text)
        elif args.command == "extract":
            extract_document(args, db)
        elif args.command == "delete-document":
            db.delete_document(args.document_id)
            print(f"deleted={args.document_id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
