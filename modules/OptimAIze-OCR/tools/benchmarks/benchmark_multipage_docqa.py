from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import setup_cpu_optimization
from optimaize_ocr.storage.history_db import DEFAULT_HISTORY_DB, HistoryDB, image_sha256
from scripts.benchmark_ocr_dataset import (
    DEFAULT_MODELS,
    PRESETS,
    compact,
    load_dataset_rows,
    similarity,
    token_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multi-page single-call OCR and DocQA.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="funsd")
    parser.add_argument("--dataset")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--image-column")
    parser.add_argument("--text-column")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--models", default="glm-ocr,paddleocr-vl,surya-ocr")
    parser.add_argument("--pages-per-call", type=int, default=2)
    parser.add_argument("--limit-groups", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mode", choices=("docqa", "ocr-pages", "both"), default="docqa")
    parser.add_argument("--single-call", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--qa-mode", choices=("generic", "manifest"), default="generic")
    parser.add_argument("--qa-index", type=int, default=0)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--fallback-docqa-mode", choices=("none", "ocr_concat"), default="ocr_concat")
    parser.add_argument(
        "--question",
        default="What are the main recipient, sender, date, and subject information across these pages?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--auto-runtime", choices=("off", "conservative", "speed", "experimental"), default="conservative")
    parser.add_argument("--quantize-mode", default="auto")
    parser.add_argument("--quantize-int8", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--paddle-table-prompt", choices=("fast", "official"), default="fast")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpu-percent", type=float)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/multipage_docqa/summary.json"))
    parser.add_argument("--ocr-cache", type=Path, default=Path("outputs/multipage_docqa/ocr_cache.json"))
    parser.add_argument("--no-ocr-cache", action="store_true")
    parser.add_argument("--history-db", type=Path, default=DEFAULT_HISTORY_DB)
    parser.add_argument("--no-history-db", action="store_true")
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--use-history-ocr", action="store_true")
    parser.add_argument("--use-original-dots", action="store_true")
    parser.add_argument("--dots-fuse-mlp-swiglu", choices=("true", "false"), default="true")
    parser.add_argument("--dots-int8-lm-head", choices=("true", "false"), default="true")
    return parser.parse_args()


def _norm(text: str) -> str:
    text = re.sub(r"<\|[^>]+\|>", " ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.casefold().split())


def answer_contains(expected: str, candidate: str) -> bool:
    exp_tokens = _norm(expected).split()
    cand = _norm(candidate)
    return bool(exp_tokens) and all(token in cand.split() for token in exp_tokens)


def required_term_recall(required_terms: list[str], candidate: str) -> float:
    if not required_terms:
        return 1.0
    cand = _norm(candidate)
    hits = sum(1 for term in required_terms if _norm(term) in cand)
    return hits / len(required_terms)


def ocr_parse_issue(page_texts: list[str]) -> str | None:
    markers = ("[parse_error", "[page_count_mismatch", "[page_schema_mismatch")
    issues = [text.split("]", 1)[0] + "]" for text in page_texts if text.startswith(markers)]
    return "; ".join(issues) if issues else None


def extract_docqa_json_answer(text: str) -> tuple[str, list[Any] | None]:
    raw = text.strip()
    candidates = [raw]
    for start, end in (("{", "}"), ("[", "]")):
        left = raw.find(start)
        right = raw.rfind(end)
        if left != -1 and right > left:
            candidates.append(raw[left:right + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            answer = parsed.get("answer") or parsed.get("value") or parsed.get("result")
            evidence_pages = parsed.get("evidence_pages") or parsed.get("pages")
            if answer is not None:
                return str(answer), evidence_pages if isinstance(evidence_pages, list) else None
    return text, None


def evidence_scores(expected_pages: list[Any], predicted_pages: list[Any] | None) -> tuple[float, bool]:
    expected = {int(page) for page in expected_pages if str(page).isdigit()}
    predicted = {int(page) for page in predicted_pages or [] if str(page).isdigit()}
    if not expected:
        return 1.0, predicted == expected
    recall = len(expected & predicted) / len(expected)
    return recall, predicted == expected


def fallback_answer_from_evidence(page_texts: list[str], evidence_pages: list[Any]) -> str:
    page_numbers = []
    for page in evidence_pages:
        try:
            number = int(page)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= len(page_texts):
            page_numbers.append(number)
    if not page_numbers:
        page_numbers = list(range(1, len(page_texts) + 1))
    return "\n\n".join(f"Page {number}: {page_texts[number - 1]}" for number in page_numbers)


def group_rows(rows: list[dict[str, Any]], pages_per_call: int, limit_groups: int | None) -> list[dict[str, Any]]:
    groups = []
    for start in range(0, len(rows), pages_per_call):
        group = rows[start:start + pages_per_call]
        if len(group) < pages_per_call:
            break
        groups.append({
            "group_id": f"dataset_{start // pages_per_call:04d}",
            "pages": group,
            "questions": [],
        })
        if limit_groups is not None and len(groups) >= limit_groups:
            break
    return groups


def load_manifest_groups(path: Path, limit_groups: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    groups = []
    for group in manifest.get("groups", []):
        pages = []
        for page in group.get("pages", []):
            image_path = Path(page["image"])
            if not image_path.is_absolute():
                image_path = base / image_path if not image_path.exists() else image_path
            pages.append({
                "idx": f"{group['group_id']}:p{page['page']}",
                "image": Image.open(image_path).convert("RGB"),
                "image_path": str(image_path),
                "reference": page.get("reference", ""),
                "page": page.get("page"),
            })
        groups.append({
            "group_id": group.get("group_id", f"manifest_{len(groups):04d}"),
            "pages": pages,
            "questions": group.get("questions", []),
        })
        if limit_groups is not None and len(groups) >= limit_groups:
            break
    info = {
        "manifest": str(path),
        "version": manifest.get("version"),
        "seed": manifest.get("seed"),
        "pages_per_group": manifest.get("pages_per_group"),
    }
    return groups, info


def selected_questions(group: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = group.get("questions") or []
    if args.qa_mode == "manifest" or args.manifest:
        if args.all_questions:
            return questions
        if questions:
            return [questions[min(args.qa_index, len(questions) - 1)]]
    return [{
        "type": "generic",
        "question": args.question,
        "answer": "",
        "required_terms": [],
        "evidence_pages": [],
    }]


def load_backend(model: str, args: argparse.Namespace):
    quantize_int8 = None
    if args.quantize_int8 == "true":
        quantize_int8 = True
    elif args.quantize_int8 == "false":
        quantize_int8 = False
    backend = get_vlm_backend(
        model,
        device="cpu",
        quantize_int8=quantize_int8,
        use_optimized_dots=not args.use_original_dots,
        quantize_mode=args.quantize_mode,
        dots_fuse_mlp_swiglu=(args.dots_fuse_mlp_swiglu == "true"),
        dots_int8_lm_head=(args.dots_int8_lm_head == "true"),
        auto_runtime=args.auto_runtime,
        paddle_table_prompt=args.paddle_table_prompt,
    )
    if hasattr(backend, "max_new_tokens"):
        backend.max_new_tokens = args.max_new_tokens
    return backend


def should_use_single_call(backend, args: argparse.Namespace) -> bool:
    if args.single_call == "true":
        return True
    if args.single_call == "false":
        return False
    return bool(backend.supports_multi_image_single_call())


def load_ocr_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_ocr_cache or not args.ocr_cache.exists():
        return {}
    try:
        return json.loads(args.ocr_cache.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_ocr_cache(args: argparse.Namespace, cache: dict[str, Any]) -> None:
    if args.no_ocr_cache:
        return
    args.ocr_cache.parent.mkdir(parents=True, exist_ok=True)
    args.ocr_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def ocr_cache_key(model: str, group: dict[str, Any], single_call: bool, category: str = "plain") -> str:
    idxs = ",".join(str(row["idx"]) for row in group["pages"])
    mode = "single" if single_call else "per_page"
    return f"{model}|{group.get('group_id')}|{mode}|{category}|{idxs}"



def group_page_records(group: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(group["pages"], start=1):
        image = row["image"]
        records.append({
            "page_number": int(row.get("page") or idx),
            "image_sha256": image_sha256(image),
            "image_path": row.get("image_path"),
            "width": image.width,
            "height": image.height,
        })
    return records


def history_runtime_config(args: argparse.Namespace, single_call: bool) -> dict[str, Any]:
    return {
        "single_call": single_call,
        "fallback_docqa_mode": args.fallback_docqa_mode,
        "auto_runtime": args.auto_runtime,
        "quantize_mode": args.quantize_mode,
        "quantize_int8": args.quantize_int8,
        "paddle_table_prompt": args.paddle_table_prompt,
        "max_new_tokens": args.max_new_tokens,
    }


def ensure_history_document(history_db: HistoryDB | None, group: dict[str, Any], dataset_info: dict[str, Any]) -> str | None:
    if history_db is None:
        return None
    return history_db.upsert_document(
        group_page_records(group),
        source_path=str(dataset_info.get("manifest") or dataset_info.get("dataset") or ""),
        metadata={"group_id": group.get("group_id"), "idxs": [row.get("idx") for row in group["pages"]], "dataset": dataset_info},
    )

def get_cached_ocr(
    backend,
    model: str,
    group: dict[str, Any],
    images: list[Any],
    args: argparse.Namespace,
    single_call: bool,
    cache: dict[str, Any],
    history_db: HistoryDB | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> tuple[list[str], float, bool, bool, str | None, str | None]:
    dataset_info = dataset_info or {}
    runtime_config = history_runtime_config(args, single_call)
    document_id = ensure_history_document(history_db, group, dataset_info) if (args.save_history or args.use_history_ocr) else None
    if args.use_history_ocr and history_db is not None and document_id is not None:
        history_run = history_db.find_ocr_run(document_id, model, model, "plain", runtime_config)
        if history_run is not None:
            return list(map(str, history_run["pages"])), 0.0, False, True, document_id, str(history_run["run_id"])
    key = ocr_cache_key(model, group, single_call)
    if not args.no_ocr_cache and key in cache:
        page_texts = list(map(str, cache[key].get("pages", [])))
        run_id = None
        if args.save_history and history_db is not None and document_id is not None:
            run_id = history_db.add_ocr_run(document_id, model, model, "plain", runtime_config, page_texts, 0.0)
        return page_texts, 0.0, True, False, document_id, run_id
    t0 = time.perf_counter()
    page_texts = backend.generate_ocr_pages(images, category="plain") if single_call else [
        backend.generate_ocr(row["image"], category="plain") for row in group["pages"]
    ]
    elapsed = time.perf_counter() - t0
    if not args.no_ocr_cache:
        cache[key] = {"pages": page_texts, "seconds": elapsed}
    run_id = None
    if args.save_history and history_db is not None and document_id is not None:
        parse_errors = [text.split("]", 1)[0] + "]" if text.startswith(("[parse_error", "[page_count_mismatch", "[page_schema_mismatch")) else None for text in page_texts]
        run_id = history_db.add_ocr_run(document_id, model, model, "plain", runtime_config, page_texts, elapsed, parse_errors)
    return page_texts, elapsed, False, False, document_id, run_id


def run_group(
    backend,
    model: str,
    group: dict[str, Any],
    args: argparse.Namespace,
    single_call: bool,
    ocr_cache: dict[str, Any],
    history_db: HistoryDB | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pages = group["pages"]
    images = [row["image"] for row in pages]
    reference = " ".join(row["reference"] for row in pages)
    result: dict[str, Any] = {
        "group_id": group.get("group_id"),
        "idxs": [row["idx"] for row in pages],
        "pages": len(pages),
        "single_call": single_call,
    }

    cached_page_texts: list[str] | None = None
    if args.mode in ("ocr-pages", "both") or (args.mode == "docqa" and not single_call and args.fallback_docqa_mode == "ocr_concat"):
        t0 = time.perf_counter()
        try:
            cached_page_texts, ocr_elapsed, cache_hit, history_hit, document_id, run_id = get_cached_ocr(backend, model, group, images, args, single_call, ocr_cache, history_db, dataset_info)
            ocr_text = "\n".join(cached_page_texts)
            result["ocr_seconds"] = ocr_elapsed
            result["ocr_wall_seconds"] = time.perf_counter() - t0
            result["ocr_cache_hit"] = cache_hit
            result["history_ocr_hit"] = history_hit
            result["history_document_id"] = document_id
            result["history_ocr_run_id"] = run_id
            result["ocr_pages"] = cached_page_texts
            result["ocr_error"] = ocr_parse_issue(cached_page_texts)
            result["ocr_token_f1"] = token_f1(reference, ocr_text)
            result["ocr_similarity"] = similarity(reference, ocr_text)
        except Exception as exc:
            result["ocr_seconds"] = time.perf_counter() - t0
            result["ocr_wall_seconds"] = result["ocr_seconds"]
            result["ocr_cache_hit"] = False
            result["ocr_pages"] = []
            result["ocr_error"] = repr(exc)
            result["ocr_token_f1"] = 0.0
            result["ocr_similarity"] = 0.0
            cached_page_texts = []

    if args.mode in ("docqa", "both"):
        qa_rows = []
        for qa in selected_questions(group, args):
            question = qa.get("question", args.question)
            t0 = time.perf_counter()
            try:
                if single_call:
                    answer = backend.docqa_pages(images, question, max_new_tokens=args.max_new_tokens)
                elif args.fallback_docqa_mode == "ocr_concat":
                    if cached_page_texts is None:
                        cached_page_texts = [backend.generate_ocr(row["image"], category="plain") for row in pages]
                    answer = "\n\n".join(f"Page {i + 1}: {text}" for i, text in enumerate(cached_page_texts))
                else:
                    raise NotImplementedError("DocQA fallback disabled")
                error = ocr_parse_issue(cached_page_texts) if not single_call and cached_page_texts is not None else None
            except Exception as exc:
                answer = ""
                error = repr(exc)
            elapsed = time.perf_counter() - t0
            expected = qa.get("answer", "")
            expected_evidence_pages = qa.get("evidence_pages", [])
            scored_answer, predicted_evidence_pages = (
                extract_docqa_json_answer(answer) if single_call else (answer, list(expected_evidence_pages))
            )
            evidence_recall, evidence_exact = evidence_scores(expected_evidence_pages, predicted_evidence_pages)
            required_terms = list(map(str, qa.get("required_terms", [])))
            qa_scores = {
                "qa_token_f1": token_f1(expected, scored_answer) if expected else 0.0,
                "qa_contains_answer": answer_contains(expected, scored_answer) if expected else False,
                "qa_required_term_recall": required_term_recall(required_terms, scored_answer),
                "qa_evidence_page_recall": evidence_recall,
                "qa_evidence_page_exact": evidence_exact,
            }
            if args.save_history and history_db is not None:
                doc_id = result.get("history_document_id") or ensure_history_document(history_db, group, dataset_info or {})
                if doc_id is not None:
                    history_db.add_qa_run(str(doc_id), result.get("history_ocr_run_id"), question, scored_answer, predicted_evidence_pages or [], qa_scores, elapsed)
            qa_rows.append({
                "type": qa.get("type"),
                "question": question,
                "expected_answer": expected,
                "required_terms": required_terms,
                "evidence_pages": expected_evidence_pages,
                "predicted_evidence_pages": predicted_evidence_pages,
                "qa_evidence_page_recall": evidence_recall,
                "qa_evidence_page_exact": evidence_exact,
                "docqa_seconds": elapsed,
                "docqa_answer": answer,
                "scored_docqa_answer": scored_answer,
                "docqa_error": error,
                "qa_token_f1": qa_scores["qa_token_f1"],
                "qa_contains_answer": qa_scores["qa_contains_answer"],
                "qa_required_term_recall": qa_scores["qa_required_term_recall"],
            })
        result["docqa_items"] = qa_rows
        result["docqa_seconds"] = sum(row["docqa_seconds"] for row in qa_rows)
        result["docqa_answer"] = qa_rows[0]["docqa_answer"] if qa_rows else ""
        result["docqa_error"] = qa_rows[0]["docqa_error"] if qa_rows else None
        scored = [row for row in qa_rows if row["expected_answer"]]
        result["avg_qa_token_f1"] = sum(row["qa_token_f1"] for row in scored) / len(scored) if scored else 0.0
        result["avg_qa_required_term_recall"] = sum(row["qa_required_term_recall"] for row in scored) / len(scored) if scored else 0.0
        result["qa_contains_rate"] = sum(1 for row in scored if row["qa_contains_answer"]) / len(scored) if scored else 0.0
        result["avg_qa_evidence_page_recall"] = sum(row["qa_evidence_page_recall"] for row in scored) / len(scored) if scored else 0.0
        result["qa_evidence_page_exact_rate"] = sum(1 for row in scored if row["qa_evidence_page_exact"]) / len(scored) if scored else 0.0

    return result


def run_model(model: str, groups: list[dict[str, Any]], args: argparse.Namespace, ocr_cache: dict[str, Any], history_db: HistoryDB | None, dataset_info: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        backend = load_backend(model, args)
    except Exception as exc:
        return {
            "model": model,
            "load_seconds": time.perf_counter() - t0,
            "load_error": repr(exc),
            "groups": [],
        }
    load_seconds = time.perf_counter() - t0
    single_call = should_use_single_call(backend, args)
    rows = []
    for group in groups:
        row = run_group(backend, model, group, args, single_call, ocr_cache, history_db, dataset_info)
        rows.append(row)
        docqa_status = "ERR" if row.get("docqa_error") else "OK"
        ocr_status = "ERR" if row.get("ocr_error") else "OK"
        print(
            f"{model} group={row.get('group_id')} idxs={row['idxs']} single_call={single_call} "
            f"docqa={docqa_status} {row.get('docqa_seconds', 0):.2f}s "
            f"ocr={ocr_status} {row.get('ocr_seconds', 0):.2f}s "
            f"ocr_f1={row.get('ocr_token_f1', 0):.3f} "
            f"qa_f1={row.get('avg_qa_token_f1', 0):.3f} "
            f"qa_terms={row.get('avg_qa_required_term_recall', 0):.3f} "
            f"evidence={row.get('avg_qa_evidence_page_recall', 0):.3f} "
            f"answer={compact(row.get('docqa_answer', ''), 120)!r}",
            flush=True,
        )
    docqa_times = [r.get("docqa_seconds", 0.0) for r in rows if "docqa_seconds" in r]
    ocr_times = [r.get("ocr_seconds", 0.0) for r in rows if "ocr_seconds" in r]
    ocr_wall_times = [r.get("ocr_wall_seconds", r.get("ocr_seconds", 0.0)) for r in rows if "ocr_seconds" in r]
    ocr_cache_hits = [1.0 if r.get("ocr_cache_hit") else 0.0 for r in rows if "ocr_cache_hit" in r]
    history_hits = [1.0 if r.get("history_ocr_hit") else 0.0 for r in rows if "history_ocr_hit" in r]
    f1s = [r.get("ocr_token_f1", 0.0) for r in rows if "ocr_token_f1" in r]
    qa_f1s = [r.get("avg_qa_token_f1", 0.0) for r in rows if "avg_qa_token_f1" in r]
    qa_terms = [r.get("avg_qa_required_term_recall", 0.0) for r in rows if "avg_qa_required_term_recall" in r]
    qa_contains = [r.get("qa_contains_rate", 0.0) for r in rows if "qa_contains_rate" in r]
    qa_evidence_recall = [r.get("avg_qa_evidence_page_recall", 0.0) for r in rows if "avg_qa_evidence_page_recall" in r]
    qa_evidence_exact = [r.get("qa_evidence_page_exact_rate", 0.0) for r in rows if "qa_evidence_page_exact_rate" in r]
    return {
        "model": model,
        "load_seconds": load_seconds,
        "runtime_policy": getattr(backend, "runtime_policy", None).to_dict() if getattr(backend, "runtime_policy", None) is not None else None,
        "single_call": single_call,
        "supports_multi_image_single_call": backend.supports_multi_image_single_call(),
        "groups_count": len(rows),
        "avg_docqa_seconds": sum(docqa_times) / len(docqa_times) if docqa_times else 0.0,
        "avg_ocr_seconds": sum(ocr_times) / len(ocr_times) if ocr_times else 0.0,
        "avg_ocr_wall_seconds": sum(ocr_wall_times) / len(ocr_wall_times) if ocr_wall_times else 0.0,
        "ocr_cache_hit_rate": sum(ocr_cache_hits) / len(ocr_cache_hits) if ocr_cache_hits else 0.0,
        "history_ocr_hit_rate": sum(history_hits) / len(history_hits) if history_hits else 0.0,
        "avg_ocr_token_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "avg_qa_token_f1": sum(qa_f1s) / len(qa_f1s) if qa_f1s else 0.0,
        "avg_qa_required_term_recall": sum(qa_terms) / len(qa_terms) if qa_terms else 0.0,
        "avg_qa_contains_rate": sum(qa_contains) / len(qa_contains) if qa_contains else 0.0,
        "avg_qa_evidence_page_recall": sum(qa_evidence_recall) / len(qa_evidence_recall) if qa_evidence_recall else 0.0,
        "avg_qa_evidence_page_exact_rate": sum(qa_evidence_exact) / len(qa_evidence_exact) if qa_evidence_exact else 0.0,
        "groups": rows,
    }


def main() -> None:
    args = parse_args()
    setup_cpu_optimization(args.threads, args.cpu_percent)
    if args.manifest:
        groups, dataset_info = load_manifest_groups(args.manifest, args.limit_groups)
        if args.qa_mode == "generic":
            args.qa_mode = "manifest"
    else:
        row_limit = args.pages_per_call * args.limit_groups
        args_for_loader = argparse.Namespace(**vars(args))
        args_for_loader.limit = row_limit
        rows, dataset_info = load_dataset_rows(args_for_loader)
        groups = group_rows(rows, args.pages_per_call, args.limit_groups)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"dataset={dataset_info} groups={len(groups)} pages_per_call={args.pages_per_call} models={models}")

    ocr_cache = load_ocr_cache(args)
    history_db = None if args.no_history_db else HistoryDB(args.history_db)
    results = []
    for model in models:
        print("=" * 100)
        print(f"MODEL {model}")
        results.append(run_model(model, groups, args, ocr_cache, history_db, dataset_info))
    save_ocr_cache(args, ocr_cache)
    if history_db is not None:
        history_db.close()

    summary = {
        "dataset": dataset_info,
        "pages_per_call": args.pages_per_call,
        "limit_groups": args.limit_groups,
        "mode": args.mode,
        "single_call": args.single_call,
        "question": args.question,
        "auto_runtime": args.auto_runtime,
        "quantize_mode": args.quantize_mode,
        "ocr_cache": None if args.no_ocr_cache else str(args.ocr_cache),
        "history_db": None if args.no_history_db else str(args.history_db),
        "save_history": args.save_history,
        "use_history_ocr": args.use_history_ocr,
        "paddle_table_prompt": args.paddle_table_prompt,
        "models": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 100)
    for result in results:
        if result.get("load_error"):
            print(f"SUMMARY model={result['model']} load_error={result['load_error']}")
        else:
            print(
                f"SUMMARY model={result['model']} single_call={result['single_call']} "
                f"groups={result['groups_count']} avg_docqa={result['avg_docqa_seconds']:.2f}s "
                f"avg_ocr={result['avg_ocr_seconds']:.2f}s wall_ocr={result.get('avg_ocr_wall_seconds', 0):.2f}s "
                f"cache_hit={result.get('ocr_cache_hit_rate', 0):.2f} history_hit={result.get('history_ocr_hit_rate', 0):.2f} "
                f"avg_f1={result['avg_ocr_token_f1']:.3f} "
                f"qa_f1={result.get('avg_qa_token_f1', 0):.3f} "
                f"qa_terms={result.get('avg_qa_required_term_recall', 0):.3f} "
                f"qa_contains={result.get('avg_qa_contains_rate', 0):.3f} "
                f"qa_evidence={result.get('avg_qa_evidence_page_recall', 0):.3f} "
                f"load={result['load_seconds']:.2f}s"
            )
    print(f"wrote={args.output_json}")


if __name__ == "__main__":
    main()
