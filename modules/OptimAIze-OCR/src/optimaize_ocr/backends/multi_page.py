import json
import re
from typing import Any

import torch
from PIL import Image


def normalize_pages(images: list[Image.Image]) -> list[Image.Image]:
    if not images:
        raise ValueError("At least one page image is required")
    return [image.convert("RGB") if image.mode != "RGB" else image for image in images]


def build_multipage_content(images: list[Image.Image], instruction: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for idx, image in enumerate(normalize_pages(images), start=1):
        content.append({"type": "text", "text": f"Page {idx}:"})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": instruction})
    return content


def build_page_ocr_instruction(category: str) -> str:
    cat = category.strip().lower()
    if cat == "table":
        task = "transcribe tables as HTML"
    elif cat == "formula":
        task = "transcribe formulas as LaTeX"
    else:
        task = "transcribe all visible text as Markdown"
    return (
        f"For each page, {task}. Return strict JSON only, with exactly one object per input page, "
        "in the same order: [{\"page\": 1, \"text\": \"...\"}]. "
        "Do not merge pages. Do not omit blank pages. Preserve the original text."
    )


def build_docqa_instruction(question: str) -> str:
    return (
        "Answer the question using all provided document pages. "
        "If the answer depends on multiple pages, combine the evidence. "
        "Return a concise answer and include evidence page numbers. "
        f"Question: {question}"
    )


def build_docqa_json_instruction(question: str) -> str:
    return (
        "Use all provided document pages to answer the question. "
        "Return strict JSON only with this schema: "
        "{\"answer\": \"\", \"evidence_pages\": []}. "
        "The evidence_pages value must contain 1-based page numbers used for the answer. "
        f"Question: {question}"
    )


def move_inputs_to_device(inputs, device: torch.device):
    inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
    inputs.pop("token_type_ids", None)
    inputs.pop("mm_token_type_ids", None)
    return inputs


def generation_kwargs(tokenizer, max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        if getattr(tokenizer, "eos_token_id", None) is not None:
            kwargs["eos_token_id"] = tokenizer.eos_token_id
    return kwargs


def decode_generated(processor, tokenizer, inputs, outputs) -> str:
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated = outputs[0][input_len:]
    decoder = tokenizer if tokenizer is not None else processor
    return decoder.decode(generated, skip_special_tokens=True).strip()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_page_json_array(text: str, expected_pages: int) -> list[str]:
    raw = text.strip()
    candidate = _strip_json_fence(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return [f"[parse_error expected_pages={expected_pages}] {raw}"]
    if isinstance(parsed, dict):
        parsed = parsed.get("pages") or parsed.get("results") or parsed.get("documents")
    if not isinstance(parsed, list):
        return [f"[parse_error expected_pages={expected_pages}] {raw}"]
    pages: list[str] = []
    schema_errors = 0
    for item in parsed:
        if isinstance(item, dict):
            value = item.get("text") or item.get("content") or item.get("markdown")
            if value is None:
                schema_errors += 1
                value = ""
            pages.append(str(value))
        else:
            pages.append(str(item))
    if schema_errors:
        return [f"[page_schema_mismatch expected_text_fields={expected_pages} invalid={schema_errors}] {raw}"]
    if len(pages) != expected_pages:
        return [f"[page_count_mismatch expected={expected_pages} actual={len(pages)}] {raw}"]
    return pages
