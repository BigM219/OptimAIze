# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Post-processor for dots.mocr DOTS_MOCR_PROMPT_LAYOUT_ALL_EN output.
#
# dots.mocr is trained to output a single JSON object describing layout
# elements.  This module strips reasoning traces and extracts the clean
# text field(s) appropriate for the requested category.

import json
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _strip_descriptive_prefix(text: str) -> str:
    """Remove ``The text in the image is "X".`` style wrapper.

    dots.mocr (especially the INT8 variant) sometimes ignores the OCR
    instruction and answers conversationally. The actual transcription
    is always inside the first quoted span; everything outside is noise.
    """
    patterns = [
        r'^\s*(?:The\s+)?(?:text|content|caption|header|title|word|words?|writing)\s+'
        r'(?:shown\s+|displayed\s+|that\s+(?:appears?|is\s+visible)\s+)?'
        r'in\s+(?:this|the)\s+image\s+(?:is|reads?|says?|shows)[: ]*\s*'
        r'["\u201c\u201d\']?(.*?)["\u201c\u201d\'\.\s]*$',
        r'^\s*(?:The|This)\s+image\s+(?:shows|contains|displays)\s+'
        r'(?:the\s+)?(?:text|words?)[: ]*\s*'
        r'["\u201c\u201d\']?(.*?)["\u201c\u201d\'\.\s]*$',
    ]
    for pat in patterns:
        m = re.match(pat, text.strip(), re.IGNORECASE | re.DOTALL)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return text


def _extract_svg_text(text: str) -> str | None:
    """Extract text content from SVG <text> elements.

    When dots.mocr is asked to convert a small table to HTML it often
    outputs an SVG mock-up instead. The actual cell strings still land
    inside ``<text>`` elements; the SVG scaffolding is throw-away.
    """
    if "<text" not in text and "<svg" not in text:
        return None
    matches = re.findall(r"<text[^>]*>([\s\S]*?)</text>", text)
    if not matches:
        return None
    cells = []
    for m in matches:
        # Strip nested SVG tags + whitespace
        clean = re.sub(r"<[^>]+>", "", m).strip()
        if clean and clean not in cells:
            cells.append(clean)
    if not cells:
        return None
    return " | ".join(cells)


def _strip_repetition_loops(text: str) -> str:
    """Cut output once we detect a degenerate repeating loop.

    dots.mocr (and Qwen2 in general) sometimes falls into greedy
    repetition: the same phrase or line printed dozens of times until
    ``max_new_tokens`` is hit. The original content is preserved in the
    *first* occurrence; everything after it is noise.
    """
    # Phrase-level: any 30+ char substring repeated 3+ times in a row
    m = re.search(r"(.{30,}?)\1{2,}", text, re.DOTALL)
    if m:
        text = text[: m.start() + len(m.group(1))]
    # Line-level: same line repeated 3+ times in a row
    lines = text.splitlines()
    cleaned: list[str] = []
    prev = None
    streak = 0
    for line in lines:
        norm = line.strip()
        if norm and norm == prev:
            streak += 1
            if streak >= 2:  # already saw it twice, skip further duplicates
                continue
        else:
            streak = 0
            prev = norm
        cleaned.append(line)
    return "\n".join(cleaned)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_special_tokens(text: str) -> str:
    text = re.sub(r"<\|[^>]+>", " ", text)
    return " ".join(text.split())


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_from_json(text: str, category: str) -> str | None:
    """Try to parse JSON layout output and return the text for this category."""
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not m:
        return None

    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError:
        # Try to find a partial JSON object before truncation
        partial = m.group()
        # If it ends mid-token, try truncating to last complete element
        for end_char in ["},", "}", "]"]:
            idx = partial.rfind(end_char)
            if idx > 0:
                try:
                    parsed = json.loads(partial[: idx + len(end_char.rstrip(","))])
                    break
                except json.JSONDecodeError:
                    continue
        else:
            return None

    # Normalise to a flat list of element dicts
    elements: list[dict] = []
    if isinstance(parsed, list):
        elements = [e for e in parsed if isinstance(e, dict)]
    elif isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("layout", "elements", "blocks", "result", "content", "items"):
            if key in parsed and isinstance(parsed[key], list):
                elements = [e for e in parsed[key] if isinstance(e, dict)]
                break
        if not elements:
            # Single element with a "text" key
            if "text" in parsed:
                return str(parsed["text"])
            return None

    if not elements:
        return None

    texts: list[str] = []
    for el in elements:
        el_text = el.get("text", "")
        if el_text and str(el_text).strip():
            texts.append(str(el_text).strip())

    if not texts:
        return None

    if category == "table":
        # Prefer the first HTML table block
        for t in texts:
            if t.lstrip().startswith("<table"):
                return t
    if category == "formula":
        # Prefer LaTeX
        for t in texts:
            if "$" in t or "\\" in t:
                return t

    return "\n".join(texts)


def _extract_coord_text(text: str) -> str | None:
    """Extract text from 4-corner coordinate format.

    dots.mocr sometimes falls back to:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4)TEXT_CONTENT
    or a truncated 3-corner variant when max_new_tokens is tight.
    """
    pattern = re.compile(
        r"(?:\(\s*\d+\s*,\s*\d+\s*\)\s*,?\s*){3,4}([^\(]+)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if matches:
        return " ".join(m.strip() for m in matches if m.strip())
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dots_mocr_output(raw_text: str, category: str) -> str:
    """Parse dots.mocr layout-format output and return clean text.

    Handles:
    * <think>...</think> reasoning chains
    * \\boxed{answer} format
    * JSON layout objects (layout/elements key, or bare list)
    * Coordinate-based OCR format (x1,y1),(x2,y2),(x3,y3),(x4,y4)TEXT
    * Plain markdown / raw text fallback
    """
    text = raw_text.strip()

    # 1. Strip reasoning chain
    text = _strip_think(text)

    # 1b. Strip special tokens that sometimes leak into decoded OCR text
    text = _strip_special_tokens(text)

    # 1c. Cut runaway repetition loops (greedy decode degeneracy)
    text = _strip_repetition_loops(text)

    # 1d. Strip descriptive prefixes (model answering conversationally)
    text = _strip_descriptive_prefix(text)

    # 2. Check for \boxed{answer} — model uses this for short single-token answers
    boxed = _extract_boxed(text)
    if boxed:
        remainder = re.sub(r"\\boxed\{.*?\}", "", text, flags=re.DOTALL).strip()
        # Only use boxed if it's the dominant content
        if not remainder or len(boxed) >= len(remainder) * 0.3:
            logger.debug("[dots-mocr parser] boxed answer extracted")
            return boxed

    # 3. Strip markdown code fences
    text = _strip_code_fences(text)

    # 4. Try JSON layout extraction
    result = _extract_from_json(text, category)
    if result:
        logger.debug("[dots-mocr parser] JSON extraction succeeded, category=%s", category)
        return result

    # 5. Try SVG/HTML <text> extraction (model emits SVG mockups for tables)
    result = _extract_svg_text(text)
    if result:
        logger.debug("[dots-mocr parser] SVG <text> extraction succeeded, category=%s", category)
        return result

    # 6. Try coordinate-based extraction
    result = _extract_coord_text(text)
    if result:
        logger.debug("[dots-mocr parser] coordinate extraction succeeded, category=%s", category)
        return result

    # 7. Return cleaned text as-is
    logger.debug("[dots-mocr parser] fallback raw text, category=%s", category)
    return text
