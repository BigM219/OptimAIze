# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# SVG-mode helpers for dots.mocr.
#
# The model's ``prompt_image_to_svg`` mode reproduces the source page as
# an SVG. Each glyph/line lands inside ``<text>`` elements with x/y/font
# attributes that we can mine to recover both the transcription and an
# approximate bounding box. This is significantly more accurate than
# per-crop OCR on small / quantized models because the SVG prompt
# anchors the model to actual pixel coordinates via the viewBox.
#
# Vendor: dots.mocr/dots_mocr/utils/svg_utils.py — ``fix_svg`` and
# ``extract_svg_from_response`` are ported here verbatim (Apache-2.0).

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vendor-ported SVG repair (handles truncated <path d="..."> and stack
# imbalances caused by max_new_tokens cutoff).
# ---------------------------------------------------------------------------

def fix_svg(svg: str) -> str:
    """Repair common SVG truncation artefacts.

    1. Close a dangling ``d="..."`` path attribute.
    2. Trim a trailing partial tag.
    3. Walk the tag stream and emit closers for every still-open tag.
    """
    if re.search(r'(<path\b[^>]*\bd="[^">]*$)', svg):
        svg += '">'

    svg = re.sub(r'<[^>]*$', '', svg)

    stack: list[str] = []
    tag_re = re.compile(r'</?\s*([a-zA-Z][\w:-]*)\b[^>]*?/?>')
    for m in tag_re.finditer(svg):
        name = m.group(1)
        token = m.group(0)
        is_close = token.lstrip().startswith("</")
        is_self_close = token.rstrip().endswith("/>")
        if is_self_close:
            continue
        if not is_close:
            stack.append(name)
        else:
            if name in stack[::-1]:
                while stack and stack[-1] != name:
                    stack.pop()
                if stack and stack[-1] == name:
                    stack.pop()

    while stack:
        svg += f'</{stack.pop()}>'
    return svg


def extract_svg_from_response(response: str) -> Tuple[Optional[str], bool]:
    """Pull the first ``<svg>...</svg>`` block out of a model response.

    Returns ``(svg_string, ok)``. If the SVG is truncated mid-stream we
    repair it via :func:`fix_svg`.
    """
    response = response.replace("svg:", "").strip()

    m = re.search(r'<svg[^>]*>(.*?)</svg>', response, re.DOTALL)
    if m:
        return m.group(0), True

    m = re.search(r'<svg[^>]*>.*', response, re.DOTALL)
    if m:
        return fix_svg(m.group(0)), True

    return None, False


# ---------------------------------------------------------------------------
# Layout-element extraction from SVG
# ---------------------------------------------------------------------------

_TEXT_BLOCK = re.compile(
    r"<text\b([^>]*)>([\s\S]*?)</text>",
    re.IGNORECASE,
)
_ATTR = re.compile(r'(\w[\w:-]*)\s*=\s*"([^"]*)"')
_INNER_TAG = re.compile(r"<[^>]+>")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _attrs(s: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _ATTR.finditer(s)}


def _to_float(v: str | None, default: float = 0.0) -> float:
    if not v:
        return default
    m = _NUM.search(v)
    if not m:
        return default
    try:
        return float(m.group(0))
    except ValueError:
        return default


def _decode_entities(text: str) -> str:
    """Decode the small subset of HTML entities the model emits."""
    return (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
    )


def svg_to_layout_elements(
    svg: str,
    image_width: int,
    image_height: int,
    line_merge_threshold: float = 0.6,
) -> list[dict]:
    """Convert an SVG transcription into a flat list of layout dicts.

    Each ``<text>`` element becomes one ``{"category", "bbox", "score",
    "text"}`` record. Adjacent text elements on the same baseline are
    merged into a single line so downstream markdown looks natural.

    ``line_merge_threshold`` controls how aggressively elements are
    grouped — values express a fraction of the local font height.
    """
    items: list[dict] = []
    for m in _TEXT_BLOCK.finditer(svg):
        attrs = _attrs(m.group(1))
        inner_raw = m.group(2)
        # Strip nested SVG tags (tspan, etc.) but keep their text.
        inner = _INNER_TAG.sub("", inner_raw)
        text = _decode_entities(inner).strip()
        if not text:
            continue

        x = _to_float(attrs.get("x"))
        y = _to_float(attrs.get("y"))
        font_size = _to_float(attrs.get("font-size"), default=14.0)

        # Try to read text-anchor for x adjustment.
        anchor = (attrs.get("text-anchor") or "start").lower()

        # Approximate width: 0.55 * font_size per char is a reasonable
        # default for proportional fonts (we never need pixel-perfect
        # bboxes here, only reading order).
        approx_w = max(1.0, 0.55 * font_size * max(1, len(text)))
        approx_h = font_size * 1.2

        if anchor == "middle":
            x1 = x - approx_w / 2
        elif anchor == "end":
            x1 = x - approx_w
        else:
            x1 = x

        # In SVG, y is the baseline → ascender goes up.
        y1 = y - approx_h * 0.85
        x2 = x1 + approx_w
        y2 = y1 + approx_h

        items.append({
            "_x": x,
            "_y": y,
            "_font": font_size,
            "_anchor": anchor,
            "bbox": [
                max(0, int(x1)),
                max(0, int(y1)),
                min(image_width, int(x2)),
                min(image_height, int(y2)),
            ],
            "category": "text",
            "score": 1.0,
            "text": text,
        })

    if not items:
        return []

    # Sort by reading order (top-to-bottom, left-to-right).
    items.sort(key=lambda it: (it["_y"], it["_x"]))

    # Merge adjacent items that share a baseline into one line.
    merged: list[dict] = []
    for it in items:
        if merged:
            last = merged[-1]
            same_line = abs(it["_y"] - last["_y"]) <= line_merge_threshold * max(
                last["_font"], it["_font"]
            )
            if same_line:
                last["text"] = (last["text"] + " " + it["text"]).strip()
                last["bbox"] = [
                    min(last["bbox"][0], it["bbox"][0]),
                    min(last["bbox"][1], it["bbox"][1]),
                    max(last["bbox"][2], it["bbox"][2]),
                    max(last["bbox"][3], it["bbox"][3]),
                ]
                continue
        merged.append(it.copy())

    # Heuristic category tagging based on font size (top-quintile = title).
    if merged:
        font_sizes = sorted({m_.get("_font", 14.0) for m_ in merged}, reverse=True)
        if len(font_sizes) >= 2:
            title_threshold = font_sizes[0] - 0.1
            header_threshold = font_sizes[min(2, len(font_sizes) - 1)] - 0.1
        else:
            title_threshold = font_sizes[0] + 1
            header_threshold = font_sizes[0] + 1
        for m_ in merged:
            fs = m_.get("_font", 14.0)
            if fs >= title_threshold:
                m_["category"] = "title"
            elif fs >= header_threshold:
                m_["category"] = "section-header"

    # Strip private fields before returning.
    clean: list[dict] = []
    for m_ in merged:
        clean.append({
            "category": m_["category"],
            "bbox": m_["bbox"],
            "score": m_["score"],
            "text": m_["text"],
        })
    return clean
