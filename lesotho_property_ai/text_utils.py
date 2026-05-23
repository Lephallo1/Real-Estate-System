"""Text-cleaning helpers shared across scraping, pipelines, and the Flask UI."""

from __future__ import annotations

from html import unescape
import re


_BREAK_TAG_PATTERN = re.compile(r"<\s*(?:br|/p|/div|/li|/ul|/ol|/section|/article)\b[^>]*>", re.I)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_html_text(value: object) -> str:
    """Return clean plain text even when the input contains embedded HTML."""

    raw = str(value or "")
    if not raw.strip():
        return ""

    cleaned = raw
    for _ in range(2):
        cleaned = unescape(cleaned)
    cleaned = _BREAK_TAG_PATTERN.sub(" ", cleaned)
    cleaned = _TAG_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned

