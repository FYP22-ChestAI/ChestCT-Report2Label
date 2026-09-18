"""Low-level text cleanup applied right after extraction, before any parsing."""

from __future__ import annotations

import re

_DEHYPHENATE_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_FORM_FEED_RE = re.compile(r"\f")


def clean_text(text: str) -> str:
    """Normalize whitespace and undo PDF line-wrap hyphenation artifacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FORM_FEED_RE.sub("\n", text)
    text = _DEHYPHENATE_RE.sub(r"\1\2", text)
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
