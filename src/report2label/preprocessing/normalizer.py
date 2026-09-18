"""Surface-form normalization for radiology report text.

Applied after cleaning so downstream sentence splitting and the classifier
see a consistent surface form, independent of how a given reporting system
happened to space out its measurements or bullets.
"""

from __future__ import annotations

import re

_UNIT_RE = re.compile(r"(?<=\d)(mm|cm|ml|cc)\b", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[•●–—*]\s*", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_COLON_DASH_RE = re.compile(r":-\s*")


def normalize_text(text: str) -> str:
    text = _UNIT_RE.sub(r" \1", text)
    text = _BULLET_RE.sub("- ", text)
    text = _COLON_DASH_RE.sub(": ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text
