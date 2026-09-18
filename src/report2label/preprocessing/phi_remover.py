"""Strip identifying information out of report text before it is persisted or logged.

This is pattern-based on generic reporting conventions (letterhead, signature
blocks, demographic key/value lines) rather than any specific institution, so
the same rules apply regardless of which facility a report came from.

The CT/accession id is deliberately NOT touched here — parsing.ct_id_parser
extracts it upstream (it's needed to match a report back to its scan), and
callers are responsible for treating it as an identifier once extracted.
"""

from __future__ import annotations

import re

# Whole-line patterns: if a line matches, the entire line is dropped.
_LINE_PATTERNS = [
    re.compile(r"department\s+of\s+radiology", re.IGNORECASE),
    re.compile(r"\b(hospital|clinic|medical\s+centre|medical\s+center|institute)\b", re.IGNORECASE),
    re.compile(r"^\s*(consultant|reporting)\s+radiologist\b", re.IGNORECASE),
    re.compile(r"^\s*dr\.?\s*[a-z]", re.IGNORECASE),
    re.compile(r"thank you for referring", re.IGNORECASE),
    re.compile(r"^\s*(reported|signed|verified|approved)\s+by\b", re.IGNORECASE),
]

# Key/value demographic fields: the line is dropped if it starts with one of
# these labels, wherever the report places them.
_FIELD_PATTERNS = [
    re.compile(r"^\s*(patient\s*name|name)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*(date\s+of\s+birth|dob)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*(age\s*/?\s*sex|age|sex|gender)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*(address|phone|contact|email)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*(referring\s+physician|referred\s+by|referring\s+doctor)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^\s*(ward|bed|admission\s+no|hospital\s+no|nic|passport)\s*[:\-]", re.IGNORECASE),
]

_ALL_PATTERNS = _LINE_PATTERNS + _FIELD_PATTERNS


def remove_phi(text: str) -> str:
    """Drop lines that look like letterhead, signature blocks, or patient demographics."""
    kept_lines = []
    for line in text.split("\n"):
        if any(pattern.search(line) for pattern in _ALL_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)
