"""Extract the CT/accession identifier a report is filed under.

The identifier is the key the output labels are filed under, so this runs
before any PHI scrubbing touches the text.
"""

from __future__ import annotations

import re

_ID_PATTERNS = [
    re.compile(r"\bCT\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*)", re.IGNORECASE),
    re.compile(r"\bAccession\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*)", re.IGNORECASE),
    re.compile(r"\b(?:Exam|Study)\s*(?:No\.?|ID)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/]*)", re.IGNORECASE),
]

def extract_ct_id(text: str) -> str | None:
    """Return the first CT/accession id found in the report text, or None."""
    for pattern in _ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().rstrip(".")
    return None
