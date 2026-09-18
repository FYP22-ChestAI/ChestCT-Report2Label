"""Extract plain text from PDF radiology reports."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def read_pdf(path: str | Path) -> str:
    """Extract text from every page of a PDF, in reading order, joined by blank lines."""
    path = Path(path)
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages).strip()
