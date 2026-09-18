"""Extract plain text from DOCX radiology reports."""

from __future__ import annotations

from pathlib import Path

import docx


def read_docx(path: str | Path) -> str:
    """Extract paragraph text (and table cell text) from a .docx file, in document order."""
    document = docx.Document(str(path))
    parts: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(part for part in parts if part.strip())
