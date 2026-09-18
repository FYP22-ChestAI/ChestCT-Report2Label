"""Raw report ingestion: turn a file on disk into plain text, regardless of format."""

from __future__ import annotations

from pathlib import Path

from .docx_reader import read_docx
from .pdf_reader import read_pdf
from .txt_reader import read_txt

_READERS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".txt": read_txt,
}


def read_report_file(path: str | Path) -> str:
    """Dispatch to the right reader based on file extension and return raw text."""
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported report file type '{path.suffix}' for {path}. "
            f"Supported: {sorted(_READERS)}"
        )
    return reader(path)


__all__ = ["read_report_file", "read_pdf", "read_docx", "read_txt"]
