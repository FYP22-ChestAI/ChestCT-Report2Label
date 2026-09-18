"""Read plain-text radiology reports."""

from __future__ import annotations

from pathlib import Path


def read_txt(path: str | Path, encoding: str = "utf-8") -> str:
    path = Path(path)
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Some exported reports carry a Windows codepage instead of UTF-8.
        return path.read_text(encoding="cp1252", errors="replace")
