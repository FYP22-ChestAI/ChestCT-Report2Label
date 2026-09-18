"""Write dataset records out to disk in a couple of common formats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from report2label.utils.io import ensure_parent_dir, write_json

from .schema import DatasetRecord


def export_to_json(records: list[DatasetRecord], path: str | Path) -> None:
    write_json([r.to_dict() for r in records], path)


def export_to_csv(records: list[DatasetRecord], path: str | Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    pd.DataFrame([r.to_flat_dict() for r in records]).to_csv(path, index=False)
