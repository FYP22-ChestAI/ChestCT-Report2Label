"""Build and persist the report<->scan<->label manifest used to assemble the dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from report2label.utils.io import ensure_parent_dir, write_json

from .ct_report_matcher import MatchResult


def build_manifest(
    match_results: list[MatchResult],
    predictions_by_path: dict[str, dict] | None = None,
) -> list[dict]:
    """`predictions_by_path` maps report source_path -> a LabelPrediction.to_dict()."""
    predictions_by_path = predictions_by_path or {}
    rows = []
    for match in match_results:
        row = {
            "ct_id": match.ct_id,
            "report_path": match.report_path,
            "scan_path": match.scan_path,
            "matched": match.matched,
        }
        prediction = predictions_by_path.get(match.report_path)
        if prediction:
            for label, value in prediction["labels"].items():
                row[f"label__{label}"] = value
        rows.append(row)
    return rows


def save_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    if path.suffix.lower() == ".json":
        write_json(rows, path)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)
