"""Assemble final dataset records from a manifest plus saved predictions."""

from __future__ import annotations

from pathlib import Path

from .schema import DatasetRecord


def build_dataset_records(
    manifest_rows: list[dict],
    predictions_by_path: dict[str, dict],
) -> list[DatasetRecord]:
    records = []
    for row in manifest_rows:
        report_path = row["report_path"]
        prediction = predictions_by_path.get(report_path)
        if prediction is None:
            continue
        records.append(
            DatasetRecord(
                report_id=Path(report_path).stem,
                ct_id=row.get("ct_id"),
                report_path=report_path,
                scan_path=row.get("scan_path"),
                labels=prediction["labels"],
                probabilities=prediction["probabilities"],
                evidence=prediction.get("evidence"),
            )
        )
    return records
