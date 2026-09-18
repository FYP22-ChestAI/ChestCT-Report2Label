"""The record shape written out by the dataset builder/exporter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetRecord:
    report_id: str
    ct_id: str | None
    report_path: str
    scan_path: str | None
    labels: dict[str, int]
    probabilities: dict[str, float]
    evidence: dict[str, list[dict]] | None = None

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "ct_id": self.ct_id,
            "report_path": self.report_path,
            "scan_path": self.scan_path,
            "labels": self.labels,
            "probabilities": self.probabilities,
            "evidence": self.evidence,
        }

    def to_flat_dict(self) -> dict:
        """One row per record with a column per label — the tabular dataset form."""
        row = {
            "report_id": self.report_id,
            "ct_id": self.ct_id,
            "report_path": self.report_path,
            "scan_path": self.scan_path,
        }
        row.update(self.labels)
        return row
