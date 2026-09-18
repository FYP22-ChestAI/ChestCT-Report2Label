"""Match parsed reports to their CT scan files/folders by CT/accession id."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from report2label.parsing.ct_id_parser import normalize_ct_id


@dataclass
class MatchResult:
    ct_id: str | None
    report_path: str
    scan_path: str | None

    @property
    def matched(self) -> bool:
        return self.scan_path is not None


def _full_stem(path: Path) -> str:
    """Strip every suffix, not just the last one — scans are often named e.g. '1234.nii.gz'."""
    name = path.name
    while "." in name:
        name = Path(name).stem
    return name


def build_scan_id_index(scans_dir: str | Path, id_normalize_pattern: str) -> dict[str, Path]:
    """Index every file/folder directly under `scans_dir` by its normalized stem."""
    scans_dir = Path(scans_dir)
    index: dict[str, Path] = {}
    if not scans_dir.exists():
        return index
    for entry in scans_dir.iterdir():
        key = normalize_ct_id(_full_stem(entry), id_normalize_pattern)
        if key:
            index[key] = entry
    return index


def match_reports_to_scans(
    reports: list[dict],
    scans_dir: str | Path,
    id_normalize_pattern: str = r"[^0-9A-Za-z]",
) -> list[MatchResult]:
    """`reports` is a list of {"ct_id": ..., "source_path": ...} dicts."""
    index = build_scan_id_index(scans_dir, id_normalize_pattern)

    results = []
    for report in reports:
        ct_id = report.get("ct_id")
        scan_path = None
        if ct_id:
            key = normalize_ct_id(ct_id, id_normalize_pattern)
            match = index.get(key)
            scan_path = str(match) if match else None
        results.append(
            MatchResult(ct_id=ct_id, report_path=report["source_path"], scan_path=scan_path)
        )
    return results
