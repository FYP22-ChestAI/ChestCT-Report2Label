"""Match processed reports to CT scan files/folders by CT id, producing a manifest.

    python scripts/match_ct_reports.py --scans-dir path/to/scans
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from report2label.matching.ct_report_matcher import match_reports_to_scans
from report2label.matching.manifest_builder import build_manifest, save_manifest
from report2label.utils.io import list_report_files, read_json, read_yaml
from report2label.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    pipeline_config = read_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["processed_dir"])
    )
    parser.add_argument(
        "--scans-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["scans_dir"])
    )
    parser.add_argument(
        "--predictions-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["predictions_dir"])
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / pipeline_config["paths"]["manifests_dir"] / "manifest.csv"),
    )
    args = parser.parse_args()

    reports = [read_json(p) for p in list_report_files(args.reports_dir, [".json"])]

    predictions_by_path = {}
    for pred_path in list_report_files(args.predictions_dir, [".json"]):
        prediction = read_json(pred_path)
        predictions_by_path[prediction["source_path"]] = prediction

    id_pattern = pipeline_config["matching"]["id_normalize_pattern"]
    match_results = match_reports_to_scans(reports, args.scans_dir, id_pattern)

    matched = sum(1 for m in match_results if m.matched)
    logger.info("Matched %d/%d report(s) to a scan under %s", matched, len(match_results), args.scans_dir)

    rows = build_manifest(match_results, predictions_by_path)
    save_manifest(rows, args.output)
    logger.info("Manifest saved to %s", args.output)


if __name__ == "__main__":
    main()
