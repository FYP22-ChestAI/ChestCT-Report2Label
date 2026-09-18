"""Combine the report<->scan manifest and saved predictions into the final labeled dataset.

    python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from report2label.dataset.builder import build_dataset_records
from report2label.dataset.exporter import export_to_csv, export_to_json
from report2label.utils.io import list_report_files, read_json, read_yaml
from report2label.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    pipeline_config = read_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / pipeline_config["paths"]["manifests_dir"] / "manifest.csv"),
    )
    parser.add_argument(
        "--predictions-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["predictions_dir"])
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "outputs"))
    parser.add_argument("--format", choices=["csv", "json", "both"], default="both")
    args = parser.parse_args()

    manifest_rows = pd.read_csv(
        args.manifest, dtype={"ct_id": str, "report_path": str, "scan_path": str}
    ).to_dict(orient="records")

    predictions_by_path = {}
    for pred_path in list_report_files(args.predictions_dir, [".json"]):
        prediction = read_json(pred_path)
        predictions_by_path[prediction["source_path"]] = prediction

    records = build_dataset_records(manifest_rows, predictions_by_path)
    logger.info("Built %d dataset record(s)", len(records))

    output_dir = Path(args.output_dir)
    if args.format in ("csv", "both"):
        csv_path = output_dir / "dataset.csv"
        export_to_csv(records, csv_path)
        logger.info("Saved %s", csv_path)
    if args.format in ("json", "both"):
        json_path = output_dir / "dataset.json"
        export_to_json(records, json_path)
        logger.info("Saved %s", json_path)


if __name__ == "__main__":
    main()
