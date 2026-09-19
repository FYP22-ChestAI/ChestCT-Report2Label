"""Ingest raw reports (PDF/DOCX/TXT) into cleaned, section-split JSON plus a reports.csv.

    python scripts/extract_reports.py [--input-dir data/raw] [--output-dir data/processed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from report2label.ingestion import read_report_file
from report2label.parsing.report_parser import parse_report
from report2label.utils.io import ensure_dir, list_report_files, read_yaml, write_json
from report2label.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    pipeline_config = read_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["raw_reports_dir"])
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["processed_dir"])
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    extensions = pipeline_config["ingestion"]["supported_extensions"]
    files = list_report_files(input_dir, extensions)
    if not files:
        logger.warning("No report files with extensions %s found under %s", extensions, input_dir)
        return

    section_headers = pipeline_config["sections"]["headers"]
    classifier_input_sections = pipeline_config["sections"]["classifier_input_sections"]
    phi_removal_enabled = pipeline_config["phi_removal"]["enabled"]

    section_names = list(section_headers)
    rows = []
    for file_path in files:
        raw_text = read_report_file(file_path)
        parsed = parse_report(
            raw_text,
            section_headers=section_headers,
            classifier_input_sections=classifier_input_sections,
            phi_removal_enabled=phi_removal_enabled,
            source_path=str(file_path),
        )
        out_path = output_dir / f"{file_path.stem}.json"
        write_json(parsed.to_dict(), out_path)
        rows.append(parsed.to_row(section_names))
        logger.info("%s -> ct_id=%s -> %s", file_path.name, parsed.ct_id, out_path.name)

    reports_path = output_dir / "reports.csv"
    pd.DataFrame(rows).to_csv(reports_path, index=False)
    logger.info("Processed %d report(s) into %s (table: %s)", len(files), output_dir, reports_path.name)


if __name__ == "__main__":
    main()
