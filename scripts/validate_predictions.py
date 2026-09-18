"""Score saved predictions against a small manually annotated CSV and rank mismatches.

    python scripts/validate_predictions.py --annotations data/annotations/manual_labels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from report2label.extraction.label_mapper import LabelVocabulary
from report2label.utils.io import read_yaml, write_json
from report2label.utils.logging import get_logger
from report2label.validation.error_analysis import find_mismatches
from report2label.validation.evaluator import evaluate_predictions, load_predictions

logger = get_logger(__name__)


def main() -> None:
    pipeline_config = read_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml")
    label_vocab = LabelVocabulary.from_yaml(PROJECT_ROOT / "configs" / "labels.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["predictions_dir"])
    )
    parser.add_argument(
        "--annotations", required=True, help="CSV with a ct_id column plus one 0/1 column per label."
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / pipeline_config["paths"]["manifests_dir"] / "validation_report.json"),
    )
    args = parser.parse_args()

    predictions = load_predictions(args.predictions_dir)
    metrics = evaluate_predictions(predictions, args.annotations, label_vocab.names)

    annotations = pd.read_csv(args.annotations, dtype={"ct_id": str})
    mismatches = find_mismatches(predictions, annotations, label_vocab.names)

    logger.info("Matched %d annotated report(s)", metrics["n_matched"])
    logger.info(
        "Overall: f1_micro=%.3f f1_macro=%.3f accuracy_flat=%.3f",
        metrics["overall"]["f1_micro"],
        metrics["overall"]["f1_macro"],
        metrics["overall"]["accuracy_flat"],
    )
    for mismatch in mismatches[:10]:
        logger.info(
            "  [%s] %s: true=%d predicted=%d (p=%.3f)",
            mismatch["ct_id"],
            mismatch["label"],
            mismatch["true"],
            mismatch["predicted"],
            mismatch["probability"],
        )

    write_json({"metrics": metrics, "mismatches": mismatches}, args.output)
    logger.info("Full validation report saved to %s", args.output)


if __name__ == "__main__":
    main()
