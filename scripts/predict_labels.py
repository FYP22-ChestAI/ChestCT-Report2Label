"""Run the RadBERT classifier over processed reports, saving per-label probabilities and evidence.

    python scripts/predict_labels.py [--input-dir data/processed] [--output-dir data/outputs/predictions] [--no-evidence]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from report2label.extraction.label_mapper import LabelVocabulary
from report2label.extraction.predictor import LabelPredictor
from report2label.extraction.thresholding import resolve_thresholds
from report2label.models.model_loader import ModelConfig
from report2label.utils.io import ensure_dir, list_report_files, read_json, read_yaml, write_json
from report2label.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    pipeline_config = read_yaml(PROJECT_ROOT / "configs" / "pipeline.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["processed_dir"])
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / pipeline_config["paths"]["predictions_dir"])
    )
    parser.add_argument(
        "--no-evidence", action="store_true", help="Skip sentence-level evidence attribution."
    )
    args = parser.parse_args()

    model_config = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    label_vocab = LabelVocabulary.from_yaml(PROJECT_ROOT / "configs" / "labels.yaml")
    thresholds = resolve_thresholds(
        label_vocab.names, model_config.default_threshold, pipeline_config.get("label_thresholds")
    )

    evidence_config = dict(pipeline_config.get("evidence", {}))
    if args.no_evidence:
        evidence_config["enabled"] = False

    predictor = LabelPredictor(model_config, label_vocab, thresholds, evidence_config)

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)
    files = list_report_files(input_dir, [".json"])
    if not files:
        logger.warning("No processed reports under %s — run extract_reports.py first.", input_dir)
        return

    summary_rows = []
    label_rows = []
    for file_path in files:
        parsed = read_json(file_path)
        prediction = predictor.predict(
            parsed["classifier_text"],
            ct_id=parsed.get("ct_id"),
            source_path=parsed["source_path"],
        )
        write_json(prediction.to_dict(), output_dir / file_path.name)

        positive = [name for name, value in prediction.labels.items() if value == 1]
        logger.info("%s (ct_id=%s): %s", file_path.stem, prediction.ct_id, positive or "no findings above threshold")

        row = {"report_id": file_path.stem, "ct_id": prediction.ct_id}
        row.update(prediction.probabilities)
        summary_rows.append(row)

        if prediction.ct_id is None:
            logger.warning("%s has no CT id — left out of labels.csv (see its JSON instead)", file_path.stem)
        else:
            label_rows.append(
                {"ct_id": prediction.ct_id, **{name: prediction.labels[name] for name in label_vocab.names}}
            )

    summary_path = output_dir / "_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    labels_path = output_dir / "labels.csv"
    pd.DataFrame(label_rows, columns=["ct_id", *label_vocab.names]).to_csv(labels_path, index=False)

    logger.info(
        "Saved %d prediction(s) to %s (probabilities: %s, labels: %s)",
        len(files),
        output_dir,
        summary_path.name,
        labels_path.name,
    )


if __name__ == "__main__":
    main()
