"""Compare saved predictions against a small manually annotated ground-truth CSV.

The annotations CSV needs a `ct_id` column plus one 0/1 column per label name
in configs/labels.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from report2label.utils.io import list_report_files, read_json

from .metrics import compute_multilabel_metrics


def load_predictions(predictions_dir: str | Path) -> list[dict]:
    return [read_json(p) for p in list_report_files(predictions_dir, [".json"])]


def evaluate_predictions(
    predictions: list[dict], annotations_path: str | Path, label_names: list[str]
) -> dict:
    annotations = pd.read_csv(annotations_path, dtype={"ct_id": str})
    predictions_by_ct_id = {p["ct_id"]: p for p in predictions if p.get("ct_id")}

    y_true, y_pred, matched_ids = [], [], []
    for _, row in annotations.iterrows():
        ct_id = str(row["ct_id"])
        prediction = predictions_by_ct_id.get(ct_id)
        if prediction is None:
            continue
        y_true.append([int(row[name]) for name in label_names])
        y_pred.append([prediction["labels"][name] for name in label_names])
        matched_ids.append(ct_id)

    if not matched_ids:
        raise ValueError(
            "No annotated report matched a saved prediction by ct_id — check that "
            "predictions were generated for the same reports as the annotations."
        )

    metrics = compute_multilabel_metrics(np.array(y_true), np.array(y_pred), label_names)
    metrics["n_matched"] = len(matched_ids)
    metrics["matched_ct_ids"] = matched_ids
    return metrics
