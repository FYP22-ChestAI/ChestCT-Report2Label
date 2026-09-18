"""Multi-label classification metrics against manually annotated ground truth."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_multilabel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = {
        name: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for name, p, r, f, s in zip(label_names, precision, recall, f1, support)
    }
    overall = {
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "accuracy_flat": round(float(accuracy_score(y_true.flatten(), y_pred.flatten())), 4),
    }
    return {"per_label": per_label, "overall": overall}
