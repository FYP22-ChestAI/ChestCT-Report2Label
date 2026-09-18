"""List individual label mismatches, with evidence text, for manual review."""

from __future__ import annotations

import pandas as pd


def find_mismatches(
    predictions: list[dict], annotations: pd.DataFrame, label_names: list[str]
) -> list[dict]:
    predictions_by_ct_id = {p["ct_id"]: p for p in predictions if p.get("ct_id")}

    mismatches = []
    for _, row in annotations.iterrows():
        ct_id = str(row["ct_id"])
        prediction = predictions_by_ct_id.get(ct_id)
        if prediction is None:
            continue
        for name in label_names:
            true_value = int(row[name])
            predicted_value = prediction["labels"][name]
            if true_value == predicted_value:
                continue
            probability = prediction["probabilities"][name]
            mismatches.append(
                {
                    "ct_id": ct_id,
                    "label": name,
                    "true": true_value,
                    "predicted": predicted_value,
                    "probability": probability,
                    "kind": "false_positive" if predicted_value == 1 else "false_negative",
                    "evidence": (prediction.get("evidence") or {}).get(name, []),
                }
            )

    # Most confidently-wrong predictions first — those are the ones worth
    # inspecting manually before the noisier borderline cases.
    mismatches.sort(key=lambda m: abs(m["probability"] - 0.5), reverse=True)
    return mismatches
