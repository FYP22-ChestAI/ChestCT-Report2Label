"""Smoke-test that the RadBERT classifier loads and runs, before pointing it at real reports.

    python scripts/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from report2label.extraction.label_mapper import LabelVocabulary
from report2label.extraction.predictor import LabelPredictor
from report2label.extraction.thresholding import resolve_thresholds
from report2label.models.model_loader import ModelConfig
from report2label.utils.logging import get_logger

logger = get_logger(__name__)

SAMPLE_SENTENCES = [
    "There is a pleural effusion in the right hemithorax.",
    "The lungs are clear without focal consolidation, nodule, or effusion.",
    "Mild cardiomegaly with mediastinal lymphadenopathy.",
]


def main() -> None:
    model_config = ModelConfig.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    label_vocab = LabelVocabulary.from_yaml(PROJECT_ROOT / "configs" / "labels.yaml")
    thresholds = resolve_thresholds(label_vocab.names, model_config.default_threshold)

    predictor = LabelPredictor(model_config, label_vocab, thresholds, evidence_config={"enabled": False})

    for sentence in SAMPLE_SENTENCES:
        prediction = predictor.predict(sentence, with_evidence=False)
        top = sorted(prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:5]
        logger.info("Text: %s", sentence)
        for name, prob in top:
            logger.info("    %-40s %.3f", name, prob)


if __name__ == "__main__":
    main()
