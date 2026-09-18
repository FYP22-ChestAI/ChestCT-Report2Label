"""End-to-end label prediction: report text in, probabilities + evidence out."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from report2label.models.model_loader import ModelConfig, load_model_and_tokenizer

from .evidence import select_evidence, split_into_sentences
from .label_mapper import LabelVocabulary
from .thresholding import apply_thresholds


@dataclass
class LabelPrediction:
    source_path: str
    ct_id: str | None
    probabilities: dict[str, float]
    labels: dict[str, int]
    evidence: dict[str, list[dict]] | None = field(default=None)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "ct_id": self.ct_id,
            "probabilities": self.probabilities,
            "labels": self.labels,
            "evidence": self.evidence,
        }


class LabelPredictor:
    def __init__(
        self,
        model_config: ModelConfig,
        label_vocab: LabelVocabulary,
        thresholds: dict[str, float],
        evidence_config: dict | None = None,
        batch_size: int = 16,
    ):
        self.tokenizer, self.model, self.device = load_model_and_tokenizer(model_config)
        self.label_vocab = label_vocab
        self.max_length = model_config.max_length
        self.thresholds = thresholds
        self.evidence_config = evidence_config or {}
        self.batch_size = batch_size

    def _score_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.label_vocab)))

        all_probs = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            encodings = self.tokenizer(
                chunk,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    def predict(
        self,
        text: str,
        *,
        ct_id: str | None = None,
        source_path: str = "",
        with_evidence: bool = True,
    ) -> LabelPrediction:
        probs = self._score_texts([text])[0]
        probabilities = {name: float(p) for name, p in zip(self.label_vocab.names, probs)}
        labels = apply_thresholds(probabilities, self.thresholds)

        evidence = None
        if with_evidence and self.evidence_config.get("enabled", True):
            sentences = split_into_sentences(
                text, min_chars=self.evidence_config.get("min_sentence_chars", 6)
            )
            if sentences:
                sentence_probs = self._score_texts(sentences)
                sentence_scores = {
                    name: sentence_probs[:, i].tolist()
                    for i, name in enumerate(self.label_vocab.names)
                }
                evidence = select_evidence(
                    sentence_scores,
                    sentences,
                    self.evidence_config.get("max_sentences_per_label", 3),
                )
            else:
                evidence = {name: [] for name in self.label_vocab.names}

        return LabelPrediction(
            source_path=source_path,
            ct_id=ct_id,
            probabilities=probabilities,
            labels=labels,
            evidence=evidence,
        )
