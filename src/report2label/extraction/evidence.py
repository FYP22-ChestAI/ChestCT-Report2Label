"""Sentence-level evidence attribution.

The classifier itself has no built-in attention/highlighting output, so
evidence is produced by re-scoring the report one sentence at a time with
the same model and keeping the top-scoring sentences per label. This is
what powers the probability + evidence-text debug output used to manually
sanity-check predictions.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_sentences(text: str, min_chars: int = 6) -> list[str]:
    """Split report text into candidate evidence sentences."""
    candidates = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in candidates if len(s.strip()) >= min_chars]


def select_evidence(
    sentence_scores: dict[str, list[float]],
    sentences: list[str],
    max_sentences_per_label: int = 3,
) -> dict[str, list[dict]]:
    """For each label, keep the top-scoring sentences (by that label's probability).

    `sentence_scores[label]` is a list of probabilities, aligned index-for-index
    with `sentences`.
    """
    evidence: dict[str, list[dict]] = {}
    for label, scores in sentence_scores.items():
        ranked = sorted(zip(sentences, scores), key=lambda pair: pair[1], reverse=True)
        evidence[label] = [
            {"sentence": sentence, "probability": round(score, 4)}
            for sentence, score in ranked[:max_sentences_per_label]
        ]
    return evidence
