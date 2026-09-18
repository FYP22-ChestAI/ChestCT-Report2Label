from pathlib import Path

from report2label.extraction.evidence import select_evidence, split_into_sentences
from report2label.extraction.label_mapper import LabelVocabulary
from report2label.extraction.thresholding import apply_thresholds, resolve_thresholds

LABELS_PATH = Path(__file__).resolve().parents[1] / "configs" / "labels.yaml"


def test_label_vocabulary_loads_18_labels_in_order():
    vocab = LabelVocabulary.from_yaml(LABELS_PATH)
    assert len(vocab) == 18
    assert vocab.names[0] == "Medical material"
    assert vocab.names[-1] == "Interlobular septal thickening"
    assert vocab.index_of("Cardiomegaly") == 2
    assert vocab.name_at(2) == "Cardiomegaly"


def test_resolve_thresholds_applies_override():
    thresholds = resolve_thresholds(["A", "B"], 0.5, {"B": 0.3})
    assert thresholds == {"A": 0.5, "B": 0.3}


def test_apply_thresholds_binarizes_probabilities():
    labels = apply_thresholds({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5})
    assert labels == {"A": 1, "B": 0}


def test_split_into_sentences_filters_short_fragments():
    sentences = split_into_sentences("Lungs clear. No. Heart size normal.", min_chars=5)
    assert sentences == ["Lungs clear.", "Heart size normal."]


def test_select_evidence_ranks_by_label_score():
    sentences = ["Heart size normal.", "Pleural effusion present."]
    sentence_scores = {
        "Pleural effusion": [0.1, 0.9],
        "Cardiomegaly": [0.2, 0.1],
    }
    evidence = select_evidence(sentence_scores, sentences, max_sentences_per_label=1)
    assert evidence["Pleural effusion"][0]["sentence"] == "Pleural effusion present."
    assert evidence["Cardiomegaly"][0]["sentence"] == "Heart size normal."
