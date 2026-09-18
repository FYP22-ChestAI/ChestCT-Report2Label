"""Load the tokenizer + RadBertClassifier from configs/model.yaml, once per process."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from report2label.utils.io import read_yaml
from report2label.utils.logging import get_logger

from .radbert import RadBertClassifier

logger = get_logger(__name__)

# Buffers whose presence/absence differs across transformers versions (e.g.
# RobertaEmbeddings.position_ids was a persistent buffer in older releases,
# non-persistent in newer ones) — a mismatch here is not a real weight
# mismatch, so it's safe to ignore when loading an older checkpoint.
_BENIGN_STATE_DICT_KEY_SUFFIXES = ("position_ids", "token_type_ids")


@dataclass
class ModelConfig:
    base_model: str
    checkpoint_path: str
    num_labels: int
    max_length: int
    do_lower_case: bool
    device: str
    default_threshold: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        path = Path(path).resolve()
        raw = read_yaml(path)

        # A relative checkpoint_path in the YAML is meant relative to the
        # project root (configs/model.yaml's grandparent), not to whatever
        # directory the current process happens to be running from — that
        # varies (project root for scripts/*.py, the notebook's own folder
        # for notebooks/*.ipynb executed via nbconvert/Jupyter).
        checkpoint_path = Path(raw["checkpoint_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = path.parent.parent / checkpoint_path

        return cls(
            base_model=raw["base_model"],
            checkpoint_path=str(checkpoint_path),
            num_labels=int(raw["num_labels"]),
            max_length=int(raw["max_length"]),
            do_lower_case=bool(raw["do_lower_case"]),
            device=raw.get("device", "auto"),
            default_threshold=float(raw.get("default_threshold", 0.5)),
        )

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def _check_state_dict_diff(missing: list[str], unexpected: list[str]) -> None:
    is_benign = lambda k: k.endswith(_BENIGN_STATE_DICT_KEY_SUFFIXES)
    bad_missing = [k for k in missing if not is_benign(k)]
    bad_unexpected = [k for k in unexpected if not is_benign(k)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint does not match the RadBertClassifier architecture — "
            f"missing keys: {bad_missing}, unexpected keys: {bad_unexpected}"
        )
    if missing or unexpected:
        logger.info(
            "Ignored benign buffer mismatch (transformers version drift): missing=%s unexpected=%s",
            missing,
            unexpected,
        )


_CACHE: dict[str, tuple[PreTrainedTokenizerBase, RadBertClassifier, torch.device]] = {}


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[PreTrainedTokenizerBase, RadBertClassifier, torch.device]:
    """Load (and cache) the tokenizer and classifier described by `config`."""
    cache_key = f"{config.base_model}|{config.checkpoint_path}|{config.device}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classifier checkpoint not found at '{checkpoint_path}'. Download "
            "RadBertClassifier.pth from the CT-RATE models repository on Hugging "
            "Face (gated dataset — requires accepting its terms) and place it there."
        )

    device = config.resolve_device()
    logger.info("Loading tokenizer/encoder '%s' on %s", config.base_model, device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, do_lower_case=config.do_lower_case)

    model = RadBertClassifier(base_model=config.base_model, n_classes=config.num_labels)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _check_state_dict_diff(missing, unexpected)
    model.to(device)
    model.eval()

    _CACHE[cache_key] = (tokenizer, model, device)
    return tokenizer, model, device
