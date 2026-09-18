"""Load the label vocabulary and index<->name mapping from configs/labels.yaml.

The list order must match the classifier head's output dimension exactly —
it is not re-sorted anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from report2label.utils.io import read_yaml


@dataclass
class LabelVocabulary:
    names: list[str]
    descriptions: dict[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelVocabulary":
        raw = read_yaml(path)
        entries = raw["labels"]
        names = [e["name"] for e in entries]
        descriptions = {e["name"]: e.get("description", "") for e in entries}
        return cls(names=names, descriptions=descriptions)

    def __len__(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def name_at(self, index: int) -> str:
        return self.names[index]
