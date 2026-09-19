"""Orchestrate cleaning, PHI removal, and section splitting into one parsed report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from report2label.preprocessing.cleaner import clean_text
from report2label.preprocessing.normalizer import normalize_text
from report2label.preprocessing.phi_remover import remove_phi

from .ct_id_parser import extract_ct_id
from .section_parser import split_sections


@dataclass
class ParsedReport:
    source_path: str
    ct_id: str | None
    sections: dict[str, str]
    classifier_text: str
    raw_char_count: int = field(default=0)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "ct_id": self.ct_id,
            "sections": self.sections,
            "classifier_text": self.classifier_text,
            "raw_char_count": self.raw_char_count,
        }

    def to_row(self, section_names: list[str]) -> dict:
        """Flat single-line-per-cell row for the reports CSV.

        Columns follow `section_names` (the configured sections), so the table
        shape doesn't depend on which headers any particular report happened
        to print — a missing section is just an empty cell.
        """
        one_line = lambda text: " ".join(text.split())
        row = {"report_id": Path(self.source_path).stem, "ct_id": self.ct_id}
        for name in section_names:
            row[name] = one_line(self.sections.get(name, ""))
        row["report_text"] = one_line(self.classifier_text)
        return row


def parse_report(
    raw_text: str,
    *,
    section_headers: dict[str, list[str]],
    classifier_input_sections: list[str],
    phi_removal_enabled: bool = True,
    source_path: str = "",
) -> ParsedReport:
    cleaned = clean_text(raw_text)
    ct_id = extract_ct_id(cleaned)

    scrubbed = remove_phi(cleaned) if phi_removal_enabled else cleaned
    normalized = normalize_text(scrubbed)

    sections = split_sections(normalized, section_headers)

    parts = []
    for key in classifier_input_sections:
        content = sections.get(key, "").strip()
        if content:
            parts.append(f"{key.capitalize()}: {content}")
    classifier_text = " ".join(parts) if parts else normalized.strip()

    return ParsedReport(
        source_path=source_path,
        ct_id=ct_id,
        sections=sections,
        classifier_text=classifier_text,
        raw_char_count=len(raw_text),
    )
