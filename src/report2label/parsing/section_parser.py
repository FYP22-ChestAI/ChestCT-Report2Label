"""Segment free-text radiology report bodies into named sections.

Reports are inconsistent about which headers they actually print — the
sample set has "Indication:-" and "Impression:-" but no explicit "Findings:"
header, with the findings body simply sitting between the two. This parser
matches whatever headers *are* present and falls back to treating the gap
between "indication" and "impression" as the findings body when no explicit
findings header exists.
"""

from __future__ import annotations

import re

_HEADER_SUFFIX = r"\s*:?-?\s*"

# These headers conventionally hold a single short value on their own line
# (e.g. "Indication:- asthma"). Unlike "findings"/"impression" they do NOT
# swallow the lines that follow, so a report with no explicit "Findings:"
# header doesn't have its findings body silently absorbed into "indication".
_SHORT_FORM_SECTIONS = {"indication", "technique", "comparison"}


def _build_header_regex(aliases: list[str]) -> re.Pattern:
    alt = "|".join(re.escape(a) for a in aliases)
    return re.compile(rf"^\s*(?:{alt}){_HEADER_SUFFIX}(.*)$", re.IGNORECASE)


def split_sections(text: str, section_headers: dict[str, list[str]]) -> dict[str, str]:
    """Split `text` into {canonical_section_name: body_text} using alias headers.

    `section_headers` maps a canonical name (e.g. "findings") to the list of
    header spellings that introduce it (e.g. ["findings"]).
    """
    lines = text.split("\n")
    patterns = {name: _build_header_regex(aliases) for name, aliases in section_headers.items()}

    matches: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        for name, pattern in patterns.items():
            m = pattern.match(line)
            if m:
                matches.append((idx, name, m.group(1).strip()))
                break
    matches.sort(key=lambda m: m[0])

    sections: dict[str, list[str]] = {}
    for i, (start, name, inline_remainder) in enumerate(matches):
        if name in _SHORT_FORM_SECTIONS:
            body_lines = [inline_remainder] if inline_remainder else []
        else:
            end = matches[i + 1][0] if i + 1 < len(matches) else len(lines)
            body_lines = ([inline_remainder] if inline_remainder else []) + lines[start + 1 : end]
        body = "\n".join(body_lines).strip()
        if body:
            sections.setdefault(name, []).append(body)

    merged = {name: "\n\n".join(parts) for name, parts in sections.items()}

    if "findings" not in merged:
        _fill_findings_gap(merged, matches, lines)

    return merged


def _fill_findings_gap(
    merged: dict[str, str], matches: list[tuple[int, str, str]], lines: list[str]
) -> None:
    """Treat the text between the short-form headers and impression as findings
    when no section was explicitly labelled "Findings"."""
    short_form_end = 0
    impression_start = len(lines)
    for start, name, _ in matches:
        if name in _SHORT_FORM_SECTIONS:
            short_form_end = max(short_form_end, start + 1)
        if name == "impression":
            impression_start = min(impression_start, start)

    gap = "\n".join(lines[short_form_end:impression_start]).strip()
    if gap:
        merged["findings"] = gap
