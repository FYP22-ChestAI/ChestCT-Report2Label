from report2label.parsing.section_parser import split_sections

HEADERS = {
    "indication": ["indication", "clinical indication", "history"],
    "findings": ["findings"],
    "impression": ["impression", "conclusion"],
}


def test_fills_findings_gap_when_no_explicit_header():
    text = (
        "Indication:- Follow-up nodule.\n"
        "Both lungs are normally aerated.\n"
        "No pleural effusion.\n"
        "Impression:-\n"
        "Stable nodule.\n"
    )
    sections = split_sections(text, HEADERS)
    assert sections["indication"] == "Follow-up nodule."
    assert "normally aerated" in sections["findings"]
    assert "No pleural effusion" in sections["findings"]
    assert sections["impression"] == "Stable nodule."


def test_uses_explicit_findings_header_when_present():
    text = (
        "Indication: chest pain\n"
        "Findings:\n"
        "Heart size normal.\n"
        "Impression: normal study\n"
    )
    sections = split_sections(text, HEADERS)
    assert sections["findings"] == "Heart size normal."
    assert sections["impression"] == "normal study"


def test_whole_text_becomes_findings_when_no_headers_recognized():
    text = "Everything looks unremarkable today."
    sections = split_sections(text, HEADERS)
    assert sections == {"findings": "Everything looks unremarkable today."}
