from report2label.parsing.report_parser import parse_report

HEADERS = {
    "indication": ["indication"],
    "technique": ["technique"],
    "findings": ["findings"],
    "impression": ["impression"],
}
SECTION_NAMES = list(HEADERS)

RAW = (
    "CT NO: 1001/26\n"
    "Indication:- Follow-up nodule.\n"
    "Both lungs are normally aerated.\n"
    "No pleural effusion.\n"
    "Impression:-\n"
    "Stable nodule.\n"
)


def _parse(raw: str, source_path: str = "data/raw/1001.pdf"):
    return parse_report(
        raw,
        section_headers=HEADERS,
        classifier_input_sections=["findings", "impression"],
        source_path=source_path,
    )


def test_to_row_has_stable_columns_in_configured_order():
    row = _parse(RAW).to_row(SECTION_NAMES)
    assert list(row) == ["report_id", "ct_id", *SECTION_NAMES, "report_text"]
    assert row["report_id"] == "1001"
    assert row["ct_id"] == "1001/26"


def test_to_row_cells_are_single_line_and_missing_sections_are_empty():
    row = _parse(RAW).to_row(SECTION_NAMES)
    assert row["findings"] == "Both lungs are normally aerated. No pleural effusion."
    assert row["technique"] == ""
    assert all("\n" not in str(value) for value in row.values())
    assert row["report_text"].startswith("Findings: Both lungs")


def test_to_row_keeps_report_without_ct_id():
    row = _parse("Findings: clear lungs.\n", source_path="x/unnamed.txt").to_row(SECTION_NAMES)
    assert row["ct_id"] is None
    assert row["report_id"] == "unnamed"
    assert row["findings"] == "clear lungs."
