from report2label.parsing.ct_id_parser import extract_ct_id, normalize_ct_id


def test_extracts_ct_no_with_slash():
    text = "HRCT CHEST CT NO: 4215/26\nIndication:- asthma"
    assert extract_ct_id(text) == "4215/26"


def test_extracts_accession_number():
    text = "Accession Number: ABC-123\nFindings: ..."
    assert extract_ct_id(text) == "ABC-123"


def test_extracts_exam_id_when_ct_no_absent():
    text = "Exam ID: EX-42\nFindings: ..."
    assert extract_ct_id(text) == "EX-42"


def test_returns_none_when_no_identifier_present():
    assert extract_ct_id("No identifiers appear in this report body.") is None


def test_normalize_ct_id_strips_punctuation_and_uppercases():
    assert normalize_ct_id("4215/26") == "421526"
    assert normalize_ct_id("4215-26") == "421526"
    assert normalize_ct_id("abc-123") == "ABC123"
