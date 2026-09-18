from report2label.preprocessing.cleaner import clean_text
from report2label.preprocessing.normalizer import normalize_text
from report2label.preprocessing.phi_remover import remove_phi


def test_clean_text_dehyphenates_line_wraps_and_collapses_blank_lines():
    raw = "solid perifis-\nsural nodule\n\n\n\nNo effusion."
    cleaned = clean_text(raw)
    assert "perifissural nodule" in cleaned
    assert "\n\n\n" not in cleaned


def test_normalize_text_spaces_out_units():
    normalized = normalize_text("3mm nodule and 4cm mass")
    assert "3 mm" in normalized
    assert "4 cm" in normalized


def test_normalize_text_converts_colon_dash_header():
    assert normalize_text("Indication:- asthma") == "Indication: asthma"


def test_phi_remover_strips_letterhead_and_signature_lines():
    text = (
        "Department of Radiology and Clinical Imaging\n"
        "Some Hospital, Somewhere\n"
        "Thank you for referring this patient.\n"
        "Both lungs are normally aerated.\n"
        "Dr. Jane Doe\n"
        "Consultant Radiologist.\n"
    )
    scrubbed = remove_phi(text)
    assert "Department" not in scrubbed
    assert "Hospital" not in scrubbed
    assert "Thank you" not in scrubbed
    assert "Dr. Jane Doe" not in scrubbed
    assert "Consultant Radiologist" not in scrubbed
    assert "Both lungs are normally aerated." in scrubbed


def test_phi_remover_strips_signature_initials_without_space():
    text = "Rest of the study is unremarkable.\nDr.BN- SR\n"
    scrubbed = remove_phi(text)
    assert "Dr.BN" not in scrubbed
    assert "Rest of the study is unremarkable." in scrubbed


def test_phi_remover_strips_demographic_fields():
    text = "Patient Name: John Smith\nDOB: 1990-01-01\nFindings: clear lungs."
    scrubbed = remove_phi(text)
    assert "John Smith" not in scrubbed
    assert "1990-01-01" not in scrubbed
    assert "Findings: clear lungs." in scrubbed
