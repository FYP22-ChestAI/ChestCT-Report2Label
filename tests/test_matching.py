from report2label.matching.ct_report_matcher import build_scan_id_index, match_reports_to_scans

ID_PATTERN = r"[^0-9A-Za-z]"


def test_build_scan_id_index_normalizes_multi_suffix_and_plain_names(tmp_path):
    (tmp_path / "4215_26.nii.gz").write_text("x")
    (tmp_path / "OTHER-001").mkdir()
    index = build_scan_id_index(tmp_path, ID_PATTERN)
    assert "421526" in index
    assert "OTHER001" in index


def test_match_reports_to_scans_finds_and_misses(tmp_path):
    (tmp_path / "4215-26").mkdir()
    reports = [
        {"ct_id": "4215/26", "source_path": "report_a.json"},
        {"ct_id": None, "source_path": "report_b.json"},
        {"ct_id": "9999", "source_path": "report_c.json"},
    ]
    results = match_reports_to_scans(reports, tmp_path, ID_PATTERN)
    by_path = {r.report_path: r for r in results}

    assert by_path["report_a.json"].matched
    assert not by_path["report_b.json"].matched
    assert not by_path["report_c.json"].matched
