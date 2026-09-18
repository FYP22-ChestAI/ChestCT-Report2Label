import pandas as pd

from report2label.dataset.builder import build_dataset_records
from report2label.dataset.exporter import export_to_csv, export_to_json
from report2label.dataset.schema import DatasetRecord
from report2label.utils.io import read_json


def test_build_dataset_records_joins_manifest_and_predictions():
    manifest_rows = [
        {"ct_id": "4215/26", "report_path": "r1.json", "scan_path": "s1", "matched": True},
        {"ct_id": "9999", "report_path": "r2.json", "scan_path": None, "matched": False},
    ]
    predictions_by_path = {
        "r1.json": {
            "labels": {"Cardiomegaly": 1, "Pleural effusion": 0},
            "probabilities": {"Cardiomegaly": 0.9, "Pleural effusion": 0.1},
            "evidence": None,
        }
    }
    records = build_dataset_records(manifest_rows, predictions_by_path)
    assert len(records) == 1
    assert records[0].report_id == "r1"
    assert records[0].labels["Cardiomegaly"] == 1


def test_export_to_json_round_trips(tmp_path):
    record = DatasetRecord(
        report_id="r1",
        ct_id="4215",
        report_path="r1.json",
        scan_path=None,
        labels={"Cardiomegaly": 1},
        probabilities={"Cardiomegaly": 0.9},
        evidence=None,
    )
    out_path = tmp_path / "dataset.json"
    export_to_json([record], out_path)
    loaded = read_json(out_path)
    assert loaded[0]["report_id"] == "r1"
    assert loaded[0]["labels"]["Cardiomegaly"] == 1


def test_export_to_csv_flattens_labels_into_columns(tmp_path):
    record = DatasetRecord(
        report_id="r1",
        ct_id="4215",
        report_path="r1.json",
        scan_path=None,
        labels={"Cardiomegaly": 1, "Pleural effusion": 0},
        probabilities={},
        evidence=None,
    )
    out_path = tmp_path / "dataset.csv"
    export_to_csv([record], out_path)
    df = pd.read_csv(out_path)
    assert df.loc[0, "Cardiomegaly"] == 1
    assert df.loc[0, "Pleural effusion"] == 0
