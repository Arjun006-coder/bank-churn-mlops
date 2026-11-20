import json

import pandas as pd

from monitoring.prepare_data import prepare_data


def make_dataset(customer_ids, targets=None):
    targets = targets or [0, 1, 0, 1, 0, 1]
    return pd.DataFrame(
        {
            "CustomerId": customer_ids,
            "CreditScore": [600] * len(customer_ids),
            "Geography": ["France"] * len(customer_ids),
            "Gender": ["Female"] * len(customer_ids),
            "Age": [40] * len(customer_ids),
            "Tenure": [3] * len(customer_ids),
            "Balance": [100.0] * len(customer_ids),
            "NumOfProducts": [1] * len(customer_ids),
            "HasCrCard": [1] * len(customer_ids),
            "IsActiveMember": [1] * len(customer_ids),
            "EstimatedSalary": [50000.0] * len(customer_ids),
            "Exited": targets,
        }
    )


def test_prepare_data_splits_stratified_batch_and_writes_report(tmp_path):
    incoming_path = tmp_path / "incoming.csv"
    old_training_path = tmp_path / "old_training.csv"
    update_path = tmp_path / "data" / "update_batch.csv"
    evaluation_path = tmp_path / "data" / "evaluation_batch.csv"
    report_path = tmp_path / "reports" / "data_validation.json"
    make_dataset(range(100, 106)).to_csv(incoming_path, index=False)
    make_dataset(range(1, 7)).to_csv(old_training_path, index=False)

    report = prepare_data(
        incoming_path,
        old_training_path,
        update_path,
        evaluation_path,
        report_path,
    )

    assert report["validation_passed"] is True
    assert report["rows"] == 6
    assert report["columns"] == 12
    assert report["class_distribution"] == {"0": 3, "1": 3}
    assert report["overlapping_customer_ids"] == 0
    update = pd.read_csv(update_path)
    evaluation = pd.read_csv(evaluation_path)
    assert len(update) + len(evaluation) == 6
    assert set(update["CustomerId"]).isdisjoint(evaluation["CustomerId"])
    assert set(update["Exited"].value_counts().index) == {0, 1}
    assert set(evaluation["Exited"].value_counts().index) == {0, 1}
    assert json.loads(report_path.read_text())["validation_passed"] is True


def test_prepare_data_reports_invalid_records_and_does_not_split(tmp_path):
    incoming_path = tmp_path / "incoming.csv"
    old_training_path = tmp_path / "old_training.csv"
    update_path = tmp_path / "update.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    report_path = tmp_path / "validation.json"
    incoming = make_dataset([10, 11, 12, 13, 14, 15])
    incoming.loc[1, "Age"] = None
    incoming.loc[2, "CustomerId"] = 10
    incoming = incoming.drop(columns=["EstimatedSalary"])
    incoming.to_csv(incoming_path, index=False)
    make_dataset([10, 20, 30, 40, 50, 60]).to_csv(old_training_path, index=False)

    report = prepare_data(
        incoming_path,
        old_training_path,
        update_path,
        evaluation_path,
        report_path,
    )

    assert report["validation_passed"] is False
    assert report["missing_values"]["Age"] == 1
    assert report["duplicate_customer_ids"] == 1
    assert report["overlapping_customer_ids"] == 1
    assert any("EstimatedSalary" in issue for issue in report["issues"])
    assert not update_path.exists()
    assert not evaluation_path.exists()
    assert report_path.exists()


def test_prepare_data_writes_failure_report_when_old_training_file_is_missing(
    tmp_path,
):
    incoming_path = tmp_path / "incoming.csv"
    old_training_path = tmp_path / "missing_old_training.csv"
    report_path = tmp_path / "validation.json"
    make_dataset(range(100, 106)).to_csv(incoming_path, index=False)

    report = prepare_data(
        incoming_path,
        old_training_path,
        tmp_path / "update.csv",
        tmp_path / "evaluation.csv",
        report_path,
    )

    assert report["validation_passed"] is False
    assert report["old_training_file_found"] is False
    assert report["overlapping_customer_ids"] is None
    assert report_path.exists()
