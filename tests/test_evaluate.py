import json

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from monitoring.evaluate import evaluate_model


def test_evaluate_model_saves_churn_metrics(tmp_path):
    training_features = pd.DataFrame({"Age": [20, 30]})
    model = DummyClassifier(strategy="constant", constant=1).fit(
        training_features, [0, 1]
    )
    model_path = tmp_path / "model.joblib"
    dataset_path = tmp_path / "evaluation.csv"
    report_path = tmp_path / "reports" / "champion_metrics.json"

    joblib.dump(model, model_path)
    pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "Age": [25, 45],
            "Exited": [0, 1],
        }
    ).to_csv(dataset_path, index=False)

    metrics = evaluate_model(model_path, dataset_path, report_path)

    assert metrics == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "f1_score": pytest.approx(2 / 3),
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == metrics


def test_evaluate_model_rejects_missing_target(tmp_path):
    model = DummyClassifier(strategy="constant", constant=1).fit(
        pd.DataFrame({"Age": [20, 30]}), [0, 1]
    )
    model_path = tmp_path / "model.joblib"
    dataset_path = tmp_path / "evaluation.csv"
    joblib.dump(model, model_path)
    pd.DataFrame({"Age": [25]}).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Exited"):
        evaluate_model(model_path, dataset_path, tmp_path / "metrics.json")
