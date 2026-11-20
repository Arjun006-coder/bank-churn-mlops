import json

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from monitoring.retrain import retrain


FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def _make_data(rows):
    return pd.DataFrame(
        [
            {
                "CreditScore": 600 + index,
                "Geography": ("France", "Spain")[index % 2],
                "Gender": ("Female", "Male")[index % 2],
                "Age": 25 + index,
                "Tenure": index % 10,
                "Balance": float(index * 100),
                "NumOfProducts": index % 4 + 1,
                "HasCrCard": index % 2,
                "IsActiveMember": (index + 1) % 2,
                "EstimatedSalary": float(50000 + index * 100),
                "Exited": index % 2,
            }
            for index in range(rows)
        ]
    )


@pytest.fixture
def retraining_files(tmp_path):
    original_data = tmp_path / "original.csv"
    update_data = tmp_path / "update.csv"
    champion_path = tmp_path / "model.joblib"
    candidate_path = tmp_path / "models" / "candidate_model.joblib"
    summary_path = tmp_path / "reports" / "retraining_summary.json"

    _make_data(20).to_csv(original_data, index=False)
    _make_data(4).to_csv(update_data, index=False)
    champion = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["Geography", "Gender"],
                        ),
                        (
                            "numeric",
                            "passthrough",
                            [
                                feature
                                for feature in FEATURES
                                if feature not in ("Geography", "Gender")
                            ],
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(n_estimators=2, random_state=42),
            )
        ]
    )
    champion.fit(_make_data(20)[FEATURES], _make_data(20)["Exited"])
    joblib.dump(champion, champion_path)

    return {
        "original_data_path": original_data,
        "update_data_path": update_data,
        "champion_model_path": champion_path,
        "candidate_model_path": candidate_path,
        "summary_path": summary_path,
    }


def test_retrain_clones_champion_and_keeps_its_file_unchanged(retraining_files):
    champion_path = retraining_files["champion_model_path"]
    champion_before = champion_path.read_bytes()

    summary = retrain(**retraining_files)

    candidate = joblib.load(retraining_files["candidate_model_path"])
    saved_summary = json.loads(retraining_files["summary_path"].read_text())
    assert champion_path.read_bytes() == champion_before
    assert candidate.named_steps["classifier"].n_estimators == 2
    assert summary["original_training_rows"] == 16
    assert summary["update_rows"] == 4
    assert summary["combined_training_rows"] == 20
    assert saved_summary == summary
    assert summary["mlflow_run_id"] is None


def test_retrain_rejects_update_with_different_feature_columns(retraining_files):
    update_data = _make_data(4).drop(columns=["Age"])
    update_data.to_csv(retraining_files["update_data_path"], index=False)

    with pytest.raises(ValueError, match="feature columns must match"):
        retrain(**retraining_files)

    assert not retraining_files["candidate_model_path"].exists()


def test_retrain_rejects_candidate_path_that_would_replace_champion(retraining_files):
    retraining_files["candidate_model_path"] = retraining_files["champion_model_path"]

    with pytest.raises(ValueError, match="must not overwrite the champion"):
        retrain(**retraining_files)
