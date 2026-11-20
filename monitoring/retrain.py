import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "Exited"
IGNORED_COLUMNS = ("RowNumber", "CustomerId", "Surname")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _validate_target(data: pd.DataFrame, source: Path) -> pd.Series:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"{source} must contain a '{TARGET_COLUMN}' column.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"{source} contains missing '{TARGET_COLUMN}' values.")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise")
    if not target.isin((0, 1)).all():
        raise ValueError(f"{source} has '{TARGET_COLUMN}' values outside 0 and 1.")
    return target.astype(int)


def _load_original_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    data = data.drop(columns=list(IGNORED_COLUMNS), errors="ignore").dropna()
    target = _validate_target(data, path)
    features = data.drop(columns=[TARGET_COLUMN])
    if features.empty:
        raise ValueError(f"{path} has no complete rows to use for training.")
    return features, target


def _load_update_data(
    path: Path, expected_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    target = _validate_target(data, path)
    features = data.drop(columns=[TARGET_COLUMN, *IGNORED_COLUMNS], errors="ignore")

    missing = sorted(set(expected_features) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(expected_features))
    if missing or unexpected:
        raise ValueError(
            f"{path} feature columns must match the original dataset; "
            f"missing={missing}, unexpected={unexpected}."
        )
    if features.empty:
        raise ValueError(f"{path} contains no update rows.")
    return features[expected_features], target


def _log_mlflow(
    model: Any,
    summary: dict[str, Any],
    experiment_name: str,
    tracking_uri: str | None,
) -> str:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking was requested, but MLflow is not installed. "
            "Install the project requirements and try again."
        ) from error

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="candidate-retraining") as run:
        mlflow.log_params(
            {
                "original_training_rows": summary["original_training_rows"],
                "update_rows": summary["update_rows"],
                "combined_training_rows": summary["combined_training_rows"],
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "estimator": summary["estimator"],
            }
        )
        mlflow.log_metric("training_accuracy", summary["training_accuracy"])
        mlflow.sklearn.log_model(model, artifact_path="candidate_model")
        return run.info.run_id


def retrain(
    original_data_path: Path = ROOT_DIR / "Churn_Modelling.csv",
    update_data_path: Path = ROOT_DIR / "data" / "update_batch.csv",
    champion_model_path: Path = ROOT_DIR / "model.joblib",
    candidate_model_path: Path = ROOT_DIR / "models" / "candidate_model.joblib",
    summary_path: Path = ROOT_DIR / "reports" / "retraining_summary.json",
    *,
    use_mlflow: bool = False,
    experiment_name: str = "bank-churn-retraining",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    original_data_path = Path(original_data_path)
    update_data_path = Path(update_data_path)
    champion_model_path = Path(champion_model_path)
    candidate_model_path = Path(candidate_model_path)
    summary_path = Path(summary_path)

    if not champion_model_path.is_file():
        raise FileNotFoundError(f"Champion model not found: {champion_model_path}")
    if candidate_model_path.resolve() == champion_model_path.resolve():
        raise ValueError("Candidate model path must not overwrite the champion model.")
    if summary_path.resolve() in {
        champion_model_path.resolve(),
        candidate_model_path.resolve(),
    }:
        raise ValueError("Retraining summary path must be separate from model artifacts.")

    original_features, original_target = _load_original_data(original_data_path)
    update_features, update_target = _load_update_data(
        update_data_path, original_features.columns.tolist()
    )
    original_train_features, _, original_train_target, _ = train_test_split(
        original_features,
        original_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=original_target,
    )

    training_features = pd.concat(
        [original_train_features, update_features], ignore_index=True
    )
    training_target = pd.concat(
        [original_train_target, update_target], ignore_index=True
    )

    champion = joblib.load(champion_model_path)
    candidate = clone(champion)
    candidate.fit(training_features, training_target)
    training_accuracy = float(
        accuracy_score(training_target, candidate.predict(training_features))
    )

    summary: dict[str, Any] = {
        "original_data": str(original_data_path),
        "update_data": str(update_data_path),
        "candidate_model": str(candidate_model_path),
        "original_training_rows": len(original_train_features),
        "update_rows": len(update_features),
        "combined_training_rows": len(training_features),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "estimator": (
            f"{candidate.__class__.__module__}.{candidate.__class__.__name__}"
        ),
        "training_accuracy": training_accuracy,
        "mlflow_run_id": None,
    }

    candidate_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, candidate_model_path)
    if use_mlflow:
        summary["mlflow_run_id"] = _log_mlflow(
            candidate, summary, experiment_name, tracking_uri
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain a candidate churn model without replacing the champion."
    )
    parser.add_argument(
        "--original-data",
        type=Path,
        default=ROOT_DIR / "Churn_Modelling.csv",
        help="Original labelled dataset used by notebooks/training.ipynb.",
    )
    parser.add_argument(
        "--update-data",
        type=Path,
        default=ROOT_DIR / "data" / "update_batch.csv",
        help="New labelled rows to include in candidate training.",
    )
    parser.add_argument(
        "--champion-model",
        type=Path,
        default=ROOT_DIR / "model.joblib",
        help="Existing model pipeline to clone.",
    )
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=ROOT_DIR / "models" / "candidate_model.joblib",
        help="Output path for the newly fitted candidate.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT_DIR / "reports" / "retraining_summary.json",
        help="Output path for the retraining summary.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log parameters, training accuracy, and the candidate artifact to MLflow.",
    )
    parser.add_argument(
        "--experiment-name",
        default="bank-churn-retraining",
        help="MLflow experiment name when --mlflow is enabled.",
    )
    parser.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI; otherwise MLflow's configured default is used.",
    )
    args = parser.parse_args()

    summary = retrain(
        original_data_path=args.original_data,
        update_data_path=args.update_data,
        champion_model_path=args.champion_model,
        candidate_model_path=args.candidate_model,
        summary_path=args.summary,
        use_mlflow=args.mlflow,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
