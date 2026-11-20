import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

TARGET_COLUMN = "Exited"
IDENTIFIER_COLUMNS = ("RowNumber", "CustomerId", "Surname")


def evaluate_model(
    model_path: str | Path,
    dataset_path: str | Path,
    report_path: str | Path,
) -> dict[str, float]:
    model = joblib.load(model_path)
    dataset = pd.read_csv(dataset_path)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Evaluation dataset must contain a '{TARGET_COLUMN}' column")
    if dataset.empty:
        raise ValueError("Evaluation dataset must contain at least one row")

    y_true = dataset[TARGET_COLUMN]
    if y_true.isna().any() or not y_true.isin([0, 1]).all():
        raise ValueError(f"'{TARGET_COLUMN}' must contain only non-null 0 or 1 labels")

    features = dataset.drop(columns=[TARGET_COLUMN])
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        missing_features = [name for name in model_features if name not in features.columns]
        if missing_features:
            raise ValueError(f"Evaluation dataset is missing model features: {missing_features}")
        features = features.loc[:, list(model_features)]
    else:
        features = features.drop(
            columns=[name for name in IDENTIFIER_COLUMNS if name in features.columns]
        )

    if features.empty:
        raise ValueError("Evaluation dataset must contain model input features")

    predictions = model.predict(features)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a churn model on labelled customer data.")
    parser.add_argument("--model", required=True, help="Path to the serialized model")
    parser.add_argument("--dataset", required=True, help="Path to the labelled evaluation CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON metrics report")
    args = parser.parse_args()

    metrics = evaluate_model(args.model, args.dataset, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
