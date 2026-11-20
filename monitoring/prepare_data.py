"""Validate and split a labelled incoming churn batch."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCOMING_PATH = PROJECT_ROOT / "data" / "incoming_churn.csv"
DEFAULT_UPDATE_PATH = PROJECT_ROOT / "data" / "update_batch.csv"
DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation_batch.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_validation.json"
TARGET_COLUMN = "Exited"
ID_COLUMN = "CustomerId"
REQUIRED_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
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
}


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def prepare_data(
    incoming_path: Path,
    old_training_path: Path,
    update_path: Path = DEFAULT_UPDATE_PATH,
    evaluation_path: Path = DEFAULT_EVALUATION_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Validate incoming records, compare IDs, split, and save validation evidence."""
    incoming_path = Path(incoming_path)
    old_training_path = Path(old_training_path)
    update_path = Path(update_path)
    evaluation_path = Path(evaluation_path)
    report_path = Path(report_path)

    frame = pd.read_csv(incoming_path)
    old_training = (
        pd.read_csv(old_training_path) if old_training_path.is_file() else None
    )

    report: dict[str, Any] = {
        "incoming_file": str(incoming_path),
        "old_training_file": str(old_training_path),
        "old_training_file_found": old_training is not None,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": frame.columns.tolist(),
        "missing_values": {
            column: int(count)
            for column, count in frame.isnull().sum().items()
        },
        "duplicate_records": int(frame.duplicated().sum()),
        "duplicate_customer_ids": (
            int(frame[ID_COLUMN].duplicated().sum())
            if ID_COLUMN in frame.columns
            else None
        ),
        "overlapping_customer_ids": None,
        "old_training_missing_customer_ids": None,
        "old_training_duplicate_customer_ids": None,
        "class_distribution": {"0": 0, "1": 0},
        "split": None,
        "issues": [],
    }

    issues = report["issues"]
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        issues.append(f"Missing required incoming columns: {missing_columns}")
    if old_training is None:
        issues.append(f"Old training data file not found: {old_training_path}")
    elif ID_COLUMN not in old_training.columns:
        issues.append(f"Old training data is missing required ID column: {ID_COLUMN}")
    else:
        report["old_training_missing_customer_ids"] = int(
            old_training[ID_COLUMN].isna().sum()
        )
        report["old_training_duplicate_customer_ids"] = int(
            old_training[ID_COLUMN].duplicated().sum()
        )
        if report["old_training_missing_customer_ids"]:
            issues.append("Old training data contains missing CustomerId values")
        if report["old_training_duplicate_customer_ids"]:
            issues.append("Old training data contains duplicate CustomerId values")

    if TARGET_COLUMN in frame.columns:
        counts = frame[TARGET_COLUMN].value_counts()
        report["class_distribution"] = {
            "0": int(counts.get(0, 0)),
            "1": int(counts.get(1, 0)),
        }
        invalid_targets = frame.loc[
            frame[TARGET_COLUMN].notna() & ~frame[TARGET_COLUMN].isin([0, 1]),
            TARGET_COLUMN,
        ].unique()
        if len(invalid_targets):
            issues.append(
                f"Target {TARGET_COLUMN} contains values outside 0 and 1: "
                f"{[str(value) for value in invalid_targets]}"
            )

    missing_value_columns = [
        column for column, count in report["missing_values"].items() if count
    ]
    if missing_value_columns:
        issues.append(f"Missing values found in columns: {missing_value_columns}")
    if report["duplicate_records"]:
        issues.append(f"Found {report['duplicate_records']} duplicate records")
    if report["duplicate_customer_ids"]:
        issues.append(
            f"Found {report['duplicate_customer_ids']} duplicate {ID_COLUMN} values"
        )

    if (
        ID_COLUMN in frame.columns
        and old_training is not None
        and ID_COLUMN in old_training.columns
    ):
        incoming_ids = set(frame[ID_COLUMN].dropna().astype("string").str.strip())
        old_ids = set(old_training[ID_COLUMN].dropna().astype("string").str.strip())
        overlap_count = len(incoming_ids & old_ids)
        report["overlapping_customer_ids"] = overlap_count
        if overlap_count:
            issues.append(
                f"Incoming data overlaps old training data for "
                f"{overlap_count} {ID_COLUMN} values"
            )

    classes_present = report["class_distribution"]
    if classes_present["0"] == 0 or classes_present["1"] == 0:
        issues.append("Incoming data must contain both Exited classes 0 and 1")
    if not 0 < test_size < 1:
        issues.append("test_size must be greater than 0 and less than 1")

    class_count = sum(count > 0 for count in classes_present.values())
    evaluation_rows = math.ceil(len(frame) * test_size) if 0 < test_size < 1 else 0
    if evaluation_rows < class_count or len(frame) - evaluation_rows < class_count:
        issues.append(
            "Incoming data is too small to stratify both classes across the "
            "update and evaluation portions"
        )

    report["validation_passed"] = not issues
    if not issues:
        update_frame, evaluation_frame = train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame[TARGET_COLUMN],
        )
        update_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        update_frame.to_csv(update_path, index=False)
        evaluation_frame.to_csv(evaluation_path, index=False)
        report["split"] = {
            "update_file": str(update_path),
            "update_rows": int(len(update_frame)),
            "evaluation_file": str(evaluation_path),
            "evaluation_rows": int(len(evaluation_frame)),
            "test_size": test_size,
            "random_state": random_state,
        }

    _write_report(report_path, report)
    print(f"Dataset size: {report['rows']} rows x {report['columns']} columns")
    print(
        "Exited distribution: "
        f"0 = {classes_present['0']}, 1 = {classes_present['1']}"
    )
    print(f"Validation {'passed' if report['validation_passed'] else 'failed'}")
    print(f"Validation report: {report_path}")
    if issues:
        for issue in issues:
            print(f"- {issue}")
    else:
        print(f"Update batch: {update_path} ({len(update_frame)} rows)")
        print(f"Evaluation batch: {evaluation_path} ({len(evaluation_frame)} rows)")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and split an incoming labelled churn CSV."
    )
    parser.add_argument(
        "--incoming",
        type=Path,
        default=DEFAULT_INCOMING_PATH,
        help=f"Incoming labelled CSV (default: {DEFAULT_INCOMING_PATH})",
    )
    parser.add_argument(
        "--old-training",
        type=Path,
        required=True,
        help="CSV containing previously used training records and CustomerId values",
    )
    parser.add_argument("--update-output", type=Path, default=DEFAULT_UPDATE_PATH)
    parser.add_argument(
        "--evaluation-output", type=Path, default=DEFAULT_EVALUATION_PATH
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    report = prepare_data(
        incoming_path=args.incoming,
        old_training_path=args.old_training,
        update_path=args.update_output,
        evaluation_path=args.evaluation_output,
        report_path=args.report,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
