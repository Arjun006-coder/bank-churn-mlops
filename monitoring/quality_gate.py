"""Compare model evaluation reports without changing the deployed model."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1")


class QualityGateError(ValueError):
    """An evaluation report or quality-gate setting is invalid."""


def validate_metrics(payload):
    """Accept a flat metrics object or an object with a nested 'metrics' key."""
    if not isinstance(payload, dict):
        raise QualityGateError("Evaluation report must contain a JSON object")

    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise QualityGateError("The 'metrics' field must be a JSON object")

    result = {}
    for metric in REQUIRED_METRICS:
        value = metrics.get(metric)
        if metric == "f1" and value is None:
            value = metrics.get("f1_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QualityGateError(f"Missing or non-numeric metric: {metric}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise QualityGateError(f"Metric {metric} must be finite and between 0 and 1")
        result[metric] = float(value)
    return result


def load_metrics(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return validate_metrics(json.load(stream))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityGateError(f"Cannot read evaluation report {path}: {exc}") from exc


def evaluate_quality_gate(champion, candidate, min_f1_gain=0.01, max_recall_drop=0.02):
    """Return a recorded recommendation; never replace the champion artifact."""
    champion = validate_metrics(champion)
    candidate = validate_metrics(candidate)

    for name, value in (("min_f1_gain", min_f1_gain), ("max_recall_drop", max_recall_drop)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QualityGateError(f"{name} must be numeric")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise QualityGateError(f"{name} must be between 0 and 1")

    f1_gain = candidate["f1"] - champion["f1"]
    recall_drop = champion["recall"] - candidate["recall"]
    f1_pass = f1_gain + 1e-12 >= min_f1_gain
    recall_pass = recall_drop <= max_recall_drop + 1e-12
    eligible = f1_pass and recall_pass

    reasons = []
    if not f1_pass:
        reasons.append(
            f"F1 improvement {f1_gain:+.4f} is below the required +{min_f1_gain:.4f}."
        )
    if not recall_pass:
        reasons.append(
            f"Recall decreased by {recall_drop:.4f}, exceeding the allowed {max_recall_drop:.4f}."
        )
    if eligible:
        reasons.append("Candidate meets both quality criteria; manual promotion may be considered.")

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAIN_CHAMPION",
        "eligible_for_promotion": eligible,
        "selected_metric": "f1",
        "metric_rationale": "Churn is a classification task; F1 balances precision and recall for customers predicted to leave.",
        "criteria": {"minimum_f1_gain": float(min_f1_gain), "maximum_recall_drop": float(max_recall_drop)},
        "champion_metrics": champion,
        "candidate_metrics": candidate,
        "comparison": {
            "f1_gain": round(f1_gain, 6),
            "recall_drop": round(recall_drop, 6),
            "f1_pass": f1_pass,
            "recall_pass": recall_pass,
        },
        "reasons": reasons,
        "production_model_changed": False,
        "promotion_requires_manual_approval": True,
    }


def save_json_report(report, path):
    """Write the decision atomically to avoid leaving an incomplete report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".quality_gate_", suffix=".tmp",
            dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare the current and retrained bank churn models")
    parser.add_argument("--champion-report", default="reports/champion_metrics.json")
    parser.add_argument("--candidate-report", default="reports/candidate_metrics.json")
    parser.add_argument("--output", default="reports/quality_gate_decision.json")
    parser.add_argument("--champion-model", default="model.joblib")
    parser.add_argument("--candidate-model", default="models/candidate_model.joblib")
    parser.add_argument("--min-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    args = parser.parse_args(argv)

    try:
        champion = load_metrics(args.champion_report)
        candidate = load_metrics(args.candidate_report)
        report = evaluate_quality_gate(champion, candidate, args.min_f1_gain, args.max_recall_drop)
        report["source_reports"] = {
            "champion": str(args.champion_report), "candidate": str(args.candidate_report)
        }
        report["model_artifacts"] = {
            "champion": str(args.champion_model), "candidate": str(args.candidate_model)
        }
        save_json_report(report, args.output)
    except (OSError, QualityGateError) as exc:
        print(f"Quality gate error: {exc}", file=sys.stderr)
        return 1

    print(f"Decision: {report['decision']}")
    print(f"Champion F1: {champion['f1']:.4f}; Candidate F1: {candidate['f1']:.4f}")
    print(f"Reason: {' '.join(report['reasons'])}")
    print(f"Decision report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
