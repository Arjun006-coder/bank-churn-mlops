"""Run the bank churn monitoring pipeline and preserve the current champion."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bank churn monitoring workflow")
    parser.add_argument("--new-data", default="data/incoming_churn.csv")
    parser.add_argument("--original-data", default="Churn_Modelling.csv")
    parser.add_argument("--update-data", default="data/update_batch.csv")
    parser.add_argument("--evaluation-data", default="data/evaluation_batch.csv")
    parser.add_argument("--champion-model", default="model.joblib")
    parser.add_argument("--candidate-model", default="models/candidate_model.joblib")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--min-f1-gain", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    parser.add_argument("--quality-gate-only", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args(argv)


def workflow_steps(args):
    reports = Path(args.report_dir)
    champion_report = reports / "champion_metrics.json"
    candidate_report = reports / "candidate_metrics.json"
    decision_report = reports / "quality_gate_decision.json"
    gate = (
        "quality_gate", "monitoring.quality_gate",
        ["--champion-report", str(champion_report),
         "--candidate-report", str(candidate_report),
         "--output", str(decision_report),
         "--champion-model", args.champion_model,
         "--candidate-model", args.candidate_model,
         "--min-f1-gain", str(args.min_f1_gain),
         "--max-recall-drop", str(args.max_recall_drop)],
        [decision_report],
    )
    if args.quality_gate_only:
        return [gate]

    retrain_args = [
        "--original-data", args.original_data,
        "--update-data", args.update_data,
        "--champion-model", args.champion_model,
        "--candidate-model", args.candidate_model,
        "--summary", str(reports / "retraining_summary.json"),
    ]
    if args.mlflow:
        retrain_args.append("--mlflow")

    return [
        (
            "prepare_data", "monitoring.prepare_data",
            ["--incoming", args.new_data,
             "--old-training", args.original_data,
             "--update-output", args.update_data,
             "--evaluation-output", args.evaluation_data,
             "--report", str(reports / "data_validation.json")],
            [Path(args.update_data), Path(args.evaluation_data), reports / "data_validation.json"],
        ),
        (
            "evaluate_champion", "monitoring.evaluate",
            ["--model", args.champion_model,
             "--dataset", args.evaluation_data,
             "--output", str(champion_report)],
            [champion_report],
        ),
        (
            "retrain_candidate", "monitoring.retrain", retrain_args,
            [Path(args.candidate_model), reports / "retraining_summary.json"],
        ),
        (
            "evaluate_candidate", "monitoring.evaluate",
            ["--model", args.candidate_model,
             "--dataset", args.evaluation_data,
             "--output", str(candidate_report)],
            [candidate_report],
        ),
        gate,
    ]


def _disk_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_log(report_path, log):
    path = _disk_path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    report_path = Path(args.report_dir) / "workflow_execution.json"
    log = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "completed_steps": [],
        "champion_model_changed": False,
    }

    def fail(message):
        log["status"] = "failed"
        log["error"] = message
        log["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _save_log(report_path, log)
        print(f"Workflow failed: {message}", file=sys.stderr)
        return 1

    champion_path = _disk_path(args.champion_model)
    before_hash = None
    if not args.quality_gate_only:
        for item in (args.new_data, args.original_data, args.champion_model):
            if not _disk_path(item).is_file():
                return fail(f"Required input missing: {item}")
        for module in ("prepare_data", "evaluate", "retrain"):
            path = PROJECT_ROOT / "monitoring" / f"{module}.py"
            if not path.is_file():
                return fail(f"Team module missing: {path}")
        if _disk_path(args.candidate_model).resolve() == champion_path.resolve():
            return fail("Candidate model must not overwrite the champion")
        before_hash = _checksum(champion_path)

    for name, module, flags, outputs in workflow_steps(args):
        print(f"Running: {name}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", module, *flags],
                cwd=PROJECT_ROOT, check=False,
            )
        except OSError as error:
            return fail(f"Cannot start {name}: {error}")
        if result.returncode != 0:
            return fail(f"{name} failed (exit code {result.returncode})")
        missing = [str(path) for path in outputs if not _disk_path(path).is_file()]
        if missing:
            return fail(f"{name} did not create: {', '.join(missing)}")
        log["completed_steps"].append(name)

    if before_hash is not None:
        if not champion_path.is_file() or _checksum(champion_path) != before_hash:
            log["champion_model_changed"] = True
            return fail("The champion artifact was modified")

    log["status"] = "completed"
    log["decision_report"] = str(Path(args.report_dir) / "quality_gate_decision.json")
    log["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _save_log(report_path, log)
    print(f"Workflow completed. See {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
