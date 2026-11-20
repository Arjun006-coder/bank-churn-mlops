"""Unit tests for the quality gate and orchestration module."""

import json
from pathlib import Path
import subprocess

import pytest

from monitoring import quality_gate, run_workflow


CHAMPION = {"accuracy": 0.85, "precision": 0.75, "recall": 0.70, "f1": 0.72}


def test_candidate_eligible():
    candidate = {"accuracy": 0.87, "precision": 0.77, "recall": 0.69, "f1": 0.74}
    decision = quality_gate.evaluate_quality_gate(CHAMPION, candidate)
    assert decision["decision"] == "ELIGIBLE_FOR_PROMOTION"
    assert decision["production_model_changed"] is False


def test_reject_small_f1_gain():
    candidate = {"accuracy": 0.86, "precision": 0.76, "recall": 0.71, "f1": 0.725}
    decision = quality_gate.evaluate_quality_gate(CHAMPION, candidate)
    assert decision["decision"] == "RETAIN_CHAMPION"
    assert not decision["comparison"]["f1_pass"]


def test_reject_large_recall_drop():
    candidate = {"accuracy": 0.88, "precision": 0.80, "recall": 0.65, "f1": 0.75}
    decision = quality_gate.evaluate_quality_gate(CHAMPION, candidate)
    assert decision["decision"] == "RETAIN_CHAMPION"
    assert not decision["comparison"]["recall_pass"]


@pytest.mark.parametrize("bad", [
    {"accuracy": 0.8, "precision": 0.6, "recall": 0.7},
    {"accuracy": True, "precision": 0.6, "recall": 0.7, "f1": 0.7},
    {"accuracy": 0.8, "precision": 1.5, "recall": 0.7, "f1": 0.7},
    {"accuracy": 0.8, "precision": 0.6, "recall": 0.7, "f1": float("nan")},
])
def test_invalid_metrics(bad):
    with pytest.raises(quality_gate.QualityGateError):
        quality_gate.evaluate_quality_gate(CHAMPION, bad)


def test_accepts_f1_score_alias(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({**CHAMPION, "f1_score": 0.72, "f1": None}))
    assert quality_gate.load_metrics(path)["f1"] == 0.72


def test_gate_writes_report_without_overwriting_champion(tmp_path):
    champion = tmp_path / "model.joblib"
    champion.write_bytes(b"original artifact")
    old_report = tmp_path / "champion.json"
    new_report = tmp_path / "candidate.json"
    decision_file = tmp_path / "decision.json"
    old_report.write_text(json.dumps(CHAMPION))
    new_report.write_text(json.dumps({"accuracy": 0.87, "precision": 0.77, "recall": 0.69, "f1_score": 0.74}))
    status = quality_gate.main([
        "--champion-report", str(old_report),
        "--candidate-report", str(new_report),
        "--output", str(decision_file),
        "--champion-model", str(champion),
    ])
    assert status == 0
    assert json.loads(decision_file.read_text())["decision"] == "ELIGIBLE_FOR_PROMOTION"
    assert champion.read_bytes() == b"original artifact"


def test_workflow_uses_existing_member_cli_flags():
    steps = run_workflow.workflow_steps(run_workflow.parse_args([]))
    assert [s[0] for s in steps] == [
        "prepare_data", "evaluate_champion", "retrain_candidate", "evaluate_candidate", "quality_gate"
    ]
    assert "--incoming" in steps[0][2]
    assert "--old-training" in steps[0][2]
    assert "--dataset" in steps[1][2]
    assert "--champion-model" in steps[2][2]
    assert "--candidate-model" in steps[2][2]
    assert "--summary" in steps[2][2]
    assert "--dataset" in steps[3][2]


def test_workflow_runs_all_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(run_workflow, "PROJECT_ROOT", tmp_path)
    (tmp_path / "monitoring").mkdir()
    (tmp_path / "data").mkdir()
    for name in ("prepare_data", "evaluate", "retrain"):
        (tmp_path / "monitoring" / f"{name}.py").write_text("# stub")
    for name in ("Churn_Modelling.csv", "model.joblib", "data/incoming_churn.csv"):
        (tmp_path / name).write_text("fixture")

    steps = run_workflow.workflow_steps(run_workflow.parse_args([]))
    calls = []

    def fake_run(command, cwd, check):
        name, _, _, outputs = steps[len(calls)]
        calls.append((name, command))
        for output in outputs:
            dest = Path(cwd) / output
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("fixture")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_workflow.subprocess, "run", fake_run)
    assert run_workflow.main([]) == 0
    assert len(calls) == 5
    log = json.loads((tmp_path / "reports/workflow_execution.json").read_text())
    assert log["status"] == "completed"
    assert log["champion_model_changed"] is False


def test_workflow_stops_when_input_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_workflow, "PROJECT_ROOT", tmp_path)
    assert run_workflow.main([]) == 1
    log = json.loads((tmp_path / "reports/workflow_execution.json").read_text())
    assert log["status"] == "failed"


def test_workflow_logs_failed_step(tmp_path, monkeypatch):
    monkeypatch.setattr(run_workflow, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        run_workflow.subprocess, "run",
        lambda command, cwd, check: subprocess.CompletedProcess(command, 1),
    )
    assert run_workflow.main(["--quality-gate-only"]) == 1
    log = json.loads((tmp_path / "reports/workflow_execution.json").read_text())
    assert log["status"] == "failed"
    assert log["completed_steps"] == []
