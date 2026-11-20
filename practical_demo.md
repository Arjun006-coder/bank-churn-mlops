# MLOps Practical: Model-quality gate and workflow demonstration

**Member 4:** Sarvesh Singh
**Project:** Bank Churn Prediction

## Contribution

My module compares the performance of the deployed champion model and a newly retrained candidate. It records whether the candidate meets the quality policy. The workflow runner executes the team modules in order, reports errors, and checks that `model.joblib` was not overwritten.

## Decision policy

F1-score for churn class `Exited = 1` is the primary metric because customer churn detection requires balancing precision and recall. Recall is an additional safeguard.

Candidate passes when both conditions are true:

1. Candidate F1 - champion F1 >= 0.01.
2. Candidate recall >= champion recall - 0.02.

A passing decision means **eligible for manual promotion**, not an automatic deployment.

## Run tests

```bash
python -m pytest tests/test_monitoring.py -v
```

## Run complete workflow

```bash
python -m monitoring.run_workflow --new-data data/incoming_churn.csv
```

This invokes, in order: `prepare_data` → `evaluate` (champion) → `retrain` → `evaluate` (candidate) → `quality_gate`.

To log the training run to MLflow, use `--mlflow` if the tracking setup is configured:

```bash
python -m monitoring.run_workflow --new-data data/incoming_churn.csv --mlflow
```

## Run only the quality gate

```bash
python -m monitoring.run_workflow --quality-gate-only
```

Both `reports/champion_metrics.json` and `reports/candidate_metrics.json` must already exist. Both evaluations must use the **same untouched labelled evaluation batch**. The evaluation script writes `f1_score`, which the quality gate accepts.

## Evidence files

- `reports/data_validation.json`: input data validation result.
- `reports/champion_metrics.json`: existing-model evaluation.
- `models/candidate_model.joblib`: separately saved candidate artifact.
- `reports/candidate_metrics.json`: new-model evaluation.
- `reports/quality_gate_decision.json`: comparison, thresholds, reasons and decision.
- `reports/workflow_execution.json`: successful or failed step record.

The prepared incoming file is synthetic demonstration data; don't describe its evaluation scores as evidence of real-world bank model performance. Neither model should be fitted on the evaluation batch.

## Viva explanation

"My contribution is the model-quality gate and workflow integration. I compare champion and candidate F1 and recall on the same labelled evaluation batch. If the candidate improves enough without a large recall drop, it becomes eligible for manual promotion. I record the decision in JSON, and I do not automatically replace the deployed model."
