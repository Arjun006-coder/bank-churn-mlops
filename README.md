# Bank Customer Churn Prediction - ML API & Monitoring Pipeline

This repository implements an end-to-end machine-learning and monitoring
pipeline for the **Bank Customer Churn** case study using the Kaggle
`Churn_Modelling.csv` dataset.

## Features
- **Model training**: A serialized scikit-learn pipeline in `model.joblib`.
- **FastAPI backend**: REST endpoints for health checks and real-time churn predictions.
- **Data preparation and evaluation**: Validation, stratified batch splitting, and model metrics.
- **Candidate retraining and monitoring**: Train and compare a candidate without replacing the champion model.
- **Automated testing**: `pytest` coverage for the API, data preparation, evaluation, retraining, and monitoring workflow.
- **Containerization and CI**: Docker image build and GitHub Actions test/build workflow.

---

## Project Structure
```
.
├── .github/workflows/ci.yml  # GitHub Actions tests and Docker build
├── data/                     # Incoming, update, and evaluation CSV batches
├── monitoring/
│   ├── prepare_data.py       # Validate incoming data and split batches
│   ├── evaluate.py           # Evaluate a model and write metrics
│   ├── retrain.py            # Train a candidate model
│   ├── quality_gate.py       # Compare champion and candidate metrics
│   └── run_workflow.py       # Run the complete monitoring workflow
├── notebooks/training.ipynb  # Model training notebook
├── tests/                    # API and monitoring tests
├── reports/                  # Validation, metrics, and workflow reports
├── Churn_Modelling.csv       # Original labelled dataset
├── main.py                   # FastAPI application
├── model.joblib              # Deployed champion model
├── Dockerfile                # API container configuration
└── requirements.txt          # Python dependencies
```

---

## Run the Tests

Run the complete test suite:
```bash
python -m pytest -v
```

---

## Task 1 - Data Preparation (Tejaswani Mishra)

Place the labelled incoming batch at `data/incoming_churn.csv`. The batch must
include `CustomerId`, the model input columns, and the `Exited` target. Provide
a separate CSV of previously used training records so incoming `CustomerId`
values can be checked for overlap:

```bash
python monitoring/prepare_data.py --old-training path/to/old_training.csv
```

The script validates required columns, missing values, duplicate rows and IDs,
and overlap with old training records. It prints the row/column count and
`Exited` class distribution. On success, it creates
`data/update_batch.csv` and `data/evaluation_batch.csv` using a stratified
80/20 split. It also writes `reports/data_validation.json` with the validation
evidence, including failures. No output batches are created when validation
fails.

Use `--incoming Churn_Modelling.csv` to use the dataset included in this
repository as the incoming batch instead of the default path.

---

## Task 2 - Model Evaluation (Arjun Agrawal)

The evaluation module reads a labelled CSV containing an `Exited` target,
aligns input columns to the loaded model's feature schema, and writes accuracy,
precision, recall, and F1-score (`f1_score`) to JSON. Use the same evaluation
batch to compare the champion and candidate models.

Evaluate the current champion:
```bash
python -m monitoring.evaluate \
  --model model.joblib \
  --dataset data/evaluation_batch.csv \
  --output reports/champion_metrics.json
```

Evaluate a retrained candidate:
```bash
python -m monitoring.evaluate \
  --model models/candidate_model.joblib \
  --dataset data/evaluation_batch.csv \
  --output reports/candidate_metrics.json
```

---

## Task 3 - Model Retraining (Shaurya Tripathi)

After Task 1 creates the update and evaluation batches, retraining uses only
`data/update_batch.csv`; the evaluation batch is reserved for model comparison.
The retraining script reproduces the notebook's stratified 80/20 split with
`random_state=42`, combines the original training partition with the update
batch, and fits a clone of the champion model:

```bash
python -m monitoring.retrain
```

It saves the candidate to `models/candidate_model.joblib` and writes row counts
and training accuracy to `reports/retraining_summary.json`. The existing
`model.joblib` is not replaced.

MLflow tracking is optional. Configure `MLFLOW_TRACKING_URI` if needed and run:

```bash
python -m monitoring.retrain --mlflow
```

The MLflow run records training parameters, training accuracy, and the candidate
model artifact.

---

## Task 4 (Task D) - Model Quality Gate (Sarvesh Singh)

The quality gate compares champion and candidate metrics produced from the
same evaluation batch. A candidate is eligible for **manual promotion** only
when its F1-score improves by at least `0.01` and its recall does not drop by
more than `0.02`. The decision is recorded in
`reports/quality_gate_decision.json`; the deployed champion is never replaced
automatically.

```bash
python -m monitoring.quality_gate \
  --champion-report reports/champion_metrics.json \
  --candidate-report reports/candidate_metrics.json \
  --output reports/quality_gate_decision.json
```

---

## Task 5 (Task E) - Workflow Integration and Demonstration (Sarvesh Singh)

Run the end-to-end demonstration after the incoming data is ready:

```bash
python -m monitoring.run_workflow --new-data data/incoming_churn.csv
```

The workflow runs data preparation, champion evaluation, candidate retraining,
candidate evaluation, and the quality gate in order. It writes the validation,
evaluation, retraining, gate-decision, and workflow-execution reports under
`reports/`, saves the candidate under `models/`, and verifies that the champion
model remains unchanged.

To run only the quality-gate demonstration, first ensure both evaluation
reports already exist:

```bash
python -m monitoring.run_workflow --quality-gate-only
```

Run the focused quality-gate and workflow integration tests:

```bash
python -m pytest tests/test_monitoring.py -v
```

Incoming sample data in this repository is for demonstration; its evaluation
scores should not be treated as evidence of real-world model performance.

---

## API Containerization and CI/CD

Build and run the API container:
```bash
docker build -t bank-churn-ml-api:latest .
docker run -d -p 8000:8000 --name bank-churn-app bank-churn-ml-api:latest
```

Check that the model loaded and the API is healthy:
```bash
curl http://localhost:8000/health
```

Make a prediction (PowerShell):
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body (@{
    CreditScore = 619
    Geography = "France"
    Gender = "Female"
    Age = 42
    Tenure = 2
    Balance = 0.0
    NumOfProducts = 1
    HasCrCard = 1
    IsActiveMember = 1
    EstimatedSalary = 101348.88
} | ConvertTo-Json)
```

Interactive API documentation:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

The workflow in `.github/workflows/ci.yml` runs for pull requests and pushes to
`main` or `master`. It sets up Python 3.10, installs `requirements.txt`, runs
the test suite, and builds the Docker image.
