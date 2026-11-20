# Bank Customer Churn Prediction - ML API & CI/CD Pipeline

This repository implements an end-to-end Machine Learning deployment pipeline for the **Bank Customer Churn** case study (Kaggle `Churn_Modelling.csv` dataset).

## Features
- **Model Training**: Random Forest Classifier with preprocessing pipeline (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) saved as `model.joblib`.
- **FastAPI Backend**: High-performance REST API for real-time customer churn prediction.
- **Automated Testing**: Test suite built with `pytest` covering health, prediction, and edge validation cases.
- **Containerization**: Optimized `Dockerfile` based on `python:3.10-slim`.
- **CI/CD Pipeline**: GitHub Actions workflow (`ci.yml`) automating dependencies installation, unit testing, and Docker image build.

---

## Project Structure
```
ca-2/
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI workflow
├── notebooks/
│   └── training.ipynb       # Model training notebook
├── tests/
│   └── test_api.py          # Pytest test cases
├── .dockerignore            # Excluded files for Docker build
├── .gitignore               # Git ignored files & environments
├── Churn_Modelling.csv      # Kaggle dataset
├── Dockerfile               # Container build configuration
├── main.py                  # FastAPI application & prediction logic
├── model.joblib             # Serialized ML pipeline model
├── README.md                # Documentation & instructions
└── requirements.txt         # Pinned Python dependencies
```

---

## 1. Local Testing with Pytest
Run tests locally:
```bash
python -m pytest -v
```

---

## 2. Prepare an Incoming Data Batch

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
evidence, including failures.
Use `--incoming Churn_Modelling.csv` to use the dataset included in this repo
as the incoming batch instead of placing it at the default path. No output
batches are created when validation fails.

---

## 3. Docker Container Execution

### Build Docker Image
```bash
docker build -t bank-churn-ml-api:latest .
```

### Run Docker Container
```bash
docker run -d -p 8000:8000 --name bank-churn-app bank-churn-ml-api:latest
```

### Test Docker API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Predict Churn (PowerShell)
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

#### Interactive Documentation
Open in browser:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 4. GitHub Actions CI/CD Pipeline
Whenever code is pushed to `main` or `master`, the workflow:
1. Checks out repository code.
2. Sets up Python 3.10 with dependency caching.
3. Installs requirements from `requirements.txt`.
4. Executes unit tests with `pytest`.
5. Builds the Docker container image.

## 5. Evaluating Models
The evaluation module reads a labelled CSV containing an `Exited` target, evaluates
the requested model, and writes Accuracy, Precision, Recall, and F1-score to JSON.
The evaluation CSV is supplied by Member 1; input columns are aligned to the
loaded model's feature schema.

Evaluate the current model:
```bash
python -m monitoring.evaluate \
  --model model.joblib \
  --dataset path/to/evaluation.csv \
  --output reports/champion_metrics.json
```

Use the same module for a retrained candidate model:
```bash
python -m monitoring.evaluate \
  --model path/to/candidate_model.joblib \
  --dataset path/to/evaluation.csv \
  --output reports/candidate_metrics.json
```
