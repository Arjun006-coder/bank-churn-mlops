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

## 2. Docker Container Execution

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

## 3. GitHub Actions CI/CD Pipeline
Whenever code is pushed to `main` or `master`, the workflow:
1. Checks out repository code.
2. Sets up Python 3.10 with dependency caching.
3. Installs requirements from `requirements.txt`.
4. Executes unit tests with `pytest`.
5. Builds the Docker container image.

---

## 4. Candidate Model Retraining

Task A prepares `data/incoming_churn.csv` and splits it into a retraining batch
and a separate evaluation batch. Run this step only after the labelled
`data/update_batch.csv` has been produced; Task C does not train on the raw
incoming file or the reserved evaluation batch. Train a candidate from the
original training partition and the update batch:

```bash
python -m monitoring.retrain
```

The script reproduces the notebook's stratified 80/20 split with
`random_state=42`, clones `model.joblib`, and writes the new model to
`models/candidate_model.joblib` with its row counts and training accuracy in
`reports/retraining_summary.json`. It does not use the original test partition
or overwrite the champion model.

MLflow logging is optional. Configure a tracking server with
`MLFLOW_TRACKING_URI`, then run:

```bash
python -m monitoring.retrain --mlflow
```

The run records training parameters, training accuracy, and the candidate
artifact. Use `--tracking-uri` to specify a URI directly.
