import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "endpoints" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_churn_customer():
    payload = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert isinstance(data["churn"], bool)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "result" in data

def test_predict_retained_customer():
    payload = {
        "CreditScore": 850,
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 30,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "churn_probability" in data

def test_predict_invalid_data():
    payload = {
        "CreditScore": 100,  # Below minimum 300
        "Geography": "France",
        "Gender": "Female"
        # Missing required fields
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422