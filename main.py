from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Locate and load the trained Bank Churn classification model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Machine Learning API to predict customer churn based on the Kaggle Churn Modelling case study.",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, description="Customer credit score (300-850)", examples=[619])
    Geography: str = Field(..., description="Country of customer: France, Spain, or Germany", examples=["France"])
    Gender: str = Field(..., description="Customer gender: Female or Male", examples=["Female"])
    Age: int = Field(..., ge=18, le=100, description="Customer age in years", examples=[42])
    Tenure: int = Field(..., ge=0, le=10, description="Years customer has been with the bank (0-10)", examples=[2])
    Balance: float = Field(..., ge=0.0, description="Account balance", examples=[0.0])
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of bank products (1-4)", examples=[1])
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (1 = Yes, 0 = No)", examples=[1])
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (1 = Yes, 0 = No)", examples=[1])
    EstimatedSalary: float = Field(..., ge=0.0, description="Estimated annual salary", examples=[101348.88])

@app.get("/")
def home():
    return {
        "title": "Bank Customer Churn Prediction API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Format customer data into DataFrame with exact feature names
    data = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(data)[0])
    
    # Calculate probability if model supports predict_proba
    churn_probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[0]
        churn_probability = round(float(probabilities[1]), 4)

    return {
        "prediction": prediction,
        "churn": bool(prediction == 1),
        "churn_probability": churn_probability,
        "result": "Customer is likely to churn" if prediction == 1 else "Customer is likely to stay"
    }
