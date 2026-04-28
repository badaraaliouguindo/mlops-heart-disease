from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps pipeline — XGBoost model served via FastAPI",
    version="1.0.0"
)

# --- Chargement du modèle ---
try:
    model         = joblib.load("models/model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
except Exception as e:
    raise RuntimeError(f"Erreur chargement modèle : {e}")

# --- Schéma d'entrée ---
class PatientData(BaseModel):
    age:      float
    sex:      float
    trestbps: float
    chol:     float
    fbs:      float
    thalach:  float
    exang:    float
    oldpeak:  float
    ca:       float
    cp_0:     float = 0
    cp_1:     float = 0
    cp_2:     float = 0
    cp_3:     float = 0
    restecg_0: float = 0
    restecg_1: float = 0
    restecg_2: float = 0
    slope_0:  float = 0
    slope_1:  float = 0
    slope_2:  float = 0
    thal_0:   float = 0
    thal_1:   float = 0
    thal_2:   float = 0
    thal_3:   float = 0

# --- Endpoints ---
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "features": len(feature_names)}

@app.post("/predict")
def predict(patient: PatientData):
    try:
        # Construction du vecteur
        data = pd.DataFrame([patient.dict()])[feature_names]

        # Scaling
        data_scaled = scaler.transform(data)

        # Prédiction
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": round(float(probability), 4),
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "algorithm":    "XGBoost",
        "n_estimators": 100,
        "max_depth":    4,
        "learning_rate": 0.1,
        "auc_roc":      0.8571,
        "cv_auc":       0.8843,
        "features":     feature_names
    }