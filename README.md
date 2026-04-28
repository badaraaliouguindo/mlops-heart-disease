#  MLOps Heart Disease — End-to-End Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.11-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Docker](https://img.shields.io/badge/Docker-28.5-blue)
![CI/CD](https://github.com/badaraaliouguindo/mlops-heart-disease/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade MLOps pipeline for heart disease prediction. Covers the full lifecycle: experiment tracking with MLflow, model serving with FastAPI, containerization with Docker, and automated CI/CD with GitHub Actions.

 **[Live API on Render](https://mlops-heart-disease.onrender.com/docs)**

---

## Project Overview

This project demonstrates how to take a trained ML model and deploy it as a production-ready service:

- **MLflow** — experiment tracking, hyperparameter logging, model registry
- **FastAPI** — REST API with auto-generated OpenAPI documentation
- **Docker** — containerization for environment reproducibility
- **GitHub Actions** — automated testing and Docker build on every push

---

##  Architecture
```bash
Data → train.py → MLflow tracking → models/
↓
FastAPI (api/main.py)
↓
Dockerfile (container)
↓
GitHub Actions CI/CD pipeline
↓
Render (production)
```

---

##  Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC (test) | 0.857 |
| CV AUC (5-fold) | 0.884 |
| Accuracy | 0.770 |
| F1-Score | 0.800 |

---

##  Project Structure
```bash
mlops-heart-disease/
├── src/
│   └── train.py              # Training script with MLflow logging
├── api/
│   └── main.py               # FastAPI application
├── models/
│   ├── model.pkl             # Trained XGBoost model
│   ├── scaler.pkl            # StandardScaler
│   └── feature_names.pkl     # Feature names
├── data/
│   └── heart_disease.csv     # UCI Heart Disease dataset
├── mlruns/                   # MLflow experiment tracking
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI/CD pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/badaraaliouguindo/mlops-heart-disease
cd mlops-heart-disease

# Install
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train model
python src/train.py

# View MLflow UI
mlflow ui  # Open http://localhost:5000

# Start API
uvicorn api.main:app --reload  # Open http://localhost:8000/docs
```

---

## 🐳 Run with Docker

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Service health check |
| POST | `/predict` | Predict heart disease |
| GET | `/model-info` | Model metadata |

### Example Request

```bash
curl -X POST "https://mlops-heart-disease.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": 1, "trestbps": 130, "chol": 250,
       "fbs": 0, "thalach": 150, "exang": 0, "oldpeak": 1.5,
       "ca": 0, "cp_1": 1, "restecg_1": 1, "slope_1": 1, "thal_2": 1}'
```

---

##  CI/CD Pipeline

On every push to `main`:

1. **Test job** — installs dependencies, loads model, runs validation
2. **Docker job** — builds image, runs container, tests `/health` endpoint

Both jobs must pass before deployment.

---

##  Key Learnings

- MLflow makes experiments reproducible and comparable across runs
- FastAPI generates OpenAPI documentation automatically — no extra work
- Docker eliminates "works on my machine" problems entirely
- GitHub Actions enables zero-touch deployment on every code change
- Separating training (`src/`) from serving (`api/`) is a production best practice

---

##  Tech Stack

`Python` `XGBoost` `MLflow` `FastAPI` `Docker` `GitHub Actions` `Render` `scikit-learn` `Pydantic`

---

##  Author

**Guindo Badara Aliou** — Master's student in Data Science & AI
•[GitHub](https://github.com/badaraaliouguindo) 
• [Portfolio](https://badaraaliouguindo.github.io)
