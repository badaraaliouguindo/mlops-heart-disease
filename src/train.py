import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib
import os

# --- Configuration MLflow ---
mlflow.set_experiment("heart-disease-prediction")

# --- Chargement des données ---
df = pd.read_csv("data/heart_disease.csv")

cat_cols = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --- Hyperparamètres ---
params = {
    "n_estimators":  100,
    "max_depth":     4,
    "learning_rate": 0.1,
    "random_state":  42,
    "eval_metric":   "logloss"
}

# --- Run MLflow ---
with mlflow.start_run():

    # Log des hyperparamètres
    mlflow.log_params(params)

    # Entraînement
    model = XGBClassifier(**params)
    model.fit(X_train_sc, y_train)

    # Prédictions
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    # Métriques
    auc      = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    cv_auc   = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

    # Log des métriques
    mlflow.log_metric("auc_roc",  auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("cv_auc",   cv_auc)

    print(f"AUC-ROC  : {auc:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"CV AUC   : {cv_auc:.4f}")

    # Sauvegarde du modèle et du scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,  "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    # Log des artifacts dans MLflow
    mlflow.log_artifact("models/model.pkl")
    mlflow.log_artifact("models/scaler.pkl")
    mlflow.log_artifact("models/feature_names.pkl")

    # Log du modèle sklearn
    mlflow.sklearn.log_model(model, "xgboost_model")

    print("\nRun MLflow terminé ✓")
    print("Lance 'mlflow ui' pour visualiser les résultats")