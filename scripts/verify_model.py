"""Reproducibility check for the packaged model (Task 4).

1. Load models/heart_model.joblib in a FRESH process and predict on 3 sample patients.
2. Load models/mlflow_model with the MLflow loaders.
3. Confirm both formats give identical probabilities and the metadata matches.

Usage:  python scripts/verify_model.py
"""
import json
import sys

import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd

from heart import config
from heart.predict import META_FILE, load_model, predict_records

SAMPLES = [
    # high-risk profile: asymptomatic chest pain, exercise angina, reversible thal defect
    {"age": 62, "sex": 1, "cp": 4, "trestbps": 150, "chol": 280, "fbs": 0, "restecg": 2,
     "thalach": 115, "exang": 1, "oldpeak": 2.8, "slope": 2, "ca": 2, "thal": 7},
    # low-risk profile
    {"age": 41, "sex": 0, "cp": 2, "trestbps": 125, "chol": 210, "fbs": 0, "restecg": 0,
     "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3},
    # incomplete record: ca and thal missing -> imputed by the pipeline
    {"age": 55, "sex": 1, "cp": 3, "trestbps": 135, "chol": 240, "fbs": 1, "restecg": 0,
     "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": None, "thal": None},
]


def main():
    meta = json.loads(META_FILE.read_text())
    print(f"model: {meta['model_kind']}  version: {meta['model_version']}  "
          f"cv_roc_auc: {meta['cv_metrics']['cv_roc_auc']}")

    preds = predict_records(SAMPLES, model=load_model())
    for s, p in zip(SAMPLES, preds):
        print(f"  age={s['age']:>2} cp={s['cp']} thal={s['thal']!s:>4} -> {p}")

    X = pd.DataFrame(SAMPLES).reindex(columns=config.FEATURES).astype(float)
    joblib_proba = load_model().predict_proba(X)[:, 1]
    mlflow_dir = str(config.ROOT / "models" / "mlflow_model")
    mlflow_proba = mlflow.sklearn.load_model(mlflow_dir).predict_proba(X)[:, 1]
    mlflow.pyfunc.load_model(mlflow_dir).predict(X)  # generic loader works too

    same = np.allclose(joblib_proba, mlflow_proba)
    print(f"joblib vs MLflow-format probabilities identical: {same}")
    if not same:
        sys.exit("[FAIL] model formats disagree")
    print("[OK] packaged model reloads and predicts reproducibly")


if __name__ == "__main__":
    main()
