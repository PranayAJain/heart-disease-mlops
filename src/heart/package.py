"""Task 4 - package the selected model for reuse and serving.

Steps:
  1. Read reports/model_comparison.csv -> winner (highest CV ROC-AUC) + its best params.
  2. Re-fit that exact pipeline (preprocessing + classifier) on ALL 303 rows.
  3. Save three reusable formats into models/:
       heart_model.joblib     - full sklearn Pipeline (used by the API)
       mlflow_model/          - MLflow model format (MLmodel, conda.yaml, requirements.txt)
       model_metadata.json    - version, params, metrics, features, library versions
  4. Log a "final" MLflow run and register it in the Model Registry.

Usage:  python -m heart.package
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature

from heart import config, tracking
from heart.data import load_clean
from heart.features import split_xy
from heart.train import MODELS, make_pipeline

MODEL_DIR = config.ROOT / "models"
MODEL_FILE = MODEL_DIR / "heart_model.joblib"
META_FILE = MODEL_DIR / "model_metadata.json"
MLFLOW_DIR = MODEL_DIR / "mlflow_model"
REGISTERED_NAME = "heart-disease-classifier"


def select_winner() -> pd.Series:
    results = pd.read_csv(config.ROOT / "reports" / "model_comparison.csv")
    return results.sort_values("cv_roc_auc", ascending=False).iloc[0]


def build_final_model(winner: pd.Series):
    estimator, _grid = MODELS[winner["model"]]
    params = {f"clf__{k}": v for k, v in json.loads(winner["best_params"]).items()}
    pipe = make_pipeline(sklearn.base.clone(estimator))
    pipe.set_params(**params)
    return pipe


def main():
    winner = select_winner()
    df = load_clean()
    X, y = split_xy(df)
    model = build_final_model(winner).fit(X, y)
    print(f"[package] winner={winner['model']}  params={winner['best_params']}  fitted on {len(X)} rows")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    meta = {
        "model_name": REGISTERED_NAME,
        "model_version": version,
        "model_kind": winner["model"],
        "best_params": json.loads(winner["best_params"]),
        "selection_metric": "cv_roc_auc",
        "cv_metrics": {k: round(float(winner[k]), 4) for k in winner.index
                       if k.startswith("cv_") and not k.endswith("_std")},
        "test_metrics": {k: round(float(winner[k]), 4) for k in winner.index
                         if k.startswith("test_")},
        "decision_threshold": 0.5,
        "features": config.FEATURES,
        "trained_rows": int(len(X)),
        "data_md5": tracking.data_md5(),
        "library_versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                             "mlflow": mlflow.__version__},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "author": tracking.AUTHOR,
    }
    META_FILE.write_text(json.dumps(meta, indent=2))

    signature = infer_signature(X.astype(float), model.predict_proba(X)[:, 1])
    if MLFLOW_DIR.exists():
        shutil.rmtree(MLFLOW_DIR)
    mlflow.sklearn.save_model(model, str(MLFLOW_DIR), signature=signature,
                              input_example=X.head(2).astype(float))

    tracking.setup()
    with mlflow.start_run(run_name=f"final-{winner['model']}") as run:
        mlflow.set_tags({"author": tracking.AUTHOR, "stage": "task4-final-model",
                         "model_kind": winner["model"]})
        mlflow.log_params({"model_kind": winner["model"], "trained_rows": len(X),
                           **{f"best_{k}": v for k, v in meta["best_params"].items()}})
        mlflow.log_metrics({**meta["cv_metrics"], **meta["test_metrics"]})
        mlflow.log_artifact(str(META_FILE))
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                 registered_model_name=REGISTERED_NAME)
        print(f"[package] MLflow run {run.info.run_id} registered as '{REGISTERED_NAME}'")

    for p in (MODEL_FILE, META_FILE, MLFLOW_DIR):
        print(f"[package] saved {p.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
