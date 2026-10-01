"""Task 3 - MLflow experiment tracking helpers.

What gets logged for every model (one parent run per model):
  * params   : model kind, best hyper-parameters, CV folds, seed, data md5
  * metrics  : CV mean/std for 5 metrics + held-out test metrics
  * artifacts: confusion matrix, ROC curve, feature importance plot,
               full grid-search table (CSV), the fitted sklearn pipeline (with signature)
  * children : one nested run per grid-search configuration (params + mean CV ROC-AUC)
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

from heart import config  # noqa: E402

TRACKING_URI = f"sqlite:///{config.ROOT / 'mlflow.db'}"
EXPERIMENT = "heart-disease-classifier"
AUTHOR = "Pranay Jain"


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def data_md5() -> str:
    return hashlib.md5(config.CLEAN_FILE.read_bytes()).hexdigest()[:10]


def _plot_confusion(model, X_te, y_te, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, ax=ax, cmap="Blues",
                                          display_labels=["no disease", "disease"])
    ax.set_title("Confusion matrix (test)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(model, X_te, y_te, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_title("ROC curve (test)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(model, path):
    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values, label = clf.feature_importances_, "importance"
    else:
        values, label = np.abs(clf.coef_[0]), "|coefficient|"
    s = pd.Series(values, index=names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    s.plot.barh(ax=ax, color="#C44E52")
    ax.set_title(f"Top features ({label})")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return s.sort_values(ascending=False)


def log_model_run(name, search, best, row, X_tr, X_te, y_te, cv_folds, seed):
    """One parent run per model + nested child runs for each grid config."""
    with mlflow.start_run(run_name=f"{name}-tuned") as parent:
        mlflow.set_tags({"author": AUTHOR, "model_kind": name,
                         "stage": "task3-tracking", "selection_metric": "cv_roc_auc"})
        mlflow.log_params({"model_kind": name, "cv_folds": cv_folds, "seed": seed,
                           "n_train": len(X_tr), "n_test": len(X_te),
                           "data_md5": data_md5(), "n_configs_tried": row["n_configs"]})
        mlflow.log_params({f"best_{k.replace('clf__', '')}": v
                           for k, v in search.best_params_.items()})
        mlflow.log_metrics({k: float(v) for k, v in row.items()
                            if k.startswith(("cv_", "test_"))})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _plot_confusion(best, X_te, y_te, tmp / "confusion_matrix.png")
            _plot_roc(best, X_te, y_te, tmp / "roc_curve.png")
            top = _plot_importance(best, tmp / "feature_importance.png")
            top.to_csv(tmp / "feature_importance.csv", header=["value"])
            grid = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]]
            grid.sort_values("rank_test_score").to_csv(tmp / "grid_search_results.csv",
                                                       index=False)
            mlflow.log_artifacts(str(tmp), artifact_path="plots")

        signature = infer_signature(X_tr.astype(float), best.predict_proba(X_tr)[:, 1])
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_tr.head(3).astype(float))

        # nested runs: every configuration the grid search tried
        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-cfg{i:02d}", nested=True):
                mlflow.set_tags({"author": AUTHOR, "model_kind": name, "type": "grid-config"})
                mlflow.log_params({k.replace("clf__", ""): v for k, v in params.items()})
                mlflow.log_metrics({"cv_roc_auc": float(res["mean_test_score"][i]),
                                    "cv_roc_auc_std": float(res["std_test_score"][i])})
        return parent.info.run_id
