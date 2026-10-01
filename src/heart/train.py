"""Task 2 - train, tune and compare classifiers with cross-validation.

Protocol:
  1. Stratified 80/20 train/test split (test set is touched only once, at the end).
  2. For each model: GridSearchCV (5-fold stratified, scoring = ROC-AUC) on train.
  3. Re-score the best config with 5-fold CV on 5 metrics (mean +/- std).
  4. Evaluate each tuned model once on the held-out test set.
  5. Pick the winner by CV ROC-AUC (not by test score - avoids test-set overfitting).

Usage:
    python -m heart.train                 # all models
    python -m heart.train --models rf xgb # subset
    python -m heart.train --no-track      # skip MLflow logging
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from heart import config
from heart.data import load_clean
from heart.features import build_preprocessor, split_xy
from heart import tracking

SEED = 42
REPORT_DIR = config.ROOT / "reports"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# model name -> (estimator, hyper-parameter grid). Grid keys use the pipeline step "clf__".
MODELS = {
    "logreg": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 0.3, 1.0, 3.0],
         "clf__class_weight": [None, "balanced"]},
    ),
    "rf": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3, 5],
         "clf__max_features": ["sqrt", 0.5]},
    ),
    "xgb": (
        XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [150, 300],
         "clf__max_depth": [2, 3, 4],
         "clf__learning_rate": [0.03, 0.1],
         "clf__subsample": [0.8, 1.0],
         "clf__colsample_bytree": [0.7, 1.0]},
    ),
}


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def test_metrics(model, X_te, y_te) -> dict:
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }


def run(model_names, track=True):
    df = load_clean()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"train={X_tr.shape}  test={X_te.shape}  positives(train)={y_tr.mean():.0%}\n")

    if track:
        tracking.setup()
    rows, fitted = [], {}
    for name in model_names:
        estimator, grid = MODELS[name]
        search = GridSearchCV(make_pipeline(estimator), grid, scoring="roc_auc",
                              cv=cv, n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_

        cv_scores = cross_validate(best, X_tr, y_tr, cv=cv, scoring=METRICS, n_jobs=-1)
        test = test_metrics(best, X_te, y_te)
        row = {"model": name, "n_configs": len(search.cv_results_["params"])}
        for m in METRICS:
            row[f"cv_{m}"] = cv_scores[f"test_{m}"].mean()
            row[f"cv_{m}_std"] = cv_scores[f"test_{m}"].std()
            row[f"test_{m}"] = test[m]
        row["best_params"] = json.dumps({k.replace("clf__", ""): v
                                         for k, v in search.best_params_.items()})
        rows.append(row)
        fitted[name] = best
        if track:
            row["mlflow_run_id"] = tracking.log_model_run(
                name, search, best, row, X_tr, X_te, y_te, cv_folds=5, seed=SEED)
        print(f"[{name:6s}] tried {row['n_configs']:3d} configs | "
              f"CV ROC-AUC {row['cv_roc_auc']:.3f} +/- {row['cv_roc_auc_std']:.3f} | "
              f"test ROC-AUC {row['test_roc_auc']:.3f} | best {row['best_params']}")

    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    show = results[["model"] + [f"cv_{m}" for m in METRICS] + ["test_roc_auc", "test_recall"]]
    print("\n=== Model comparison (5-fold CV on train, mean) ===")
    print(show.round(3).to_string(index=False))
    winner = results.iloc[0]["model"]
    print(f"\nSelected model (highest CV ROC-AUC): {winner}")
    print(f"Saved {(REPORT_DIR / 'model_comparison.csv').relative_to(config.ROOT)}")
    return results, fitted, (X_tr, X_te, y_tr, y_te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--no-track", action="store_true", help="disable MLflow logging")
    args = ap.parse_args()
    run(args.models, track=not args.no_track)
