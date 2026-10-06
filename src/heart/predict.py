"""Inference helper - one place that turns raw patient records into predictions.
Used by the API, the tests and scripts/verify_model.py.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart import config

MODEL_FILE = config.ROOT / "models" / "heart_model.joblib"
META_FILE = config.ROOT / "models" / "model_metadata.json"


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_FILE):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run `python -m heart.package` first")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata(path: Path = META_FILE) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def predict_records(records: list[dict], model=None, threshold: float | None = None) -> list[dict]:
    """records: list of dicts with the 13 feature keys (missing keys/None -> imputed)."""
    model = model or load_model()
    if threshold is None:
        threshold = load_metadata().get("decision_threshold", 0.5)
    X = pd.DataFrame(records).reindex(columns=config.FEATURES).astype(float)
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= threshold)
        out.append({
            "prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
        })
    return out


def main(argv=None):
    """CLI inference: python -m heart.predict samples/patient_high_risk.json [more.json ...]"""
    import argparse
    ap = argparse.ArgumentParser(description="Predict heart-disease risk for patient JSON file(s)")
    ap.add_argument("files", nargs="+", help="JSON file with one patient (dict) or a list of patients")
    args = ap.parse_args(argv)
    meta = load_metadata()
    print(f"model: {meta.get('model_kind')}  version: {meta.get('model_version')}")
    for f in args.files:
        data = json.loads(Path(f).read_text())
        records = data if isinstance(data, list) else [data]
        for rec, res in zip(records, predict_records(records)):
            print(f"{f}: age={rec.get('age')} cp={rec.get('cp')} thal={rec.get('thal')} -> "
                  f"{res['label']} (p={res['probability_disease']}, confidence={res['confidence']})")


if __name__ == "__main__":
    main()
