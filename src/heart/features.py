"""Feature engineering + preprocessing pipeline (shared by training, tests and the API).

Design:
  * add_features: one engineered clinical feature -
      hr_pct_max = thalach / (220 - age)  -> share of age-predicted max heart rate reached
  * numeric  -> median impute -> StandardScaler
  * binary   -> most-frequent impute (already 0/1)
  * nominal  -> most-frequent impute -> OneHotEncoder(handle_unknown="ignore")
Everything lives in ONE sklearn Pipeline, so the exact same transforms run at
training time, in cross-validation folds (no leakage) and inside the API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from heart import config

ENGINEERED = ["hr_pct_max"]
NUMERIC_ALL = config.NUMERIC + ENGINEERED


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X[config.FEATURES].astype(float).copy()   # nullable Int64 -> float (NaN-safe)
    X["hr_pct_max"] = X["thalach"] / (220.0 - X["age"])
    return X


def build_preprocessor() -> Pipeline:
    columns = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_ALL),
            ("bin", SimpleImputer(strategy="most_frequent"), config.BINARY),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                       sparse_output=False))]),
             config.CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("engineer", FunctionTransformer(add_features, feature_names_out=_engineered_names)),
        ("columns", columns),
    ])


def _engineered_names(_transformer, _input_features):
    return np.array(config.FEATURES + ENGINEERED, dtype=object)


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET].astype(int)
