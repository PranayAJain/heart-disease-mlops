"""Load the raw UCI file and produce the cleaned dataset.

Cleaning decisions (documented for the report):
  1. '?' markers -> NaN (only `ca` and `thal` have them in the Cleveland file).
  2. Missing values are NOT filled here. They are imputed inside the sklearn
     pipeline (fit on training folds only) so there is no leakage and the
     deployed API can handle a missing field the same way.
  3. Target `num` (0-4 severity) -> binary `target` (0 = absent, 1 = present).
  4. Integer-coded columns cast to nullable Int64; exact duplicates dropped.
  5. Rows outside plausible physiological ranges are dropped.
"""
from __future__ import annotations

import pandas as pd

from heart import config


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/download_data.py` first")
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 3. binary target
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # 4. dtypes + duplicates
    int_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "slope", "ca", "thal"]
    for col in int_cols:
        df[col] = df[col].round().astype("Int64")
    df["oldpeak"] = df["oldpeak"].astype(float)
    df = df.drop_duplicates().reset_index(drop=True)

    # 5. range check (NaN passes - the pipeline imputer handles it)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in config.VALID_RANGES.items():
        mask &= df[col].isna() | df[col].between(lo, hi)
    dropped = int((~mask).sum())
    if dropped:
        print(f"[clean] dropped {dropped} out-of-range rows")
    return df[mask].reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.Series:
    miss = df.isna().sum()
    return miss[miss > 0]


def build_clean_dataset() -> pd.DataFrame:
    raw = load_raw()
    df = clean(raw)
    config.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    print(f"[clean] raw={raw.shape} -> clean={df.shape}; missing={missing_report(df).to_dict()}")
    print(f"[clean] target balance: {df[config.TARGET].value_counts().to_dict()}")
    print(f"[clean] wrote {config.CLEAN_FILE.relative_to(config.ROOT)}")
    return df


def load_clean() -> pd.DataFrame:
    return pd.read_csv(config.CLEAN_FILE, dtype={c: "Int64" for c in ["ca", "thal"]})


if __name__ == "__main__":
    build_clean_dataset()
