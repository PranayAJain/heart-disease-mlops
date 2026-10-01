"""Unit tests for data loading and cleaning (src/heart/data.py)."""
import pandas as pd
import pytest

from heart import config
from heart.data import clean, load_raw, missing_report


def test_target_is_binary(raw_df):
    out = clean(raw_df)
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns


def test_severity_levels_map_to_disease(raw_df):
    out = clean(raw_df)
    # rows with num 1-4 in the raw data must become target 1
    kept = raw_df.drop_duplicates()
    kept = kept[kept["trestbps"] <= config.VALID_RANGES["trestbps"][1]]
    assert out[config.TARGET].sum() == (kept["num"] > 0).sum()


def test_duplicates_and_out_of_range_rows_removed(raw_df):
    out = clean(raw_df)
    assert len(out) == len(raw_df) - 2          # 1 duplicate + 1 trestbps=999
    assert out["trestbps"].max() <= config.VALID_RANGES["trestbps"][1]
    assert not out.duplicated().any()


def test_missing_values_kept_for_pipeline_imputation(raw_df):
    out = clean(raw_df)
    miss = missing_report(out)
    assert miss.to_dict() == {"ca": 1, "thal": 1}


def test_columns_and_dtypes(raw_df):
    out = clean(raw_df)
    expected = [c for c in config.RAW_COLUMNS if c != "num"] + [config.TARGET]
    assert list(out.columns) == expected
    assert str(out["ca"].dtype) == "Int64"
    assert out["oldpeak"].dtype == float


def test_load_raw_parses_question_marks(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n")
    df = load_raw(f)
    assert df.shape == (1, 14)
    assert pd.isna(df.loc[0, "ca"])


def test_load_raw_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_raw(tmp_path / "nope.data")
