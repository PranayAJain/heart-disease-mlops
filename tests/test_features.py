"""Unit tests for feature engineering + preprocessing (src/heart/features.py)."""
import numpy as np
import pandas as pd
import pytest

from heart import config
from heart.data import clean
from heart.features import add_features, build_preprocessor, split_xy


@pytest.fixture
def xy(raw_df):
    return split_xy(clean(raw_df))


def test_engineered_feature_formula(xy):
    X, _ = xy
    out = add_features(X)
    expected = X["thalach"].astype(float) / (220 - X["age"].astype(float))
    np.testing.assert_allclose(out["hr_pct_max"], expected)


def test_preprocessor_output_shape_and_no_nans(xy):
    X, _ = xy
    out = build_preprocessor().fit(X).transform(X)
    assert out.shape[0] == len(X)
    assert not np.isnan(out).any(), "imputer must remove all missing values"


def test_numeric_features_are_scaled(xy):
    X, _ = xy
    prep = build_preprocessor().fit(X)
    names = list(prep.get_feature_names_out())
    out = pd.DataFrame(prep.transform(X), columns=names)
    assert abs(out["age"].mean()) < 1e-6
    assert out["age"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_unseen_category_does_not_crash(xy):
    X, _ = xy
    prep = build_preprocessor().fit(X)
    new = X.head(1).copy()
    new["cp"] = 9                                # category never seen in training
    out = prep.transform(new)
    assert out.shape[1] == len(prep.get_feature_names_out())


def test_feature_names_include_onehot_and_engineered(xy):
    X, _ = xy
    names = build_preprocessor().fit(X).get_feature_names_out()
    assert "hr_pct_max" in names
    assert any(n.startswith("cp_") for n in names)
    assert set(config.BINARY) <= set(names)
