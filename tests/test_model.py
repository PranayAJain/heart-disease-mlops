"""Model-level tests: training pipeline, inference helper, packaged artefact, quality gate."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.data import clean, load_clean
from heart.features import split_xy
from heart.predict import MODEL_FILE, load_model, predict_records
from heart.train import MODELS, make_pipeline


def test_all_model_specs_build_valid_pipelines():
    for name, (estimator, grid) in MODELS.items():
        pipe = make_pipeline(estimator)
        assert all(k.startswith("clf__") for k in grid), name
        pipe.set_params(**{k: v[0] for k, v in grid.items()})


def test_pipeline_trains_and_outputs_probabilities(raw_df):
    X, y = split_xy(clean(raw_df))
    model = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_records_contract(raw_df, sample_patient):
    X, y = split_xy(clean(raw_df))
    model = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    incomplete = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    out = predict_records([sample_patient, incomplete], model=model, threshold=0.5)
    assert len(out) == 2
    for r in out:
        assert set(r) == {"prediction", "label", "probability_disease", "confidence"}
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0


needs_model = pytest.mark.skipif(not MODEL_FILE.exists(), reason="run heart.package first")
needs_data = pytest.mark.skipif(not config.CLEAN_FILE.exists(), reason="no cleaned dataset")


@needs_model
def test_packaged_model_predicts_high_risk_patient(sample_patient):
    out = predict_records([sample_patient], model=load_model())[0]
    assert out["prediction"] == 1


@needs_model
@needs_data
def test_quality_gate_auc_on_full_data():
    """Fail the pipeline if a retrained/packaged model is clearly worse than expected."""
    from sklearn.metrics import roc_auc_score
    X, y = split_xy(load_clean())
    auc = roc_auc_score(y, load_model().predict_proba(X)[:, 1])
    assert auc > 0.85, f"model AUC {auc:.3f} below quality gate"
    assert np.isfinite(auc)
