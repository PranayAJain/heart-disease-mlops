"""API tests (FastAPI TestClient - no server or Docker needed)."""
import pytest
from fastapi.testclient import TestClient

from heart.predict import MODEL_FILE

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="run heart.package first")


@pytest.fixture(scope="module")
def client():
    from heart.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    patient = {k: v for k, v in sample_patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=patient).status_code == 200


def test_invalid_input_is_rejected(client, sample_patient):
    bad = {**sample_patient, "age": 250, "cp": 9}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    fields = {e["loc"][-1] for e in r.json()["detail"]}
    assert {"age", "cp"} <= fields


def test_missing_required_field_is_rejected(client, sample_patient):
    patient = {k: v for k, v in sample_patient.items() if k != "thalach"}
    assert client.post("/predict", json=patient).status_code == 422


def test_batch_predict(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient, sample_patient]})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_metrics_endpoint_exposes_custom_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "heart_predictions_total" in text
    assert "http_requests_total" in text
