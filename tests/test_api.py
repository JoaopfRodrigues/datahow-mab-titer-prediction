"""Tests for the REST API (src/api.py)."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_payload():
    """The example payload from the OpenAPI spec."""
    return {
        "timestamps": list(range(15)),
        "values": {
            "Z:FeedStart": [3.0],
            "Z:FeedEnd": [11.0],
            "Z:FeedRateGlc": [5.473684211],
            "Z:FeedRateGln": [6.263157895],
            "Z:phStart": [7.473684211],
            "Z:phEnd": [6.289473684],
            "Z:phShift": [13.0],
            "Z:tempStart": [36.26315789],
            "Z:tempEnd": [36.94736842],
            "Z:tempShift": [10.0],
            "Z:Stir": [194.7368421],
            "Z:DO": [76.05263158],
            "Z:ExpDuration": [14.0],
            "W:temp": [36.26] * 10 + [36.95] * 5,
            "W:pH": [7.47] * 13 + [6.29] * 2,
            "W:FeedGlc": [0, 0, 0] + [5.47] * 8 + [0] * 4,
            "W:FeedGln": [0, 0, 0] + [6.26] * 8 + [0] * 4,
            "X:VCD": [
                0.55,
                1.53,
                3.39,
                5.45,
                8.95,
                14.74,
                19.67,
                24.45,
                26.98,
                30.07,
                32.33,
                34.61,
                31.91,
                29.50,
                27.40,
            ],
            "X:Glc": [
                5.64,
                5.92,
                5.00,
                2.87,
                7.46,
                9.78,
                12.17,
                13.79,
                15.06,
                15.88,
                16.28,
                16.72,
                15.34,
                12.70,
                9.41,
            ],
            "X:Gln": [
                5.55,
                4.59,
                2.21,
                0.0,
                1.63,
                1.30,
                0.0,
                0.06,
                0.17,
                0.08,
                0.14,
                0.14,
                0.11,
                0.02,
                0.0,
            ],
            "X:Amm": [
                0.1,
                0.16,
                0.65,
                0.52,
                1.06,
                1.75,
                2.14,
                2.46,
                2.77,
                3.14,
                3.42,
                3.63,
                3.78,
                3.72,
                3.57,
            ],
            "X:Lac": [
                0.1,
                0.45,
                1.56,
                2.77,
                4.36,
                5.94,
                6.97,
                5.49,
                4.92,
                5.38,
                5.64,
                4.75,
                4.26,
                3.60,
                3.58,
            ],
            "X:Lysed": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.01,
                0.003,
                0.02,
                0.02,
                0.04,
                0.06,
                0.10,
                0.17,
                0.25,
                0.39,
            ],
        },
    }


# --- Health endpoint ---


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Predict endpoint ---


def test_predict_returns_200(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_titer" in data
    assert isinstance(data["predicted_titer"], float)
    assert data["predicted_titer"] > 0
    assert "confidence_lower" in data
    assert "confidence_upper" in data


def test_predict_titer_in_plausible_range(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    data = response.json()
    titer = data["predicted_titer"]
    lower = data["confidence_lower"]
    upper = data["confidence_upper"]
    # Training titer range is ~283-4823; predictions should be in a reasonable range
    assert 0 < titer < 10000
    # Intervals should be ordered: lower <= titer <= upper
    assert lower <= titer <= upper


def test_predict_missing_variable_returns_400(client):
    """Spec says 400 for invalid payload."""
    payload = {
        "timestamps": [0, 1, 2],
        "values": {"Z:FeedStart": [3.0]},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_empty_timestamps_returns_400(client):
    payload = {"timestamps": [], "values": {}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_mismatched_lengths_returns_400(client, sample_payload):
    sample_payload["values"]["X:VCD"] = [1.0, 2.0]  # wrong length
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 400


def test_predict_spec_example_payload(client, sample_payload):
    """Run the spec-example payload end-to-end."""
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_titer"] > 0
    assert data["confidence_lower"] <= data["predicted_titer"] <= data["confidence_upper"]


def test_predict_negative_vcd(client, sample_payload):
    """Negative VCD is physically impossible but the API should still return a prediction."""
    sample_payload["values"]["X:VCD"] = [-1.0] * 15
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200


def test_predict_nan_in_values_returns_400(client, sample_payload):
    """NaN in sensor data should be rejected, not produce garbage predictions.

    NaN is not valid JSON per RFC 8259, but Python's json module accepts it
    by default. A permissive client could send it, so we validate server-side.
    """
    import json

    sample_payload["values"]["X:VCD"][0] = float("nan")
    body = json.dumps(sample_payload, allow_nan=True)
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400


def test_predict_nan_in_timestamps_returns_400(client, sample_payload):
    import json

    sample_payload["timestamps"][0] = float("nan")
    body = json.dumps(sample_payload, allow_nan=True)
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400


def test_predict_inf_in_values_returns_400(client, sample_payload):
    """Inf from sensor overflow should be rejected."""
    import json

    sample_payload["values"]["X:VCD"][0] = float("inf")
    body = json.dumps(sample_payload, allow_nan=True)
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400


# --- Model unavailable ---


def test_health_returns_503_when_model_missing(client):
    """Health check should report unhealthy when model is not loaded."""
    from src.api import app, get_model_bundle

    app.dependency_overrides[get_model_bundle] = lambda: None
    try:
        response = client.get("/health")
    finally:
        app.dependency_overrides.pop(get_model_bundle, None)
    assert response.status_code == 503


def test_predict_returns_503_when_model_missing(client, sample_payload):
    """Predict should fail gracefully when model is not loaded."""
    from src.api import app, get_model_bundle

    app.dependency_overrides[get_model_bundle] = lambda: None
    try:
        response = client.post("/predict", json=sample_payload)
    finally:
        app.dependency_overrides.pop(get_model_bundle, None)
    assert response.status_code == 503


def test_predict_returns_500_on_internal_error(client, sample_payload):
    """Internal errors during prediction should return 500, not leak details."""
    from unittest.mock import patch

    with patch("src.api.extract_features_from_payload", side_effect=RuntimeError("boom")):
        response = client.post("/predict", json=sample_payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal prediction error"


# --- Output validation ---


def test_predict_nonfinite_output_returns_500(client, sample_payload):
    """If the model produces non-finite output (e.g. exp overflow), return 500."""
    from unittest.mock import MagicMock, patch

    from src.api import app, get_model_bundle

    mock_bundle = {
        "model": MagicMock(predict=MagicMock(return_value=[1e308])),
        "feature_names": ["Z:ExpDuration"],
        "log_target": True,
        "bootstrap_models": [],
    }
    app.dependency_overrides[get_model_bundle] = lambda: mock_bundle
    try:
        with patch("src.api.extract_features_from_payload", return_value={"Z:ExpDuration": 14.0}):
            response = client.post("/predict", json=sample_payload)
    finally:
        app.dependency_overrides.pop(get_model_bundle, None)
    assert response.status_code == 500


# --- Timestamp validation ---


def test_predict_reversed_timestamps_returns_400(client, sample_payload):
    """Reversed timestamps should be rejected -- they produce wrong features."""
    sample_payload["timestamps"] = list(range(14, -1, -1))
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 400


def test_predict_duplicate_timestamps_returns_400(client, sample_payload):
    """Duplicate timestamps violate monotonicity."""
    sample_payload["timestamps"] = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 400


def test_predict_negative_timestamps_returns_400(client, sample_payload):
    """Negative timestamps are physically meaningless for bioprocess data."""
    sample_payload["timestamps"] = list(range(-5, 10))
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 400
    assert "negative" in response.json()["detail"].lower()


def test_predict_expduration_mismatch_returns_400(client, sample_payload):
    """Z:ExpDuration must match the actual timestamp span."""
    sample_payload["values"]["Z:ExpDuration"] = [7.0]  # but timestamps go to 14
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 400
    assert "ExpDuration" in response.json()["detail"]


# --- Error message safety ---


def test_400_error_does_not_leak_paths(client):
    """Validation errors should not expose server filesystem paths."""
    payload = {"timestamps": [0, 1, 2], "values": {"Z:FeedStart": [3.0]}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    body = response.text
    assert "/home/" not in body
    assert "src/" not in body
