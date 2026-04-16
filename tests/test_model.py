"""Tests for model serialization and prediction round-trip."""

import joblib
import numpy as np
import pytest
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / "models" / "titer_model.pkl"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model file not available (run modeling notebook first)",
)
class TestModelRoundTrip:
    """Test that the serialized model loads and predicts correctly."""

    def test_model_loads(self):
        bundle = joblib.load(MODEL_PATH)
        assert "model" in bundle
        assert "feature_names" in bundle
        assert "log_target" in bundle

    def test_model_predicts(self):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        n_features = len(bundle["feature_names"])
        # Random input
        X = np.random.RandomState(42).randn(1, n_features)
        pred = model.predict(X)
        assert pred.shape == (1,)
        assert np.isfinite(pred[0])

    def test_model_prediction_deterministic(self):
        """Same input → same output."""
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        n_features = len(bundle["feature_names"])
        X = np.random.RandomState(42).randn(1, n_features)
        pred1 = model.predict(X)[0]
        pred2 = model.predict(X)[0]
        assert pred1 == pred2

    def test_log_target_inverse(self):
        """If log_target=True, exp(prediction) should be positive."""
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        n_features = len(bundle["feature_names"])
        X = np.random.RandomState(42).randn(1, n_features)
        pred = model.predict(X)[0]
        if bundle["log_target"]:
            titer = np.exp(pred)
            assert titer > 0

    def test_feature_names_match_model(self):
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        n_features = len(bundle["feature_names"])
        # Verify model accepts the right number of features
        X = np.random.RandomState(42).randn(1, n_features)
        pred = model.predict(X)
        assert pred.shape[0] == 1
