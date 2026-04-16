"""Tests for feature extraction (src/features.py)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.features import (
    extract_experiment_features,
    extract_features_from_dataframe,
    extract_features_from_payload,
)


DATA_DIR = Path(__file__).parent.parent / "interview_files"


# --- Unit tests for core extraction ---


def _make_simple_experiment(duration=7):
    """Create a minimal synthetic experiment for testing."""
    n = duration + 1
    time = np.arange(n, dtype=float)
    z_params = {
        "Z:FeedStart": 3.0,
        "Z:FeedEnd": 6.0,
        "Z:FeedRateGlc": 5.0,
        "Z:FeedRateGln": 4.0,
        "Z:phStart": 7.2,
        "Z:phEnd": 6.8,
        "Z:phShift": 10.0,
        "Z:tempStart": 37.0,
        "Z:tempEnd": 33.0,
        "Z:tempShift": 8.0,
        "Z:Stir": 200.0,
        "Z:DO": 60.0,
        "Z:ExpDuration": float(duration),
    }
    x_series = {
        "X:VCD": np.linspace(0.5, 20.0, n),
        "X:Glc": np.linspace(5.0, 15.0, n),
        "X:Gln": np.linspace(5.0, 0.5, n),
        "X:Amm": np.linspace(0.1, 3.0, n),
        "X:Lac": np.concatenate(
            [np.linspace(0.1, 5.0, n // 2 + 1), np.linspace(4.5, 3.0, n - n // 2 - 1)]
        ),
        "X:Lysed": np.linspace(0.0, 0.5, n),
    }
    w_series = {
        "W:temp": np.full(n, 37.0),
        "W:pH": np.full(n, 7.2),
        "W:FeedGlc": np.where((time >= 3) & (time < 6), 5.0, 0.0),
        "W:FeedGln": np.where((time >= 3) & (time < 6), 4.0, 0.0),
    }
    return time, z_params, x_series, w_series


def test_extract_returns_dict():
    time, z, x, w = _make_simple_experiment()
    features = extract_experiment_features(time, z, x, w)
    assert isinstance(features, dict)
    assert len(features) == 44  # 38 original + 6 common-window/structural


def test_no_nan_or_inf():
    time, z, x, w = _make_simple_experiment()
    features = extract_experiment_features(time, z, x, w)
    for key, val in features.items():
        assert np.isfinite(val), f"Feature {key} is not finite: {val}"


def test_z_params_pass_through():
    time, z, x, w = _make_simple_experiment()
    features = extract_experiment_features(time, z, x, w)
    for key, val in z.items():
        assert features[key] == val, f"Z: param {key} mismatch"


def test_ivcd_positive():
    time, z, x, w = _make_simple_experiment()
    features = extract_experiment_features(time, z, x, w)
    assert features["IVCD"] > 0
    assert features["IVCD_0to7"] > 0


def test_common_window_equals_full_for_7d():
    """For a 7-day experiment, common-window features should match full-duration."""
    time, z, x, w = _make_simple_experiment(duration=7)
    features = extract_experiment_features(time, z, x, w)
    assert abs(features["IVCD"] - features["IVCD_0to7"]) < 1e-6
    assert abs(features["VCD_end"] - features["VCD_at_d7"]) < 1e-6


def test_common_window_less_than_full_for_14d():
    """For a 14-day experiment, common-window IVCD should be less than full."""
    time, z, x, w = _make_simple_experiment(duration=14)
    features = extract_experiment_features(time, z, x, w)
    assert features["IVCD_0to7"] < features["IVCD"]


def test_structural_features_7d():
    """For a 7-day experiment, fraction=1, acceleration=0, shape=VCD_d7/VCD_max."""
    time, z, x, w = _make_simple_experiment(duration=7)
    features = extract_experiment_features(time, z, x, w)
    assert abs(features["fraction_ivcd_0to7"] - 1.0) < 1e-6
    assert abs(features["growth_acceleration"]) < 1e-6
    assert features["vcd_trajectory_shape"] > 0


def test_structural_features_14d():
    """For a 14-day experiment, fraction<1, acceleration nonzero."""
    time, z, x, w = _make_simple_experiment(duration=14)
    features = extract_experiment_features(time, z, x, w)
    assert 0 < features["fraction_ivcd_0to7"] < 1.0
    assert features["growth_acceleration"] != 0  # growth rate changes
    assert 0 < features["vcd_trajectory_shape"] <= 1.0


# --- Integration test with real data ---


@pytest.mark.skipif(
    not (DATA_DIR / "datahow_interview_train_data.csv").exists(),
    reason="Training data not available",
)
def test_extract_from_dataframe():
    data = pd.read_csv(DATA_DIR / "datahow_interview_train_data.csv")
    result = extract_features_from_dataframe(data)
    assert result.shape == (100, 44)
    assert result.isna().sum().sum() == 0
    assert np.isfinite(result.values).all()


# --- Payload extraction (API interface) ---


def test_extract_from_payload():
    time, z, x, w = _make_simple_experiment(duration=14)
    timestamps = time.tolist()
    values = {}
    for k, v in z.items():
        values[k] = [v]
    for k, v in {**x, **w}.items():
        values[k] = v.tolist()

    features = extract_features_from_payload(timestamps, values)
    assert isinstance(features, dict)
    assert len(features) == 44
    for val in features.values():
        assert np.isfinite(val)
