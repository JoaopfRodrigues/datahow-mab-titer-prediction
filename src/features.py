"""Feature engineering for mAb bioprocess titer prediction.

Extracts per-experiment features from time-series data (X: observations,
W: inputs) and scalar DoE parameters (Z:).
"""

import numpy as np
import pandas as pd


# Column group definitions matching the challenge data schema

Z_COLS = [
    "Z:FeedStart",
    "Z:FeedEnd",
    "Z:FeedRateGlc",
    "Z:FeedRateGln",
    "Z:phStart",
    "Z:phEnd",
    "Z:phShift",
    "Z:tempStart",
    "Z:tempEnd",
    "Z:tempShift",
    "Z:Stir",
    "Z:DO",
    "Z:ExpDuration",
]
W_COLS = ["W:temp", "W:pH", "W:FeedGlc", "W:FeedGln"]
X_COLS = ["X:VCD", "X:Glc", "X:Gln", "X:Amm", "X:Lac", "X:Lysed"]


# --- Core feature extraction ---


def extract_experiment_features(
    time: np.ndarray,
    z_params: dict[str, float],
    x_series: dict[str, np.ndarray],
    w_series: dict[str, np.ndarray],
) -> dict[str, float]:
    """Extract features from a single experiment.

    Parameters
    ----------
    time : array of shape (n_timepoints,)
        Timestamps in days.
    z_params : dict
        Scalar DoE parameters (Z: prefix). One value per parameter.
    x_series : dict
        Time-series observations (X: prefix). Arrays of length n_timepoints.
    w_series : dict
        Time-series inputs (W: prefix). Arrays of length n_timepoints.

    Returns
    -------
    dict
        Feature name -> value mapping.
    """
    features = {}
    duration = z_params["Z:ExpDuration"]
    vcd = x_series["X:VCD"]
    dt = np.diff(time)

    # Group 1: Z: scalar DoE parameters (pass through)
    for col in Z_COLS:
        features[col] = z_params[col]

    # Group 2: VCD dynamics
    # IVCD, VCD_end, VCD_max are the strongest independent predictors
    features["IVCD"] = float(np.trapezoid(vcd, time))
    features["VCD_max"] = float(np.max(vcd))
    features["VCD_end"] = float(vcd[-1])
    features["VCD_mean"] = float(np.mean(vcd))
    features["time_to_peak_VCD"] = float(time[np.argmax(vcd)])

    # Specific growth rate: mu = d(ln(VCD))/dt
    vcd_safe = np.maximum(vcd, 1e-6)
    log_vcd = np.log(vcd_safe)
    mu = np.diff(log_vcd) / np.maximum(dt, 1e-6)
    features["mu_max"] = float(np.max(mu)) if len(mu) > 0 else 0.0
    early_mask = time[:-1] < 3
    features["mu_early"] = float(np.mean(mu[early_mask])) if np.any(early_mask) else 0.0

    # Group 3: Metabolite endpoints
    features["Glc_end"] = float(x_series["X:Glc"][-1])
    features["Gln_end"] = float(x_series["X:Gln"][-1])
    features["Lac_end"] = float(x_series["X:Lac"][-1])
    features["Lac_max"] = float(np.max(x_series["X:Lac"]))
    features["Amm_end"] = float(x_series["X:Amm"][-1])
    features["Lysed_end"] = float(x_series["X:Lysed"][-1])
    features["Lysed_max"] = float(np.max(x_series["X:Lysed"]))

    # Group 4: Integrated time-series
    # Glutamine integral — important after duration correction
    features["int_Gln"] = float(np.trapezoid(x_series["X:Gln"], time))
    features["int_Lac"] = float(np.trapezoid(x_series["X:Lac"], time))
    features["int_Amm"] = float(np.trapezoid(x_series["X:Amm"], time))
    # Cumulative feed: equivalent to FeedRate * max(0, min(FeedEnd, duration) - FeedStart)
    # since W:Feed is a deterministic step function from Z: (EDA Finding 4).
    # We integrate W: directly for consistency with other time-series features.
    features["cum_FeedGlc"] = float(np.trapezoid(w_series["W:FeedGlc"], time))
    features["cum_FeedGln"] = float(np.trapezoid(w_series["W:FeedGln"], time))

    # Group 5: Metabolic efficiency indicators
    # Specific rates: overall metabolite change per unit cell activity
    ivcd = features["IVCD"]
    if ivcd > 0:
        features["qGlc"] = (x_series["X:Glc"][-1] - x_series["X:Glc"][0]) / ivcd
        features["qLac"] = (x_series["X:Lac"][-1] - x_series["X:Lac"][0]) / ivcd
    else:
        features["qGlc"] = 0.0
        features["qLac"] = 0.0

    # Lactate metabolic switch: does lactate decline after peak?
    lac = x_series["X:Lac"]
    lac_peak_idx = int(np.argmax(lac))
    features["lac_decline"] = (
        float(lac[lac_peak_idx] - lac[-1]) if lac_peak_idx < len(lac) - 1 else 0.0
    )

    # Glutamine accumulation ratio — linked to low VCD
    gln_start = max(x_series["X:Gln"][0], 1e-6)
    features["gln_accumulation_ratio"] = float(x_series["X:Gln"][-1] / gln_start)

    # Group 6: Phase/shift features
    features["days_post_temp_shift"] = max(0.0, duration - z_params["Z:tempShift"])
    features["days_post_ph_shift"] = max(0.0, duration - z_params["Z:phShift"])

    # Group 7: Common-window features (days 0-7)
    # All experiments share the first 7 days. Features from this window
    # address the train/test covariate shift on experiment duration.
    WINDOW_DAY = 7.0
    w_mask = time <= WINDOW_DAY
    time_w = time[w_mask]
    vcd_w = vcd[w_mask]

    if len(time_w) >= 2:
        features["VCD_at_d7"] = float(vcd_w[-1])
        features["IVCD_0to7"] = float(np.trapezoid(vcd_w, time_w))

        vcd_w_safe = np.maximum(vcd_w, 1e-6)
        mu_w = np.diff(np.log(vcd_w_safe)) / np.maximum(np.diff(time_w), 1e-6)
        features["mu_0to7"] = float(np.mean(mu_w))

        # Structural features comparing common window to full experiment.
        # fraction_ivcd_0to7 is correlated with duration (1.0 for 7d, ~0.5 for 14d),
        # but retains *within-duration* variance: among the 10 fourteen-day experiments
        # it ranges ~0.35-0.55 depending on whether biomass growth is front-loaded
        # (early high VCD, then plateau/decline) vs sustained. That signal is not
        # recoverable from Z:ExpDuration alone and is what earns the feature its
        # non-zero ElasticNet coefficient at LOO-14d evaluation.
        features["fraction_ivcd_0to7"] = (
            features["IVCD_0to7"] / features["IVCD"] if features["IVCD"] > 0 else 1.0
        )
        post_mask = time[:-1] >= WINDOW_DAY
        mu_post = float(np.mean(mu[post_mask])) if np.any(post_mask) else features["mu_0to7"]
        features["growth_acceleration"] = mu_post - features["mu_0to7"]
        features["vcd_trajectory_shape"] = (
            float(vcd_w[-1]) / features["VCD_max"] if features["VCD_max"] > 0 else 0.0
        )
    else:
        for key in [
            "VCD_at_d7",
            "IVCD_0to7",
            "mu_0to7",
            "fraction_ivcd_0to7",
            "growth_acceleration",
            "vcd_trajectory_shape",
        ]:
            features[key] = 0.0

    return features


# --- Batch extraction from DataFrames ---


def extract_features_from_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Extract features from all experiments in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Raw experiment data with columns: Exp, Time[day], Z:*, W:*, X:*

    Returns
    -------
    pd.DataFrame
        One row per experiment, indexed by experiment identifier.
    """
    records = []
    for exp_id, exp_df in data.groupby("Exp"):
        exp_df = exp_df.sort_values("Time[day]")
        time = exp_df["Time[day]"].values
        z_params = {col: float(exp_df[col].iloc[0]) for col in Z_COLS}
        x_series = {col: exp_df[col].values for col in X_COLS}
        w_series = {col: exp_df[col].values for col in W_COLS}

        feat = extract_experiment_features(time, z_params, x_series, w_series)
        feat["Exp"] = exp_id
        records.append(feat)

    result = pd.DataFrame(records).set_index("Exp")
    result = result.loc[sorted(result.index, key=_exp_sort_key)]
    return result


# --- Extraction from API payload ---


def extract_features_from_payload(
    timestamps: list[float],
    values: dict[str, list[float]],
) -> dict[str, float]:
    """Extract features from an API request payload.

    Parameters
    ----------
    timestamps : list of float
        Time points for the experiment.
    values : dict
        Variable name -> array mapping. Z: keys have single-element arrays,
        W: and X: keys have arrays matching timestamps length.

    Returns
    -------
    dict
        Feature name -> value mapping.
    """
    time = np.array(timestamps, dtype=float)
    z_params = {k: v[0] for k, v in values.items() if k.startswith("Z:")}
    x_series = {k: np.array(v, dtype=float) for k, v in values.items() if k.startswith("X:")}
    w_series = {k: np.array(v, dtype=float) for k, v in values.items() if k.startswith("W:")}
    return extract_experiment_features(time, z_params, x_series, w_series)


# --- Utilities ---


def _exp_sort_key(exp_name: str) -> tuple[str, int]:
    """Sort key for experiment names like 'Exp 1', 'Test Exp 20'."""
    parts = exp_name.rsplit(" ", 1)
    try:
        return (parts[0], int(parts[1]))
    except (ValueError, IndexError):
        return (exp_name, 0)
