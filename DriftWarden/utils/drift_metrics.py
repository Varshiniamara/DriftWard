"""
Drift metrics: PSI, Kolmogorov-Smirnov, and ADWIN streaming detection.
Used by the drift_detector agent to quantify distribution shift.
"""

import numpy as np
import pandas as pd
from scipy import stats
from river import drift

# Sensor columns monitored for drift (real AI4I feature names)
FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# PSI thresholds (common industry rule-of-thumb)
PSI_STABLE = 0.1
PSI_WARNING = 0.2


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between reference and current distributions.
    Higher PSI => more drift. Typical: <0.1 stable, 0.1-0.2 moderate, >0.2 significant.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0

    # Shared bin edges from reference quantiles
    breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0) with small epsilon
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    stat, pval = stats.ks_2samp(ref, cur)
    return float(stat), float(pval)


def run_adwin_on_series(values: np.ndarray, delta: float = 0.002) -> dict:
    """
    ADWIN streaming drift detector over a numeric series.
    Returns whether drift was detected and approximate drift index.
    """
    detector = drift.ADWIN(delta=delta)
    drift_points = []
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        detector.update(float(v))
        if detector.drift_detected:
            drift_points.append(i)
            # Reset after detection to catch subsequent shifts
            detector = drift.ADWIN(delta=delta)

    return {
        "drift_detected": len(drift_points) > 0,
        "drift_indices": drift_points,
        "n_drifts": len(drift_points),
    }


def psi_severity(psi: float) -> str:
    if psi < PSI_STABLE:
        return "low"
    if psi < PSI_WARNING:
        return "medium"
    return "high"


def analyze_all_features(
    reference_df: pd.DataFrame,
    live_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> list[dict]:
    """Run PSI + KS on each feature; return structured per-feature results."""
    cols = feature_cols or FEATURE_COLS
    results = []
    for col in cols:
        if col not in reference_df.columns or col not in live_df.columns:
            continue
        ref = reference_df[col].values.astype(float)
        cur = live_df[col].values.astype(float)
        psi = compute_psi(ref, cur)
        ks_stat, ks_p = compute_ks(ref, cur)
        adwin = run_adwin_on_series(cur)
        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "psi_severity": psi_severity(psi),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 6),
            "ks_significant": ks_p < 0.05,
            "adwin_drift": adwin["drift_detected"],
            "adwin_n_drifts": adwin["n_drifts"],
        })
    return results


def overall_drift_score(feature_results: list[dict]) -> float:
    """Aggregate PSI across features (mean) for dashboard summary."""
    if not feature_results:
        return 0.0
    return float(np.mean([r["psi"] for r in feature_results]))
