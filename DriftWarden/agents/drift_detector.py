"""
Drift Detector — PSI, KS-test, ADWIN. Severity computed from actual metric values.
"""

from datetime import datetime, timezone
import pandas as pd

from utils.drift_metrics import analyze_all_features, overall_drift_score, FEATURE_COLS


def _compute_severity(feature_results: list[dict], f1_drop: float = 0.0) -> tuple[str, float]:
    """
    Dynamic severity from max PSI, KS signals, and model F1 degradation.
    Returns (severity_label, max_psi).
    """
    if not feature_results:
        return "LOW", 0.0

    max_psi = max(r["psi"] for r in feature_results)
    n_ks = sum(1 for r in feature_results if r.get("ks_significant"))
    n_adwin = sum(1 for r in feature_results if r.get("adwin_drift"))

    # Primary: PSI thresholds (industry standard)
    if max_psi > 0.25 or f1_drop > 0.20:
        level = "HIGH"
    elif max_psi > 0.10 or f1_drop > 0.08 or n_ks >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Escalate if many signals agree
    if n_adwin >= 3 and level == "MEDIUM":
        level = "HIGH"

    return level, max_psi


def detect_drift(
    reference_df: pd.DataFrame,
    live_df: pd.DataFrame,
    f1_drop: float = 0.0,
) -> dict:
    feature_results = analyze_all_features(reference_df, live_df, FEATURE_COLS)
    agg_psi = overall_drift_score(feature_results)
    severity, max_psi = _compute_severity(feature_results, f1_drop)

    drifted_features = [
        r["feature"]
        for r in feature_results
        if r["psi"] >= 0.10 or r.get("ks_significant") or r.get("adwin_drift")
    ]

    drift_detected = (
        severity != "LOW"
        or len(drifted_features) > 0
        or agg_psi >= 0.05
    )

    # Map to internal status keys used elsewhere
    severity_map = {"LOW": "normal", "MEDIUM": "warning", "HIGH": "critical"}

    return {
        "event_id": f"DRIFT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": drift_detected,
        "severity": severity_map[severity],
        "severity_label": severity,
        "max_psi": round(max_psi, 4),
        "aggregate_psi": round(agg_psi, 4),
        "drifted_features": drifted_features,
        "feature_metrics": feature_results,
        "reference_samples": len(reference_df),
        "live_samples": len(live_df),
    }
