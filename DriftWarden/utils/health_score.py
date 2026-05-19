"""
ML Reliability Index (MRI) — decreases when PSI, F1 drop, or confidence worsen.
"""

import math


def normalize_psi(raw_psi: float) -> float:
    return min(math.log1p(max(raw_psi, 0)) / math.log1p(0.35), 1.0)


def compute_ml_reliability_index(
    drift_event: dict | None,
    model_f1: float,
    confidence: float | None = None,
    f1_drop: float | None = None,
) -> dict:
    if drift_event is None:
        max_psi, severity = 0.0, "unknown"
    else:
        max_psi = drift_event.get("max_psi", drift_event.get("aggregate_psi", 0))
        severity = drift_event.get("severity_label", drift_event.get("severity", "LOW"))

    drift_risk = normalize_psi(max_psi)
    f1_drop = f1_drop if f1_drop is not None else (drift_event or {}).get("f1_drop", 0)
    f1_risk = min(f1_drop * 2.0, 1.0)
    conf = confidence if confidence is not None else 0.75

    mri = 100 * (
        0.35 * (1 - drift_risk)
        + 0.30 * model_f1
        + 0.20 * (1 - f1_risk)
        + 0.15 * conf
    )
    mri = round(max(0, min(100, mri)), 1)

    if mri >= 75:
        status, status_class = "OPERATIONAL", "status-operational"
    elif mri >= 50:
        status, status_class = "DEGRADED", "status-degraded"
    else:
        status, status_class = "CRITICAL", "status-critical"

    return {
        "mri": mri,
        "status": status,
        "status_class": status_class,
        "drift_risk_pct": round(drift_risk * 100, 1),
        "model_health_pct": round(model_f1 * 100, 1),
        "f1_drop_pct": round(f1_drop * 100, 1),
        "governance_confidence": round(conf, 3),
        "severity": severity,
    }
