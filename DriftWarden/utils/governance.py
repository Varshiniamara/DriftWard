"""
Governance: confidence, human escalation, drift ranking — all computed from live metrics.
"""

import math

FEATURE_LABELS = {
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}

CONFIDENCE_THRESHOLD = 0.70


def rank_drifted_features(feature_metrics: list[dict], top_n: int = 5) -> list[dict]:
    ranked = sorted(feature_metrics, key=lambda x: x["psi"], reverse=True)
    return [
        {
            "feature": r["feature"],
            "label": FEATURE_LABELS.get(r["feature"], r["feature"]),
            "psi": r["psi"],
            "psi_severity": r["psi_severity"],
            "ks_significant": r.get("ks_significant", False),
        }
        for r in ranked[:top_n]
    ]


def compute_confidence(
    drift_event: dict,
    analysis: dict | None = None,
    model_f1: float | None = None,
    f1_drop: float | None = None,
) -> float:
    """Confidence drops as PSI and F1 degradation increase."""
    max_psi = drift_event.get("max_psi", drift_event.get("aggregate_psi", 0))
    f1_drop = f1_drop if f1_drop is not None else drift_event.get("f1_drop", 0)
    model_f1 = model_f1 if model_f1 is not None else drift_event.get("live_f1", 0.5)

    psi_penalty = min(math.log1p(max_psi) / math.log1p(0.35), 1.0) * 0.35
    f1_penalty = min(f1_drop * 1.2, 0.35)
    severity_penalty = {"normal": 0.0, "warning": 0.12, "critical": 0.22}.get(
        drift_event.get("severity", "normal"), 0.08
    )

    confidence = 0.95 - psi_penalty - f1_penalty - severity_penalty + (model_f1 * 0.08)
    return round(max(0.10, min(1.0, confidence)), 3)


def resolve_action(strategy: str, confidence: float) -> dict:
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "action": "human_review_required",
            "autonomous_allowed": False,
            "message": (
                f"Confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}. "
                "Engineer approval required before retraining."
            ),
            "proposed_strategy": strategy,
        }
    return {
        "action": strategy,
        "autonomous_allowed": True,
        "message": f"Confidence {confidence:.2f} — autonomous execution allowed.",
        "proposed_strategy": strategy,
    }
