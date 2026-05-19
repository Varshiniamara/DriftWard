"""
Strategy selection driven by measured PSI, F1 drop, and confidence — no hardcoded outputs.
"""

from utils.governance import (
    compute_confidence,
    resolve_action,
    rank_drifted_features,
    CONFIDENCE_THRESHOLD,
)


def select_strategy(
    drift_event: dict,
    analysis: dict,
    current_f1: float | None = None,
    reference_f1: float | None = None,
) -> dict:
    max_psi = drift_event.get("max_psi", 0)
    agg_psi = drift_event.get("aggregate_psi", 0)
    f1_drop = drift_event.get("f1_drop", 0)
    if reference_f1 is not None and current_f1 is not None:
        f1_drop = max(0.0, reference_f1 - current_f1)

    severity_label = drift_event.get("severity_label", "LOW")

    # Dynamic rules from actual measurements
    if max_psi < 0.10 and f1_drop < 0.05:
        strategy = "monitor_only"
        rationale = (
            f"Stable: max PSI={max_psi:.3f}, F1 drop={f1_drop:.3f}. "
            "Continue monitoring without model change."
        )
    elif max_psi < 0.25 and f1_drop < 0.15:
        strategy = "fine_tune"
        rationale = (
            f"Moderate drift: max PSI={max_psi:.3f}, F1 drop={f1_drop:.3f}. "
            "Incremental retrain on recent production labels."
        )
    else:
        strategy = "full_retrain"
        rationale = (
            f"Significant shift: max PSI={max_psi:.3f}, F1 drop={f1_drop:.3f}, "
            f"severity={severity_label}. Full retrain required."
        )

    # If model still excellent despite mild PSI, prefer monitor
    if strategy == "fine_tune" and current_f1 is not None and current_f1 >= 0.92 and f1_drop < 0.03:
        strategy = "monitor_only"
        rationale += " (downgraded: live F1 still strong.)"

    confidence = compute_confidence(drift_event, analysis, current_f1, f1_drop)
    governance = resolve_action(strategy, confidence)

    return {
        "strategy": strategy,
        "rationale": rationale,
        "requires_retrain": strategy in ("fine_tune", "full_retrain"),
        "confidence": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "governance": governance,
        "top_drifted_features": rank_drifted_features(drift_event.get("feature_metrics", [])),
        "inputs": {"max_psi": max_psi, "agg_psi": agg_psi, "f1_drop": f1_drop, "live_f1": current_f1},
    }
