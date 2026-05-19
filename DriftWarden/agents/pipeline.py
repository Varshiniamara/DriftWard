"""
Full pipeline using reactive state — logs real measured values at each step.
"""

import time
import pandas as pd

from utils.reactive_engine import compute_system_state
from agents.retrainer import retrain
from agents.audit_logger import log_action


def run_autonomous_pipeline(
    reference_df: pd.DataFrame,
    live_baseline: pd.DataFrame,
    temp_shift: float,
    torque_noise: float,
    tool_wear: int,
    auto_approve: bool = False,
) -> dict:
    time.sleep(0.5)
    state = compute_system_state(reference_df, live_baseline, temp_shift, torque_noise, tool_wear)

    event = state["drift_event"]
    analysis = state["analysis"]
    strategy_result = state["strategy"]

    log_action("drift_detection", {
        "max_psi": event["max_psi"],
        "aggregate_psi": event["aggregate_psi"],
        "severity": event["severity_label"],
        "f1_drop": state["f1_drop"],
        "live_f1": state["current_f1"],
        "drift_params": state["drift_params"],
        "drifted_features": event["drifted_features"],
    })

    log_action("strategy_selection", strategy_result)

    steps = [
        {"step": 1, "name": "Drift Detection", "status": "complete",
         "detail": f"{event['severity_label']} max_psi={event['max_psi']:.3f}"},
        {"step": 2, "name": "Root Cause Analysis", "status": "complete",
         "detail": analysis.get("severity", "")},
        {"step": 3, "name": "Strategy", "status": "complete",
         "detail": strategy_result["strategy"]},
    ]

    gov = strategy_result["governance"]
    human_required = gov["action"] == "human_review_required"
    approved = auto_approve or not human_required

    retrain_result = None
    if approved and strategy_result["requires_retrain"]:
        retrain_result = retrain(reference_df, state["live_df"], strategy_result["strategy"])
        log_action("retrain", {
            "status": retrain_result["status"],
            "strategy": retrain_result["strategy"],
            "old_f1": retrain_result["old_metrics"]["f1"],
            "new_f1": retrain_result["new_metrics"]["f1"],
            "comparison": retrain_result["comparison"],
            "version_before": retrain_result["version_before"],
            "version_after": retrain_result["version_after"],
        })
        steps.append({
            "step": 4, "name": "Retrain + F1 Gate", "status": retrain_result["status"],
            "detail": retrain_result["comparison"]["reason"],
        })
    elif human_required:
        steps.append({"step": 4, "name": "Retrain", "status": "pending_approval",
                      "detail": "Awaiting sign-off"})
    else:
        steps.append({"step": 4, "name": "Retrain", "status": "skipped",
                      "detail": strategy_result["strategy"]})

    return {
        **state,
        "pipeline_steps": steps,
        "retrain_result": retrain_result,
        "human_required": human_required,
    }
