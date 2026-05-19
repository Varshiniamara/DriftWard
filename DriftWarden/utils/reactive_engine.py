"""
Reactive engine — recomputes ALL metrics from reference + baseline + drift sliders.
"""

import hashlib
import pandas as pd

from utils.drift_injector import inject_custom_drift, describe_injection
from agents.drift_detector import detect_drift
from agents.llm_analyst import analyze_drift
from agents.strategy_selector import select_strategy
from utils.model_utils import load_model, evaluate_model
from utils.health_score import compute_ml_reliability_index


def _drift_seed(temp_shift: float, torque_noise: float, tool_wear: int) -> int:
    key = f"{temp_shift:.2f}_{torque_noise:.2f}_{tool_wear}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def build_live_df(
    live_baseline: pd.DataFrame,
    temp_shift: float,
    torque_noise: float,
    tool_wear: int,
) -> pd.DataFrame:
    if temp_shift == 0 and torque_noise == 0 and tool_wear == 0:
        return live_baseline.copy()
    return inject_custom_drift(
        live_baseline,
        temp_shift_k=temp_shift,
        torque_noise_std=torque_noise,
        tool_wear_increment=tool_wear,
        seed=_drift_seed(temp_shift, torque_noise, tool_wear),
    )


def compute_system_state(
    reference_df: pd.DataFrame,
    live_baseline: pd.DataFrame,
    temp_shift: float,
    torque_noise: float,
    tool_wear: int,
) -> dict:
    live_df = build_live_df(live_baseline, temp_shift, torque_noise, tool_wear)

    model = load_model()
    baseline_metrics = evaluate_model(model, live_baseline)
    live_metrics = evaluate_model(model, live_df)

    reference_f1 = baseline_metrics["f1"]
    current_f1 = live_metrics["f1"]
    f1_drop = round(max(0.0, reference_f1 - current_f1), 4)

    drift_event = detect_drift(reference_df, live_df, f1_drop=f1_drop)
    drift_event["reference_f1"] = reference_f1
    drift_event["live_f1"] = current_f1
    drift_event["f1_drop"] = f1_drop

    analysis = analyze_drift(drift_event)
    strategy = select_strategy(
        drift_event, analysis,
        current_f1=current_f1,
        reference_f1=reference_f1,
    )

    health = compute_ml_reliability_index(
        drift_event, current_f1,
        confidence=strategy.get("confidence"),
        f1_drop=f1_drop,
    )

    return {
        "live_df": live_df,
        "drift_event": drift_event,
        "analysis": analysis,
        "strategy": strategy,
        "live_metrics": live_metrics,
        "baseline_metrics": baseline_metrics,
        "reference_f1": reference_f1,
        "current_f1": current_f1,
        "f1_drop": f1_drop,
        "health": health,
        "drift_params": {
            "temp_shift_k": temp_shift,
            "torque_noise_std": torque_noise,
            "tool_wear_increment": tool_wear,
            "description": describe_injection(temp_shift, torque_noise, tool_wear),
        },
        "params_hash": _drift_seed(temp_shift, torque_noise, tool_wear),
    }
