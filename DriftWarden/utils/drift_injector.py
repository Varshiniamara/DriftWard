"""
Apply realistic industrial drift to live production data.
Uses transformations on the real AI4I dataset — not synthetic mock data.
"""

import pandas as pd
import numpy as np

from utils.drift_metrics import FEATURE_COLS


def inject_seasonal_aging_drift(
    df: pd.DataFrame,
    temp_shift_k: float = 12.0,
    torque_noise_std: float = 2.5,
    tool_wear_increment: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Demo scenario: Seasonal Temperature Shift + Equipment Aging.

    - +12 K on air and process temperature (cooler training → warmer live)
    - Gaussian noise on torque (sensor instability)
    - Increased tool wear (mechanical aging)
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if "Air temperature [K]" in out.columns:
        out["Air temperature [K]"] = out["Air temperature [K]"] + temp_shift_k
    if "Process temperature [K]" in out.columns:
        out["Process temperature [K]"] = out["Process temperature [K]"] + temp_shift_k

    if "Torque [Nm]" in out.columns:
        noise = rng.normal(0, torque_noise_std, size=len(out))
        out["Torque [Nm]"] = out["Torque [Nm]"] + noise
        out["Torque [Nm]"] = out["Torque [Nm]"].clip(lower=3.0)  # physical lower bound

    if "Tool wear [min]" in out.columns:
        out["Tool wear [min]"] = (out["Tool wear [min]"] + tool_wear_increment).clip(upper=250)

    return out


def inject_custom_drift(
    df: pd.DataFrame,
    temp_shift_k: float = 0.0,
    torque_noise_std: float = 0.0,
    tool_wear_increment: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Flexible drift injection from dashboard sliders."""
    return inject_seasonal_aging_drift(
        df,
        temp_shift_k=temp_shift_k,
        torque_noise_std=torque_noise_std,
        tool_wear_increment=tool_wear_increment,
        seed=seed,
    )


def describe_injection(
    temp_shift_k: float,
    torque_noise_std: float,
    tool_wear_increment: int,
) -> str:
    """Human-readable summary for audit log."""
    parts = []
    if temp_shift_k:
        parts.append(f"+{temp_shift_k:.1f} K air/process temperature")
    if torque_noise_std:
        parts.append(f"torque noise σ={torque_noise_std:.1f} Nm")
    if tool_wear_increment:
        parts.append(f"+{tool_wear_increment} min tool wear")
    return "; ".join(parts) if parts else "no drift applied"
