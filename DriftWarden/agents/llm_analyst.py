"""
LLM Analyst Agent — explains drift cause, severity, and recommended action.

Uses rule-based industrial reasoning by default (no API key required for demo).
Optional: set OPENAI_API_KEY env var for GPT-enhanced narratives.
"""

import os
from typing import Any


def _rule_based_analysis(drift_event: dict) -> dict:
    """
    Expert-system style analysis aligned with AI4I sensor semantics.
    Sounds industrial without requiring cloud LLM for the 4-day demo.
    """
    features = drift_event.get("feature_metrics", [])
    drifted = drift_event.get("drifted_features", [])
    severity = drift_event.get("severity", "normal")

    causes = []
    if any("temperature" in f.lower() for f in drifted):
        causes.append(
            "Environmental shift: air/process temperature distributions diverged "
            "from the training baseline — consistent with seasonal warming or "
            "changed cooling conditions on the shop floor."
        )
    if any("Torque" in f for f in drifted):
        causes.append(
            "Mechanical/sensor instability: torque readings show increased variance, "
            "possibly from bearing wear, load changes, or sensor calibration drift."
        )
    if any("Tool wear" in f for f in drifted):
        causes.append(
            "Equipment aging: tool wear values are elevated relative to reference, "
            "indicating accelerated component degradation."
        )
    if any("Rotational speed" in f for f in drifted):
        causes.append(
            "Operating regime change: rotational speed profile shifted — "
            "check for process setpoint changes or drive controller updates."
        )

    if not causes:
        causes.append(
            "No major per-feature drift detected; continue monitoring. "
            "Aggregate metrics remain within acceptable bounds."
        )

    if severity == "critical":
        action = (
            "Immediate review recommended. Initiate full retrain on recent labeled "
            "production window after validating sensor health with maintenance team."
        )
        rec_severity = "high"
    elif severity == "warning":
        action = (
            "Schedule retraining within the next maintenance window. "
            "Increase monitoring frequency on temperature and wear sensors."
        )
        rec_severity = "medium"
    else:
        action = "Continue autonomous monitoring. No retrain required at this time."
        rec_severity = "low"

    narrative = (
        "The model was originally trained under cooler, lower-wear operating conditions. "
        "As environmental and mechanical conditions evolved, the live feature distribution "
        "diverged from the training baseline."
        if any("temperature" in f.lower() or "Tool wear" in f for f in drifted)
        else "Production telemetry remains largely aligned with the reference baseline."
    )

    return {
        "probable_causes": causes,
        "severity": rec_severity,
        "recommended_action": action,
        "narrative": narrative,
        "analyst_mode": "rule_based",
    }


def _openai_analysis(drift_event: dict) -> dict | None:
    """Optional GPT layer if API key is configured."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are an industrial ML engineer at a motor predictive maintenance plant. "
            f"Analyze this drift detection JSON and respond in 3 short paragraphs: "
            f"cause, severity, recommended action.\n\n{drift_event}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        text = resp.choices[0].message.content
        return {
            "probable_causes": [text],
            "severity": drift_event.get("severity", "medium"),
            "recommended_action": "See narrative above.",
            "narrative": text,
            "analyst_mode": "openai",
        }
    except Exception:
        return None


def analyze_drift(drift_event: dict) -> dict:
    """Run analysis layer on a drift event."""
    enhanced = _openai_analysis(drift_event)
    if enhanced:
        return enhanced
    return _rule_based_analysis(drift_event)
