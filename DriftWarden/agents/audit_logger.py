"""
Audit Logger Agent — append-only governance log for every autonomous action.
Formats entries as industrial event reports for demo impact.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_log.json"


def _load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_log(entries: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def log_action(
    action_type: str,
    details: dict[str, Any],
    actor: str = "DriftWarden",
) -> dict:
    """Append one audit record with human-readable report."""
    entries = _load_log()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action_type": action_type,
        "actor": actor,
        "details": details,
        "report": format_audit_report({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": action_type,
            "details": details,
        }),
    }
    entries.append(record)
    _save_log(entries)
    return record


def format_audit_report(entry: dict) -> str:
    """Render one log entry as an industrial event report."""
    ts = entry.get("timestamp", "")[:19].replace("T", " ")
    action = entry.get("action_type", "unknown")
    d = entry.get("details", {})
    lines = [f"[{ts}] {action.upper().replace('_', ' ')}", ""]

    if action == "drift_detection":
        ev = d.get("event", d)
        analysis = d.get("analysis", {})
        sev = ev.get("severity", "—").upper()
        lines.append(f"SEVERITY: {sev} DRIFT DETECTED" if ev.get("drift_detected") else "STATUS: STABLE")
        lines.append(f"Aggregate PSI: {ev.get('aggregate_psi', '—')}")
        for feat in ev.get("drifted_features", [])[:3]:
            lines.append(f"Feature: {feat}")
        lines.append("")
        lines.append("LLM Analysis:")
        for cause in analysis.get("probable_causes", [])[:2]:
            lines.append(f"  • {cause[:120]}{'…' if len(cause) > 120 else ''}")
        if analysis.get("narrative"):
            lines.append(f"  {analysis['narrative'][:160]}…" if len(analysis.get("narrative", "")) > 160 else f"  {analysis.get('narrative', '')}")

    elif action == "strategy_selection":
        gov = d.get("governance", {})
        lines.append(f"Proposed strategy: {d.get('strategy', '—').upper().replace('_', ' ')}")
        lines.append(f"Confidence: {d.get('confidence', '—')} (threshold {d.get('confidence_threshold', 0.7)})")
        lines.append(f"Action: {gov.get('action', '—').upper().replace('_', ' ')}")
        lines.append("")
        lines.append("Top drifted features:")
        for f in d.get("top_drifted_features", [])[:5]:
            lines.append(f"  {f.get('label', f.get('feature')):22s}  PSI = {f.get('psi', 0):.4f}")
        lines.append("")
        lines.append(f"Rationale: {d.get('rationale', '')[:200]}")

    elif action == "retrain":
        lines.append(f"Strategy: {d.get('strategy', '—').upper().replace('_', ' ')}")
        lines.append("Action:")
        lines.append(f"  {d.get('strategy', 'retrain').upper()} initiated")
        comp = d.get("comparison") or {}
        old_m = d.get("old_metrics_slim") or d.get("old_metrics") or {}
        new_m = d.get("new_metrics_slim") or d.get("new_metrics") or {}
        if old_m and new_m:
            lines.append("")
            lines.append("Validation:")
            lines.append(
                f"  F1 improved {old_m.get('f1', 0):.2f} → {new_m.get('f1', 0):.2f} "
                f"(Δ {comp.get('f1_delta', 0):+.3f})"
            )
            lines.append(
                f"  {d.get('version_before', '?')} → {d.get('version_after', '?')}"
            )
        lines.append("")
        lines.append("Deployment:")
        if d.get("status") == "promoted":
            lines.append("  ✓ NEW MODEL PROMOTED")
        elif d.get("status") == "rejected":
            lines.append("  ✗ CANDIDATE REJECTED (accuracy gate)")
        else:
            lines.append(f"  — {d.get('status', 'skipped').upper()}")

    elif action == "drift_injection":
        lines.append(f"Simulation: {d.get('description', 'manual drift')}")

    elif action == "drift_recalc":
        lines.append(f"Params: {d.get('drift_params', {}).get('description', '')}")
        lines.append(f"Max PSI: {d.get('max_psi')}  Mean PSI: {d.get('aggregate_psi')}")
        lines.append(f"Severity: {d.get('severity')}")
        lines.append(f"F1: {d.get('reference_f1')} → {d.get('live_f1')} (drop {d.get('f1_drop')})")
        lines.append(f"Strategy: {d.get('strategy')}  Confidence: {d.get('confidence')}")

    else:
        lines.append(json.dumps(d, indent=2, default=str)[:500])

    return "\n".join(lines)


def get_audit_log(limit: int | None = 50) -> list[dict]:
    entries = _load_log()
    if limit:
        return entries[-limit:]
    return entries


def clear_audit_log() -> None:
    _save_log([])
