"""
DriftWarden — Reactive ML monitoring prototype.
Every slider movement recalculates drift, metrics, strategy, and charts from real data.

Run: streamlit run app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from ui.theme import inject_theme
from ui.components import render_header, render_safe_banner, render_mri_row, render_pipeline_timeline
from utils.reactive_engine import compute_system_state
from agents.pipeline import run_autonomous_pipeline
from agents.retrainer import retrain
from agents.audit_logger import log_action, get_audit_log, clear_audit_log, format_audit_report
from utils.drift_metrics import FEATURE_COLS
from utils.governance import rank_drifted_features, CONFIDENCE_THRESHOLD
from utils.model_utils import MODEL_PATH
from utils.model_versioning import get_current_version
from utils.report_export import build_audit_report_text
from utils.visualizations import (
    feature_distribution_plot,
    psi_bar_chart,
    metrics_comparison_chart,
    confusion_matrix_heatmap,
    version_f1_table,
    mri_gauge_chart,
)

# fix import
from utils.model_utils import MODEL_PATH

DATA_DIR = ROOT / "data"
REF_PATH = DATA_DIR / "ai4i_reference.csv"
BASELINE_PATH = DATA_DIR / "ai4i_live_baseline.csv"
LIVE_PATH = DATA_DIR / "ai4i_live.csv"


@st.cache_data
def load_reference():
    return pd.read_csv(REF_PATH)


def load_live_baseline() -> pd.DataFrame:
    if BASELINE_PATH.exists():
        return pd.read_csv(BASELINE_PATH)
    if LIVE_PATH.exists():
        return pd.read_csv(LIVE_PATH)
    raise FileNotFoundError("Run: python scripts/setup_data.py")


def ensure_setup():
    if not MODEL_PATH.exists() or not REF_PATH.exists():
        st.error("Run: `pip install -r requirements.txt && python scripts/setup_data.py`")
        st.stop()


def init_session():
    defaults = {
        "retrain_result": None,
        "human_approved": False,
        "last_audit_hash": None,
        "pipeline_steps": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def maybe_log_drift_change(state: dict):
    """Append audit entry when drift sliders produce a new configuration."""
    h = state["params_hash"]
    if st.session_state.last_audit_hash != h:
        st.session_state.last_audit_hash = h
        log_action("drift_recalc", {
            "drift_params": state["drift_params"],
            "max_psi": state["drift_event"]["max_psi"],
            "aggregate_psi": state["drift_event"]["aggregate_psi"],
            "severity": state["drift_event"]["severity_label"],
            "live_f1": state["current_f1"],
            "reference_f1": state["reference_f1"],
            "f1_drop": state["f1_drop"],
            "strategy": state["strategy"]["strategy"],
            "confidence": state["strategy"]["confidence"],
        })


# ─── Bootstrap ───
st.set_page_config(page_title="DriftWarden", page_icon="⚙️", layout="wide")
inject_theme()
init_session()
ensure_setup()

ref_df = load_reference()
live_baseline = load_live_baseline()
model_version = get_current_version()

# ─── Sidebar: drift controls (reactive — no separate Apply button) ───
with st.sidebar:
    st.markdown("### Drift simulator")
    st.caption("Move sliders — all metrics recalculate instantly.")

    temp_shift = st.slider("Temperature +K", 0.0, 20.0, 0.0, 0.5, key="temp")
    torque_noise = st.slider("Torque noise σ", 0.0, 8.0, 0.0, 0.1, key="torque")
    tool_wear = st.slider("Tool wear +min", 0, 50, 0, 1, key="wear")

    st.divider()
    auto_approve = st.checkbox("Auto-approve retrain (confidence ≥ 0.70)", value=False)

    if st.button("▶ Run full pipeline", type="primary", use_container_width=True):
        with st.spinner("Pipeline running…"):
            pr = run_autonomous_pipeline(
                ref_df, live_baseline, temp_shift, torque_noise, tool_wear, auto_approve
            )
        st.session_state.pipeline_steps = pr["pipeline_steps"]
        st.session_state.retrain_result = pr.get("retrain_result")
        if not pr["human_required"]:
            st.session_state.human_approved = True
        st.rerun()

    st.download_button(
        "Export audit log",
        build_audit_report_text(),
        file_name="driftwarden_audit.txt",
        use_container_width=True,
    )

# ─── CORE: recompute everything from current sliders ───
state = compute_system_state(ref_df, live_baseline, temp_shift, torque_noise, tool_wear)
maybe_log_drift_change(state)

live_df = state["live_df"]
event = state["drift_event"]
analysis = state["analysis"]
strategy = state["strategy"]
health = state["health"]

# ─── Header + KPIs ───
render_header(model_version)
render_safe_banner()
render_mri_row(health)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Max PSI", f"{event['max_psi']:.3f}")
c2.metric("Mean PSI", f"{event['aggregate_psi']:.3f}")
c3.metric("Severity", event["severity_label"])
c4.metric("Live F1", f"{state['current_f1']:.3f}", f"Δ −{state['f1_drop']:.3f}" if state["f1_drop"] else None)
c5.metric("Confidence", f"{strategy['confidence']:.2f}")
c6.metric("Strategy", strategy["strategy"].replace("_", " "))

tabs = st.tabs(["Dashboard", "Drift", "Model & retrain", "Audit"])

# ═══ Dashboard ═══
with tabs[0]:
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(mri_gauge_chart(health["mri"]), use_container_width=True)
    with col_r:
        st.plotly_chart(psi_bar_chart(event["feature_metrics"]), use_container_width=True)

    sensor = st.selectbox("Sensor", FEATURE_COLS)
    st.plotly_chart(feature_distribution_plot(ref_df, live_df, sensor), use_container_width=True)

    if st.session_state.pipeline_steps:
        st.markdown("#### Last pipeline run")
        render_pipeline_timeline(st.session_state.pipeline_steps)

# ═══ Drift ═══
with tabs[1]:
    st.markdown("### Live drift analysis")
    st.caption("PSI · KS-test · ADWIN — recalculated from current slider values")

    ranked = rank_drifted_features(event["feature_metrics"])
    st.dataframe(pd.DataFrame([{
        "Sensor": r["label"], "PSI": r["psi"],
        "Level": r["psi_severity"].upper(),
        "KS alert": "YES" if r["ks_significant"] else "no",
    } for r in ranked]), use_container_width=True, hide_index=True)

    st.info(analysis.get("narrative", ""))
    for line in analysis.get("probable_causes", []):
        st.markdown(f"- {line}")

    st.progress(strategy["confidence"], text=f"Confidence {strategy['confidence']:.2f} (need {CONFIDENCE_THRESHOLD})")
    st.markdown(f"**Recommended action:** {analysis.get('recommended_action', '')}")
    st.markdown(f"**Auto strategy:** {strategy['strategy']} — {strategy['rationale']}")

    gov = strategy["governance"]
    if gov["action"] == "human_review_required":
        st.warning(gov["message"])
        if st.button("Engineer sign-off"):
            st.session_state.human_approved = True
            log_action("human_approval", {"strategy": gov["proposed_strategy"]})
            st.rerun()
    else:
        st.success(gov["message"])

# ═══ Model ═══
with tabs[2]:
    can_retrain = gov["action"] != "human_review_required" or st.session_state.human_approved
    chosen = st.selectbox("Strategy override", ["full_retrain", "fine_tune", "monitor_only"],
                          index=["full_retrain", "fine_tune", "monitor_only"].index(strategy["strategy"]))

    if st.button("Retrain + validate F1 gate", type="primary", disabled=not can_retrain):
        with st.spinner("Training (~2s)…"):
            st.session_state.retrain_result = retrain(ref_df, live_df, chosen)
        r = st.session_state.retrain_result
        log_action("retrain", {
            "status": r["status"], "strategy": r["strategy"],
            "old_f1": r["old_metrics"]["f1"], "new_f1": r["new_metrics"]["f1"] if r["new_metrics"] else None,
            "comparison": r.get("comparison"),
        })
        st.rerun()

    res = st.session_state.retrain_result
    if res and res.get("new_metrics"):
        vb, va = res["version_before"], res["version_after"]
        st.table(version_f1_table(vb, va, res["old_metrics"]["f1"], res["new_metrics"]["f1"]))
        if res["status"] == "promoted":
            st.success(res["comparison"]["reason"])
        else:
            st.error(res["comparison"]["reason"])
        st.plotly_chart(metrics_comparison_chart(res["old_metrics"], res["new_metrics"]), use_container_width=True)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(confusion_matrix_heatmap(res["old_metrics"]["confusion_matrix"], f"{vb} before"), use_container_width=True)
        with b:
            st.plotly_chart(confusion_matrix_heatmap(res["new_metrics"]["confusion_matrix"], f"{va} after"), use_container_width=True)
    elif res and res["status"] == "skipped":
        st.warning("Monitor-only — no retrain executed.")

# ═══ Audit ═══
with tabs[3]:
    if st.button("Clear audit log"):
        clear_audit_log()
        st.session_state.last_audit_hash = None
        st.rerun()
    for e in reversed(get_audit_log(80)[-15:]):
        st.code(e.get("report") or format_audit_report(e))
