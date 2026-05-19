"""Reusable enterprise UI components."""

import streamlit as st


def render_header(model_version: str):
    st.markdown(
        f"""
        <motion.div class="dw-header">
            <span class="dw-badge">INDUSTRIAL ML RELIABILITY</span>
            <h1>DriftWarden Command Center</h1>
            <p class="tagline">Autonomous reliability layer · AI4I Predictive Maintenance · Model <b>{model_version}</b></p>
        </motion.div>
        """.replace("motion.", ""),
        unsafe_allow_html=True,
    )


def render_safe_banner():
    st.markdown(
        """
        <motion.div class="dw-safe-banner">
        <strong>⬢ SAFE AUTONOMOUS MODE ACTIVE</strong><br/>
        Retraining only after: ✓ Drift confirmation · ✓ Confidence &gt;0.70 · ✓ F1 gate · ✓ Audit log
        </motion.div>
        """.replace("motion.", ""),
        unsafe_allow_html=True,
    )


def render_mri_row(health: dict):
    def card(label, value, sub, vclass=""):
        return (
            f'<motion.div class="dw-metric-card">'
            f'<motion.div class="label">{label}</motion.div>'
            f'<motion.div class="value {vclass}">{value}</motion.div>'
            f'<motion.div class="delta">{sub}</motion.div>'
            f"</motion.div>"
        ).replace("motion.", "")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(card("ML Reliability Index", str(health["mri"]), health["status"], health.get("status_class", "")), unsafe_allow_html=True)
    with c2:
        st.markdown(card("Drift Risk", f'{health["drift_risk_pct"]}%', "distribution shift"), unsafe_allow_html=True)
    with c3:
        st.markdown(card("F1 degradation", f'{health.get("f1_drop_pct", 0)}%', "vs clean live"), unsafe_allow_html=True)
    with c4:
        st.markdown(card("Confidence", str(health["governance_confidence"]), "threshold 0.70"), unsafe_allow_html=True)


def render_compliance_checklist(strategy, retrain_result, human_approved: bool):
    gov_action = (strategy or {}).get("governance", {}).get("action", "")
    checks = [
        ("Multi-signal drift (PSI + KS + ADWIN)", strategy is not None),
        ("Root cause analysis", strategy is not None),
        ("Confidence gate", "confidence" in (strategy or {})),
        ("Human approval (if required)", gov_action != "human_review_required" or human_approved),
        ("F1 gate — model promoted", (retrain_result or {}).get("status") == "promoted"),
        ("Audit trail persisted", True),
    ]
    st.markdown("#### Governance & compliance")
    for label, ok in checks:
        st.markdown(f"{'✅' if ok else '⬜'} {label}")


def render_pipeline_timeline(steps: list):
    icons = {"complete": "✓", "promoted": "✓", "rejected": "✗", "pending_approval": "◷", "skipped": "—"}
    for s in steps:
        icon = icons.get(s.get("status"), "●")
        html = (
            f'<motion.div class="pipeline-step">'
            f'<strong>{icon} Step {s["step"]}: {s["name"]}</strong> — {s.get("detail", "")}'
            f"</motion.div>"
        )
        st.markdown(html.replace("motion.", ""), unsafe_allow_html=True)
