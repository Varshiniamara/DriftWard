"""
Plot helpers for the Streamlit dashboard — clean industrial style.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

INDUSTRIAL_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E8EDF4", family="IBM Plex Sans"),
)


def feature_distribution_plot(
    reference_df: pd.DataFrame,
    live_df: pd.DataFrame,
    feature: str,
) -> go.Figure:
    """Overlay reference vs live histogram for one sensor."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=reference_df[feature],
        name="Reference (training)",
        opacity=0.6,
        nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=live_df[feature],
        name="Live (production)",
        opacity=0.6,
        nbinsx=30,
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"{feature} — Reference vs Live",
        xaxis_title=feature,
        yaxis_title="Count",
        height=380,
        **INDUSTRIAL_LAYOUT,
    )
    return fig


def feature_importance_chart(imp_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        imp_df,
        x="importance_pct",
        y="feature",
        orientation="h",
        title="Model Explainability — Feature Importance",
        labels={"importance_pct": "Contribution %", "feature": ""},
        color="importance_pct",
        color_continuous_scale=["#1a2744", "#FF000F"],
    )
    fig.update_layout(height=360, showlegend=False, **INDUSTRIAL_LAYOUT)
    return fig


def mri_gauge_chart(mri: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mri,
        title={"text": "ML Reliability Index", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FF000F"},
            "steps": [
                {"range": [0, 50], "color": "#3d1f1f"},
                {"range": [50, 75], "color": "#3d3520"},
                {"range": [75, 100], "color": "#1a3d2e"},
            ],
            "threshold": {
                "line": {"color": "#52b788", "width": 4},
                "thickness": 0.8,
                "value": 75,
            },
        },
    ))
    fig.update_layout(height=280, **INDUSTRIAL_LAYOUT)
    return fig


def psi_bar_chart(feature_results: list[dict]) -> go.Figure:
    """Bar chart of PSI per feature."""
    df = pd.DataFrame(feature_results)
    colors = df["psi_severity"].map({
        "low": "#2ecc71",
        "medium": "#f39c12",
        "high": "#e74c3c",
    })
    fig = px.bar(
        df,
        x="feature",
        y="psi",
        title="Population Stability Index by Sensor",
        labels={"psi": "PSI", "feature": "Sensor"},
    )
    fig.update_traces(marker_color=colors)
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warning (0.1)")
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Alert (0.2)")
    fig.update_layout(height=400, **INDUSTRIAL_LAYOUT)
    return fig


def confusion_matrix_heatmap(cm: list, title: str) -> go.Figure:
    """2x2 confusion matrix for fault vs no-fault."""
    labels = ["No failure", "Failure"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=360,
        **INDUSTRIAL_LAYOUT,
    )
    return fig


def version_f1_table(version_before: str, version_after: str, old_f1: float, new_f1: float) -> pd.DataFrame:
    """Simple version vs F1 table for judges."""
    return pd.DataFrame([
        {"Model Version": version_before, "F1 Score": round(old_f1, 3), "Status": "Deployed"},
        {"Model Version": version_after, "F1 Score": round(new_f1, 3), "Status": "Candidate"},
    ])


def metrics_comparison_chart(old_m: dict, new_m: dict) -> go.Figure:
    """Grouped bar chart: old vs new model metrics."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig = go.Figure(data=[
        go.Bar(name="Current model", x=metrics, y=[old_m[m] for m in metrics]),
        go.Bar(name="Candidate model", x=metrics, y=[new_m[m] for m in metrics]),
    ])
    fig.update_layout(
        barmode="group",
        title="Model Performance — Before vs After Retrain",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        height=400,
        **INDUSTRIAL_LAYOUT,
    )
    return fig
