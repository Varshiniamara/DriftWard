"""Enterprise industrial theme for DriftWarden command center."""

ENTERPRISE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 1.2rem;
    max-width: 1400px;
}

.dw-header {
    background: linear-gradient(135deg, #0B1120 0%, #1a2744 50%, #0d1f3c 100%);
    border: 1px solid #2a3f5f;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.dw-header h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}

.dw-header .tagline {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

.dw-badge {
    display: inline-block;
    background: #FF000F;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.dw-safe-banner {
    background: linear-gradient(90deg, #0f2a1f 0%, #1a3d2e 100%);
    border: 1px solid #2d6a4f;
    border-left: 4px solid #52b788;
    border-radius: 8px;
    padding: 16px 20px;
    color: #d8f3dc;
    margin-bottom: 20px;
    font-size: 0.92rem;
}

.dw-metric-card {
    background: #151D2E;
    border: 1px solid #2a3f5f;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    height: 100%;
}

.dw-metric-card .label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.dw-metric-card .value {
    color: #fff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 6px 0;
}

.dw-metric-card .delta {
    font-size: 0.85rem;
}

.status-operational { color: #52b788; }
.status-degraded { color: #f4a261; }
.status-critical { color: #e76f51; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: #151D2E;
    border-radius: 8px 8px 0 0;
    border: 1px solid #2a3f5f;
    padding: 10px 20px;
    font-weight: 600;
}

div[data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid #2a3f5f;
}

.pipeline-step {
    background: #151D2E;
    border-left: 3px solid #FF000F;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
}
</style>
"""


def inject_theme():
    import streamlit as st
    st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)
