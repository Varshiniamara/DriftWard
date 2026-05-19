# DriftWarden — Enterprise ML Reliability Layer

**Autonomous monitoring, governed retraining, and audit-compliant deployment** for industrial predictive maintenance models.

Built on the [UCI AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) dataset — real motor sensor telemetry (temperature, speed, torque, tool wear, failure labels).

---

## Quick start

```bash
cd DriftWarden
pip install -r requirements.txt
python scripts/setup_data.py
streamlit run app.py
```

Open **http://localhost:8501**

---

## What it does

| Capability | Implementation |
|------------|----------------|
| Drift detection | PSI, Kolmogorov–Smirnov, ADWIN (River) |
| Root cause analysis | Expert analyst + optional OpenAI (`OPENAI_API_KEY`) |
| Strategy selection | monitor_only · fine_tune · full_retrain |
| Confidence gate | Escalates to human if confidence < 0.70 |
| Accuracy gate | Promotes model only if F1 improves |
| ML Reliability Index | Composite 0–100 health KPI |
| Explainability | Random Forest feature importance |
| Audit | Immutable JSON log + TXT export |

---

## Demo script (2 min)

1. Click **▶ RUN FULL PIPELINE** in sidebar  
2. Show **ML Reliability Index** + drift drivers  
3. Show **Expert analysis** narrative  
4. **Engineer sign-off** if prompted → retrain promotes model  
5. Show **version table** + confusion matrices  
6. **Export audit report**

**Key sentence:** *"The model was trained under cooler conditions; live telemetry diverged — DriftWarden detected, explained, retrained, and gated deployment."*

---

## Project structure

```
DriftWarden/
├── app.py                 # Enterprise command center (Streamlit)
├── agents/                # Autonomous agents (detect, analyze, retrain, audit)
├── utils/                 # Metrics, governance, explainability, health score
├── ui/                    # Industrial theme + components
├── data/                  # AI4I reference + live splits
├── models/                # fault_detector.pkl + version registry
└── logs/                  # audit_log.json + exports
```

---

## Positioning

> **DriftWarden is an autonomous reliability layer for industrial machine learning systems.**

Designed for ABB-style motor predictive maintenance: safe autonomy, measurable improvement, full traceability.
