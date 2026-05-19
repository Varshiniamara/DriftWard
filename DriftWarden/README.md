<div align="center">

# 🛡️ DriftWarden — Autonomous ML Reliability Orchestration
**Industrial Drift Detection · Governed Retraining · Audit-Compliant Deployment**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

*Autonomous monitoring that detects drift, explains root cause, and gates model updates with safety constraints and immutable audit logging.*

---

</div>

## 📌 1. Abstract
DriftWarden is an end-to-end **industrial ML reliability layer** for predictive maintenance models. It continuously monitors live telemetry for distribution drift, corroborates statistical signals, performs governed root-cause analysis, and only promotes retrained models when performance improves and safety gates pass.

Built on the **UCI AI4I 2020** motor telemetry dataset (temperature, speed, torque, tool wear, failure labels), while preserving the runtime architecture for real streaming production signals.

---

## ⚡ 2. Quick start


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
