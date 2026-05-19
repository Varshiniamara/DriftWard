#!/usr/bin/env python3
"""
One-time setup: download is done manually; this script splits AI4I into
reference (training baseline) and live (production) CSVs, then trains the initial model.

Run from DriftWarden/:  python scripts/setup_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.model_utils import train_model, save_model, evaluate_model
from utils.drift_injector import inject_seasonal_aging_drift
from utils.model_versioning import register_version

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "ai4i_raw.csv"
REF_PATH = DATA_DIR / "ai4i_reference.csv"
LIVE_PATH = DATA_DIR / "ai4i_live.csv"


def main():
    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}. Download from UCI:")
        print("  curl -L -o data/ai4i_raw.csv "
              "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    print(f"Loaded AI4I dataset: {len(df)} rows, {len(df.columns)} columns")

    # Chronological split simulates deployment timeline (by UDI order)
    n_ref = int(len(df) * 0.7)
    reference = df.iloc[:n_ref].copy()
    live_base = df.iloc[n_ref:].copy()

    reference.to_csv(REF_PATH, index=False)
    print(f"Reference (training baseline): {len(reference)} rows -> {REF_PATH}")

    # Live starts as clean production copy; drift applied via dashboard or below
    live_base.to_csv(LIVE_PATH, index=False)
    print(f"Live (production, no drift yet): {len(live_base)} rows -> {LIVE_PATH}")

    # Train and save initial model on reference only
    model = train_model(reference)
    path = save_model(model)
    metrics = evaluate_model(model, reference)
    print(f"Model saved: {path}")
    print(f"Reference F1: {metrics['f1']:.4f}  Accuracy: {metrics['accuracy']:.4f}")
    register_version(metrics["f1"], promoted=True, note="initial_deploy")

    # Save clean baseline copy — drift is applied in-memory by the UI sliders
    baseline_path = DATA_DIR / "ai4i_live_baseline.csv"
    live_base.to_csv(baseline_path, index=False)
    print(f"Clean live baseline -> {baseline_path}")
    print("Setup complete. Drift is controlled live in the app. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
