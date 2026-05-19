"""
Model training, evaluation, and persistence for the fault detector.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.drift_metrics import FEATURE_COLS

TARGET_COL = "Machine failure"
TYPE_COL = "Type"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fault_detector.pkl"

# Features used by the predictive maintenance model
NUMERIC_FEATURES = list(FEATURE_COLS)
CATEGORICAL_FEATURES = [TYPE_COL]


def build_pipeline() -> Pipeline:
    """Random forest pipeline with scaling for numeric sensors."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract feature matrix X and binary target y."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).values
    return X, y


def train_model(df: pd.DataFrame) -> Pipeline:
    """Train fault detector on reference dataset."""
    X, y = prepare_xy(df)
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def evaluate_model(model: Pipeline, df: pd.DataFrame) -> dict:
    """Compute classification metrics for model governance gate."""
    X, y_true = prepare_xy(df)
    y_pred = model.predict(X)
    # Handle edge case: no positive class in slice
    zero_div = 0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_div)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_div)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_div)),
        "n_samples": len(y_true),
        "failure_rate": float(y_true.mean()),
        "report": classification_report(y_true, y_pred, zero_division=zero_div),
        "confusion_matrix": cm.tolist(),
    }


def save_model(model: Pipeline, path: Path | None = None) -> Path:
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | None = None) -> Pipeline:
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run scripts/setup_data.py first."
        )
    return joblib.load(path)


def compare_models(old_metrics: dict, new_metrics: dict) -> dict:
    """Compare old vs new model; F1 gate for promotion."""
    f1_delta = new_metrics["f1"] - old_metrics["f1"]
    promote = new_metrics["f1"] >= old_metrics["f1"]
    return {
        "old": old_metrics,
        "new": new_metrics,
        "f1_delta": round(f1_delta, 4),
        "promote": promote,
        "reason": (
            "New model promoted: F1 improved or matched baseline."
            if promote
            else "New model rejected: F1 did not improve (accuracy gate)."
        ),
    }
