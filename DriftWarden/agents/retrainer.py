"""
Retrainer — real sklearn retrain, F1 gate, realistic delay, promote only on measurable gain.
"""

import time
import pandas as pd

from utils.model_utils import (
    train_model,
    evaluate_model,
    compare_models,
    save_model,
    load_model,
    build_pipeline,
    prepare_xy,
)
from utils.model_versioning import get_current_version, register_version

# Must beat deployed model by at least this much to promote
F1_PROMOTE_MARGIN = 0.001


def retrain(
    reference_df: pd.DataFrame,
    live_df: pd.DataFrame,
    strategy: str = "full_retrain",
) -> dict:
    time.sleep(2)  # realistic training feel for demo

    current_model = load_model()
    version_before = get_current_version()
    old_metrics = evaluate_model(current_model, live_df)

    if strategy == "monitor_only":
        return {
            "status": "skipped",
            "strategy": strategy,
            "version_before": version_before,
            "version_after": version_before,
            "old_metrics": old_metrics,
            "new_metrics": None,
            "comparison": None,
        }

    if strategy == "full_retrain":
        train_df = pd.concat([reference_df, live_df], ignore_index=True)
        new_model = train_model(train_df)
    else:
        train_df = pd.concat([
            reference_df.sample(frac=0.5, random_state=42),
            live_df,
        ], ignore_index=True)
        pipe = build_pipeline()
        pipe.named_steps["clf"].set_params(n_estimators=120, max_depth=14)
        X, y = prepare_xy(train_df)
        pipe.fit(X, y)
        new_model = pipe

    new_metrics = evaluate_model(new_model, live_df)
    comparison = compare_models(old_metrics, new_metrics)

    # Strict gate: must improve by margin, not just tie
    f1_gain = new_metrics["f1"] - old_metrics["f1"]
    promote = f1_gain >= F1_PROMOTE_MARGIN
    comparison["promote"] = promote
    comparison["f1_delta"] = round(f1_gain, 4)
    comparison["reason"] = (
        f"Promoted: F1 improved {old_metrics['f1']:.3f} → {new_metrics['f1']:.3f} (+{f1_gain:.4f})"
        if promote
        else f"Rejected: F1 gain {f1_gain:+.4f} below required margin ({F1_PROMOTE_MARGIN})"
    )

    if promote:
        save_model(new_model)
        ver = register_version(new_metrics["f1"], promoted=True, note=strategy)
        status = "promoted"
    else:
        ver = register_version(old_metrics["f1"], promoted=False, note="rejected")
        status = "rejected"

    def _slim(m: dict) -> dict:
        return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "confusion_matrix")}

    return {
        "status": status,
        "strategy": strategy,
        "version_before": version_before,
        "version_after": ver["new_version"],
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "old_metrics_slim": _slim(old_metrics),
        "new_metrics_slim": _slim(new_metrics),
        "comparison": comparison,
    }
