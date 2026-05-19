"""Model explainability — feature importance for governance and RCA."""

import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

from utils.model_utils import prepare_xy, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def get_feature_importance(model, df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Extract Random Forest feature importances (numeric + aggregated categorical).
    """
    X, y = prepare_xy(df)
    clf = model.named_steps["clf"]
    preprocess = model.named_steps["preprocess"]

    try:
        X_t = preprocess.transform(X)
        importances = clf.feature_importances_
        # Feature names after preprocessing
        cat_encoder = preprocess.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = list(NUMERIC_FEATURES) + cat_names
    except Exception:
        feature_names = list(NUMERIC_FEATURES) + CATEGORICAL_FEATURES
        importances = clf.feature_importances_[: len(feature_names)]

    imp_df = pd.DataFrame({
        "feature": feature_names[: len(importances)],
        "importance": importances[: len(feature_names)],
    })
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
    imp_df["importance_pct"] = (imp_df["importance"] / imp_df["importance"].sum() * 100).round(1)
    return imp_df.reset_index(drop=True)
