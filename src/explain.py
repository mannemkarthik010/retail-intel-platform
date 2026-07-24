"""Forecast-driver explainability for the global GBM model, without SHAP.

`shap` could not be installed in this sandbox (outbound package installs
are restricted to a pre-cached set -- see docs/ARCHITECTURE.md), so this
module implements two honest, transparent substitutes instead of quietly
skipping explainability:

1. **Local, occlusion-based attribution** (`occlusion_attribution`): for
   one specific prediction row, replace each feature one at a time with
   its GLOBAL MEDIAN (computed across all training data at pipeline-build
   time -- see scripts/run_pipeline.py) and measure how much the
   prediction moves. A feature whose median-occlusion swings the
   prediction a lot mattered a lot for THIS particular row. This is a
   simplified, single-order analogue of SHAP's Shapley-value attribution:
   it has no interaction/coalition terms and doesn't average over feature
   orderings the way a true Shapley value does, so it can misattribute
   credit when features are correlated (e.g. rollmean_4 and rollmean_8
   moving together). Documented as an approximation, not sold as SHAP.
2. **Global permutation importance** (`global_permutation_importance`):
   `sklearn.inspection.permutation_importance` -- a real, standard,
   non-approximated method that measures how much held-out error
   increases when a single feature column is randomly shuffled. Computed
   against the model's own training data in this build (the final serving
   model is trained on ALL available history, so there is no leftover
   held-out slice to score against without shrinking training data) --
   that's an in-sample importance estimate, not a held-out one, and that
   tradeoff is stated here rather than hidden.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import features as feat

# Store/Dept are identifiers used by the model as categorical context, not
# "drivers" in the explanatory sense a human is asking about -- excluded
# from the local attribution ranking (they're still in the model's inputs).
NON_EXPLANATORY_FEATURES = {"Store", "Dept"}


def occlusion_attribution(model, row: pd.DataFrame, feature_medians: dict, top_n: int = 6) -> dict:
    """`row` must be a single-row DataFrame with all of feat.FEATURE_COLUMNS
    already populated (exactly as used at prediction time). Returns the
    baseline prediction plus the top_n features ranked by |impact|."""
    row = row[feat.FEATURE_COLUMNS].copy()
    baseline_pred = float(model.predict(row)[0])
    impacts = []
    for col in feat.FEATURE_COLUMNS:
        if col in NON_EXPLANATORY_FEATURES:
            continue
        median_val = feature_medians.get(col)
        if median_val is None or pd.isna(row[col].iloc[0]):
            continue
        occluded = row.copy()
        occluded[col] = median_val
        occluded_pred = float(model.predict(occluded)[0])
        impacts.append({
            "feature": col,
            "actual_value": round(float(row[col].iloc[0]), 3),
            "median_value": round(float(median_val), 3),
            # positive impact = this feature's actual value PUSHED the
            # forecast UP relative to a "typical" (median) series-week
            "impact": round(baseline_pred - occluded_pred, 2),
        })
    impacts.sort(key=lambda d: abs(d["impact"]), reverse=True)
    return {
        "baseline_prediction": round(baseline_pred, 2),
        "top_drivers": impacts[:top_n],
        "method": "occlusion (median-replacement), not SHAP -- see src/explain.py docstring",
    }


def global_permutation_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5,
                                   random_state: int = 42, top_n: int = 12) -> list:
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    return [
        {
            "feature": X.columns[i],
            "importance_mean": round(float(result.importances_mean[i]), 4),
            "importance_std": round(float(result.importances_std[i]), 4),
        }
        for i in order
    ]
