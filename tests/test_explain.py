"""Unit tests for src/explain.py -- the occlusion-based local attribution
and global permutation importance, using small synthetic setups where the
"correct" answer is known by construction (no need for the real trained
pipeline model here)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import explain
from src import features as feat


class _LinearInOneFeature:
    """A fake 'model' whose prediction depends ONLY on `driver_col`, times
    `coef` -- so occlusion attribution has a known-correct answer: every
    other feature should show ~zero impact, and `driver_col` should show
    exactly coef * (actual - median)."""
    def __init__(self, driver_col, coef=2.0):
        self.driver_col = driver_col
        self.coef = coef

    def predict(self, df):
        return (self.coef * df[self.driver_col]).values


class TestOcclusionAttribution(unittest.TestCase):
    def test_only_the_true_driver_shows_nonzero_impact(self):
        row = pd.DataFrame({c: [1.0] for c in feat.FEATURE_COLUMNS})
        row["lag_1"] = 10.0
        medians = {c: 1.0 for c in feat.FEATURE_COLUMNS}
        medians["lag_1"] = 4.0  # actual (10) differs from median (4)

        model = _LinearInOneFeature("lag_1", coef=3.0)
        result = explain.occlusion_attribution(model, row, medians, top_n=len(feat.FEATURE_COLUMNS))

        top = result["top_drivers"][0]
        self.assertEqual(top["feature"], "lag_1")
        self.assertAlmostEqual(top["impact"], 3.0 * (10.0 - 4.0), places=6)
        # every OTHER feature occludes to a prediction identical to baseline
        # (the fake model ignores them entirely) -> impact should be ~0
        for d in result["top_drivers"][1:]:
            self.assertAlmostEqual(d["impact"], 0.0, places=6)

    def test_store_and_dept_excluded_from_ranking(self):
        row = pd.DataFrame({c: [1.0] for c in feat.FEATURE_COLUMNS})
        medians = {c: 1.0 for c in feat.FEATURE_COLUMNS}
        model = _LinearInOneFeature("lag_1", coef=1.0)
        result = explain.occlusion_attribution(model, row, medians, top_n=len(feat.FEATURE_COLUMNS))
        features_shown = {d["feature"] for d in result["top_drivers"]}
        self.assertNotIn("Store", features_shown)
        self.assertNotIn("Dept", features_shown)


class TestGlobalPermutationImportance(unittest.TestCase):
    def test_ranks_the_informative_feature_above_pure_noise(self):
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(0)
        n = 500
        X = pd.DataFrame({
            "signal": rng.uniform(0, 10, n),
            "noise_1": rng.uniform(0, 10, n),
            "noise_2": rng.uniform(0, 10, n),
        })
        y = 5 * X["signal"] + rng.normal(0, 0.1, n)
        model = LinearRegression().fit(X, y)

        importance = explain.global_permutation_importance(model, X, y, n_repeats=5, top_n=3)
        self.assertEqual(importance[0]["feature"], "signal")
        self.assertGreater(importance[0]["importance_mean"], importance[1]["importance_mean"])


if __name__ == "__main__":
    unittest.main()
