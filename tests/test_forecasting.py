"""Unit tests for src/forecasting.py (stdlib unittest -- pytest could not be
installed in the build sandbox; `python -m unittest discover` runs these
the same way `pytest` would)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import forecasting as fc


class TestSeasonalNaive(unittest.TestCase):
    def test_uses_value_from_one_season_ago(self):
        history = np.arange(1, 105, dtype=float)  # 104 points, 2 full seasons of 52
        preds = fc.seasonal_naive_forecast(history, horizon=4, season_length=52)
        # forecast for h=0 should equal history[104-52] = history[52] = 53.0
        self.assertAlmostEqual(preds[0], history[104 - 52])

    def test_short_history_falls_back_to_last_value(self):
        history = np.array([10.0, 20.0, 30.0])
        preds = fc.seasonal_naive_forecast(history, horizon=3, season_length=52)
        self.assertTrue(np.all(preds == 30.0))

    def test_output_length_matches_horizon(self):
        history = np.random.default_rng(0).uniform(100, 200, size=200)
        preds = fc.seasonal_naive_forecast(history, horizon=8)
        self.assertEqual(len(preds), 8)


class TestHoltWinters(unittest.TestCase):
    def test_falls_back_when_insufficient_history(self):
        history = np.random.default_rng(1).uniform(100, 200, size=50)
        preds = fc.holt_winters_forecast(history, horizon=8, season_length=52)
        # with < 2 seasons of history this should equal the seasonal-naive fallback
        expected = fc.seasonal_naive_forecast(history, 8, 52)
        np.testing.assert_allclose(preds, expected)

    def test_produces_nonnegative_forecast_on_seasonal_data(self):
        rng = np.random.default_rng(2)
        t = np.arange(160)
        y = 1000 + 200 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 10, len(t))
        preds = fc.holt_winters_forecast(y, horizon=8)
        self.assertEqual(len(preds), 8)
        self.assertTrue(np.all(preds >= 0))


class TestGlobalGBM(unittest.TestCase):
    def test_fit_predict_roundtrip(self):
        import pandas as pd
        from src import features as feat
        rng = np.random.default_rng(3)
        n = 300
        df = pd.DataFrame({c: rng.uniform(0, 1, n) for c in feat.FEATURE_COLUMNS})
        df["Weekly_Sales"] = rng.uniform(1000, 5000, n)
        model = fc.GlobalGBMModel.fit(df)
        preds = model.predict(df)
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all(preds >= 0))  # predictions are clipped at 0


if __name__ == "__main__":
    unittest.main()
