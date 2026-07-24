"""Unit tests for src/intervals.py -- the interval-assembly logic itself
(quantile-crossing correction, normal-approximation band), independent of
any trained model, using small hand-built fakes."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src import intervals


class _FakeQuantileModel:
    """Returns a fixed value regardless of input, so we can test the
    crossing-correction logic in isolation from any real model."""
    def __init__(self, value):
        self.value = value

    def predict(self, df):
        return np.full(len(df), self.value)


class TestRmseNormalInterval(unittest.TestCase):
    def test_widens_around_point_forecast_symmetrically_in_z_units(self):
        point = np.array([100.0, 200.0, 300.0])
        lo, hi = intervals.rmse_normal_interval(point, rmse=10.0)
        np.testing.assert_allclose(hi - point, point - lo)
        np.testing.assert_allclose(hi - point, intervals.Z80 * 10.0)

    def test_low_is_clipped_at_zero(self):
        point = np.array([1.0])
        lo, hi = intervals.rmse_normal_interval(point, rmse=100.0)
        self.assertGreaterEqual(lo[0], 0.0)

    def test_missing_rmse_collapses_to_point_forecast(self):
        point = np.array([50.0, 60.0])
        lo, hi = intervals.rmse_normal_interval(point, rmse=None)
        np.testing.assert_allclose(lo, point)
        np.testing.assert_allclose(hi, point)
        lo2, hi2 = intervals.rmse_normal_interval(point, rmse=float("nan"))
        np.testing.assert_allclose(lo2, point)
        np.testing.assert_allclose(hi2, point)


class TestGbmQuantileInterval(unittest.TestCase):
    def test_enforces_ordering_even_if_models_cross(self):
        """Two independently trained quantile models have no mathematical
        guarantee that q10 <= q90 -- feed in a deliberately CROSSED pair
        (q10 model predicts higher than q90 model) and confirm the
        interval still comes out correctly ordered and containing the
        point forecast."""
        series_hist = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=60, freq="W-SUN"),
            "Weekly_Sales": np.linspace(100, 160, 60),
        })
        future_cov = pd.DataFrame({"Date": pd.date_range("2021-03-01", periods=2, freq="W-SUN")})
        point_path = np.array([150.0, 155.0])

        crossed_low_model = _FakeQuantileModel(200.0)   # "q10" that predicts HIGH
        crossed_high_model = _FakeQuantileModel(50.0)   # "q90" that predicts LOW

        lo, hi = intervals.gbm_quantile_interval(crossed_low_model, crossed_high_model,
                                                  series_hist, future_cov, point_path)
        self.assertTrue(np.all(lo <= hi))
        self.assertTrue(np.all(lo <= point_path))
        self.assertTrue(np.all(hi >= point_path))

    def test_normal_noncrossed_case_brackets_point_forecast(self):
        series_hist = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=60, freq="W-SUN"),
            "Weekly_Sales": np.linspace(100, 160, 60),
        })
        future_cov = pd.DataFrame({"Date": pd.date_range("2021-03-01", periods=2, freq="W-SUN")})
        point_path = np.array([150.0, 155.0])

        q10_model = _FakeQuantileModel(120.0)
        q90_model = _FakeQuantileModel(180.0)

        lo, hi = intervals.gbm_quantile_interval(q10_model, q90_model, series_hist, future_cov, point_path)
        np.testing.assert_allclose(lo, [120.0, 120.0])
        np.testing.assert_allclose(hi, [180.0, 180.0])


if __name__ == "__main__":
    unittest.main()
