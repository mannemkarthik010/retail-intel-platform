"""Unit tests for src/anomaly.py using small synthetic series with a KNOWN
injected shock, so we assert on ground truth rather than eyeballing output."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.anomaly import detect_anomalies_for_series


def make_series(n_weeks=160, shock_start=100, shock_len=5, shock_mult=0.3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-05", periods=n_weeks, freq="W-SUN")
    t = np.arange(n_weeks)
    base = 1000 + 150 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 15, n_weeks)
    base = np.clip(base, 100, None)
    y = base.copy()
    y[shock_start:shock_start + shock_len] *= shock_mult
    return pd.DataFrame({"Date": dates, "Weekly_Sales": y, "markdown_active_count": 0})


class TestAnomalyDetection(unittest.TestCase):
    def test_flags_injected_shock(self):
        df = make_series()
        result = detect_anomalies_for_series(df)
        shocked = result.iloc[100:105]
        self.assertTrue(shocked["needs_investigation"].sum() >= 3,
                         "at least most of the injected shock weeks should be flagged")

    def test_does_not_flag_normal_seasonal_variation(self):
        df = make_series(shock_mult=1.0)  # no actual shock
        result = detect_anomalies_for_series(df)
        # after warmup, false positive rate should be low
        warm = result.iloc[60:]
        false_positive_rate = warm["needs_investigation"].mean()
        self.assertLess(false_positive_rate, 0.15)

    def test_negative_sales_always_flagged_as_data_quality_error(self):
        df = make_series()
        df.loc[50, "Weekly_Sales"] = -500.0
        result = detect_anomalies_for_series(df)
        self.assertEqual(result.loc[50, "anomaly_type"], "data_quality_error")
        self.assertTrue(result.loc[50, "needs_investigation"])

    def test_markdown_active_spike_marked_explained_not_unexplained(self):
        df = make_series(shock_mult=2.5)  # a real spike
        df.loc[100:104, "markdown_active_count"] = 2
        result = detect_anomalies_for_series(df)
        flagged = result.iloc[100:105]
        # anything flagged during an active promo should be "explained", never "unexplained"
        self.assertFalse((flagged["anomaly_type"] == "unexplained_anomaly").any())


if __name__ == "__main__":
    unittest.main()
