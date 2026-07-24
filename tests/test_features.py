"""Tests for feature engineering -- mainly guarding against lookahead leakage,
which is the single easiest mistake to make (and the one the target JDs
explicitly call out knowing how to avoid: 'know ... how to avoid leakage
and train/serve skew')."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import add_lag_and_rolling_features


class TestNoLookaheadLeakage(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-05", periods=20, freq="W-SUN")
        self.df = pd.DataFrame({
            "Store": 1, "Dept": 1, "Date": dates,
            "Weekly_Sales": np.arange(1, 21, dtype=float),
        })

    def test_lag_1_is_previous_row_not_current(self):
        out = add_lag_and_rolling_features(self.df)
        # row index 5 (Weekly_Sales=6) should have lag_1 == 5 (the prior week), never 6
        self.assertEqual(out.loc[5, "lag_1"], 5.0)
        self.assertNotEqual(out.loc[5, "lag_1"], out.loc[5, "Weekly_Sales"])

    def test_rolling_mean_excludes_current_row(self):
        out = add_lag_and_rolling_features(self.df)
        # rollmean_4 at row 10 (Weekly_Sales=11) should be mean of rows 6..9 (values 7,8,9,10) = 8.5
        self.assertAlmostEqual(out.loc[10, "rollmean_4"], 8.5)

    def test_series_do_not_bleed_into_each_other(self):
        df2 = self.df.copy()
        df2["Store"] = 2
        df2["Weekly_Sales"] = df2["Weekly_Sales"] * 100
        both = pd.concat([self.df, df2], ignore_index=True).sort_values(["Store", "Date"])
        out = add_lag_and_rolling_features(both)
        store2 = out[out["Store"] == 2].reset_index(drop=True)
        # store 2's lag_1 must never equal a store-1-scale value
        self.assertTrue((store2["lag_1"].dropna() >= 100).all())


if __name__ == "__main__":
    unittest.main()
