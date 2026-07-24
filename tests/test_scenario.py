"""Tests for src/scenario.py (what-if simulation). Requires the trained
GBM artifacts scripts/run_pipeline.py persists under reports/models/, since
this is the one part of the system that genuinely recomputes rather than
reading a precomputed CSV -- see scenario.py's module docstring."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPORTS = Path(__file__).parent.parent / "reports"
MODELS_DIR = REPORTS / "models"


@unittest.skipUnless((MODELS_DIR / "gbm_point.joblib").exists(),
                      "run scripts/run_pipeline.py first to persist reports/models/*.joblib")
class TestScenarioCalendarHandling(unittest.TestCase):
    def test_is_holiday_none_preserves_real_calendar_not_all_false(self):
        """Regression test for a real bug found during development: an
        earlier version defaulted IsHoliday to 0 whenever the caller only
        asked about a markdown, which silently stripped a genuine holiday
        flag off any week that happened to actually be one. is_holiday=None
        must NOT be equivalent to is_holiday=False."""
        import json
        import pandas as pd
        from src import scenario
        from src.data_io import load_all_merged
        from data.generate_data import _us_holidays_for_weekending

        raw = load_all_merged()
        sel = json.loads((REPORTS / "selected_model.json").read_text())
        gbm_series = [k for k, v in sel.items() if v == "global_gbm"]
        if not gbm_series:
            self.skipTest("no global_gbm series in this run's selected_model.json")
        store, dept = int(gbm_series[0][1:3]), int(gbm_series[0][-2:])

        hist = raw[(raw.Store == store) & (raw.Dept == dept)].sort_values("Date")
        last_date = hist["Date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=4, freq="W-SUN")
        real_flags = _us_holidays_for_weekending(future_dates)["IsHoliday"].tolist()

        _, cov_default = scenario._build_hypothetical_covariates(raw, store, dept, 4, False, None)
        _, cov_forced_false = scenario._build_hypothetical_covariates(raw, store, dept, 4, False, False)

        self.assertEqual(cov_default["IsHoliday"].tolist(), [int(f) for f in real_flags])
        # forcing False must actually zero it out, proving the two modes differ
        self.assertTrue(all(v == 0 for v in cov_forced_false["IsHoliday"].tolist()))

    def test_markdown_only_scenario_does_not_change_length(self):
        import json
        from src import scenario
        from src.data_io import load_all_merged

        raw = load_all_merged()
        sel = json.loads((REPORTS / "selected_model.json").read_text())
        gbm_series = [k for k, v in sel.items() if v == "global_gbm"]
        if not gbm_series:
            self.skipTest("no global_gbm series in this run's selected_model.json")
        store, dept = int(gbm_series[0][1:3]), int(gbm_series[0][-2:])

        preds = scenario.simulate_scenario(raw, store, dept, markdown_active=True, is_holiday=None, weeks=3)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(p >= 0 for p in preds))


if __name__ == "__main__":
    unittest.main()
