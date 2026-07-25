"""Tests for scripts/retrain_flagged.py -- the script that closes the loop
from scripts/run_monitoring_sim.py's per-series drift alerts to an actual
re-selection decision. Requires reports/ pipeline artifacts + persisted
models to exist (run scripts/run_pipeline.py + scripts/run_monitoring_sim.py
first), same precondition as tests/test_agent.py."""
import json
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
REPORTS = Path(__file__).parent.parent / "reports"


class TestParseSeriesKey(unittest.TestCase):
    def test_parses_store_and_dept(self):
        from scripts.retrain_flagged import _parse_series_key
        self.assertEqual(_parse_series_key("S13_D08"), (13, 8))
        self.assertEqual(_parse_series_key("S01_D01"), (1, 1))
        self.assertEqual(_parse_series_key("S20_D12"), (20, 12))


class TestLatestCutoffWapeByModel(unittest.TestCase):
    def test_picks_only_the_most_recent_cutoff_per_model(self):
        from scripts.retrain_flagged import _latest_cutoff_wape_by_model
        per_cutoff = pd.DataFrame([
            {"series": "S01_D01", "cutoff": "2023-01-22", "model": "seasonal_naive", "wape": 0.10},
            {"series": "S01_D01", "cutoff": "2023-01-22", "model": "global_gbm", "wape": 0.08},
            {"series": "S01_D01", "cutoff": "2023-10-29", "model": "seasonal_naive", "wape": 0.30},
            {"series": "S01_D01", "cutoff": "2023-10-29", "model": "global_gbm", "wape": 0.05},
            {"series": "S02_D02", "cutoff": "2023-10-29", "model": "seasonal_naive", "wape": 0.99},
        ])
        result = _latest_cutoff_wape_by_model(per_cutoff, "S01_D01")
        self.assertEqual(result, {"seasonal_naive": 0.30, "global_gbm": 0.05})

    def test_returns_empty_dict_for_unknown_series(self):
        from scripts.retrain_flagged import _latest_cutoff_wape_by_model
        per_cutoff = pd.DataFrame([{"series": "S01_D01", "cutoff": "2023-01-22",
                                     "model": "seasonal_naive", "wape": 0.10}])
        self.assertEqual(_latest_cutoff_wape_by_model(per_cutoff, "S99_D99"), {})


@unittest.skipUnless((REPORTS / "monitoring_series_alerts.csv").exists()
                      and (REPORTS / "models" / "gbm_point.joblib").exists(),
                      "run scripts/run_pipeline.py + scripts/run_monitoring_sim.py first")
class TestRetrainFlaggedEndToEnd(unittest.TestCase):
    """Runs the real script against the real reports/ artifacts (same
    precondition style as the rest of the suite), snapshotting and
    restoring the files it mutates so running this test doesn't leave the
    checked-in reports/ state permanently changed or grow
    retraining_log.jsonl on every test run."""

    MUTATED_FILES = ["selected_model.json", "current_forecasts.csv", "retraining_log.jsonl"]

    def setUp(self):
        self._backups = {}
        for name in self.MUTATED_FILES:
            path = REPORTS / name
            self._backups[name] = path.read_bytes() if path.exists() else None

    def tearDown(self):
        for name, content in self._backups.items():
            path = REPORTS / name
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)

    def test_every_alert_gets_a_logged_decision_and_forecast_row_count_is_preserved(self):
        from scripts.retrain_flagged import main

        alerts = pd.read_csv(REPORTS / "monitoring_series_alerts.csv")
        before_forecasts = pd.read_csv(REPORTS / "current_forecasts.csv")
        before_selected = json.loads((REPORTS / "selected_model.json").read_text())
        before_log_lines = 0
        if (REPORTS / "retraining_log.jsonl").exists():
            with open(REPORTS / "retraining_log.jsonl") as f:
                before_log_lines = len(f.readlines())

        main()

        after_forecasts = pd.read_csv(REPORTS / "current_forecasts.csv")
        after_selected = json.loads((REPORTS / "selected_model.json").read_text())
        with open(REPORTS / "retraining_log.jsonl") as f:
            after_log_lines = len(f.readlines())

        # row count per series is preserved -- re-selection swaps WHICH
        # model produced a series' 8 rows, never adds/drops rows
        self.assertEqual(len(before_forecasts), len(after_forecasts))
        self.assertEqual(set(before_forecasts["series"]), set(after_forecasts["series"]))

        # every flagged series has a valid (possibly unchanged) selection
        for series in alerts["series"]:
            self.assertIn(after_selected[series], {"seasonal_naive", "holt_winters", "global_gbm"})

        # exactly one new audit line per flagged series
        self.assertEqual(after_log_lines - before_log_lines, len(alerts))

        # any series whose model changed now forecasts with the new model
        for series in alerts["series"]:
            rows = after_forecasts[after_forecasts["series"] == series]
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows["model_used"].nunique(), 1)
            self.assertEqual(rows["model_used"].iloc[0], after_selected[series])


if __name__ == "__main__":
    unittest.main()
