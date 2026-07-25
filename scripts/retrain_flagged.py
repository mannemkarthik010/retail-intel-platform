"""Closes the loop from scripts/run_monitoring_sim.py's per-series drift
alerts to an actual action, instead of leaving `monitoring_series_alerts.csv`
as a table nobody acts on. Run this AFTER run_monitoring_sim.py (it reads
that script's output):

    python scripts/run_pipeline.py
    python scripts/run_monitoring_sim.py
    python scripts/retrain_flagged.py

Scope, disclosed honestly, same spirit as every other "what this actually
does" disclosure in this repo (see docs/ARCHITECTURE.md):

- This does NOT refit the global GBM. That model is trained ONCE across all
  240 series (see docs/ARCHITECTURE.md's "why a global model" section) --
  refitting it is a full-pipeline operation (`scripts/run_pipeline.py`), not
  something that makes sense to trigger for a handful of flagged series.
  It reuses the already-persisted `reports/models/gbm_point.joblib` (+
  q10/q90), which the most recent `run_pipeline.py` run already fit on ALL
  available history.
- What it DOES do, per flagged series: re-examines that series' three
  candidate models (seasonal_naive, holt_winters, global_gbm) using their
  WAPE at the SAME LATEST rolling-origin cutoff the drift alert itself fired
  on (reports/per_cutoff_metrics.csv already has this, computed honestly and
  leakage-free by the original run_backtest() -- nothing new is scored here
  that risks leakage). The series' ORIGINAL selection
  (reports/selected_model.json) picked whichever model had the lowest MEAN
  WAPE across all 6 historical cutoffs; this asks a narrower, more current
  question: "given what just happened most recently, is a different model
  now the better choice for this series?" If yes, it re-selects, recomputes
  that series' forward forecast (reports/current_forecasts.csv) with the
  newly chosen model, and logs the decision. If the currently-deployed model
  is still the best available choice even at the latest cutoff, it's left
  alone and that's logged too -- a retraining review that concludes "no
  change needed" is a real, useful outcome, not a null result to hide.
- Comparing against a single cutoff (rather than a wider recent window) is
  noisier than the original 6-cutoff mean, but it's the metric the alert
  itself fired on -- see docs/EVAL_REPORT.md's "what a next iteration would
  change" for the wider-window refinement this doesn't attempt.
- Every decision (changed or not) is appended to reports/retraining_log.jsonl
  -- an audit trail in the same spirit as reports/agent_traces.jsonl.

Usage: python scripts/retrain_flagged.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
MODELS_DIR = REPORTS / "models"

from src.data_io import load_all_merged, series_key  # noqa: E402
from src import forecasting as fc  # noqa: E402
from src import intervals  # noqa: E402
from src.backtest import HORIZON, _recursive_gbm_forecast  # noqa: E402
from scripts.run_pipeline import make_future_covariates, _gbm_covariates_for_series  # noqa: E402


def _parse_series_key(key: str) -> tuple[int, int]:
    """'S13_D08' -> (13, 8) -- same convention used throughout src/agent.py."""
    store_part, dept_part = key.split("_")
    return int(store_part[1:]), int(dept_part[1:])


def _latest_cutoff_wape_by_model(per_cutoff: pd.DataFrame, key: str) -> dict:
    rows = per_cutoff[per_cutoff["series"] == key].copy()
    if rows.empty:
        return {}
    rows["cutoff"] = pd.to_datetime(rows["cutoff"])
    latest_cutoff = rows["cutoff"].max()
    latest = rows[rows["cutoff"] == latest_cutoff]
    return dict(zip(latest["model"], latest["wape"]))


def _recompute_forecast_rows(raw: pd.DataFrame, store: int, dept: int, model_name: str,
                              rmse_lookup: dict, gbm, gbm_q10, gbm_q90, future_cov: pd.DataFrame) -> list[dict]:
    """Same per-series forecast + interval logic scripts/run_pipeline.py's
    build_current_forecasts() uses -- duplicated in miniature here rather
    than refactored into a shared helper, since this needs to run for ONE
    series against already-loaded/persisted models, not fit fresh ones for
    all 240."""
    key = series_key(store, dept)
    hist = raw[(raw.Store == store) & (raw.Dept == dept)].sort_values("Date")
    y_hist = hist["Weekly_Sales"].values

    cov = None
    if model_name == "seasonal_naive":
        preds = fc.seasonal_naive_forecast(y_hist, HORIZON)
    elif model_name == "holt_winters":
        preds = fc.holt_winters_forecast(y_hist, HORIZON)
    else:
        cov = _gbm_covariates_for_series(future_cov, hist, store, dept)
        preds = _recursive_gbm_forecast(gbm, hist, cov, HORIZON)

    if model_name == "global_gbm":
        lo, hi = intervals.gbm_quantile_interval(gbm_q10, gbm_q90, hist, cov, preds)
    else:
        rmse = rmse_lookup.get((key, model_name))
        lo, hi = intervals.rmse_normal_interval(preds, rmse)

    last_date = hist["Date"].max()
    return [{
        "series": key, "Store": store, "Dept": dept,
        "model_used": model_name,
        "week_ahead": h + 1,
        "forecast_date": (last_date + pd.Timedelta(weeks=h + 1)).date(),
        "forecast_value": round(float(preds[h]), 2),
        "forecast_low": round(float(lo[h]), 2),
        "forecast_high": round(float(hi[h]), 2),
    } for h in range(HORIZON)]


def main():
    alerts_path = REPORTS / "monitoring_series_alerts.csv"
    if not alerts_path.exists():
        print("No reports/monitoring_series_alerts.csv found -- run scripts/run_monitoring_sim.py first.")
        return

    alerts = pd.read_csv(alerts_path)
    if alerts.empty:
        print("No series flagged for retraining review -- nothing to do.")
        return

    selected_model = json.loads((REPORTS / "selected_model.json").read_text())
    series_summary = pd.read_csv(REPORTS / "series_summary.csv")
    per_cutoff = pd.read_csv(REPORTS / "per_cutoff_metrics.csv")
    # Deliberately NOT parse_dates here: reading forecast_date as Timestamps
    # and writing straight back would silently reformat every untouched
    # row's date string ("2023-12-31" -> "2023-12-31 00:00:00") purely as a
    # read/write round-trip side effect, and -- worse -- leave the column
    # mixing that reformatted style with the plain date() strings the newly
    # recomputed rows use, which breaks every downstream parse_dates=[...]
    # reader's format inference. Keeping it as plain strings on read, then
    # normalizing the WHOLE column to one consistent format right before
    # writing (below), avoids both problems.
    current_forecasts = pd.read_csv(REPORTS / "current_forecasts.csv")
    rmse_lookup = series_summary.set_index(["series", "model"])["rmse"].to_dict()

    print("Loading raw data + persisted models (reused, not refit -- see module docstring)...")
    raw = load_all_merged()
    import joblib
    gbm = joblib.load(MODELS_DIR / "gbm_point.joblib")
    gbm_q10 = joblib.load(MODELS_DIR / "gbm_q10.joblib")
    gbm_q90 = joblib.load(MODELS_DIR / "gbm_q90.joblib")
    future_cov = make_future_covariates(raw, HORIZON)

    log_entries = []
    changed_count = 0
    for _, alert in alerts.iterrows():
        key = alert["series"]
        store, dept = _parse_series_key(key)
        deployed_model = selected_model.get(key, alert["model"])

        candidates = _latest_cutoff_wape_by_model(per_cutoff, key)
        if not candidates:
            print(f"  {key}: no per-cutoff data found, skipping")
            continue

        best_model = min(candidates, key=candidates.get)
        changed = best_model != deployed_model

        entry = {
            "ts": round(time.time(), 3),
            "series": key,
            "trigger": {
                "latest_wape": float(alert["latest_wape"]),
                "prior_avg_wape": float(alert["prior_avg_wape"]),
                "delta": float(alert["delta"]),
            },
            "deployed_model": deployed_model,
            "latest_cutoff_wape_by_candidate": {k: round(float(v), 4) for k, v in candidates.items()},
            "best_model_at_latest_cutoff": best_model,
            "action": "reselected" if changed else "kept_same",
        }

        if changed:
            print(f"  {key}: RESELECTING {deployed_model} -> {best_model} "
                  f"(latest-cutoff WAPE {candidates[deployed_model]:.1%} -> {candidates[best_model]:.1%})")
            selected_model[key] = best_model
            new_rows = _recompute_forecast_rows(raw, store, dept, best_model, rmse_lookup,
                                                 gbm, gbm_q10, gbm_q90, future_cov)
            current_forecasts = current_forecasts[current_forecasts["series"] != key]
            current_forecasts = pd.concat([current_forecasts, pd.DataFrame(new_rows)], ignore_index=True)
            changed_count += 1
        else:
            print(f"  {key}: kept {deployed_model} (still the best candidate at the latest cutoff)")

        log_entries.append(entry)

    (REPORTS / "selected_model.json").write_text(json.dumps(selected_model, indent=2))
    current_forecasts = current_forecasts.sort_values(["series", "week_ahead"]).reset_index(drop=True)
    # Normalize the whole column to one consistent "YYYY-MM-DD" format
    # (matching build_current_forecasts()'s original output) before writing
    # -- see the read-side comment above for why this can't just be
    # left as whatever mix of string/date-object formatting fell out of
    # the concat.
    current_forecasts["forecast_date"] = pd.to_datetime(current_forecasts["forecast_date"]).dt.strftime("%Y-%m-%d")
    current_forecasts.to_csv(REPORTS / "current_forecasts.csv", index=False)
    with open(REPORTS / "retraining_log.jsonl", "a") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{len(log_entries)} series reviewed, {changed_count} re-selected, "
          f"{len(log_entries) - changed_count} kept unchanged.")
    print(f"Updated reports/selected_model.json + reports/current_forecasts.csv "
          f"for re-selected series; appended {len(log_entries)} entries to reports/retraining_log.jsonl.")


if __name__ == "__main__":
    main()
