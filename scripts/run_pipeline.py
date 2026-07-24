"""Orchestrates the full offline pipeline and writes every artifact the API
and agent layer read at request time (this mirrors how a real system would
have a nightly batch job populate a feature/forecast store that a thin
online serving layer just queries -- the online path never re-trains models
on the fly).

Artifacts written to reports/:
  series_summary.csv     - per-series backtest WAPE per model + selected model
  per_cutoff_metrics.csv - raw rolling-origin backtest results
  anomaly_flags.csv      - every week x series, flagged or not, with type
  current_forecasts.csv  - next-8-week forecast per series using each
                           series' selected (winning) model
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_io import load_all_merged, series_key
from src.features import add_markdown_features, build_feature_table, FEATURE_COLUMNS
from src import forecasting as fc
from src.backtest import run_backtest, HORIZON
from src.anomaly import detect_anomalies_for_series, isolation_forest_crosscheck
from data.generate_data import _us_holidays_for_weekending

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"


def make_future_covariates(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build the covariate rows needed to forecast `horizon` weeks past the
    end of the dataset. Calendar/holiday info is genuinely knowable in
    advance, so it's computed properly. Macro covariates (CPI, Unemployment,
    Fuel_Price, Temperature) are NOT knowable with certainty in reality --
    here we hold them at their last-observed level, a documented
    simplifying assumption (a production system would plug in its own
    macro forecasts or published projections instead)."""
    last_date = raw["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN")
    hflags = _us_holidays_for_weekending(future_dates).set_index("Date")

    last_row = raw[raw["Date"] == last_date].iloc[0]
    future = pd.DataFrame({"Date": future_dates})
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        future[col] = last_row[col]
    future["IsHoliday"] = hflags["IsHoliday"].values
    for c in [f"MarkDown{i}" for i in range(1, 6)]:
        future[c] = 0.0  # no promo calendar assumed beyond the data horizon
    return future


def build_current_forecasts(raw: pd.DataFrame, selected_model: dict, horizon: int = HORIZON) -> pd.DataFrame:
    full_feat = build_feature_table(raw)
    future_cov = make_future_covariates(raw, horizon)

    # fit the global GBM once on ALL available data for series whose winner is global_gbm
    train_feat = full_feat.dropna(subset=FEATURE_COLUMNS)
    gbm = fc.GlobalGBMModel.fit(train_feat)

    rows = []
    pairs = raw[["Store", "Dept"]].drop_duplicates().values
    from src.backtest import _recursive_gbm_forecast  # reuse recursive logic

    for store, dept in pairs:
        key = series_key(store, dept)
        model_name = selected_model.get(key, "seasonal_naive")
        hist = raw[(raw.Store == store) & (raw.Dept == dept)].sort_values("Date")
        y_hist = hist["Weekly_Sales"].values

        if model_name == "seasonal_naive":
            preds = fc.seasonal_naive_forecast(y_hist, horizon)
        elif model_name == "holt_winters":
            preds = fc.holt_winters_forecast(y_hist, horizon)
        else:
            cov = future_cov.copy()
            cov["Store"], cov["Dept"] = store, dept
            cov["Size"] = hist["Size"].iloc[-1]
            from src.features import add_calendar_features, add_markdown_features as amd
            cov = add_calendar_features(cov)
            cov = amd(cov)
            preds = _recursive_gbm_forecast(gbm, hist, cov, horizon)

        last_date = hist["Date"].max()
        for h in range(horizon):
            rows.append({
                "series": key, "Store": store, "Dept": dept,
                "model_used": model_name,
                "week_ahead": h + 1,
                "forecast_date": (last_date + pd.Timedelta(weeks=h + 1)).date(),
                "forecast_value": round(float(preds[h]), 2),
            })
    return pd.DataFrame(rows)


def build_all_anomaly_flags(raw: pd.DataFrame) -> pd.DataFrame:
    raw = add_markdown_features(raw)
    all_results = []
    for (store, dept), g in raw.groupby(["Store", "Dept"]):
        res = detect_anomalies_for_series(g[["Date", "Weekly_Sales", "markdown_active_count", "IsHoliday"]])
        res["Store"], res["Dept"] = store, dept
        res["series"] = series_key(store, dept)
        all_results.append(res)
    all_df = pd.concat(all_results, ignore_index=True)
    all_df = isolation_forest_crosscheck(all_df)
    return all_df


def main():
    print("Loading data...")
    raw = load_all_merged()

    print("Running rolling-origin backtest (this takes ~1 minute)...")
    bt = run_backtest(verbose=True)
    bt.series_summary.to_csv(REPORTS / "series_summary.csv", index=False)
    bt.per_cutoff_metrics.to_csv(REPORTS / "per_cutoff_metrics.csv", index=False)
    (REPORTS / "selected_model.json").write_text(json.dumps(bt.selected_model, indent=2))
    print(f"  -> wrote series_summary.csv ({len(bt.series_summary)} rows), "
          f"per_cutoff_metrics.csv ({len(bt.per_cutoff_metrics)} rows)")

    print("Scanning for anomalies across all series...")
    anomalies = build_all_anomaly_flags(raw)
    anomalies.to_csv(REPORTS / "anomaly_flags.csv", index=False)
    print(f"  -> wrote anomaly_flags.csv ({len(anomalies)} rows)")

    print("Building forward-looking forecasts with each series' selected model...")
    current_fc = build_current_forecasts(raw, bt.selected_model)
    current_fc.to_csv(REPORTS / "current_forecasts.csv", index=False)
    print(f"  -> wrote current_forecasts.csv ({len(current_fc)} rows)")

    print("\nDone. Artifacts in reports/:")
    for f in sorted(REPORTS.glob("*.csv")) + sorted(REPORTS.glob("*.json")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
