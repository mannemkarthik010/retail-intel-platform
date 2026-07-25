"""Orchestrates the full offline pipeline and writes every artifact the API
and agent layer read at request time (this mirrors how a real system would
have a nightly batch job populate a feature/forecast store that a thin
online serving layer just queries -- the online path never re-trains models
on the fly).

Artifacts written to reports/:
  series_summary.csv        - per-series backtest WAPE per model + selected model
  per_cutoff_metrics.csv    - raw rolling-origin backtest results
  anomaly_flags.csv         - every week x series, flagged or not, with type
  current_forecasts.csv     - next-8-week forecast per series using each
                               series' selected (winning) model, plus an
                               80% prediction interval (forecast_low/high)
  interval_coverage.json    - honest, held-out validation of the interval
                               method (see src/intervals.py)
  models/gbm_point.joblib   - the final point global-GBM, fit on all data
  models/gbm_q10.joblib     - matching q=0.10 quantile GBM
  models/gbm_q90.joblib     - matching q=0.90 quantile GBM
  models/feature_medians.joblib - per-feature medians (occlusion baseline
                               for src/explain.py's local attribution)
  feature_importance.json   - global permutation importance (src/explain.py)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_io import load_all_merged, series_key
from src.features import add_markdown_features, build_feature_table, FEATURE_COLUMNS, TARGET_COLUMN
from src import forecasting as fc
from src import intervals
from src import explain
from src.backtest import run_backtest, HORIZON
from src.anomaly import detect_anomalies_for_series, isolation_forest_crosscheck
from data.generate_data import _us_holidays_for_weekending

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
MODELS_DIR = REPORTS / "models"


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


def _gbm_covariates_for_series(future_cov: pd.DataFrame, hist: pd.DataFrame, store: int, dept: int) -> pd.DataFrame:
    from src.features import add_calendar_features, add_markdown_features as amd
    cov = future_cov.copy()
    cov["Store"], cov["Dept"] = store, dept
    cov["Size"] = hist["Size"].iloc[-1]
    cov = add_calendar_features(cov)
    cov = amd(cov)
    return cov


def build_current_forecasts(raw: pd.DataFrame, selected_model: dict, series_summary: pd.DataFrame,
                             horizon: int = HORIZON) -> pd.DataFrame:
    full_feat = build_feature_table(raw)
    future_cov = make_future_covariates(raw, horizon)

    # Fit the point GBM plus two quantile GBMs (q10/q90) once on ALL
    # available data -- shared by every series whose winner is global_gbm,
    # and persisted to disk so the agent's explainability/what-if tools
    # (src/explain.py, simulate_scenario) can reuse the exact same trained
    # model instead of silently retraining a slightly different one.
    train_feat = full_feat.dropna(subset=FEATURE_COLUMNS)
    gbm = fc.GlobalGBMModel.fit(train_feat)
    gbm_q10 = fc.QuantileGBMModel.fit(train_feat, 0.10)
    gbm_q90 = fc.QuantileGBMModel.fit(train_feat, 0.90)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm, MODELS_DIR / "gbm_point.joblib")
    joblib.dump(gbm_q10, MODELS_DIR / "gbm_q10.joblib")
    joblib.dump(gbm_q90, MODELS_DIR / "gbm_q90.joblib")
    feature_medians = train_feat[FEATURE_COLUMNS].median(numeric_only=True).to_dict()
    joblib.dump(feature_medians, MODELS_DIR / "feature_medians.joblib")

    # Global permutation importance (real, standard method -- see
    # src/explain.py for why this is "in-sample" rather than held-out, and
    # why it's paired with the honest occlusion method for LOCAL
    # attribution). Subsampled for speed: on a model this size, permutation
    # importance re-predicts n_repeats x n_features times, which is
    # expensive against the full ~60k-row table -- a fixed random sample is
    # a documented speed/precision tradeoff, not silently full-precision.
    imp_sample = train_feat.sample(n=min(6000, len(train_feat)), random_state=42)
    # Pass the raw sklearn regressor (gbm.model), not our GlobalGBMModel
    # wrapper -- sklearn's permutation_importance needs __sklearn_tags__
    # (via BaseEstimator) to recognize it as a regressor, which our thin
    # dataclass wrapper doesn't provide. The raw model's predictions can
    # differ trivially from the wrapper's (no non-negative clipping), which
    # doesn't matter for a relative feature-importance ranking.
    importance = explain.global_permutation_importance(
        gbm.model, imp_sample[FEATURE_COLUMNS], imp_sample[TARGET_COLUMN], n_repeats=3,
    )
    (REPORTS / "feature_importance.json").write_text(json.dumps(importance, indent=2))

    rmse_lookup = series_summary.set_index(["series", "model"])["rmse"].to_dict()

    rows = []
    pairs = raw[["Store", "Dept"]].drop_duplicates().values
    from src.backtest import _recursive_gbm_forecast  # reuse recursive logic

    for store, dept in pairs:
        key = series_key(store, dept)
        model_name = selected_model.get(key, "seasonal_naive")
        hist = raw[(raw.Store == store) & (raw.Dept == dept)].sort_values("Date")
        y_hist = hist["Weekly_Sales"].values

        cov = None
        if model_name == "seasonal_naive":
            preds = fc.seasonal_naive_forecast(y_hist, horizon)
        elif model_name == "holt_winters":
            preds = fc.holt_winters_forecast(y_hist, horizon)
        else:
            cov = _gbm_covariates_for_series(future_cov, hist, store, dept)
            preds = _recursive_gbm_forecast(gbm, hist, cov, horizon)

        if model_name == "global_gbm":
            lo, hi = intervals.gbm_quantile_interval(gbm_q10, gbm_q90, hist, cov, preds)
        else:
            rmse = rmse_lookup.get((key, model_name))
            lo, hi = intervals.rmse_normal_interval(preds, rmse)

        last_date = hist["Date"].max()
        for h in range(horizon):
            rows.append({
                "series": key, "Store": store, "Dept": dept,
                "model_used": model_name,
                "week_ahead": h + 1,
                "forecast_date": (last_date + pd.Timedelta(weeks=h + 1)).date(),
                "forecast_value": round(float(preds[h]), 2),
                "forecast_low": round(float(lo[h]), 2),
                "forecast_high": round(float(hi[h]), 2),
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

    print("Validating prediction-interval coverage on a held-out backtest cutoff...")
    coverage = intervals.validate_interval_coverage(raw, bt.selected_model, bt.series_summary)
    (REPORTS / "interval_coverage.json").write_text(json.dumps(coverage, indent=2))
    print(f"  -> wrote interval_coverage.json")

    print("Building forward-looking forecasts (+ 80% prediction intervals) "
          "with each series' selected model...")
    current_fc = build_current_forecasts(raw, bt.selected_model, bt.series_summary)
    current_fc.to_csv(REPORTS / "current_forecasts.csv", index=False)
    print(f"  -> wrote current_forecasts.csv ({len(current_fc)} rows)")

    print("\nRunning agent tool-routing eval harness (scripts/run_agent_eval.py)...")
    from scripts.run_agent_eval import main as run_agent_eval
    run_agent_eval()

    print("\nDone. Artifacts in reports/:")
    for f in sorted(REPORTS.glob("*.csv")) + sorted(REPORTS.glob("*.json")):
        print(" -", f.name)
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        print(" - models/" + f.name)


if __name__ == "__main__":
    main()
