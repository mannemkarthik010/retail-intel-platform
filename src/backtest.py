"""Rolling-origin ("walk-forward") backtesting across all three models.

For each cutoff date we:
  1. Train the global GBM on everything strictly before the cutoff.
  2. Forecast `HORIZON` weeks ahead for every (Store, Dept) series with all
     three models (seasonal-naive, Holt-Winters, global GBM -- GBM forecasts
     recursively, feeding its own predictions back in as lag features since
     future actuals aren't known).
  3. Score each model against the true values with WAPE, MAPE, RMSE.

We then pick a WINNING MODEL PER SERIES based on mean WAPE across all
cutoffs -- this is the "best-fit selection" pattern several of the target
job descriptions call out explicitly (Confido's JD names it verbatim).

This is deliberately NOT a single train/test split: one split can make a
mediocre model look great (or a good one look bad) purely by luck of which
weeks land in the holdout. Rolling-origin backtesting is the standard fix
used in real forecasting teams, and it's the piece most tutorial projects
skip.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import features as feat
from . import forecasting as fc
from .data_io import load_all_merged, series_key

HORIZON = 8
N_CUTOFFS = 6
MIN_HISTORY_FOR_HW = 2 * fc.SEASON_LENGTH


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom if denom > 0 else np.nan


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _recursive_gbm_forecast(model: fc.GlobalGBMModel, series_hist: pd.DataFrame,
                             future_covariates: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forecast `horizon` steps ahead for ONE series, feeding predictions back
    in as lag features (since we don't know future actuals)."""
    history = series_hist.sort_values("Date")
    sales_hist = list(history["Weekly_Sales"].values)
    preds = []
    for h in range(horizon):
        cov_row = future_covariates.iloc[[h]].copy()
        arr = np.array(sales_hist)
        for lag in feat.LAGS:
            cov_row[f"lag_{lag}"] = arr[-lag] if len(arr) >= lag else np.nan
        for w in feat.ROLLING_WINDOWS:
            window = arr[-w:] if len(arr) >= w else arr
            cov_row[f"rollmean_{w}"] = window.mean() if len(window) else np.nan
            cov_row[f"rollstd_{w}"] = window.std() if len(window) else 0.0
        pred = float(model.predict(cov_row)[0])
        preds.append(pred)
        sales_hist.append(pred)
    return np.array(preds)


@dataclass
class BacktestResult:
    per_cutoff_metrics: pd.DataFrame
    series_summary: pd.DataFrame
    selected_model: dict = field(default_factory=dict)


def run_backtest(verbose: bool = True) -> BacktestResult:
    raw = load_all_merged()
    full_feat = feat.build_feature_table(raw)
    dates = sorted(raw["Date"].unique())
    n_weeks = len(dates)

    cutoff_idxs = [n_weeks - HORIZON - i * HORIZON for i in range(N_CUTOFFS)]
    cutoff_idxs = [i for i in cutoff_idxs if i >= MIN_HISTORY_FOR_HW]
    cutoff_idxs = sorted(cutoff_idxs)

    all_rows = []
    pairs = raw[["Store", "Dept"]].drop_duplicates().values

    for ci, cutoff_idx in enumerate(cutoff_idxs):
        cutoff_date = dates[cutoff_idx - 1]
        horizon_dates = dates[cutoff_idx: cutoff_idx + HORIZON]
        if len(horizon_dates) < HORIZON:
            continue
        if verbose:
            print(f"[backtest] cutoff {ci+1}/{len(cutoff_idxs)}: {pd.Timestamp(cutoff_date).date()}")

        train_feat = full_feat[full_feat["Date"] <= cutoff_date].dropna(subset=feat.FEATURE_COLUMNS)
        gbm = fc.GlobalGBMModel.fit(train_feat)

        # covariates for the horizon (known in advance: calendar/macro/markdown)
        horizon_feat_all = full_feat[full_feat["Date"].isin(horizon_dates)]

        for store, dept in pairs:
            hist = raw[(raw.Store == store) & (raw.Dept == dept) & (raw.Date <= cutoff_date)]
            truth = raw[(raw.Store == store) & (raw.Dept == dept) & (raw.Date.isin(horizon_dates))]
            truth = truth.sort_values("Date")
            if len(truth) < HORIZON or len(hist) < 10:
                continue
            y_true = truth["Weekly_Sales"].values

            y_naive = fc.seasonal_naive_forecast(hist.sort_values("Date")["Weekly_Sales"].values, HORIZON)
            y_hw = fc.holt_winters_forecast(hist.sort_values("Date")["Weekly_Sales"].values, HORIZON)

            cov = horizon_feat_all[(horizon_feat_all.Store == store) & (horizon_feat_all.Dept == dept)]
            cov = cov.sort_values("Date")
            y_gbm = _recursive_gbm_forecast(gbm, hist, cov, HORIZON)

            for model_name, y_pred in [("seasonal_naive", y_naive), ("holt_winters", y_hw), ("global_gbm", y_gbm)]:
                all_rows.append({
                    "cutoff": pd.Timestamp(cutoff_date).date(),
                    "Store": store, "Dept": dept,
                    "series": series_key(store, dept),
                    "model": model_name,
                    "wape": wape(y_true, y_pred),
                    "mape": mape(y_true, y_pred),
                    "rmse": rmse(y_true, y_pred),
                })

    per_cutoff = pd.DataFrame(all_rows)
    series_summary = (
        per_cutoff.groupby(["series", "Store", "Dept", "model"])[["wape", "mape", "rmse"]]
        .mean()
        .reset_index()
    )
    selected_model = (
        series_summary.loc[series_summary.groupby("series")["wape"].idxmin()]
        .set_index("series")["model"]
        .to_dict()
    )
    return BacktestResult(per_cutoff_metrics=per_cutoff, series_summary=series_summary, selected_model=selected_model)


if __name__ == "__main__":
    result = run_backtest()
    print("\n=== Mean WAPE by model (lower is better) ===")
    print(result.series_summary.groupby("model")["wape"].mean().sort_values())
    print("\n=== Model win counts across series ===")
    print(pd.Series(result.selected_model).value_counts())
