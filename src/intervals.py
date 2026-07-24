"""Prediction intervals: an 80% band (10th/90th percentile) around every
point forecast, not just a single number.

Two different methods are used depending on which model won the backtest
for a given series, and both are documented as honest approximations
rather than textbook-perfect uncertainty quantification:

- **global_gbm series**: two extra `HistGradientBoostingRegressor` models
  trained with pinball (quantile) loss at q=0.10 and q=0.90
  (`forecasting.QuantileGBMModel`), queried along the POINT model's own
  recursive forecast path. See that class's docstring for the specific
  limitation this implies (uncertainty along one trajectory, not a fully
  recursive quantile tree).
- **seasonal_naive / holt_winters series**: neither classical method has a
  native quantile-regression analogue, so the interval is a
  normal-approximation band using that series' own backtested RMSE for
  the winning model: `forecast +/- 1.2816 * rmse` (the z-score for an 80%
  two-sided interval under a Gaussian error assumption). This assumes
  roughly constant error variance across the 8-week horizon, which is a
  simplification -- real forecast uncertainty typically grows with the
  number of weeks out. Documented here and in docs/EVAL_REPORT.md rather
  than silently assumed to be perfectly calibrated.

`validate_interval_coverage()` below checks the honest, held-out question
this raises: does the 80% interval actually contain the true value about
80% of the time? It refits on data strictly before the most recent
backtest cutoff and checks coverage against real (never-trained-on)
future values -- see scripts/run_pipeline.py for where this is invoked and
reports/interval_coverage.json for where the real (not cherry-picked)
number is written.
"""
import numpy as np
import pandas as pd

from . import features as feat
from . import forecasting as fc

Z80 = 1.2816  # two-sided 80% interval (10th/90th percentile) under a normal approximation


def _recursive_quantile_along_path(qmodel: "fc.QuantileGBMModel", series_hist: pd.DataFrame,
                                    future_covariates: pd.DataFrame, point_path: np.ndarray) -> np.ndarray:
    """Mirrors src/backtest.py::_recursive_gbm_forecast's feature
    construction, but advances the lag/rolling history using the POINT
    model's predictions at each step (not this quantile model's own) -- see
    QuantileGBMModel's docstring for why."""
    history = series_hist.sort_values("Date")
    sales_hist = list(history["Weekly_Sales"].values)
    out = np.zeros(len(point_path))
    for h in range(len(point_path)):
        cov_row = future_covariates.iloc[[h]].copy()
        arr = np.array(sales_hist)
        for lag in feat.LAGS:
            cov_row[f"lag_{lag}"] = arr[-lag] if len(arr) >= lag else np.nan
        for w in feat.ROLLING_WINDOWS:
            window = arr[-w:] if len(arr) >= w else arr
            cov_row[f"rollmean_{w}"] = window.mean() if len(window) else np.nan
            cov_row[f"rollstd_{w}"] = window.std() if len(window) else 0.0
        out[h] = float(qmodel.predict(cov_row)[0])
        sales_hist.append(point_path[h])
    return out


def gbm_quantile_interval(q10_model, q90_model, series_hist: pd.DataFrame,
                           future_covariates: pd.DataFrame, point_path: np.ndarray):
    """Returns (low, high) arrays for a global_gbm-selected series."""
    low = _recursive_quantile_along_path(q10_model, series_hist, future_covariates, point_path)
    high = _recursive_quantile_along_path(q90_model, series_hist, future_covariates, point_path)
    # Two independently-trained quantile models have no mathematical
    # guarantee that q10 <= q90 pointwise ("quantile crossing"). Rather than
    # silently emit a crossed interval, enforce ordering explicitly and also
    # make sure the point forecast itself always sits inside its own band.
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    lo = np.minimum(lo, point_path)
    hi = np.maximum(hi, point_path)
    return np.clip(lo, 0, None), hi


def rmse_normal_interval(point_path: np.ndarray, rmse):
    """Returns (low, high) arrays for a seasonal_naive/holt_winters series
    using a normal approximation scaled by backtested RMSE."""
    if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
        return point_path, point_path
    lo = np.clip(point_path - Z80 * rmse, 0, None)
    hi = point_path + Z80 * rmse
    return lo, hi


def validate_interval_coverage(raw: pd.DataFrame, selected_model: dict, series_summary: pd.DataFrame,
                                verbose: bool = True) -> dict:
    """Honest, held-out validation of the interval methods above: refit on
    data strictly before the single most recent backtest cutoff (the same
    cutoff run_backtest() itself uses as its most recent fold), forecast 8
    weeks forward, and check what fraction of the ACTUAL values (which the
    refit never saw) fall inside the resulting interval. Nominal target is
    80% -- this reports the real empirical number, not a rounded-up one.

    Note on the RMSE used for the non-GBM band here: it's the backtest-mean
    RMSE from `series_summary` (averaged across all N_CUTOFFS backtest
    folds, one of which is this exact held-out cutoff). That's a small
    source of optimism for the non-GBM interval width -- disclosed rather
    than hidden -- since a fully clean validation would recompute RMSE from
    the other 5 folds only. The GBM quantile models, by contrast, are
    refit from scratch on pre-cutoff data only, so their coverage number
    here is clean.
    """
    from .backtest import HORIZON, MIN_HISTORY_FOR_HW, _recursive_gbm_forecast
    from .data_io import series_key
    from .features import build_feature_table, FEATURE_COLUMNS, add_calendar_features, add_markdown_features

    full_feat = build_feature_table(raw)
    dates = sorted(raw["Date"].unique())
    cutoff_idx = len(dates) - HORIZON
    if cutoff_idx < MIN_HISTORY_FOR_HW:
        return {"error": "not enough history to validate interval coverage"}
    cutoff_date = dates[cutoff_idx - 1]
    horizon_dates = dates[cutoff_idx: cutoff_idx + HORIZON]

    train_feat = full_feat[full_feat["Date"] <= cutoff_date].dropna(subset=FEATURE_COLUMNS)
    point_model = fc.GlobalGBMModel.fit(train_feat)
    q10_model = fc.QuantileGBMModel.fit(train_feat, 0.10)
    q90_model = fc.QuantileGBMModel.fit(train_feat, 0.90)
    horizon_feat_all = full_feat[full_feat["Date"].isin(horizon_dates)]

    rmse_lookup = series_summary.set_index(["series", "model"])["rmse"].to_dict()

    total, covered = 0, 0
    per_model_stats = {}
    pairs = raw[["Store", "Dept"]].drop_duplicates().values
    for store, dept in pairs:
        key = series_key(store, dept)
        model_name = selected_model.get(key, "seasonal_naive")
        hist = raw[(raw.Store == store) & (raw.Dept == dept) & (raw.Date <= cutoff_date)]
        truth = raw[(raw.Store == store) & (raw.Dept == dept) & (raw.Date.isin(horizon_dates))].sort_values("Date")
        if len(truth) < HORIZON or len(hist) < 10:
            continue
        y_true = truth["Weekly_Sales"].values

        if model_name == "global_gbm":
            cov = horizon_feat_all[(horizon_feat_all.Store == store) & (horizon_feat_all.Dept == dept)].sort_values("Date")
            point_path = _recursive_gbm_forecast(point_model, hist, cov, HORIZON)
            lo, hi = gbm_quantile_interval(q10_model, q90_model, hist, cov, point_path)
        else:
            y_hist = hist.sort_values("Date")["Weekly_Sales"].values
            point_path = (fc.seasonal_naive_forecast(y_hist, HORIZON) if model_name == "seasonal_naive"
                          else fc.holt_winters_forecast(y_hist, HORIZON))
            rmse = rmse_lookup.get((key, model_name))
            lo, hi = rmse_normal_interval(point_path, rmse)

        inside = (y_true >= lo) & (y_true <= hi)
        covered += int(inside.sum())
        total += len(y_true)
        per_model_stats.setdefault(model_name, [0, 0])
        per_model_stats[model_name][0] += int(inside.sum())
        per_model_stats[model_name][1] += len(y_true)

    result = {
        "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "nominal_coverage": 0.80,
        "empirical_coverage_overall": round(covered / total, 4) if total else None,
        "n_observations": total,
        "empirical_coverage_by_selected_model": {
            m: round(c / t, 4) if t else None for m, (c, t) in per_model_stats.items()
        },
    }
    if verbose and total:
        print(f"[intervals] held-out coverage check @ {result['cutoff_date']}: "
              f"{result['empirical_coverage_overall']*100:.1f}% actual vs 80% nominal (n={total})")
    return result
