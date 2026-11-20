"""Regression tests for the serving-path optimizations.

Each optimization in the request path (see the README's "Performance"
section) replaced a slow-but-obvious implementation with a faster one that
is supposed to be *exactly* equivalent, not approximately so. These tests
keep the slow reference implementation around and assert the fast one
still agrees with it, so a future change that quietly alters a number gets
caught here rather than in a dashboard nobody double-checks.
"""
import pandas as pd
import pytest

from src import agent as agent_mod
from src import explain, features as feat
from src.agent import _store
from src.backtest import HORIZON


@pytest.fixture(scope="module")
def ds():
    return _store()


@pytest.fixture(scope="module")
def gbm_series(ds):
    keys = [k for k, v in ds.selected_model.items() if v == "global_gbm"]
    return keys[:4]


# --------------------------------------------------------------------------
# 1. Batched occlusion attribution must equal the per-row loop exactly.
# --------------------------------------------------------------------------
def _occlusion_reference(model, row, feature_medians, top_n=6):
    """The original one-predict-per-feature implementation, verbatim."""
    row = row[feat.FEATURE_COLUMNS].copy()
    baseline_pred = float(model.predict(row)[0])
    impacts = []
    for col in feat.FEATURE_COLUMNS:
        if col in explain.NON_EXPLANATORY_FEATURES:
            continue
        median_val = feature_medians.get(col)
        if median_val is None or pd.isna(row[col].iloc[0]):
            continue
        occluded = row.copy()
        occluded[col] = median_val
        occluded_pred = float(model.predict(occluded)[0])
        impacts.append({
            "feature": col,
            "actual_value": round(float(row[col].iloc[0]), 3),
            "median_value": round(float(median_val), 3),
            "impact": round(baseline_pred - occluded_pred, 2),
        })
    impacts.sort(key=lambda d: abs(d["impact"]), reverse=True)
    return {
        "baseline_prediction": round(baseline_pred, 2),
        "top_drivers": impacts[:top_n],
        "method": "occlusion (median-replacement), not SHAP -- see src/explain.py docstring",
    }


def test_batched_occlusion_matches_per_row_loop(ds, gbm_series):
    model, medians = ds.point_gbm(), ds.feature_medians()
    for key in gbm_series:
        store, dept = int(key[1:3]), int(key[5:7])
        _, rows, _ = ds.gbm_forecast_rows(store, dept)
        for week_ahead in (1, HORIZON // 2, HORIZON):
            row = rows[week_ahead - 1]
            assert explain.occlusion_attribution(model, row, medians) == \
                _occlusion_reference(model, row, medians), f"{key} week {week_ahead}"


# --------------------------------------------------------------------------
# 2. The per-series row-span index must return what a full scan would.
# --------------------------------------------------------------------------
def test_anomalies_for_matches_full_scan(ds):
    for key in sorted(ds.selected_model)[:12]:
        expected = ds.anomalies[ds.anomalies["series"] == key].sort_values("Date")
        got = ds.anomalies_for(key)
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True), check_like=True
        )


def test_forecasts_for_matches_full_scan(ds):
    for key in sorted(ds.selected_model)[:12]:
        expected = ds.forecasts[ds.forecasts["series"] == key].sort_values("week_ahead")
        got = ds.forecasts_for(key)
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), expected.reset_index(drop=True), check_like=True
        )


def test_lookup_of_unknown_series_is_empty_not_an_error(ds):
    assert ds.anomalies_for("S99_D99").empty
    assert ds.forecasts_for("S99_D99").empty


# --------------------------------------------------------------------------
# 3. The vectorized year-over-year table must equal the per-series loop.
# --------------------------------------------------------------------------
def _yoy_reference(ds):
    """The original O(n_series x n_rows) loop, verbatim."""
    df = ds.anomalies.sort_values(["series", "Date"])
    latest_per_series = df.groupby("series").tail(1)
    rows = []
    for _, row in latest_per_series.iterrows():
        prior = df[(df["series"] == row["series"])
                   & (df["Date"] == row["Date"] - pd.Timedelta(weeks=52))]
        if prior.empty or not prior.iloc[0]["Weekly_Sales"]:
            continue
        pct = 100 * (row["Weekly_Sales"] - prior.iloc[0]["Weekly_Sales"]) \
            / abs(prior.iloc[0]["Weekly_Sales"])
        rows.append({"series": row["series"], "pct_change_yoy": round(pct, 1),
                     "latest_value": round(float(row["Weekly_Sales"]), 2)})
    return pd.DataFrame(rows)


def test_yoy_table_matches_per_series_loop(ds):
    pd.testing.assert_frame_equal(
        ds.yoy_table().reset_index(drop=True),
        _yoy_reference(ds).reset_index(drop=True),
    )


@pytest.mark.parametrize("direction", ["up", "down"])
def test_top_movers_ordering_unchanged(ds, direction):
    """Ranking is what a user actually sees, so pin it directly too."""
    ref = _yoy_reference(ds).sort_values("pct_change_yoy",
                                         ascending=(direction == "down"),
                                         kind="mergesort")
    got = agent_mod.TOOLS["top_movers"](direction=direction, n=5)
    assert got["movers"] == ref.head(5).to_dict("records")


# --------------------------------------------------------------------------
# 4. Caching must not make repeated calls disagree with each other.
# --------------------------------------------------------------------------
def test_cached_recursion_is_stable_and_shared_across_week_aheads(ds, gbm_series):
    key = gbm_series[0]
    store, dept = int(key[1:3]), int(key[5:7])
    first = ds.gbm_forecast_rows(store, dept)
    second = ds.gbm_forecast_rows(store, dept)
    assert first is second, "second call should hit the cache, not recompute"
    preds, rows, _ = first
    assert len(rows) == HORIZON == len(preds)
    # Every week_ahead must come out of this one recursion.
    for week_ahead in range(1, HORIZON + 1):
        out = agent_mod.TOOLS["explain_forecast_drivers"](
            store=store, dept=dept, week_ahead=week_ahead)
        assert out["week_ahead"] == week_ahead
        assert out["baseline_prediction"] == pytest.approx(round(float(preds[week_ahead - 1]), 2))


def test_missing_history_is_cached_as_an_error_not_a_crash(ds):
    assert ds.gbm_forecast_rows(999, 999) is None
    assert ds.gbm_forecast_rows(999, 999) is None  # still None on the cached path


# --------------------------------------------------------------------------
# 5. The native thread pools must be pinned before sklearn loads.
# --------------------------------------------------------------------------
def test_server_pins_native_thread_pools():
    import os
    import app.server  # noqa: F401  (import is what applies the setting)
    assert os.environ["OMP_NUM_THREADS"] == "1"
