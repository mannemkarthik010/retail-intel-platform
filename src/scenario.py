"""What-if scenario simulation for the agent's `simulate_scenario` tool.

Every OTHER tool in this project (get_forecast, get_anomalies,
explain_change, top_movers, explain_forecast_drivers) is a pure read
against artifacts scripts/run_pipeline.py already wrote offline -- the
online agent never re-trains or re-scores live (see docs/ARCHITECTURE.md
for why that split matters). A what-if is the one deliberate exception:
by definition it asks about a covariate combination ("what if a markdown
promotion were running these next few weeks?") that was never scored
offline, so there is nothing to look up. It reuses the exact point GBM
`scripts/run_pipeline.py` already persisted (models/gbm_point.joblib)
rather than retraining anything, so the model itself is identical to the
one that produced the baseline forecast -- only the hypothetical future
covariates differ.

Only meaningful for series whose SELECTED (winning) model is global_gbm:
seasonal_naive and Holt-Winters have no markdown/holiday covariate slots
at all, so a "what if we ran a promo" question on those series would be a
silent no-op. The agent tool checks this and says so rather than
fabricating a number (see src/agent.py::tool_simulate_scenario).
"""
from pathlib import Path

import joblib
import pandas as pd

from .backtest import _recursive_gbm_forecast
from .features import add_calendar_features, add_markdown_features

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "reports" / "models"

_POINT_MODEL = None


def _point_model():
    global _POINT_MODEL
    if _POINT_MODEL is None:
        _POINT_MODEL = joblib.load(MODELS_DIR / "gbm_point.joblib")
    return _POINT_MODEL


def _build_hypothetical_covariates(raw: pd.DataFrame, store: int, dept: int, weeks: int,
                                    markdown_active: bool, is_holiday):
    """Macro covariates (temperature/fuel/CPI/unemployment) are held at
    their last-observed value -- the SAME simplifying assumption
    scripts/run_pipeline.py::make_future_covariates makes for the real
    baseline forecast, so the only thing that differs between baseline and
    scenario is the two covariates this function actually varies.

    `is_holiday=None` means "don't override -- use the real calendar",
    exactly like the baseline forecast does. This matters: an earlier
    version of this function defaulted IsHoliday to 0 whenever the caller
    only wanted to ask about a markdown, which silently ALSO stripped out
    a real holiday flag on weeks that genuinely are holidays, producing a
    misleading "markdown made sales drop" result that was actually just
    "we accidentally removed Thanksgiving." `is_holiday=True/False` still
    lets a caller explicitly force the flag either way."""
    hist = raw[(raw.Store == store) & (raw.Dept == dept)].sort_values("Date")
    if hist.empty:
        return None, None
    last_date = hist["Date"].max()
    last_row = hist.iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=weeks, freq="W-SUN")

    cov = pd.DataFrame({"Date": future_dates})
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        cov[col] = last_row[col]
    if is_holiday is None:
        from data.generate_data import _us_holidays_for_weekending
        hflags = _us_holidays_for_weekending(future_dates).set_index("Date")
        cov["IsHoliday"] = hflags.reindex(future_dates)["IsHoliday"].fillna(0).astype(int).values
    else:
        cov["IsHoliday"] = 1 if is_holiday else 0
    for c in [f"MarkDown{i}" for i in range(1, 6)]:
        cov[c] = 0.0
    if markdown_active:
        # A representative "one promotion running" magnitude, not a
        # specific real markdown value -- this is a hypothetical, not a
        # calendar lookup, so an illustrative constant is honest here.
        cov["MarkDown1"] = 1000.0
    cov["Store"], cov["Dept"] = store, dept
    cov["Size"] = last_row["Size"]
    cov = add_calendar_features(cov)
    cov = add_markdown_features(cov)
    return hist, cov


def simulate_scenario(raw: pd.DataFrame, store: int, dept: int, markdown_active: bool,
                       is_holiday, weeks: int):
    """`is_holiday`: None (default, preserve the real calendar), or an
    explicit True/False to override it -- see _build_hypothetical_covariates."""
    hist, cov = _build_hypothetical_covariates(raw, store, dept, weeks, markdown_active, is_holiday)
    if hist is None:
        return None
    model = _point_model()
    return _recursive_gbm_forecast(model, hist, cov, weeks)
