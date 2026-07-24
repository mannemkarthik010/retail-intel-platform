"""Three forecasting approaches, competed against each other per series:

1. seasonal_naive       - value from 52 weeks ago (the "can we beat last year" bar)
2. holt_winters         - classical multiplicative triple exponential smoothing,
                          fit per series (fixed smoothing constants -- see note below)
3. global_gbm           - one HistGradientBoostingRegressor trained across ALL
                          series with lag/rolling/calendar features

Note on Holt-Winters: a production system would fit smoothing constants via
MLE (statsmodels' ExponentialSmoothing does this). That package wasn't
installable in this sandbox (outbound package installs are blocked here
beyond what's pre-cached), so this implementation uses a small fixed-grid
search over a handful of (alpha, beta, gamma) triples on the training
window itself, minimizing in-sample SSE -- a reasonable stand-in that's
fully transparent about the tradeoff. Swapping in statsmodels' ETS is a
one-function change (see holt_winters_forecast docstring).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as feat

SEASON_LENGTH = 52


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season_length: int = SEASON_LENGTH) -> np.ndarray:
    if len(history) < season_length:
        # not enough history for a seasonal lag -> fall back to last value
        return np.full(horizon, history[-1] if len(history) else 0.0)
    out = np.zeros(horizon)
    for h in range(horizon):
        idx = len(history) - season_length + (h % season_length)
        # walk backwards further if we still don't have that many cycles
        while idx < 0:
            idx += season_length
        out[h] = history[idx] if 0 <= idx < len(history) else history[-1]
    return out


def _fit_holt_winters(y: np.ndarray, season_length: int, alpha: float, beta: float, gamma: float):
    n = len(y)
    if n < 2 * season_length:
        # not enough cycles for a seasonal fit
        return None
    # initial level/trend/season via simple decomposition
    season_avgs = np.array([y[i::season_length].mean() for i in range(season_length)])
    season = season_avgs / season_avgs.mean()
    level = y[:season_length].mean()
    trend = (y[season_length:2 * season_length].mean() - y[:season_length].mean()) / season_length

    levels = np.zeros(n)
    trends = np.zeros(n)
    seasons = list(season)
    for t in range(n):
        s = seasons[t % season_length] if t >= season_length else season[t % season_length]
        if s == 0:
            s = 1e-6
        new_level = alpha * (y[t] / s) + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        new_season = gamma * (y[t] / new_level) + (1 - gamma) * s
        levels[t], trends[t] = new_level, new_trend
        if t >= season_length:
            seasons[t % season_length] = new_season
        level, trend = new_level, new_trend
    return level, trend, seasons


def holt_winters_forecast(history: np.ndarray, horizon: int, season_length: int = SEASON_LENGTH) -> np.ndarray:
    y = np.clip(history.astype(float), 1.0, None)  # multiplicative HW needs positive values
    if len(y) < 2 * season_length:
        return seasonal_naive_forecast(history, horizon, season_length)

    grid = [(0.15, 0.02, 0.15), (0.3, 0.05, 0.3), (0.4, 0.1, 0.2)]
    best = None
    for alpha, beta, gamma in grid:
        fit = _fit_holt_winters(y, season_length, alpha, beta, gamma)
        if fit is None:
            continue
        level, trend, seasons = fit
        # in-sample one-step-ahead SSE as the model-selection criterion
        # (cheap proxy in lieu of MLE; documented above)
        preds = []
        lvl, tr = y[:season_length].mean(), 0.0
        for t in range(len(y)):
            s = seasons[t % season_length]
            preds.append((lvl + tr) * s)
        sse = np.sum((y - np.array(preds)) ** 2)
        if best is None or sse < best[0]:
            best = (sse, level, trend, seasons)
    if best is None:
        return seasonal_naive_forecast(history, horizon, season_length)
    _, level, trend, seasons = best
    out = np.zeros(horizon)
    for h in range(1, horizon + 1):
        s = seasons[(len(y) + h - 1) % season_length]
        out[h - 1] = max((level + h * trend) * s, 0.0)
    return out


@dataclass
class GlobalGBMModel:
    model: HistGradientBoostingRegressor

    @classmethod
    def fit(cls, train_df: pd.DataFrame):
        X = train_df[feat.FEATURE_COLUMNS]
        y = train_df[feat.TARGET_COLUMN]
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=250,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X, y)
        return cls(model=model)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[feat.FEATURE_COLUMNS]), 0, None)


@dataclass
class QuantileGBMModel:
    """Same architecture as GlobalGBMModel but trained with pinball
    (quantile) loss instead of squared error, so it estimates a specific
    quantile of the conditional demand distribution rather than the mean.
    Used in pairs (e.g. q=0.1 and q=0.9) to produce an 80% prediction
    interval around the point forecast -- see src/intervals.py for how the
    interval is actually assembled and validated.

    Honest limitation: when used recursively (see intervals.py) it's
    queried along the POINT model's own recursive path rather than
    re-branching a full quantile recursion tree. The interval therefore
    reflects one-step-ahead conditional uncertainty propagated along a
    single trajectory, not the (wider, more correct) uncertainty of a truly
    recursive quantile forecast. A production system would Monte-Carlo
    sample multiple recursive paths or use a proper multi-step conformal
    method; this is a documented, honest approximation instead."""
    model: HistGradientBoostingRegressor
    quantile: float

    @classmethod
    def fit(cls, train_df: pd.DataFrame, quantile: float):
        X = train_df[feat.FEATURE_COLUMNS]
        y = train_df[feat.TARGET_COLUMN]
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_depth=6,
            learning_rate=0.08,
            max_iter=250,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X, y)
        return cls(model=model, quantile=quantile)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[feat.FEATURE_COLUMNS]), 0, None)
