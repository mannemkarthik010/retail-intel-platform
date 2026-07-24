"""Feature engineering for the global gradient-boosted demand model.

We build one flat table across ALL (store, dept) series -- a single global
model trained on lag/rolling/calendar features with store & dept as
categorical context, rather than one model per series. This mirrors how
production forecasting systems at retail scale (M5-style) actually work:
a global model generalizes across sparse/intermittent series far better
than thousands of tiny per-series models, and it's one artifact to deploy,
monitor, and retrain instead of thousands.
"""
import numpy as np
import pandas as pd

LAGS = [1, 2, 4, 8, 52]
ROLLING_WINDOWS = [4, 8, 12]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["woy_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


def add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    md_cols = [c for c in df.columns if c.startswith("MarkDown")]
    df["markdown_total"] = df[md_cols].sum(axis=1)
    df["markdown_active_count"] = (df[md_cols] > 0).sum(axis=1)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must be called on a df sorted by [Store, Dept, Date]."""
    df = df.copy()
    grp_keys = ["Store", "Dept"]
    grp = df.groupby(grp_keys, sort=False)["Weekly_Sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)
    # shift(1) first so the rolling window never sees the current target row
    df["_shifted"] = grp.shift(1)
    shifted_grp = df.groupby(grp_keys, sort=False)["_shifted"]
    for w in ROLLING_WINDOWS:
        df[f"rollmean_{w}"] = shifted_grp.transform(lambda s: s.rolling(w).mean())
        df[f"rollstd_{w}"] = shifted_grp.transform(lambda s: s.rolling(w).std())
    df = df.drop(columns=["_shifted"])
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_markdown_features(df)
    df = add_lag_and_rolling_features(df)
    return df


FEATURE_COLUMNS = (
    ["Store", "Dept", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
     "IsHoliday", "weekofyear", "month", "woy_sin", "woy_cos",
     "markdown_total", "markdown_active_count"]
    + [f"lag_{l}" for l in LAGS]
    + [f"rollmean_{w}" for w in ROLLING_WINDOWS]
    + [f"rollstd_{w}" for w in ROLLING_WINDOWS]
)
CATEGORICAL_COLUMNS = ["Store", "Dept"]
TARGET_COLUMN = "Weekly_Sales"
