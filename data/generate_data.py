"""
Synthetic-but-statistically-realistic retail sales data generator.

IMPORTANT / DATA PROVENANCE
----------------------------
This dataset is SYNTHETIC. It is generated, not scraped or downloaded. It is
deliberately modeled on the schema and statistical character of the real,
widely-used "Walmart Recruiting - Store Sales Forecasting" Kaggle dataset
(45 stores, weekly sales by department, IsHoliday flag, MarkDown promo
events, CPI/Unemployment/Fuel_Price macro features). We could not download
the real Kaggle/UCI/FRED files from inside this build environment (outbound
network to those hosts is blocked here), so this generator reproduces the
same *shape* of problem with known ground truth we can use to validate the
pipeline honestly.

To swap in the REAL Walmart dataset (or M5, Favorita, etc.) later:
  1. Download train.csv / features.csv / stores.csv from Kaggle yourself.
  2. Rename columns to match the schema below (Store, Dept, Date,
     Weekly_Sales, IsHoliday, MarkDown1..5, CPI, Unemployment, Fuel_Price,
     Temperature, Type, Size).
  3. Drop the file in data/retail_sales.csv and data/store_meta.csv.
  4. Nothing else in src/ needs to change -- the pipeline is schema-driven,
     not generator-driven.

Ground truth this generator embeds (used later to check the pipeline caught
the right things):
  - Annual seasonality + named US retail holidays (Super Bowl, Labor Day,
    Thanksgiving, Christmas) with department-specific holiday lift.
  - Slow per-store trend (growth or decline).
  - Five MarkDown promo columns, active in bursts, which lift sales while
    active (an EXPECTED, explainable spike -- not an anomaly).
  - A COVID-like demand shock in Mar-Apr 2020: apparel/electronics/dept-
    store categories crash, grocery/home-improvement categories spike --
    a regime shift that differs by department, not a single global scalar.
  - A localized, unexplained supply-disruption dip in one store/dept pair.
  - A handful of data-entry-error rows (negative sales / spikes) scattered
    at random -- the kind of garbage real POS extracts always contain.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 20240101
DATA_DIR = Path(__file__).parent

N_STORES = 20
N_DEPTS = 12
START_DATE = "2019-01-06"   # a Sunday, weekly cadence like the real dataset
N_WEEKS = 260               # 5 years -> enough history for backtesting + a COVID window

STORE_TYPES = ["A", "B", "C"]


def _us_holidays_for_weekending(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Flag the same 4 holiday weeks used in the real Walmart dataset."""
    years = sorted(set(dates.year))
    holiday_weeks = []
    for y in years:
        # Super Bowl: first Sunday in Feb (approx week containing it)
        holiday_weeks.append(pd.Timestamp(f"{y}-02-07"))
        # Labor Day: first Monday in Sept -> week ending that Sunday
        holiday_weeks.append(pd.Timestamp(f"{y}-09-10"))
        # Thanksgiving: 4th Thursday Nov
        holiday_weeks.append(pd.Timestamp(f"{y}-11-29"))
        # Christmas
        holiday_weeks.append(pd.Timestamp(f"{y}-12-25"))
    flags = pd.DataFrame({"Date": dates})
    flags["IsHoliday"] = flags["Date"].apply(
        lambda d: any(abs((d - h).days) <= 6 for h in holiday_weeks)
    )
    flags["holiday_name"] = flags["Date"].apply(
        lambda d: next(
            (name for name, h in zip(
                ["SuperBowl", "LaborDay", "Thanksgiving", "Christmas"] * len(years),
                holiday_weeks) if abs((d - h).days) <= 6),
            None,
        )
    )
    return flags


def generate():
    rng = np.random.default_rng(RNG_SEED)
    dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-SUN")
    holiday_flags = _us_holidays_for_weekending(dates).set_index("Date")

    # ---- store metadata ----
    store_ids = np.arange(1, N_STORES + 1)
    store_type = rng.choice(STORE_TYPES, size=N_STORES, p=[0.35, 0.4, 0.25])
    size_base = {"A": 175000, "B": 120000, "C": 70000}
    store_size = np.array([size_base[t] * rng.uniform(0.85, 1.15) for t in store_type]).astype(int)
    store_meta = pd.DataFrame({"Store": store_ids, "Type": store_type, "Size": store_size})

    # per-store slow trend (annual growth rate), mostly positive, a few declining stores
    store_trend = rng.normal(loc=0.02, scale=0.05, size=N_STORES)  # ~2%/yr average

    # dept baseline weekly demand (before store scaling) + holiday sensitivity + covid sensitivity
    dept_ids = np.arange(1, N_DEPTS + 1)
    dept_base_demand = rng.uniform(8000, 45000, size=N_DEPTS)
    dept_holiday_lift = rng.uniform(0.05, 0.9, size=N_DEPTS)     # how much this dept spikes on holidays
    # covid_sensitivity: negative = crashes during covid (apparel/electronics-like),
    # positive = spikes during covid (grocery/home-like)
    dept_covid_sensitivity = rng.uniform(-0.65, 0.55, size=N_DEPTS)
    rng.shuffle(dept_covid_sensitivity)

    # macro features (shared across stores, weekly)
    t = np.arange(N_WEEKS)
    cpi = 210 + 0.09 * t + rng.normal(0, 0.4, N_WEEKS).cumsum() * 0.02
    unemployment = 6.0 + 2.2 * np.exp(-((t - 62) ** 2) / (2 * 10 ** 2)) + rng.normal(0, 0.08, N_WEEKS)
    fuel_price = 2.6 + 0.5 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 0.08, N_WEEKS).cumsum() * 0.01
    temperature = 55 + 25 * np.sin(2 * np.pi * (t - 12) / 52) + rng.normal(0, 4, N_WEEKS)

    covid_start = pd.Timestamp("2020-03-08")
    covid_crash_end = pd.Timestamp("2020-05-03")
    covid_recover_end = pd.Timestamp("2020-08-30")

    rows = []
    markdown_rows = []
    # 5 markdown "campaigns" per store/dept across the whole history, each lasting 2-4 weeks
    for s in store_ids:
        for d in dept_ids:
            n_campaigns = rng.integers(4, 9)
            campaign_starts = rng.choice(dates[10:-10], size=n_campaigns, replace=False)
            for cs in campaign_starts:
                dur = rng.integers(2, 5)
                for md_col in rng.choice(range(1, 6), size=rng.integers(1, 4), replace=False):
                    markdown_rows.append({
                        "Store": s, "Dept": d, "start": cs,
                        "duration": dur, "markdown_col": md_col,
                        "amount": rng.uniform(1000, 9000),
                    })
    markdown_df = pd.DataFrame(markdown_rows)

    # index markdowns for fast lookup: (store,dept,date) -> {col: amount}
    md_lookup = {}
    for _, r in markdown_df.iterrows():
        for w in range(int(r["duration"])):
            key = (int(r["Store"]), int(r["Dept"]), r["start"] + pd.Timedelta(weeks=w))
            md_lookup.setdefault(key, {})[int(r["markdown_col"])] = r["amount"]

    # a genuinely unexplained localized anomaly: supply disruption for one store/dept
    disruption_store, disruption_dept = int(store_ids[3]), int(dept_ids[7])
    disruption_start = dates[140]
    disruption_len = 5

    for si, s in enumerate(store_ids):
        size_factor = store_size[si] / store_size.mean()
        for di, d in enumerate(dept_ids):
            base = dept_base_demand[di] * size_factor
            for wi, dt in enumerate(dates):
                # trend
                level = base * (1 + store_trend[si]) ** (wi / 52)
                # annual seasonality
                season = 1 + 0.18 * np.sin(2 * np.pi * (wi + 10) / 52)
                # holiday lift
                is_hol = bool(holiday_flags.loc[dt, "IsHoliday"])
                hol_name = holiday_flags.loc[dt, "holiday_name"]
                hol_mult = 1.0
                if is_hol:
                    lift = dept_holiday_lift[di]
                    if hol_name in ("Thanksgiving", "Christmas"):
                        hol_mult = 1 + lift * 1.4
                    else:
                        hol_mult = 1 + lift * 0.5

                # markdown lift (expected, explainable spike)
                mds = md_lookup.get((int(s), int(d), dt), {})
                md_mult = 1 + 0.06 * len(mds)

                # covid regime shift
                covid_mult = 1.0
                if covid_start <= dt <= covid_crash_end:
                    covid_mult = 1 + dept_covid_sensitivity[di] * 1.6
                elif covid_crash_end < dt <= covid_recover_end:
                    # partial recovery, damped
                    frac = (dt - covid_crash_end).days / (covid_recover_end - covid_crash_end).days
                    covid_mult = 1 + dept_covid_sensitivity[di] * 1.6 * (1 - frac)

                noise = rng.normal(1.0, 0.05)
                sales = level * season * hol_mult * md_mult * covid_mult * noise
                sales = max(sales, 500)

                # localized unexplained disruption
                is_disruption = (
                    s == disruption_store and d == disruption_dept
                    and disruption_start <= dt < disruption_start + pd.Timedelta(weeks=disruption_len)
                )
                if is_disruption:
                    sales *= 0.25

                # rare data-entry error rows (returns-heavy weeks going negative, or fat-finger spikes)
                err_roll = rng.random()
                if err_roll < 0.0015:
                    sales = -abs(sales) * rng.uniform(0.05, 0.3)
                elif err_roll < 0.003:
                    sales *= rng.uniform(3, 5)

                row = {
                    "Store": int(s), "Dept": int(d), "Date": dt,
                    "Weekly_Sales": round(float(sales), 2),
                    "IsHoliday": is_hol,
                }
                for col in range(1, 6):
                    row[f"MarkDown{col}"] = round(mds.get(col, 0.0), 2)
                rows.append(row)

    sales_df = pd.DataFrame(rows)

    features_df = pd.DataFrame({
        "Date": dates,
        "Temperature": temperature.round(1),
        "Fuel_Price": fuel_price.round(3),
        "CPI": cpi.round(3),
        "Unemployment": unemployment.round(2),
        "IsHoliday": holiday_flags["IsHoliday"].values,
    })

    sales_df.to_csv(DATA_DIR / "retail_sales.csv", index=False)
    store_meta.to_csv(DATA_DIR / "store_meta.csv", index=False)
    features_df.to_csv(DATA_DIR / "macro_features.csv", index=False)

    meta = {
        "disruption_store": disruption_store,
        "disruption_dept": disruption_dept,
        "disruption_start": str(disruption_start.date()),
        "disruption_weeks": disruption_len,
        "covid_start": str(covid_start.date()),
        "covid_crash_end": str(covid_crash_end.date()),
        "covid_recover_end": str(covid_recover_end.date()),
        "n_stores": N_STORES,
        "n_depts": N_DEPTS,
        "n_weeks": N_WEEKS,
        "n_rows": len(sales_df),
    }
    pd.Series(meta).to_json(DATA_DIR / "ground_truth_events.json", indent=2)
    print(f"Generated {len(sales_df):,} rows across {N_STORES} stores x {N_DEPTS} depts x {N_WEEKS} weeks")
    print("Ground truth events:", meta)
    return sales_df, store_meta, features_df


if __name__ == "__main__":
    generate()
