"""Central place that loads the (synthetic) retail dataset.

Swap in real data by dropping matching CSVs in data/ -- see
data/generate_data.py's module docstring for the exact schema.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "retail_sales.csv", parse_dates=["Date"])
    return df


def load_store_meta() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "store_meta.csv")


def load_macro() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "macro_features.csv", parse_dates=["Date"])


def load_all_merged() -> pd.DataFrame:
    sales = load_sales()
    store_meta = load_store_meta()
    macro = load_macro().drop(columns=["IsHoliday"])
    df = sales.merge(store_meta, on="Store", how="left")
    df = df.merge(macro, on="Date", how="left")
    return df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)


def series_key(store: int, dept: int) -> str:
    return f"S{store:02d}_D{dept:02d}"
