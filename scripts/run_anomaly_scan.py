"""Run anomaly detection across every (Store, Dept) series and validate
against the generator's embedded ground-truth events (data/ground_truth_events.json).
This is the honesty check: does the detector actually catch what we know is there?
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_io import load_all_merged, series_key
from src.features import add_markdown_features
from src.anomaly import detect_anomalies_for_series, isolation_forest_crosscheck

ROOT = Path(__file__).parent.parent


def main():
    raw = load_all_merged()
    raw = add_markdown_features(raw)
    truth = json.loads((ROOT / "data" / "ground_truth_events.json").read_text())

    all_results = []
    for (store, dept), g in raw.groupby(["Store", "Dept"]):
        res = detect_anomalies_for_series(g[["Date", "Weekly_Sales", "markdown_active_count", "IsHoliday"]])
        res["Store"], res["Dept"] = store, dept
        res["series"] = series_key(store, dept)
        all_results.append(res)
    all_df = pd.concat(all_results, ignore_index=True)
    all_df = isolation_forest_crosscheck(all_df)

    out_path = ROOT / "reports" / "anomaly_flags.csv"
    all_df.to_csv(out_path, index=False)
    print(f"Wrote {len(all_df):,} rows to {out_path}")

    print("\n=== Anomaly type counts (statistical layer) ===")
    print(all_df["anomaly_type"].value_counts())
    print(f"\nHigh-confidence anomalies (statistical AND IsolationForest agree): "
          f"{int(all_df['high_confidence_anomaly'].sum())}")

    # --- validation against embedded ground truth ---
    print("\n=== Validation against embedded ground-truth events ===")

    # 1. localized disruption
    ds, dd = truth["disruption_store"], truth["disruption_dept"]
    dstart = pd.Timestamp(truth["disruption_start"])
    dend = dstart + pd.Timedelta(weeks=truth["disruption_weeks"])
    sub = all_df[(all_df.Store == ds) & (all_df.Dept == dd)
                 & (all_df.Date >= dstart) & (all_df.Date < dend)]
    caught = sub["needs_investigation"].sum()
    print(f"Localized disruption (Store {ds}, Dept {dd}, {dstart.date()} "
          f"+{truth['disruption_weeks']}wk): {caught}/{len(sub)} weeks flagged as needing investigation")

    # 2. data-entry errors (negative sales)
    neg = all_df[all_df["is_negative_sales"]]
    print(f"Data-entry error rows (negative sales) in dataset: {len(neg)}, "
          f"all flagged as data_quality_error: {(neg['anomaly_type']=='data_quality_error').all()}")

    # 3. COVID window -- how many series flagged in the crash window
    covid_start, covid_end = pd.Timestamp(truth["covid_start"]), pd.Timestamp(truth["covid_crash_end"])
    covid_window = all_df[(all_df.Date >= covid_start) & (all_df.Date <= covid_end)]
    n_series_flagged = covid_window[covid_window["needs_investigation"]]["series"].nunique()
    n_series_total = all_df["series"].nunique()
    print(f"COVID crash window ({covid_start.date()} to {covid_end.date()}): "
          f"{n_series_flagged}/{n_series_total} series had at least one flagged week")

    # 4. promo spikes correctly marked "explained" not "unexplained"
    promo_flagged = all_df[all_df["anomaly_type"] == "explained_promo_spike"]
    print(f"Promo-driven spikes correctly separated out as 'explained', not flagged for investigation: "
          f"{len(promo_flagged)}")


if __name__ == "__main__":
    main()
