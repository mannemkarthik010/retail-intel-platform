"""Simulates ongoing production monitoring using the rolling-origin backtest
cutoffs as a stand-in for "a new week of ground truth arrived" -- for each
series we only look at ITS OWN selected/winning model's accuracy (not all
three), which is what a real monitoring dashboard would show: has the model
we actually deployed for this series drifted?

Flags:
  - fleet-level drift: mean WAPE across all series at the latest cutoff is
    more than DRIFT_MULTIPLIER x the mean of prior cutoffs
  - series-level drift: an individual series' WAPE jumped by more than
    SERIES_DRIFT_DELTA (absolute) vs. its own prior-cutoff average -- these
    are the ones you'd actually page someone about / queue for retraining
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"

DRIFT_MULTIPLIER = 1.35
SERIES_DRIFT_DELTA = 0.08  # 8 absolute points of WAPE


def main():
    per_cutoff = pd.read_csv(REPORTS / "per_cutoff_metrics.csv")
    selected_model = json.loads((REPORTS / "selected_model.json").read_text())

    per_cutoff["is_selected_model"] = per_cutoff.apply(
        lambda r: selected_model.get(r["series"]) == r["model"], axis=1
    )
    deployed = per_cutoff[per_cutoff["is_selected_model"]].copy()
    deployed["cutoff"] = pd.to_datetime(deployed["cutoff"])
    deployed = deployed.sort_values("cutoff")

    fleet = deployed.groupby("cutoff")["wape"].mean().reset_index()
    fleet["prior_avg"] = fleet["wape"].expanding().mean().shift(1)
    fleet["fleet_drift_flag"] = fleet["wape"] > DRIFT_MULTIPLIER * fleet["prior_avg"]

    print("=== Fleet-level WAPE by monitoring checkpoint (deployed model per series) ===")
    print(fleet.to_string(index=False))

    # per-series drift: compare each series' latest cutoff vs its own prior average
    series_drift_rows = []
    for series, g in deployed.groupby("series"):
        g = g.sort_values("cutoff")
        if len(g) < 2:
            continue
        latest = g.iloc[-1]
        prior_avg = g.iloc[:-1]["wape"].mean()
        delta = latest["wape"] - prior_avg
        if delta > SERIES_DRIFT_DELTA:
            series_drift_rows.append({
                "series": series, "model": latest["model"],
                "latest_wape": round(float(latest["wape"]), 4),
                "prior_avg_wape": round(float(prior_avg), 4),
                "delta": round(float(delta), 4),
            })
    series_drift = pd.DataFrame(series_drift_rows).sort_values("delta", ascending=False) if series_drift_rows else pd.DataFrame()

    print(f"\n=== Series flagged for retraining review (WAPE jumped >{SERIES_DRIFT_DELTA*100:.0f}pts vs. their own history) ===")
    if series_drift.empty:
        print("(none)")
    else:
        print(series_drift.to_string(index=False))

    fleet.to_csv(REPORTS / "monitoring_fleet_log.csv", index=False)
    series_drift.to_csv(REPORTS / "monitoring_series_alerts.csv", index=False)
    print(f"\nWrote monitoring_fleet_log.csv and monitoring_series_alerts.csv "
          f"({len(series_drift)} series flagged) to reports/")


if __name__ == "__main__":
    main()
