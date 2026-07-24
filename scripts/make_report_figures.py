"""Generates the static PNG figures used in EVAL_REPORT.md and the polished
Word/PDF report -- all computed from the actual pipeline artifacts in
reports/, never hand-drawn or faked."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(exist_ok=True, parents=True)

# palette (dataviz skill categorical order, light mode)
BLUE, ORANGE, AQUA, RED, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def fig_wape_by_model():
    ss = pd.read_csv(REPORTS / "series_summary.csv")
    means = ss.groupby("model")["wape"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [BLUE, ORANGE, AQUA]
    bars = ax.bar(means.index, means.values, color=colors[:len(means)], width=0.55, zorder=3)
    for b, v in zip(bars, means.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}%", ha="center", fontsize=10, color="#0b0b0b")
    ax.set_ylabel("Mean WAPE (%) across 6 rolling-origin cutoffs, 240 series")
    ax.set_title("Backtested accuracy by model (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "wape_by_model.png", dpi=150)
    plt.close(fig)


def fig_series_wins():
    selected = json.loads((REPORTS / "selected_model.json").read_text())
    counts = pd.Series(list(selected.values())).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"global_gbm": BLUE, "seasonal_naive": ORANGE, "holt_winters": AQUA}
    ax.bar(counts.index, counts.values, color=[colors.get(c, MUTED) for c in counts.index], width=0.55, zorder=3)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10)
    ax.set_ylabel("# of series (out of 240) where this model won")
    ax.set_title("Per-series model selection outcome")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "series_wins.png", dpi=150)
    plt.close(fig)


def fig_example_forecast(store=3, dept=5):
    from src.agent import _store
    ds = _store()
    key = ds.series_key(store, dept)
    hist = ds.anomalies[ds.anomalies["series"] == key].sort_values("Date").tail(60)
    fc = ds.forecasts[ds.forecasts["series"] == key].sort_values("week_ahead")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(pd.to_datetime(hist["Date"]), hist["Weekly_Sales"], color=BLUE, linewidth=1.8, label="Actuals")
    flagged = hist[hist["needs_investigation"]]
    ax.scatter(pd.to_datetime(flagged["Date"]), flagged["Weekly_Sales"], color=RED, zorder=5, s=35, label="Flagged anomaly")

    last_date = pd.to_datetime(hist["Date"]).max()
    fc_dates = pd.to_datetime(fc["forecast_date"])
    bridge_dates = pd.concat([pd.Series([last_date]), fc_dates])
    bridge_vals = pd.concat([pd.Series([hist["Weekly_Sales"].iloc[-1]]), fc["forecast_value"]])
    ax.plot(bridge_dates, bridge_vals, color=ORANGE, linewidth=1.8, linestyle="--", label="Forecast (next 8 weeks)")

    ax.set_title(f"Store {store} / Dept {dept} -- last 60 weeks + forecast (model: {fc['model_used'].iloc[0]})")
    ax.set_ylabel("Weekly sales ($)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "example_forecast.png", dpi=150)
    plt.close(fig)


def fig_covid_regime_shift():
    from src.data_io import load_all_merged
    raw = load_all_merged()
    covid = raw[(raw.Date >= "2020-01-01") & (raw.Date <= "2020-12-31")]
    by_dept = covid.groupby(["Dept", "Date"])["Weekly_Sales"].sum().reset_index()
    totals = covid.groupby("Dept")["Weekly_Sales"].sum()
    up_dept = totals.idxmax()
    down_dept = totals.idxmin()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for dept, color, label in [(up_dept, AQUA, f"Dept {up_dept} (COVID-era spike)"),
                                (down_dept, RED, f"Dept {down_dept} (COVID-era crash)")]:
        sub = by_dept[by_dept.Dept == dept].sort_values("Date")
        norm = sub["Weekly_Sales"] / sub["Weekly_Sales"].iloc[:4].mean()
        ax.plot(pd.to_datetime(sub["Date"]), norm, color=color, linewidth=2, label=label)
    ax.axvspan(pd.Timestamp("2020-03-08"), pd.Timestamp("2020-05-03"), color=MUTED, alpha=0.15, label="COVID shock window")
    ax.axhline(1.0, color=GRID, linewidth=1)
    ax.set_ylabel("Demand indexed to Jan 2020 = 1.0")
    ax.set_title("Same shock, opposite department-level response (all stores summed)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "covid_regime_shift.png", dpi=150)
    plt.close(fig)


def fig_monitoring_fleet():
    fleet = pd.read_csv(REPORTS / "monitoring_fleet_log.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pd.to_datetime(fleet["cutoff"]), fleet["wape"] * 100, marker="o", color=BLUE, linewidth=1.8)
    ax.set_ylabel("Fleet mean WAPE (%) -- deployed model per series")
    ax.set_title("Production monitoring: accuracy of the DEPLOYED model over time")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "monitoring_fleet.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_wape_by_model()
    fig_series_wins()
    fig_example_forecast()
    fig_covid_regime_shift()
    fig_monitoring_fleet()
    print("Wrote figures to", FIG)
    for f in sorted(FIG.glob("*.png")):
        print(" -", f.name)
