"""Thin serving layer (Flask, since FastAPI could not be installed in the
build sandbox -- see docs/ARCHITECTURE.md for the note; swapping frameworks
would not touch src/ at all, only this file).

This layer NEVER trains or re-scores models live. It reads the artifacts
`scripts/run_pipeline.py` produced (forecasts, backtest metrics, anomaly
flags) and answers requests against them -- the same separation of
"offline batch scoring" vs. "online serving" a real production system uses,
which is what keeps p99 latency low and predictable regardless of how
expensive the modeling is.
"""
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import agent as agent_mod  # noqa: E402
from src.agent import TOOLS, _store  # noqa: E402

app = Flask(__name__)


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/series")
def list_series():
    ds = _store()
    pairs = ds.forecasts[["Store", "Dept"]].drop_duplicates().sort_values(["Store", "Dept"])
    return jsonify(pairs.to_dict("records"))


@app.get("/api/forecast")
def forecast():
    store = int(request.args.get("store", 1))
    dept = int(request.args.get("dept", 1))
    return jsonify(TOOLS["get_forecast"](store=store, dept=dept))


@app.get("/api/anomalies")
def anomalies():
    store = int(request.args.get("store", 1))
    dept = int(request.args.get("dept", 1))
    only_flagged = request.args.get("only_flagged", "true") == "true"
    return jsonify(TOOLS["get_anomalies"](store=store, dept=dept, only_needs_investigation=only_flagged, limit=25))


@app.get("/api/history")
def history():
    store = int(request.args.get("store", 1))
    dept = int(request.args.get("dept", 1))
    weeks = int(request.args.get("weeks", 104))
    ds = _store()
    key = ds.series_key(store, dept)
    df = ds.anomalies[ds.anomalies["series"] == key].sort_values("Date").tail(weeks)
    return jsonify([
        {"date": str(r["Date"].date()), "value": round(float(r["Weekly_Sales"]), 2),
         "anomaly_type": r["anomaly_type"], "needs_investigation": bool(r["needs_investigation"])}
        for _, r in df.iterrows()
    ])


@app.get("/api/eval/summary")
def eval_summary():
    ds = _store()
    by_model = ds.series_summary.groupby("model")["wape"].mean().round(4)
    wins = {}
    for series, model in ds.selected_model.items():
        wins[model] = wins.get(model, 0) + 1
    return jsonify({
        "mean_wape_by_model": by_model.to_dict(),
        "series_wins_by_model": wins,
        "n_series": len(ds.selected_model),
    })


@app.post("/api/ask")
def ask():
    payload = request.get_json(force=True)
    question = payload.get("question", "")
    if not question:
        return jsonify({"error": "missing 'question'"}), 400
    trace = agent_mod.ask(question)
    return jsonify(trace.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
