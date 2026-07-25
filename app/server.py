"""Thin serving layer (Flask, since FastAPI could not be installed in the
build sandbox -- see docs/ARCHITECTURE.md for the note; swapping frameworks
would not touch src/ at all, only this file).

This layer NEVER trains or re-scores models live. It reads the artifacts
`scripts/run_pipeline.py` produced (forecasts, backtest metrics, anomaly
flags) and answers requests against them -- the same separation of
"offline batch scoring" vs. "online serving" a real production system uses,
which is what keeps p99 latency low and predictable regardless of how
expensive the modeling is.

Hardening in this file (still not auth/rate-limiting -- see
docs/ARCHITECTURE.md's "what's still a demo" disclosure, which those remain
under): every query param is validated, returning a clean 400 JSON error
instead of an unhandled 500 on bad input; CORS is opt-in via an explicit
allowlist env var rather than wide open; request bodies are size-capped;
every request is logged with method/path/status/duration; and both HTTP
errors and unexpected exceptions are caught and returned as JSON, never
Flask's default HTML error page.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request, abort
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import agent as agent_mod  # noqa: E402
from src.agent import TOOLS, _store  # noqa: E402

app = Flask(__name__)

# Every real payload this API expects (query params, a short JSON question)
# is tiny -- this just caps abuse, not legitimate use.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

# Opt-in CORS via an explicit comma-separated allowlist, e.g.
# CORS_ALLOWED_ORIGINS="https://example.com,https://foo.example.com".
# Unset (the default) means no cross-origin access at all -- the bundled
# dashboard is served same-origin and never needs it; this only matters if
# a separately-hosted frontend calls this API directly.
ALLOWED_ORIGINS = {o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("retail_intel.api")


def _int_arg(name: str, default, minimum: int = None) -> int:
    """Validates a query param as an int, aborting with a clean 400 JSON
    error (instead of letting int() raise and Flask return a bare 500) on
    anything malformed. Only a floor is enforced, not a ceiling -- store/dept
    combos outside the dataset already get a clean {"error": ...} 200 from
    the tool functions themselves (see src/agent.py::tool_get_forecast),
    which is the right response for "well-formed but unknown," as opposed
    to this validating "not even a valid integer.\""""
    raw = request.args.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        abort(400, description=f"'{name}' must be an integer, got {raw!r}")
    if minimum is not None and value < minimum:
        abort(400, description=f"'{name}' must be >= {minimum}, got {value}")
    return value


@app.before_request
def _start_timer():
    g._start_time = time.time()


@app.after_request
def _apply_cors_and_log(response):
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    duration_ms = round((time.time() - getattr(g, "_start_time", time.time())) * 1000, 1)
    logger.info(json.dumps({
        "method": request.method, "path": request.path, "status": response.status_code,
        "duration_ms": duration_ms, "remote_addr": request.remote_addr,
    }))
    return response


@app.errorhandler(HTTPException)
def _handle_http_exception(e: HTTPException):
    # Covers everything werkzeug/Flask raises natively too (404 on an
    # unknown route, 413 over MAX_CONTENT_LENGTH, 405 on a wrong method)
    # -- every error from this API is JSON, never the default HTML page.
    return jsonify({"error": e.description or e.name}), e.code


@app.errorhandler(Exception)
def _handle_unexpected_exception(e: Exception):
    logger.exception("unhandled exception")
    return jsonify({"error": "internal server error"}), 500


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
    store = _int_arg("store", 1, minimum=1)
    dept = _int_arg("dept", 1, minimum=1)
    return jsonify(TOOLS["get_forecast"](store=store, dept=dept))


@app.get("/api/anomalies")
def anomalies():
    store = _int_arg("store", 1, minimum=1)
    dept = _int_arg("dept", 1, minimum=1)
    only_flagged = request.args.get("only_flagged", "true") == "true"
    return jsonify(TOOLS["get_anomalies"](store=store, dept=dept, only_needs_investigation=only_flagged, limit=25))


@app.get("/api/history")
def history():
    store = _int_arg("store", 1, minimum=1)
    dept = _int_arg("dept", 1, minimum=1)
    weeks = _int_arg("weeks", 104, minimum=1)
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


MAX_QUESTION_LENGTH = 2000


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="request body must be a JSON object with a 'question' string field")
    question = payload.get("question", "")
    if not isinstance(question, str) or not question.strip():
        abort(400, description="missing or invalid 'question' (expected a non-empty string)")
    if len(question) > MAX_QUESTION_LENGTH:
        abort(400, description=f"'question' is too long (max {MAX_QUESTION_LENGTH} characters)")
    trace = agent_mod.ask(question)
    return jsonify(trace.to_dict())


@app.get("/api/explain")
def explain_drivers():
    store = _int_arg("store", 1, minimum=1)
    dept = _int_arg("dept", 1, minimum=1)
    week_ahead = _int_arg("week_ahead", 1, minimum=1)
    return jsonify(TOOLS["explain_forecast_drivers"](store=store, dept=dept, week_ahead=week_ahead))


@app.get("/api/simulate")
def simulate_scenario_endpoint():
    store = _int_arg("store", 1, minimum=1)
    dept = _int_arg("dept", 1, minimum=1)
    markdown_active = request.args.get("markdown_active", "false") == "true"
    is_holiday_raw = request.args.get("is_holiday")  # absent -> None -> keep real calendar
    is_holiday = None if is_holiday_raw is None else (is_holiday_raw == "true")
    weeks = _int_arg("weeks", 4, minimum=1)
    return jsonify(TOOLS["simulate_scenario"](store=store, dept=dept, markdown_active=markdown_active,
                                               is_holiday=is_holiday, weeks=weeks))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
