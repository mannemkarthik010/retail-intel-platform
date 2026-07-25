"""Agentic reasoning layer over the forecast/anomaly artifact store.

Design goals (mirroring what the target job descriptions actually ask for):
  - Tool use, not a single giant prompt: the agent calls discrete, typed
    tools (get_forecast, get_anomalies, explain_change, top_movers,
    explain_forecast_drivers, simulate_scenario) and reasons over their
    results.
  - Full audit trail: every tool call, its arguments, and its result are
    logged to an AgentTrace that's returned alongside the answer -- this is
    the auditability requirement Condor's JD calls out explicitly for
    financial data, and it generalizes to "can an enterprise user trust
    this system" (Merciv's framing).
  - Pluggable LLM backend: MockLLM (default, deterministic, no API key
    needed, good for CI/tests/demos), AnthropicLLM (real tool-calling loop
    via the Messages API), or OpenAILLM (real tool-calling loop via the
    Chat Completions API) -- both over plain HTTP, no vendor SDK, activated
    automatically based on which API key is present in the environment.
    Swapping backends doesn't touch the tools or the trace format at all --
    only which "brain" decides which tool to call next. This multi-provider
    pluggability is deliberate: Condor's job description names both
    "Anthropic Claude or OpenAI" as acceptable providers, and a system that
    hard-codes one vendor's SDK into its tool-calling logic can't honor that.
"""
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
MODELS_DIR = REPORTS / "models"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so `scripts.run_pipeline` is importable below


# --------------------------------------------------------------------------
# Data access layer the tools sit on top of (reads the artifacts produced by
# scripts/run_pipeline.py -- the agent never re-trains or re-scores live,
# with ONE deliberate exception: simulate_scenario, which by definition
# asks about a covariate combination that was never scored offline -- see
# src/scenario.py's docstring for why that one tool is different in kind.)
# --------------------------------------------------------------------------
class DataStore:
    def __init__(self):
        self.forecasts = pd.read_csv(REPORTS / "current_forecasts.csv", parse_dates=["forecast_date"])
        self.series_summary = pd.read_csv(REPORTS / "series_summary.csv")
        self.anomalies = pd.read_csv(REPORTS / "anomaly_flags.csv", parse_dates=["Date"])
        self.selected_model = json.loads((REPORTS / "selected_model.json").read_text())
        self._raw = None
        self._point_gbm = None
        self._feature_medians = None

    def series_key(self, store: int, dept: int) -> str:
        return f"S{store:02d}_D{dept:02d}"

    def raw_sales(self):
        """Lazily loaded, cached: only what simulate_scenario needs, since
        every other tool here only ever reads the offline reports/*.csv."""
        if self._raw is None:
            from .data_io import load_all_merged
            self._raw = load_all_merged()
        return self._raw

    def point_gbm(self):
        if self._point_gbm is None:
            import joblib
            self._point_gbm = joblib.load(MODELS_DIR / "gbm_point.joblib")
        return self._point_gbm

    def feature_medians(self):
        if self._feature_medians is None:
            import joblib
            self._feature_medians = joblib.load(MODELS_DIR / "feature_medians.joblib")
        return self._feature_medians


STORE = None  # lazy singleton so importing this module doesn't require artifacts to exist yet


def _store() -> DataStore:
    global STORE
    if STORE is None:
        STORE = DataStore()
    return STORE


# --------------------------------------------------------------------------
# Tools -- each returns a plain JSON-serializable dict, never raw dataframes
# --------------------------------------------------------------------------
def tool_get_forecast(store: int, dept: int) -> dict:
    ds = _store()
    key = ds.series_key(store, dept)
    rows = ds.forecasts[ds.forecasts["series"] == key].sort_values("week_ahead")
    if rows.empty:
        return {"error": f"no forecast found for store={store} dept={dept}"}
    summary = ds.series_summary[(ds.series_summary["series"] == key)]
    model_used = rows["model_used"].iloc[0]
    model_wape = summary[summary["model"] == model_used]["wape"].mean()
    return {
        "series": key,
        "model_used": model_used,
        "backtest_wape": None if pd.isna(model_wape) else round(float(model_wape), 4),
        "forecast": [
            {
                "date": str(r["forecast_date"].date()),
                "value": r["forecast_value"],
                "low": r.get("forecast_low", None),
                "high": r.get("forecast_high", None),
            }
            for _, r in rows.iterrows()
        ],
    }


def tool_get_anomalies(store: int, dept: int, only_needs_investigation: bool = True, limit: int = 10) -> dict:
    ds = _store()
    key = ds.series_key(store, dept)
    df = ds.anomalies[ds.anomalies["series"] == key].sort_values("Date", ascending=False)
    if only_needs_investigation:
        df = df[df["needs_investigation"]]
    df = df.head(limit)
    return {
        "series": key,
        "count": int(len(df)),
        "anomalies": [
            {
                "date": str(r["Date"].date()),
                "value": round(float(r["Weekly_Sales"]), 2),
                "type": r["anomaly_type"],
                "z_score": None if pd.isna(r["z_score"]) else round(float(r["z_score"]), 2),
                "high_confidence": bool(r.get("high_confidence_anomaly", False)),
            }
            for _, r in df.iterrows()
        ],
    }


def tool_explain_change(store: int, dept: int) -> dict:
    """Decompose the most recent actual vs. what the series 'normally' does:
    trend level, seasonal factor, active promos, and any flagged anomaly --
    so the answer is a reasoned explanation, not just a number."""
    ds = _store()
    key = ds.series_key(store, dept)
    df = ds.anomalies[ds.anomalies["series"] == key].sort_values("Date")
    if df.empty:
        return {"error": f"no history for store={store} dept={dept}"}
    latest = df.iloc[-1]
    prev_year = df[df["Date"] == (latest["Date"] - pd.Timedelta(weeks=52))]
    yoy = None
    if not prev_year.empty and prev_year.iloc[0]["Weekly_Sales"]:
        yoy = round(100 * (latest["Weekly_Sales"] - prev_year.iloc[0]["Weekly_Sales"]) / abs(prev_year.iloc[0]["Weekly_Sales"]), 1)
    reasons = []
    if latest.get("explained_by_markdown"):
        reasons.append("an active promotional markdown is running this week")
    if latest["anomaly_type"] == "unexplained_anomaly":
        reasons.append("statistically unusual vs. this series' own history -- flagged for investigation")
    if latest["anomaly_type"] == "data_quality_error":
        reasons.append("looks like a data quality issue (negative/implausible value)")
    if not reasons:
        reasons.append("within normal trend + seasonal expectation, nothing unusual")
    return {
        "series": key,
        "latest_date": str(latest["Date"].date()),
        "latest_value": round(float(latest["Weekly_Sales"]), 2),
        "z_score": None if pd.isna(latest["z_score"]) else round(float(latest["z_score"]), 2),
        "year_over_year_pct_change": yoy,
        "reasons": reasons,
    }


def tool_top_movers(direction: str = "down", n: int = 5) -> dict:
    ds = _store()
    df = ds.anomalies.sort_values(["series", "Date"])
    latest_per_series = df.groupby("series").tail(1)
    yoy_rows = []
    for _, row in latest_per_series.iterrows():
        prior = df[(df["series"] == row["series"]) & (df["Date"] == row["Date"] - pd.Timedelta(weeks=52))]
        if prior.empty or not prior.iloc[0]["Weekly_Sales"]:
            continue
        pct = 100 * (row["Weekly_Sales"] - prior.iloc[0]["Weekly_Sales"]) / abs(prior.iloc[0]["Weekly_Sales"])
        yoy_rows.append({"series": row["series"], "pct_change_yoy": round(pct, 1),
                          "latest_value": round(float(row["Weekly_Sales"]), 2)})
    yoy_df = pd.DataFrame(yoy_rows)
    if yoy_df.empty:
        return {"movers": []}
    yoy_df = yoy_df.sort_values("pct_change_yoy", ascending=(direction == "down"))
    return {"direction": direction, "movers": yoy_df.head(n).to_dict("records")}


def tool_explain_forecast_drivers(store: int, dept: int, week_ahead: int = 1) -> dict:
    """Local, occlusion-based feature attribution for one specific
    forecasted week (src/explain.py) -- only meaningful for series whose
    selected model is the global GBM, since seasonal_naive/Holt-Winters
    have no per-feature model to attribute anything to."""
    from .backtest import HORIZON, _recursive_gbm_forecast_with_rows
    from . import explain

    ds = _store()
    key = ds.series_key(store, dept)
    model_used = ds.selected_model.get(key)
    if model_used != "global_gbm":
        return {"error": f"driver explanations are only available for series whose selected "
                          f"model is global_gbm (this series' selected model is {model_used}, "
                          f"a classical method with no per-feature attribution to give)"}
    week_ahead = max(1, min(int(week_ahead), HORIZON))

    raw = ds.raw_sales()
    hist = raw[(raw.Store == store) & (raw.Dept == dept)].sort_values("Date")
    if hist.empty:
        return {"error": f"no history for store={store} dept={dept}"}

    from scripts.run_pipeline import make_future_covariates, _gbm_covariates_for_series
    future_cov = make_future_covariates(raw, HORIZON)
    cov = _gbm_covariates_for_series(future_cov, hist, store, dept)

    model = ds.point_gbm()
    _, rows = _recursive_gbm_forecast_with_rows(model, hist, cov, HORIZON)
    row = rows[week_ahead - 1]

    result = explain.occlusion_attribution(model, row, ds.feature_medians())
    result["series"] = key
    result["week_ahead"] = week_ahead
    result["forecast_date"] = str(cov["Date"].iloc[week_ahead - 1].date())
    return result


def tool_simulate_scenario(store: int, dept: int, markdown_active: bool = False,
                            is_holiday: bool = None, weeks: int = 4) -> dict:
    """What-if simulation (src/scenario.py) -- the one tool that genuinely
    recomputes rather than reading a precomputed artifact, since a
    hypothetical covariate combination was never scored offline. Only
    meaningful for global_gbm-selected series (see scenario.py docstring).

    `is_holiday=None` (the default) means "don't touch the real calendar,"
    so asking a markdown-only question doesn't silently strip out a real
    holiday flag on a week that genuinely is one -- pass True/False only
    when the question is explicitly ABOUT holiday-ness."""
    from .backtest import HORIZON
    from . import scenario

    ds = _store()
    key = ds.series_key(store, dept)
    model_used = ds.selected_model.get(key)
    if model_used != "global_gbm":
        return {"error": f"what-if simulation is only available for series whose selected "
                          f"model is global_gbm (this series' selected model is {model_used}, "
                          f"which has no markdown/holiday covariate slots to hypothetically change)"}
    weeks = max(1, min(int(weeks), HORIZON))

    raw = ds.raw_sales()
    scenario_preds = scenario.simulate_scenario(raw, store, dept, markdown_active, is_holiday, weeks)
    if scenario_preds is None:
        return {"error": f"no history for store={store} dept={dept}"}

    baseline_rows = ds.forecasts[ds.forecasts["series"] == key].sort_values("week_ahead").head(weeks)
    baseline = baseline_rows["forecast_value"].values
    return {
        "series": key,
        "assumptions": {"markdown_active": markdown_active, "is_holiday": is_holiday, "weeks": weeks},
        "baseline_forecast": [round(float(b), 2) for b in baseline],
        "scenario_forecast": [round(float(s), 2) for s in scenario_preds],
        "delta_vs_baseline": [round(float(s - b), 2) for s, b in zip(scenario_preds, baseline)],
        "note": "baseline uses the pipeline's default future assumptions (no active promo, "
                "real calendar holidays); this scenario overrides only markdown_active/is_holiday.",
    }


TOOLS: dict[str, Callable[..., dict]] = {
    "get_forecast": tool_get_forecast,
    "get_anomalies": tool_get_anomalies,
    "explain_change": tool_explain_change,
    "top_movers": tool_top_movers,
    "explain_forecast_drivers": tool_explain_forecast_drivers,
    "simulate_scenario": tool_simulate_scenario,
}

TOOL_SPECS = [
    {
        "name": "get_forecast",
        "description": "Get the next-8-week demand forecast for one store/department series, "
                       "including which model was selected and its backtested accuracy (WAPE).",
        "input_schema": {
            "type": "object",
            "properties": {"store": {"type": "integer"}, "dept": {"type": "integer"}},
            "required": ["store", "dept"],
        },
    },
    {
        "name": "get_anomalies",
        "description": "Get flagged anomalies (data errors, unexplained spikes/drops) for one series.",
        "input_schema": {
            "type": "object",
            "properties": {"store": {"type": "integer"}, "dept": {"type": "integer"},
                            "limit": {"type": "integer"}},
            "required": ["store", "dept"],
        },
    },
    {
        "name": "explain_change",
        "description": "Explain why a series' most recent value looks the way it does "
                       "(promo, anomaly, or normal trend/seasonality), with year-over-year context.",
        "input_schema": {
            "type": "object",
            "properties": {"store": {"type": "integer"}, "dept": {"type": "integer"}},
            "required": ["store", "dept"],
        },
    },
    {
        "name": "top_movers",
        "description": "Rank all series by year-over-year percent change; direction 'up' or 'down'.",
        "input_schema": {
            "type": "object",
            "properties": {"direction": {"type": "string", "enum": ["up", "down"]},
                            "n": {"type": "integer"}},
            "required": [],
        },
    },
    {
        "name": "explain_forecast_drivers",
        "description": "Explain which input features drove a specific forecasted week's number "
                       "for one series, via occlusion-based local attribution (not SHAP). Only "
                       "available for series whose selected model is the global GBM.",
        "input_schema": {
            "type": "object",
            "properties": {"store": {"type": "integer"}, "dept": {"type": "integer"},
                            "week_ahead": {"type": "integer", "description": "1-8, which forecasted week to explain"}},
            "required": ["store", "dept"],
        },
    },
    {
        "name": "simulate_scenario",
        "description": "What-if simulation: recompute a series' forecast under a hypothetical "
                       "active markdown promotion and/or holiday week, and report the delta vs. "
                       "the baseline forecast. Only available for series whose selected model is "
                       "the global GBM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store": {"type": "integer"}, "dept": {"type": "integer"},
                "markdown_active": {"type": "boolean", "description": "hypothetically run an active promo"},
                "is_holiday": {"type": "boolean", "description": "only set this if the question explicitly asks "
                                                                  "about holiday-ness; omit it to keep the real "
                                                                  "calendar (don't default it to false)"},
                "weeks": {"type": "integer", "description": "how many weeks ahead to simulate, 1-8"},
            },
            "required": ["store", "dept"],
        },
    },
]


# --------------------------------------------------------------------------
# Retry/backoff + cost tracking shared by the three real LLM backends below.
# MockLLM never makes a network call, so none of this applies to it -- there
# is nothing to retry and nothing to cost.
#
# Retrying blindly on ANY exception would waste attempts retrying things
# that will never succeed (a 401 from a bad API key, a 400 from a malformed
# request) -- each backend below supplies its own `retryable(exc) -> bool`
# predicate so only actually-transient failures (rate limits, momentary
# 5xx/overload, connection hiccups) get retried.
# --------------------------------------------------------------------------
def _retry_with_backoff(fn, max_retries: int = 4, base_delay: float = 1.0, retryable=None):
    """Calls fn() with exponential backoff + jitter. Returns (result,
    retries_used). Raises the last exception once max_retries is exhausted,
    or immediately if `retryable(exc)` says the failure isn't transient."""
    retryable = retryable or (lambda e: False)
    for attempt in range(max_retries):
        try:
            return fn(), attempt
        except Exception as e:
            if not retryable(e) or attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))
    raise AssertionError("unreachable")  # pragma: no cover


_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _is_retryable_requests_error(e: Exception) -> bool:
    import requests
    if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(e, requests.exceptions.HTTPError):
        status = e.response.status_code if e.response is not None else None
        return status in _RETRYABLE_HTTP_STATUS
    return False


def _is_retryable_bedrock_error(e: Exception) -> bool:
    """botocore's ClientError carries the AWS error code in
    e.response['Error']['Code'] -- checked via getattr/dict.get rather than
    importing botocore, since boto3 is an optional dependency (see
    BedrockLLM's docstring) and this predicate must not force-import it."""
    response = getattr(e, "response", None)
    if not isinstance(response, dict):
        return False
    code = response.get("Error", {}).get("Code", "")
    return code in {"ThrottlingException", "ServiceUnavailable", "ModelTimeoutException",
                     "InternalServerException", "ModelNotReadyException"}


# Approximate list prices in USD per 1M tokens, accurate as of when this was
# written -- for COST-AWARENESS in the trace/usage log (reports/llm_usage.jsonl),
# NOT billing-accurate invoicing. Provider pricing changes over time; check
# the vendor's current pricing page before relying on this for anything
# financial. Bedrock model IDs vary by region/version and aren't matched
# here individually, so BEDROCK_DEFAULT_PRICING assumes a Claude-family
# model (the common case) rather than trying to parse every possible ID.
LLM_PRICING_USD_PER_1M_TOKENS = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}
BEDROCK_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


def _estimate_cost_usd(backend: str, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    pricing = LLM_PRICING_USD_PER_1M_TOKENS.get(model)
    if pricing is None and backend == "bedrock":
        pricing = BEDROCK_DEFAULT_PRICING
    if pricing is None:
        return None
    return round(input_tokens / 1e6 * pricing["input"] + output_tokens / 1e6 * pricing["output"], 6)


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------
@dataclass
class AgentTrace:
    question: str
    steps: list = field(default_factory=list)
    final_answer: str = ""
    backend: str = ""
    model: str = ""
    token_usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    estimated_cost_usd: Optional[float] = None
    retries: int = 0

    def log_step(self, tool: str, args: dict, result: dict, reasoning: str = ""):
        self.steps.append({
            "step": len(self.steps) + 1,
            "reasoning": reasoning,
            "tool": tool,
            "args": args,
            "result": result,
            "ts": round(time.time(), 3),
        })

    def add_usage(self, input_tokens: int, output_tokens: int, model: str = ""):
        """Called once per real API round-trip (a single `ask()` can make
        several, one per tool-calling turn) -- accumulates across the whole
        answer rather than overwriting, so a multi-turn tool-calling loop
        reports its TOTAL cost, not just the last turn's."""
        self.token_usage["input_tokens"] += input_tokens
        self.token_usage["output_tokens"] += output_tokens
        if model:
            self.model = model
        self.estimated_cost_usd = _estimate_cost_usd(
            self.backend, self.model, self.token_usage["input_tokens"], self.token_usage["output_tokens"])

    def to_dict(self):
        return {"question": self.question, "backend": self.backend, "model": self.model,
                "steps": self.steps, "final_answer": self.final_answer,
                "token_usage": self.token_usage, "estimated_cost_usd": self.estimated_cost_usd,
                "retries": self.retries}

    def persist(self):
        REPORTS.mkdir(exist_ok=True)
        with open(REPORTS / "agent_traces.jsonl", "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
        # Only real (non-mock) backends ever accumulate token usage --
        # skip the cost/usage log entirely for MockLLM rather than
        # appending meaningless all-zero rows to it.
        if self.token_usage["input_tokens"] or self.token_usage["output_tokens"]:
            with open(REPORTS / "llm_usage.jsonl", "a") as f:
                f.write(json.dumps({
                    "ts": round(time.time(), 3), "backend": self.backend, "model": self.model,
                    "question": self.question, **self.token_usage,
                    "estimated_cost_usd": self.estimated_cost_usd, "retries": self.retries,
                }) + "\n")


# --------------------------------------------------------------------------
# Backend 1: MockLLM -- deterministic intent routing, zero external calls.
# This is the default so the whole system runs end-to-end with no API key.
# --------------------------------------------------------------------------
class MockLLM:
    """Regex/keyword-based router standing in for an LLM planner. It's
    intentionally simple and fully deterministic -- the point is to prove
    out the AGENT ARCHITECTURE (tools, trace, orchestration) independent of
    any specific model provider. Swap in AnthropicLLM below for real
    natural-language reasoning over the same tools."""

    name = "mock"

    def answer(self, question: str, trace: AgentTrace) -> str:
        q = question.lower()
        store_m = re.search(r"store\s*(\d+)", q)
        dept_m = re.search(r"dep(?:t|artment)\.?\s*(\d+)", q)
        store = int(store_m.group(1)) if store_m else 1
        dept = int(dept_m.group(1)) if dept_m else 1

        if "top mover" in q or "biggest drop" in q or "biggest decline" in q or "worst perform" in q:
            direction = "down" if any(w in q for w in ["drop", "decline", "worst", "down"]) else "up"
            result = TOOLS["top_movers"](direction=direction, n=5)
            trace.log_step("top_movers", {"direction": direction, "n": 5}, result,
                            reasoning=f"Question asks for biggest movers ({direction}) -> rank all series by YoY%.")
            movers = result["movers"]
            lines = [f"Store {m['series'][1:3]}, Dept {m['series'][-2:]}: {m['pct_change_yoy']:+.1f}% YoY"
                      for m in movers]
            return f"Top {direction} movers year-over-year:\n" + "\n".join(lines)

        if "what if" in q or "what-if" in q or "scenario" in q or "hypothetical" in q:
            markdown_active = any(w in q for w in ["markdown", "promo", "promotion", "discount", "sale"])
            # None = leave the real calendar alone (don't silently strip a
            # genuine holiday flag just because the question was about a
            # markdown); only override when the question explicitly names
            # holiday-ness one way or the other.
            if "not a holiday" in q or "non-holiday" in q or "no holiday" in q:
                is_holiday = False
            elif "holiday" in q:
                is_holiday = True
            else:
                is_holiday = None
            weeks_m = re.search(r"(\d+)\s*week", q)
            weeks = int(weeks_m.group(1)) if weeks_m else 4
            args = {"store": store, "dept": dept, "markdown_active": markdown_active,
                    "is_holiday": is_holiday, "weeks": weeks}
            result = TOOLS["simulate_scenario"](**args)
            trace.log_step("simulate_scenario", args, result,
                            reasoning=f"Question is a what-if -> simulate store {store} dept {dept} "
                                      f"under markdown_active={markdown_active}, is_holiday={is_holiday}.")
            if "error" in result:
                return result["error"]
            lines = [f"week {i+1}: baseline ${b:,.0f} -> scenario ${s:,.0f} ({d:+,.0f})"
                     for i, (b, s, d) in enumerate(zip(result["baseline_forecast"], result["scenario_forecast"],
                                                        result["delta_vs_baseline"]))]
            return (f"Store {store} / Dept {dept} what-if (markdown_active={markdown_active}, "
                    f"is_holiday={is_holiday}):\n" + "\n".join(lines))

        if "driver" in q or "driving the forecast" in q or "feature" in q or "what's pushing" in q:
            week_m = re.search(r"week\s*(\d+)", q)
            week_ahead = int(week_m.group(1)) if week_m else 1
            args = {"store": store, "dept": dept, "week_ahead": week_ahead}
            result = TOOLS["explain_forecast_drivers"](**args)
            trace.log_step("explain_forecast_drivers", args, result,
                            reasoning=f"Question asks what's driving the forecast for store {store} "
                                      f"dept {dept}, week {week_ahead} -> occlusion attribution.")
            if "error" in result:
                return result["error"]
            lines = [f"{d['feature']}: actual {d['actual_value']} vs. typical {d['median_value']} "
                     f"-> {d['impact']:+.0f} impact on the forecast" for d in result["top_drivers"]]
            return (f"Store {store} / Dept {dept}, week {week_ahead} forecast "
                    f"(${result['baseline_prediction']:,.0f}) -- top drivers:\n" + "\n".join(lines))

        if "why" in q or "explain" in q:
            result = TOOLS["explain_change"](store=store, dept=dept)
            trace.log_step("explain_change", {"store": store, "dept": dept}, result,
                            reasoning=f"Question asks 'why' for store {store} dept {dept} -> decompose latest value.")
            if "error" in result:
                return result["error"]
            reason_txt = "; ".join(result["reasons"])
            yoy_txt = f", {result['year_over_year_pct_change']:+.1f}% vs. the same week last year" if result["year_over_year_pct_change"] is not None else ""
            return (f"Store {store} / Dept {dept} on {result['latest_date']}: "
                    f"${result['latest_value']:,.0f}{yoy_txt}. Reason: {reason_txt}.")

        if "anomal" in q or "flag" in q or "unusual" in q:
            result = TOOLS["get_anomalies"](store=store, dept=dept)
            trace.log_step("get_anomalies", {"store": store, "dept": dept}, result,
                            reasoning=f"Question asks about anomalies for store {store} dept {dept}.")
            if result["count"] == 0:
                return f"No flagged anomalies needing investigation for Store {store} / Dept {dept}."
            lines = [f"{a['date']}: ${a['value']:,.0f} ({a['type']}, z={a['z_score']})" for a in result["anomalies"]]
            return f"Flagged anomalies for Store {store} / Dept {dept}:\n" + "\n".join(lines)

        # default: forecast
        result = TOOLS["get_forecast"](store=store, dept=dept)
        trace.log_step("get_forecast", {"store": store, "dept": dept}, result,
                        reasoning=f"Default intent: question about store {store} dept {dept} -> return forecast.")
        if "error" in result:
            return result["error"]
        fc_lines = [f"{f['date']}: ${f['value']:,.0f}" for f in result["forecast"]]
        wape_txt = f" (backtested WAPE {result['backtest_wape']*100:.1f}%)" if result["backtest_wape"] is not None else ""
        return (f"Store {store} / Dept {dept} forecast using {result['model_used']}{wape_txt}:\n"
                + "\n".join(fc_lines))


# --------------------------------------------------------------------------
# Backend 2: real Anthropic tool-calling loop, plain HTTP (no SDK dependency)
# Activated automatically if ANTHROPIC_API_KEY is set in the environment.
# --------------------------------------------------------------------------
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", max_turns: int = 4):
        import requests  # local import: only needed on this path
        self._requests = requests
        self.model = model
        self.max_turns = max_turns
        self.api_key = os.environ["ANTHROPIC_API_KEY"]

    def _call(self, messages):
        def do_call():
            resp = self._requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "tools": TOOL_SPECS,
                    "system": (
                        "You are a retail demand-forecasting analyst assistant. Use the "
                        "provided tools to answer questions about specific store/department "
                        "series. Always ground numeric claims in tool results, never guess."
                    ),
                    "messages": messages,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        return _retry_with_backoff(do_call, retryable=_is_retryable_requests_error)

    def answer(self, question: str, trace: AgentTrace) -> str:
        messages = [{"role": "user", "content": question}]
        for _ in range(self.max_turns):
            result, retries = self._call(messages)
            trace.retries += retries
            usage = result.get("usage", {})
            trace.add_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0), model=self.model)
            content = result.get("content", [])
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            text_blocks = [b["text"] for b in content if b.get("type") == "text"]
            if not tool_uses:
                return "\n".join(text_blocks) if text_blocks else "(no answer produced)"

            messages.append({"role": "assistant", "content": content})
            tool_results = []
            for tu in tool_uses:
                fn = TOOLS.get(tu["name"])
                args = tu.get("input", {})
                out = fn(**args) if fn else {"error": f"unknown tool {tu['name']}"}
                trace.log_step(tu["name"], args, out, reasoning="(reasoning happened inside the model; "
                                                                  "this step is the tool call it chose to make)")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": json.dumps(out),
                })
            messages.append({"role": "user", "content": tool_results})
        return "(hit max tool-call turns without a final answer)"


def _to_openai_tool_specs():
    """OpenAI's function-calling schema is the same JSON Schema Anthropic
    uses for `input_schema`, just wrapped differently -- no logic to
    duplicate, only a shape change."""
    return [
        {"type": "function", "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["input_schema"],
        }}
        for spec in TOOL_SPECS
    ]


# --------------------------------------------------------------------------
# Backend 3: real OpenAI tool-calling loop, plain HTTP (no SDK dependency)
# Activated automatically if OPENAI_API_KEY is set (and ANTHROPIC_API_KEY
# isn't, unless LLM_BACKEND=openai forces it -- see get_llm_backend()).
# --------------------------------------------------------------------------
class OpenAILLM:
    name = "openai"

    def __init__(self, model: str = None, max_turns: int = 4):
        import requests  # local import: only needed on this path
        self._requests = requests
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_turns = max_turns
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.tool_specs = _to_openai_tool_specs()

    def _call(self, messages):
        def do_call():
            resp = self._requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tool_specs,
                    "tool_choice": "auto",
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        return _retry_with_backoff(do_call, retryable=_is_retryable_requests_error)

    def answer(self, question: str, trace: AgentTrace) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a retail demand-forecasting analyst assistant. Use the "
                "provided tools to answer questions about specific store/department "
                "series. Always ground numeric claims in tool results, never guess."
            )},
            {"role": "user", "content": question},
        ]
        for _ in range(self.max_turns):
            result, retries = self._call(messages)
            trace.retries += retries
            usage = result.get("usage", {})
            trace.add_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), model=self.model)
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return message.get("content") or "(no answer produced)"

            messages.append(message)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = TOOLS.get(fn_name)
                out = fn(**args) if fn else {"error": f"unknown tool {fn_name}"}
                trace.log_step(fn_name, args, out, reasoning="(reasoning happened inside the model; "
                                                              "this step is the tool call it chose to make)")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(out),
                })
        return "(hit max tool-call turns without a final answer)"


def _to_bedrock_tool_config():
    """Bedrock's Converse API wants the same JSON Schema everyone else here
    uses, wrapped as {"toolSpec": {"name", "description", "inputSchema":
    {"json": <schema>}}} inside a top-level {"tools": [...]}. Same shape
    change as _to_openai_tool_specs() above -- no logic duplicated, just a
    different envelope."""
    return {
        "tools": [
            {"toolSpec": {
                "name": spec["name"],
                "description": spec["description"],
                "inputSchema": {"json": spec["input_schema"]},
            }}
            for spec in TOOL_SPECS
        ]
    }


# --------------------------------------------------------------------------
# Backend 4: real Bedrock tool-calling loop via boto3's Converse API.
# Activated automatically if BEDROCK_MODEL_ID is set in the environment (see
# get_llm_backend() below for why that's the trigger instead of a generic
# AWS credential check), or explicitly via LLM_BACKEND=bedrock.
#
# Unlike AnthropicLLM/OpenAILLM (plain HTTP, deliberately no vendor SDK --
# see docs/ARCHITECTURE.md for why: proving provider-agnosticism across two
# competing LLM vendors was the point there), this backend uses boto3. AWS
# has no plain-HTTP-friendly equivalent to a bearer-token API key -- auth is
# SigV4-signed requests, and boto3 is the standard, expected SDK for that
# (and is needed anyway for the S3/Lambda/SageMaker calls elsewhere in
# infra/). Requiring it here isn't a meaningful vendor lock-in concern the
# way requiring anthropic's/openai's SDK would have been.
# --------------------------------------------------------------------------
class BedrockLLM:
    name = "bedrock"

    def __init__(self, model_id: str = None, region: str = None, max_turns: int = 4):
        import boto3  # local import: only needed on this path, and only
                       # ever reached when BEDROCK_MODEL_ID/LLM_BACKEND=bedrock
                       # is explicitly set -- see module docstring above.
        self.model_id = model_id or os.environ["BEDROCK_MODEL_ID"]
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.max_turns = max_turns
        self._client = boto3.client("bedrock-runtime", region_name=self.region)
        self._tool_config = _to_bedrock_tool_config()

    def _call(self, messages):
        def do_call():
            return self._client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": (
                    "You are a retail demand-forecasting analyst assistant. Use the "
                    "provided tools to answer questions about specific store/department "
                    "series. Always ground numeric claims in tool results, never guess."
                )}],
                toolConfig=self._tool_config,
            )
        return _retry_with_backoff(do_call, retryable=_is_retryable_bedrock_error)

    def answer(self, question: str, trace: AgentTrace) -> str:
        messages = [{"role": "user", "content": [{"text": question}]}]
        for _ in range(self.max_turns):
            result, retries = self._call(messages)
            trace.retries += retries
            usage = result.get("usage", {})
            trace.add_usage(usage.get("inputTokens", 0), usage.get("outputTokens", 0), model=self.model_id)
            message = result["output"]["message"]
            content = message.get("content", [])
            tool_uses = [b["toolUse"] for b in content if "toolUse" in b]
            text_blocks = [b["text"] for b in content if "text" in b]
            if not tool_uses:
                return "\n".join(text_blocks) if text_blocks else "(no answer produced)"

            messages.append(message)
            tool_result_content = []
            for tu in tool_uses:
                fn = TOOLS.get(tu["name"])
                args = tu.get("input", {})
                out = fn(**args) if fn else {"error": f"unknown tool {tu['name']}"}
                trace.log_step(tu["name"], args, out, reasoning="(reasoning happened inside the model; "
                                                                  "this step is the tool call it chose to make)")
                tool_result_content.append({
                    "toolResult": {
                        "toolUseId": tu["toolUseId"],
                        "content": [{"json": out}],
                    }
                })
            messages.append({"role": "user", "content": tool_result_content})
        return "(hit max tool-call turns without a final answer)"


def get_llm_backend():
    """Selection order: an explicit LLM_BACKEND override always wins;
    otherwise prefer whichever API key is present (Anthropic first, purely
    because that's this project's primary integration target; then OpenAI;
    then Bedrock last among the real backends), falling back to the
    deterministic mock so the system always runs.

    Bedrock's auto-detection trigger is BEDROCK_MODEL_ID, not a generic AWS
    credential check (e.g. AWS_ACCESS_KEY_ID) -- AWS credentials are
    frequently present in an environment for unrelated infra reasons (S3
    access, deployment tooling), so their mere presence isn't a reliable
    signal that Bedrock should be the LLM brain. Deliberately setting
    BEDROCK_MODEL_ID is the same kind of explicit, single-purpose signal
    ANTHROPIC_API_KEY/OPENAI_API_KEY already are."""
    forced = os.environ.get("LLM_BACKEND", "").lower()
    if forced == "anthropic":
        return AnthropicLLM()
    if forced == "openai":
        return OpenAILLM()
    if forced == "bedrock":
        return BedrockLLM()
    if forced == "mock":
        return MockLLM()

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    if os.environ.get("BEDROCK_MODEL_ID"):
        return BedrockLLM()
    return MockLLM()


def ask(question: str) -> AgentTrace:
    trace = AgentTrace(question=question)
    backend = get_llm_backend()
    trace.backend = backend.name
    trace.final_answer = backend.answer(question, trace)
    trace.persist()
    return trace


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What's the forecast for store 3 dept 5?"
    t = ask(q)
    print(f"[backend: {t.backend}]")
    print(json.dumps(t.steps, indent=2, default=str))
    print("\nANSWER:", t.final_answer)
