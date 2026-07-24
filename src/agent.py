"""Agentic reasoning layer over the forecast/anomaly artifact store.

Design goals (mirroring what the target job descriptions actually ask for):
  - Tool use, not a single giant prompt: the agent calls discrete, typed
    tools (get_forecast, get_anomalies, explain_change, top_movers) and
    reasons over their results.
  - Full audit trail: every tool call, its arguments, and its result are
    logged to an AgentTrace that's returned alongside the answer -- this is
    the auditability requirement Condor's JD calls out explicitly for
    financial data, and it generalizes to "can an enterprise user trust
    this system" (Merciv's framing).
  - Pluggable LLM backend: MockLLM (default, deterministic, no API key
    needed, good for CI/tests/demos) or AnthropicLLM (real tool-calling
    loop via the Messages API over plain HTTP, activated automatically if
    ANTHROPIC_API_KEY is set). Swapping backends doesn't touch the tools
    or the trace format at all -- only which "brain" decides which tool
    to call next.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"


# --------------------------------------------------------------------------
# Data access layer the tools sit on top of (reads the artifacts produced by
# scripts/run_pipeline.py -- the agent never re-trains or re-scores live)
# --------------------------------------------------------------------------
class DataStore:
    def __init__(self):
        self.forecasts = pd.read_csv(REPORTS / "current_forecasts.csv", parse_dates=["forecast_date"])
        self.series_summary = pd.read_csv(REPORTS / "series_summary.csv")
        self.anomalies = pd.read_csv(REPORTS / "anomaly_flags.csv", parse_dates=["Date"])
        self.selected_model = json.loads((REPORTS / "selected_model.json").read_text())

    def series_key(self, store: int, dept: int) -> str:
        return f"S{store:02d}_D{dept:02d}"


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
            {"date": str(r["forecast_date"].date()), "value": r["forecast_value"]}
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


TOOLS: dict[str, Callable[..., dict]] = {
    "get_forecast": tool_get_forecast,
    "get_anomalies": tool_get_anomalies,
    "explain_change": tool_explain_change,
    "top_movers": tool_top_movers,
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
]


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------
@dataclass
class AgentTrace:
    question: str
    steps: list = field(default_factory=list)
    final_answer: str = ""
    backend: str = ""

    def log_step(self, tool: str, args: dict, result: dict, reasoning: str = ""):
        self.steps.append({
            "step": len(self.steps) + 1,
            "reasoning": reasoning,
            "tool": tool,
            "args": args,
            "result": result,
            "ts": round(time.time(), 3),
        })

    def to_dict(self):
        return {"question": self.question, "backend": self.backend,
                "steps": self.steps, "final_answer": self.final_answer}

    def persist(self):
        REPORTS.mkdir(exist_ok=True)
        with open(REPORTS / "agent_traces.jsonl", "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")


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

    def answer(self, question: str, trace: AgentTrace) -> str:
        messages = [{"role": "user", "content": question}]
        for _ in range(self.max_turns):
            result = self._call(messages)
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


def get_llm_backend():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
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
