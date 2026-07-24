# What this project maps to, and where

This project was built specifically to speak to five senior ML/AI engineering
job descriptions (Merciv, Uber Freight, Sigma, Confido, Condor) without having
the years of production experience they ask for yet. This doc is the explicit
crosswalk — useful for a resume bullet, a cover letter, or a "walk me through
a project" interview answer.

| Requirement (as phrased across the JDs) | Where it lives here |
|---|---|
| Demand forecasting / time-series prediction at scale | `src/forecasting.py`, `src/backtest.py` — 240 series, 3 competing models |
| Rolling-origin / backtesting, "best-fit selection" (Confido's exact phrase) | `src/backtest.py::run_backtest` — 6 cutoffs, per-series model selection |
| Anomaly detection | `src/anomaly.py` — decomposition + robust z-score + IsolationForest cross-check |
| Agentic AI architecture / multi-step reasoning over tools | `src/agent.py` — 6 tools (including live what-if simulation and driver explainability), pluggable planner, full trace |
| RAG / LLM integration for natural-language interfaces | `src/agent.py::AnthropicLLM` / `OpenAILLM` — real tool-calling loops, activate automatically based on which API key is present |
| "Providers such as Anthropic Claude or OpenAI" (Condor's exact phrase) | Both are implemented as interchangeable backends behind the same tool registry and trace format — see `docs/ARCHITECTURE.md` |
| "Systems enterprise users can actually trust" / interpretability (Merciv) | Every anomaly is labeled *why* (promo / holiday / genuinely unexplained); every agent answer carries its reasoning trace |
| Uncertainty quantification / calibrated forecasts, not just point estimates (all five, implicitly) | `src/intervals.py` — 80% prediction intervals on every forecast, with a real held-out coverage check reported honestly (74.7% actual vs. 80% nominal) rather than assumed calibrated — see `docs/EVAL_REPORT.md` §4 |
| Model interpretability / feature attribution (Merciv, Confido) | `src/explain.py` — occlusion-based local attribution + permutation-importance global ranking, both explicitly disclosed as SHAP-substitutes rather than passed off as SHAP |
| Scenario / what-if analysis, decision support (Sigma, Confido) | `src/scenario.py` + agent tool `simulate_scenario` — recomputes the forecast under a hypothetical markdown/holiday and reports the delta vs. baseline |
| Auditability, "correctness and auditability are non-negotiable" (Condor) | `AgentTrace` persisted to `reports/agent_traces.jsonl` for every question asked |
| Own model performance end-to-end: monitoring, retraining (all five) | `scripts/run_monitoring_sim.py` — fleet-level + per-series drift detection |
| Evaluation harnesses, "know which metric to trust... avoid leakage and train/serve skew" (Confido) | `tests/test_features.py` — explicit lookahead-leakage tests; WAPE/MAPE/RMSE tracked per series per cutoff, not one aggregate number |
| Production-grade services, backend APIs (Condor, Uber Freight) | `app/server.py` — Flask API, offline/online separation documented in `docs/ARCHITECTURE.md` |
| Full ML lifecycle: data curation → training → deployment → monitoring (Sigma) | The whole `scripts/run_pipeline.py` → `run_monitoring_sim.py` chain |
| Product sense / knowing when a simpler approach wins (Confido) | Seasonal-naive wins 90/240 series in the backtest — the system says so instead of hiding it |
| Document/messy-data understanding (Confido specifically) | Not built here — see "What this project deliberately does NOT cover" below |

## What this project deliberately does NOT cover

Being precise about scope matters more than implying broader coverage than
is true:

- **Document/information extraction from unstructured sources** (Confido's
  invoice/financial-document parsing) isn't part of this build. It's a
  distinct enough problem (structured extraction + validation from messy
  layouts, not time-series or tool-calling) that it's proposed as a
  **separate follow-on project** rather than bolted on here artificially.
- **Knowledge graphs / GNNs** (Merciv's bonus points) — not attempted. A
  graph-RAG layer over the same forecast/anomaly data would be a natural
  extension, not a rebuild.
- **AWS deployment** (Condor's SageMaker/Bedrock/Lambda preference) — this
  runs locally/in a container. The offline/online split in
  `docs/ARCHITECTURE.md` is exactly the shape you'd lift onto
  Lambda/Fargate + SageMaker, but it hasn't actually been deployed there.
- **React frontend** (Condor's nice-to-have) — the dashboard here is plain
  HTML/JS by design (zero build step, easy to read end-to-end in one
  sitting); a React rewrite would be additive polish, not a different
  architecture.

## How to talk about this in an interview

The honest framing that holds up under follow-up questions: *"I built the
full pipeline shape a production forecasting + agentic system actually has —
backtesting methodology, per-series model selection, anomaly detection that
distinguishes explained from unexplained, an agent with a real tool-calling
architecture and an audit trail, and a monitoring layer that catches
individual-series drift the fleet average would hide. The data is synthetic
because of a sandboxed build environment, generated to match a specific real
dataset's schema and statistical character, and that's disclosed everywhere
rather than implied to be real. What I haven't done yet is run any of this
against real production traffic or a cloud deployment — that's the gap
between this and the target roles, and I know exactly where it is."*
