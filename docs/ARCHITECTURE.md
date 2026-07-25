# Architecture

## System shape

```
data/generate_data.py                     scripts/run_pipeline.py
        |                                          |
        v                                          v
  retail_sales.csv  ---------------------> src/backtest.py  ------> reports/series_summary.csv
  store_meta.csv                                   |                  per_cutoff_metrics.csv
  macro_features.csv                               |                  selected_model.json
        |                                          v
        +--------------------------------> src/anomaly.py  --------> reports/anomaly_flags.csv
        |                                          |
        +--------------------------------> forward forecast  ------> reports/current_forecasts.csv
        |                                    (+ src/intervals.py       (+ forecast_low/high)
        |                                     for 80% bands)
        |                                          |
        +--------------------------------> src/intervals.py           reports/interval_coverage.json
        |                                    .validate_interval_coverage
        |                                          |
        +--------------------------------> permutation importance --> reports/feature_importance.json
                                             (src/explain.py)      --> reports/models/*.joblib
                                                    |                  (point + q10 + q90 GBM,
                                                    v                   feature medians)
                                        scripts/run_monitoring_sim.py
                                                    |
                                                    v
                                   reports/monitoring_fleet_log.csv
                                   reports/monitoring_series_alerts.csv
                                                    |
                                                    v
                                     scripts/retrain_flagged.py
                                                    |
                                                    v
                          reports/selected_model.json (updated in place)
                          reports/current_forecasts.csv (updated in place)
                          reports/retraining_log.jsonl (append-only audit trail)

                              (all of the above are OFFLINE / BATCH)
                                                    |
                                                    v
                              src/agent.py  <-- reads reports/*.csv, never re-trains
                                    |               (except simulate_scenario -- see below)
                                    v
                              app/server.py (Flask)  <-- thin online layer
                                    |
                                    v
                              app/templates/dashboard.html
```

The load-bearing design decision is the split between **offline batch scoring**
(`scripts/run_pipeline.py`, which trains models and writes every forecast/anomaly
flag to `reports/*.csv`) and a **thin online serving layer** (`app/server.py`,
`src/agent.py`) that only ever *reads* those artifacts. This is how forecasting
and agentic systems are actually deployed at scale: online request latency can't
depend on how expensive the modeling is, and it means the agent's answers are
always backed by a specific, versioned batch of scored predictions you could
audit after the fact — not a live re-computation that might differ between two
questions asked five minutes apart.

## Why three forecasting models, not one

A single model, chosen once, either overfits to whichever benchmark you first
tried it on, or quietly loses to a trivial baseline on some slice of the data
you didn't examine. Competing three qualitatively different approaches and
selecting **per series** based on rolling-origin backtest performance is the
"best-fit selection" pattern named explicitly in the Confido JD, and it produces
an honest, occasionally humbling result here: seasonal-naive — literally "same
week last year" — wins on 90 of 240 series. That's not a failure of the more
sophisticated models; those 90 series are dominated by strong, low-noise annual
seasonality where recent history genuinely doesn't help. A system that always
promotes the fanciest model regardless of the data would perform *worse* on
those series than one honest enough to fall back to the baseline. See
`EVAL_REPORT.md` for the full numbers.

## Why a global model instead of one model per series

`src/forecasting.py`'s `GlobalGBMModel` is trained ONCE across all 240 series,
with store/dept as features rather than 240 separate models. This mirrors how
production forecasting at retail scale actually works (this is the standard
approach in, e.g., M5-competition-style systems): a single global model shares
statistical strength across series — including sparse, low-volume ones that
would badly overfit a lag-feature model trained on their own ~250 data points —
and it's one artifact to version, monitor, and retrain instead of hundreds.

## Substitutions made because of this build environment

This project was built inside a sandboxed cloud container with restricted
outbound network access. Two things that would normally just be `pip install`
weren't available, so the architecture uses a documented substitute. Both
substitutions are isolated to one file each — nothing about the pipeline's
design, correctness, or the rest of this document depends on which specific
library is behind them.

- **LightGBM → `sklearn.ensemble.HistGradientBoostingRegressor`.** Same
  algorithm family (histogram-based gradient-boosted trees), broadly
  comparable accuracy/speed tradeoff. Confined to `src/forecasting.py`.
- **FastAPI → Flask.** Same REST semantics (routes, JSON in/out), no async
  support. Confined to `app/server.py`; the dashboard's fetch calls don't care
  which framework served them.
- **statsmodels' `ExponentialSmoothing` → a from-scratch multiplicative
  Holt-Winters** with a small fixed grid search over smoothing constants
  instead of MLE-fit ones (see the module docstring in `src/forecasting.py`).
  This is the one substitution with a real (small) accuracy cost, which is
  visible in the backtest: Holt-Winters is the weakest of the three models
  here. A production system would use the MLE-fit version.
- **pytest → stdlib `unittest`.** No functional difference for this test
  suite; `python -m unittest discover` runs everything pytest would.
- **Real Kaggle/UCI/FRED dataset → a synthetic-but-schema-identical
  generator.** This one isn't a "same thing, different library" swap — see
  `docs/DATA_PROVENANCE.md` for the full reasoning and how to swap in the
  real dataset yourself.

## Prediction intervals and explainability

Two capabilities were added after the initial build to make the forecasts
themselves more trustworthy, not just accurate on average:

- **`src/intervals.py`** — every forecast ships an 80% prediction interval,
  not just a point number. global_gbm series get it from two extra
  quantile-loss GBMs (`src/forecasting.py::QuantileGBMModel`, q=0.10 and
  q=0.90); seasonal_naive/Holt-Winters series (no natural quantile
  analogue) get a normal-approximation band scaled by that series'
  backtested RMSE. `scripts/run_pipeline.py` validates the real,
  held-out coverage of these bands every run (`reports/interval_coverage.json`)
  rather than asserting they're calibrated — see `docs/EVAL_REPORT.md` §4
  for the honest number (74.7% actual vs. 80% nominal) and why it runs
  low.
- **`src/explain.py`** — `shap` isn't installable in this sandbox, so local
  feature attribution uses occlusion (median-replacement per feature,
  documented as a simplified, non-Shapley approximation) and global
  importance uses `sklearn.inspection.permutation_importance` (a real,
  standard method, computed in-sample -- see `docs/EVAL_REPORT.md` §5).
- **`src/scenario.py`** — what-if simulation. This is the ONE tool in the
  whole system that doesn't just read a precomputed `reports/*.csv`
  artifact: a hypothetical covariate combination (e.g. "run a markdown
  promotion these next 3 weeks") was never scored offline by definition,
  so it reuses the persisted point GBM (`reports/models/gbm_point.joblib`)
  to recompute live. Both this and the explainability tool are only
  meaningful for series whose selected model is global_gbm; both say so
  explicitly rather than returning a fabricated number for
  seasonal_naive/Holt-Winters series.

All three trained artifacts these depend on (`gbm_point.joblib`,
`gbm_q10.joblib`, `gbm_q90.joblib`, `feature_medians.joblib`) are persisted
by `scripts/run_pipeline.py` via `joblib`, so the agent/dashboard reuse the
EXACT model that produced the baseline forecast rather than silently
retraining a slightly different one on demand.

## The agent layer

`src/agent.py` implements tool use (`get_forecast`, `get_anomalies`,
`explain_change`, `top_movers`, `explain_forecast_drivers`,
`simulate_scenario`) behind a pluggable "brain":

- **`MockLLM`** (default): a deterministic keyword/regex router. No API key,
  no external calls, fully reproducible — good for CI, tests, and demos. Its
  entire job is to prove the *architecture* (tools + audit trail +
  orchestration) works independent of any model provider.
- **`AnthropicLLM`**: a real tool-calling loop against the Anthropic Messages
  API over plain HTTP (no SDK dependency — just `requests`), activated
  automatically the moment `ANTHROPIC_API_KEY` is set in the environment.
- **`OpenAILLM`**: the same real tool-calling loop against OpenAI's Chat
  Completions API, same plain-HTTP style, activated automatically when
  `OPENAI_API_KEY` is set instead. Model defaults to `gpt-4o-mini`,
  overridable via `OPENAI_MODEL`.
- **`BedrockLLM`**: the same tool-calling loop again, this time against AWS
  Bedrock's Converse API via `boto3`, activated automatically when
  `BEDROCK_MODEL_ID` is set (deliberately not a generic AWS-credential
  check — see `get_llm_backend()`'s own docstring for why). This is the one
  backend that uses a vendor SDK rather than plain HTTP: AWS auth is
  SigV4-signed, not a bearer token, and `boto3` is the standard SDK needed
  anyway for the S3/Lambda/SageMaker calls in `infra/` — see
  `infra/README.md` for the full AWS deployment this backend is part of.

Swapping backends touches zero tool code and zero trace format — only which
component decides which tool to call next. Selection order in
`get_llm_backend()`: an explicit `LLM_BACKEND=anthropic|openai|bedrock|mock`
environment variable always wins; otherwise Anthropic is preferred if its key
is present, then OpenAI, then Bedrock, then the mock. This multi-provider
pluggability directly answers Condor's job description, which names both
"Anthropic Claude or OpenAI" (and SageMaker/Bedrock elsewhere in the same
JD) as acceptable providers — a system that hard-codes one vendor's SDK
into its tool-calling logic can't honor that requirement.

Every call, real or mock, produces an **`AgentTrace`**: the question, each
tool call with its arguments and result, and the final answer, persisted to
`reports/agent_traces.jsonl`. This is the auditability requirement called out
explicitly in the Condor JD ("correctness and auditability are non-negotiable")
generalized to any domain where "trust the AI's answer" needs to mean
"and here's exactly how it got there."

**Resilience and cost tracking for the three real backends.** A live API call
can fail transiently (a momentary rate limit, a dropped connection, a 5xx),
and every real-LLM call costs real tokens. Both are handled in `src/agent.py`,
shared across `AnthropicLLM`/`OpenAILLM`/`BedrockLLM`:

- `_retry_with_backoff()` retries only actually-transient failures (429/5xx
  HTTP status, connection/timeout errors, Bedrock's `ThrottlingException`/
  `ServiceUnavailable`/etc.) with exponential backoff + jitter, up to 4
  attempts. A 401 (bad key) or 400 (malformed request) is never retried --
  retrying something that will fail identically four times just burns
  latency and, for a paid API, money.
- Every real API response's token usage is accumulated onto the `AgentTrace`
  (`token_usage`, `estimated_cost_usd`, `retries`) across every tool-calling
  turn in a single `ask()` call, and persisted to `reports/llm_usage.jsonl`
  (skipped for `MockLLM`, which never makes a network call and so has
  nothing to cost). Pricing (`LLM_PRICING_USD_PER_1M_TOKENS`) is disclosed as
  approximate/illustrative, not billing-accurate -- provider list prices
  change, and this is for cost-awareness in the trace, not an invoice.

## AWS deployment

`infra/` deploys this platform onto AWS using the exact services Condor's
job description names — SageMaker (a nightly Processing Job runs the
batch pipeline; an optional Serverless Inference endpoint serves the point
GBM model), Bedrock (the fourth `src/agent.py` backend above), and Lambda
(the API layer, behind API Gateway). It's real Terraform + real
integration code, not a narrative description — but it was written in the
same network-restricted sandbox as the rest of this project, so it hasn't
been run against a live AWS account. See `infra/README.md` for the full
architecture diagram, deployment steps, cost estimate, and — in the same
spirit as the Docker/CI disclosure below — an explicit list of what's
verified by a passing test run here versus what's "written to spec, not
observed running."

## Closing the monitoring loop

`scripts/run_monitoring_sim.py` flags individual series whose *deployed*
model's accuracy has drifted (see `docs/EVAL_REPORT.md` §3) -- but a flagged
series that nobody acts on is just a CSV nobody reads. `scripts/retrain_flagged.py`
runs immediately after it and turns each flag into a decision:

1. For each flagged series, look up its three candidate models' WAPE at the
   exact latest rolling-origin cutoff the drift alert itself fired on
   (`reports/per_cutoff_metrics.csv` -- already computed, leakage-free, by
   the original `run_backtest()`, so nothing new is scored here that risks
   the kind of lookahead leakage `tests/test_features.py` explicitly guards
   against elsewhere).
2. If a different model now wins at that latest cutoff, re-select it,
   recompute that series' forward forecast (reusing the already-persisted
   `gbm_point`/`gbm_q10`/`gbm_q90` models -- see below for why this
   deliberately does NOT refit the global GBM), and update
   `reports/selected_model.json` + `reports/current_forecasts.csv` for that
   series only.
3. Either way -- reselected or left alone -- append one line to
   `reports/retraining_log.jsonl`, an audit trail in the same spirit as
   `reports/agent_traces.jsonl`. "Reviewed and confirmed the deployed model
   is still best" is a real, useful, loggable outcome, not a null result to
   hide.

**Why this doesn't refit the global GBM.** The GBM is trained ONCE across
all 240 series (see "Why a global model instead of one model per series"
above) -- refitting it is what `scripts/run_pipeline.py` does for the whole
fleet, not something that makes sense to trigger for a handful of flagged
series. `retrain_flagged.py` reuses the persisted model as-is; what it
retrains, in the literal sense, is the *classical* per-series
candidates (seasonal-naive/Holt-Winters are recomputed directly from each
series' latest history, which is all "fitting" ever meant for them) and the
*selection decision* among all three.

**Why "latest single cutoff" instead of a wider recent window.** It's
noisier than the 6-cutoff mean the original selection uses, but it's
literally the metric the drift alert fired on -- see
`docs/EVAL_REPORT.md`'s "what a next iteration would change" for the
wider-window refinement this doesn't attempt.

## What's still a demo, not a production system

Being direct about the gap matters more than papering over it:

- Still **no authentication, rate limiting, or multi-tenancy** on the API --
  that gap is real and unaddressed. What IS handled now (`app/server.py`):
  every query param is validated with a clean 400 JSON error instead of an
  unhandled 500 on bad input, request bodies are size-capped
  (`MAX_CONTENT_LENGTH`), CORS is opt-in via an explicit origin allowlist
  (`CORS_ALLOWED_ORIGINS`, unset by default = no cross-origin access) rather
  than wide open, every request is logged with method/path/status/duration,
  and both HTTP errors and unexpected exceptions come back as JSON, never
  Flask's default HTML error/debug page. None of that is a substitute for
  auth or rate limiting -- it closes the "malformed input 500s the server"
  and "errors leak an HTML stack trace" gaps, not the "who is allowed to
  call this at all" gap.
- The Flask dev server is explicitly not a production WSGI server (the
  console warns about this on startup). `requirements.txt` and the
  `Dockerfile` both document the gunicorn swap-in (commented out, with the
  exact commands) rather than silently applying it -- gunicorn itself
  couldn't be installed in this build sandbox to verify the swap actually
  works end-to-end, so "documented" and "verified" are kept as distinct
  claims here.
- Future macro covariates (CPI, unemployment, fuel price) are held at their
  last observed value for the forward forecast rather than forecast
  themselves — a documented simplifying assumption (see
  `scripts/run_pipeline.py::make_future_covariates`).
- A GitHub Actions workflow (`.github/workflows/tests.yml`) now runs the
  full pipeline + test suite + an API smoke test on every push/PR, and a
  `Dockerfile` bakes a reproducible image (data generation + backtest +
  monitoring run at build time, so every image is a versioned snapshot).
  Neither was validated by actually running in GitHub's infrastructure or
  a live Docker daemon from inside this build sandbox (no Docker daemon was
  available here to build against) — both follow standard, unexotic patterns,
  but "written correctly" and "observed passing in CI" are different claims,
  and only the first is certain yet.
- The monitoring "simulation" replays the backtest's historical cutoffs
  rather than watching genuinely new weekly data arrive, since this is a
  point-in-time snapshot, not a running service. The monitoring *logic*
  (fleet-level drift + per-series drift, both computed from the deployed
  model's own accuracy history) is exactly what you'd wire up to a real
  scheduler.
