"""Load-test profile for the serving API (`app/server.py`).

Run it through bench/run_load_test.sh, which starts gunicorn, warms every
worker, then drives this file headless and writes the p50/p95/throughput
CSVs into bench/results/.

The task weights approximate real dashboard + agent traffic rather than
hammering one route: the cheap artifact reads the dashboard issues on every
page load dominate by count, and the expensive model-inference routes
(/api/explain, /api/simulate, /api/ask) are the minority of requests that
nonetheless set the tail. Store/dept pairs are drawn from the series whose
selected model is the global GBM, because /api/explain and /api/simulate
short-circuit with a clean error on seasonal_naive/holt_winters series --
pointing the load test at those would "measure" a branch that never runs
the model and report a flattering number.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent
SELECTED = json.loads((ROOT / "reports" / "selected_model.json").read_text())

# Series the heavy endpoints can actually do work for.
GBM_SERIES = [(int(k[1:3]), int(k[5:7])) for k, v in SELECTED.items() if v == "global_gbm"]
# Every series, for the cheap artifact reads.
ALL_SERIES = [(int(k[1:3]), int(k[5:7])) for k in SELECTED]


class ApiUser(HttpUser):
    # A dashboard user clicking around, not a tight spin loop.
    wait_time = between(0.05, 0.2)

    @task(10)
    def forecast(self):
        s, d = random.choice(ALL_SERIES)
        self.client.get(f"/api/forecast?store={s}&dept={d}", name="/api/forecast")

    @task(6)
    def history(self):
        s, d = random.choice(ALL_SERIES)
        self.client.get(f"/api/history?store={s}&dept={d}", name="/api/history")

    @task(6)
    def anomalies(self):
        s, d = random.choice(ALL_SERIES)
        self.client.get(f"/api/anomalies?store={s}&dept={d}", name="/api/anomalies")

    @task(2)
    def series(self):
        self.client.get("/api/series", name="/api/series")

    @task(2)
    def eval_summary(self):
        self.client.get("/api/eval/summary", name="/api/eval/summary")

    @task(4)
    def explain(self):
        s, d = random.choice(GBM_SERIES)
        week = random.randint(1, 8)
        self.client.get(f"/api/explain?store={s}&dept={d}&week_ahead={week}",
                        name="/api/explain")

    @task(3)
    def simulate(self):
        s, d = random.choice(GBM_SERIES)
        self.client.get(f"/api/simulate?store={s}&dept={d}&markdown_active=true&weeks=4",
                        name="/api/simulate")

    @task(2)
    def ask_top_movers(self):
        self.client.post("/api/ask", json={"question": "what are the biggest drops this week?"},
                         name="/api/ask (top_movers)")

    @task(1)
    def ask_drivers(self):
        s, d = random.choice(GBM_SERIES)
        self.client.post("/api/ask",
                         json={"question": f"what is driving the forecast for store {s} dept {d}?"},
                         name="/api/ask (drivers)")
