"""Pre-warm every gunicorn worker before the measured window.

Each sync worker is a separate process that builds its own DataStore
(reads ~7.5 MB of pipeline CSVs) lazily on first request, and sklearn pays
a one-off cost on its first predict in a process. Those are startup
effects; leaving them inside the load-test window would show up as a fake
multi-hundred-ms tail. Enough sequential requests to hit every worker
several times over removes them.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELECTED = json.loads((ROOT / "reports" / "selected_model.json").read_text())
GBM = [(int(k[1:3]), int(k[5:7])) for k, v in SELECTED.items() if v == "global_gbm"]

base = sys.argv[1]
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def get(path):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        r.read()


def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


# 6 rounds x every route: comfortably more than enough requests for the
# round-robin accept() to have handed each of `workers` processes several.
for i in range(6 * workers):
    s, d = GBM[i % len(GBM)]
    get("/api/health")
    get("/api/series")
    get("/api/eval/summary")
    get(f"/api/forecast?store={s}&dept={d}")
    get(f"/api/history?store={s}&dept={d}")
    get(f"/api/anomalies?store={s}&dept={d}")
    get(f"/api/explain?store={s}&dept={d}&week_ahead=1")
    get(f"/api/simulate?store={s}&dept={d}&markdown_active=true&weeks=4")
    post("/api/ask", {"question": "what are the biggest drops this week?"})
print("warm")
