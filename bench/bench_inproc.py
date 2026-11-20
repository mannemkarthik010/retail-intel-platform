"""Single-threaded, in-process per-endpoint timings.

No HTTP, no gunicorn, no queueing: just the application cost of handling
one request, so before/after is attributable to the code rather than to
how saturated the worker pool happened to be. OMP is pinned explicitly in
BOTH trees here so this isolates the code changes from the thread-pool fix.
"""
import json, logging, os, statistics, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.getcwd())
logging.disable(logging.CRITICAL)
from app.server import app
from src.agent import _store

c = app.test_client()
ds = _store()
gbm = [k for k, v in ds.selected_model.items() if v == "global_gbm"]
allk = sorted(ds.selected_model)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

def sweep(label, make_path, keys, n=N, post=None):
    # warm: one pass so lazy artifact loads / first-predict costs are out
    for i in range(min(8, len(keys))):
        req(make_path(keys[i % len(keys)]), post)
    ts = []
    for i in range(n):
        p = make_path(keys[i % len(keys)])
        t = time.perf_counter(); req(p, post); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"endpoint": label, "n": n,
            "p50": round(statistics.median(ts), 2),
            "p95": round(ts[max(0, int(0.95 * n) - 1)], 2),
            "mean": round(statistics.fmean(ts), 2),
            "max": round(ts[-1], 2)}

def req(path, post):
    if post is None:
        r = c.get(path)
    else:
        r = c.post(path, json=post)
    assert r.status_code == 200, (path, r.status_code)

def sd(k): return (int(k[1:3]), int(k[5:7]))

results = []
results.append(sweep("GET /api/forecast", lambda k: "/api/forecast?store=%d&dept=%d" % sd(k), allk))
results.append(sweep("GET /api/anomalies", lambda k: "/api/anomalies?store=%d&dept=%d" % sd(k), allk))
results.append(sweep("GET /api/history", lambda k: "/api/history?store=%d&dept=%d" % sd(k), allk))
results.append(sweep("GET /api/series", lambda k: "/api/series", ["x"]))
results.append(sweep("GET /api/eval/summary", lambda k: "/api/eval/summary", ["x"]))
results.append(sweep("GET /api/simulate", lambda k: "/api/simulate?store=%d&dept=%d&markdown_active=true&weeks=4" % sd(k), gbm, n=min(N, 40)))
results.append(sweep("POST /api/ask (top_movers)", lambda k: "/api/ask", ["x"], n=min(N, 20),
                     post={"question": "what are the biggest drops this week?"}))
# /api/explain: report cold (first touch of a series on this process) and
# warm (series already in the recursion cache) separately -- they are
# genuinely different costs and averaging them hides both.
cold = []
for k in gbm[:40]:
    s, d = sd(k)
    t = time.perf_counter(); req(f"/api/explain?store={s}&dept={d}&week_ahead=1", None)
    cold.append((time.perf_counter() - t) * 1000)
cold.sort()
results.append({"endpoint": "GET /api/explain (cold series)", "n": len(cold),
                "p50": round(statistics.median(cold), 2),
                "p95": round(cold[max(0, int(0.95 * len(cold)) - 1)], 2),
                "mean": round(statistics.fmean(cold), 2), "max": round(cold[-1], 2)})
results.append(sweep("GET /api/explain (warm series)",
                     lambda k: "/api/explain?store=%d&dept=%d&week_ahead=%d" % (sd(k) + ((hash(k) % 8) + 1,)),
                     gbm[:40], n=min(N, 60)))
json.dump(results, open(sys.argv[1], "w"), indent=1)
w = max(len(r["endpoint"]) for r in results)
for r in results:
    print(f'{r["endpoint"]:<{w}}  n={r["n"]:>3}  p50={r["p50"]:>9.2f}ms  p95={r["p95"]:>9.2f}ms  mean={r["mean"]:>9.2f}ms')
