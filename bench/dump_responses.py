"""Dump every endpoint's response for a wide parameter sweep, as canonical JSON.

This is the equivalence check behind the README's "Verifying nothing
changed": run it in a git worktree of the commit before an optimization and
again in the optimized tree, then diff the two files. Any difference in any
response, for any series, week_ahead, scenario variant or error path, shows
up as a diff. The agent trace's wall-clock `ts` is the one field that is
expected to differ between runs.

    git worktree add /tmp/before <commit-before>
    cp -R reports data/*.csv /tmp/before/          # gitignored artifacts
    (cd /tmp/before && python ../bench/dump_responses.py /tmp/before.json)
    python bench/dump_responses.py /tmp/after.json
    python -c "import json;a=json.load(open('/tmp/before.json'));b=json.load(open('/tmp/after.json'));print([k for k in a if a[k]!=b[k]])"
"""
import json, logging, sys, os
sys.path.insert(0, os.getcwd())
logging.disable(logging.CRITICAL)
from app.server import app
from src.agent import _store

c = app.test_client()
ds = _store()
keys = sorted(ds.selected_model)
out = {}

def rec(label, resp):
    out[label] = json.loads(resp.data)

rec("series", c.get("/api/series"))
rec("eval", c.get("/api/eval/summary"))
for k in keys:
    s, d = int(k[1:3]), int(k[5:7])
    rec(f"forecast/{k}", c.get(f"/api/forecast?store={s}&dept={d}"))
    rec(f"anom_flagged/{k}", c.get(f"/api/anomalies?store={s}&dept={d}"))
    rec(f"anom_all/{k}", c.get(f"/api/anomalies?store={s}&dept={d}&only_flagged=false"))
    rec(f"hist104/{k}", c.get(f"/api/history?store={s}&dept={d}"))
    rec(f"hist7/{k}", c.get(f"/api/history?store={s}&dept={d}&weeks=7"))
# heavy endpoints across all series and several week_aheads / scenarios
for k in keys:
    s, d = int(k[1:3]), int(k[5:7])
    for wa in (1, 3, 8, 99):
        rec(f"explain/{k}/{wa}", c.get(f"/api/explain?store={s}&dept={d}&week_ahead={wa}"))
    for md, hol, w in (("true", None, 4), ("false", None, 8), ("true", "true", 2), ("true", "false", 6)):
        q = f"/api/simulate?store={s}&dept={d}&markdown_active={md}&weeks={w}"
        if hol: q += f"&is_holiday={hol}"
        rec(f"sim/{k}/{md}/{hol}/{w}", c.get(q))
# agent questions
qs = ["what are the biggest drops this week?", "what are the top movers up?",
      "biggest decline?", "what is driving the forecast for store 1 dept 2?",
      "what's driving the forecast for store 3 dept 2 in week 5?",
      "what if a markdown promo ran for 6 weeks at store 1 dept 2?",
      "what if it were a holiday at store 2 dept 2?",
      "forecast for store 5 dept 9", "why did store 1 dept 1 change?",
      "any anomalies for store 4 dept 7?"]
for q in qs:
    r = c.post("/api/ask", json={"question": q})
    t = json.loads(r.data)
    # the trace carries a wall-clock duration; drop it, it is not an output
    for step in t.get("steps", []):
        step.pop("duration_ms", None)
    t.pop("duration_ms", None); t.pop("total_duration_ms", None)
    out[f"ask/{q}"] = t
# error paths
for q in ["/api/forecast?store=abc&dept=1", "/api/forecast?store=0&dept=1",
          "/api/history?store=1&dept=1&weeks=0", "/api/forecast?store=999&dept=999",
          "/api/explain?store=999&dept=999", "/api/nope"]:
    r = c.get(q); out[f"err{q}"] = [r.status_code, json.loads(r.data)]
r = c.post("/api/ask", json={"nope": 1}); out["err/ask"] = [r.status_code, json.loads(r.data)]
json.dump(out, open(sys.argv[1], "w"), sort_keys=True, indent=0, default=str)
print(f"{len(out)} responses -> {sys.argv[1]}")
