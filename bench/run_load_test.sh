#!/usr/bin/env bash
# Reproducible load test of the serving API.
#
#   bench/run_load_test.sh <label> [users] [run_time]
#
# Starts gunicorn with 4 sync workers (the configuration the Dockerfile
# documents for real deployment -- NOT the Flask dev server, which is
# single-threaded and would measure the server's concurrency ceiling
# instead of the application's cost), warms every worker so the first
# request's lazy artifact load doesn't land in the measured window, then
# drives bench/locustfile.py headless and writes CSVs to bench/results/.
set -euo pipefail

LABEL="${1:?usage: run_load_test.sh <label> [users] [run_time]}"
USERS="${2:-20}"
RUN_TIME="${3:-60s}"
WORKERS=4
PORT="${PORT:-8000}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PERF_VENV:?set PERF_VENV to a venv containing gunicorn + locust}"
OUT="$ROOT/bench/results/$LABEL"
mkdir -p "$OUT"

cd "$ROOT"
"$VENV/bin/gunicorn" -w "$WORKERS" -b "127.0.0.1:$PORT" \
    --worker-class sync --timeout 120 --log-level warning \
    app.server:app &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for the port, then warm each worker on every route. gunicorn's sync
# workers each build their own DataStore lazily on first use (7.5 MB of
# CSV) and sklearn's first predict pays a one-off import/JIT cost; both
# are startup effects, not steady-state serving cost.
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break
    sleep 0.5
done
echo "warming $WORKERS workers..."
"$VENV/bin/python" "$ROOT/bench/warmup.py" "http://127.0.0.1:$PORT" "$WORKERS"

echo "load test '$LABEL': $USERS users, $RUN_TIME"
"$VENV/bin/locust" -f "$ROOT/bench/locustfile.py" \
    --headless --users "$USERS" --spawn-rate "$USERS" --run-time "$RUN_TIME" \
    --host "http://127.0.0.1:$PORT" \
    --csv "$OUT/stats" --csv-full-history --only-summary

echo "results -> $OUT"
