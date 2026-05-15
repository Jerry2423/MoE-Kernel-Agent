#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate fi-bench

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Resolve trajectory dir up front so modal_bench --json-out can write into it.
if [ -n "$LABEL" ]; then
    TRAJ_DIR="../trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="../trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
BENCH_JSON_ABS="$(cd "$TRAJ_DIR" && pwd)/bench_result.json"

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
cd .. && PYTHONPATH=. modal run -m bench.modal_bench \
    --label "$LABEL" --json-out "$BENCH_JSON_ABS" 2>&1 | tee _bench_output.txt
# NOTE: Always use `modal run -m bench.<script>` (module syntax), not `modal run bench/<script>.py`
BENCH_EXIT=$?
set -e
# --- End bench command ---

# --- Trajectory ---
TRAJ_DIR_ABS="$(cd "$(dirname "$BENCH_JSON_ABS")" && pwd)"
cp -r solution/* "$TRAJ_DIR_ABS/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR_ABS/output.txt"
echo "Trajectory saved to: $TRAJ_DIR_ABS"

exit $BENCH_EXIT
