#!/bin/bash
# Benchmark wrapper with trajectory tracking.
#
# Usage: bash scripts/bench.sh [label]
#
# Phase-2 setup (or the user) replaces the {{BENCH_COMMAND}} placeholder with
# a concrete modal invocation, e.g.:
#
#   eval "$(conda shell.bash hook)" && conda activate fi-bench && \
#     PYTHONPATH=. modal run -m bench.modal_bench \
#       --label "$LABEL" --json-out "$BENCH_JSON" 2>&1 | tee _bench_output.txt
#
# The wrapper exports BENCH_JSON ahead of {{BENCH_COMMAND}} so the bench
# invocation can opt in to a trajectory/*/bench_result.json side-channel
# matching context/schemas/bench_result.schema.json.
set -eo pipefail
cd "$(dirname "$0")"

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
export BENCH_JSON="$TRAJ_DIR/bench_result.json"

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
{{BENCH_COMMAND}}
BENCH_EXIT=$?
set -e
# --- End bench command ---

# --- Trajectory ---
cp -r solution/* "$TRAJ_DIR/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
