# Hints (Phase 3 only)

> Do not read this file in Phase 1 or Phase 2. Performance hints bias both knowledge curation and initial implementation away from simplicity.

## Profiling discipline

- Before iter 1, run full two-stage profiling on the baseline:
  1. End-to-end: `PYTHONPATH=. modal run -m bench.modal_profile --workload-index <N>` (always use `-m`). Identifies which stage is dominant.
  2. NCU deep-dive on the dominant kernel: `PYTHONPATH=. modal run -m bench.modal_ncu --workload-index <N>`.
- Fetch traces from Modal volume after end-to-end profiling: `modal volume get flashinfer-trace profiles/ ./profiler_results/`.
- NCU reports print to stdout and save to `./ncu_results/` locally; no fetch needed.

## Stall rule

- If 5 consecutive iterations show no improvement, stop editing code. Re-profile, re-read `knowledge/INDEX.md` grouped by the observed bottleneck class, and write a plan into `ITERATIONS.md` before the next iteration.

## Cleanup rule

- Every 5th iteration (N = 5, 10, 15, ...), run a cleanup pass in its own `[iter-N cleanup]` commit **before** opening iteration N+1. Full procedure: see "Cleanup Cadence" in `.claude/commands/phase3-optimize.md`.
- Minimum checklist: delete dead branches and commented-out code; drop autotune configs that have not won in the last 5 iterations; remove `tl.device_print` / debug `print`; collapse scratch reasoning in the last 5 `ITERATIONS.md` rows into a single "Summary after iter N" block.
- Re-bench after cleanup (`bash scripts/bench.sh iter-N-cleanup`). `CORRECT=True` must still hold, and medians must not regress >1% vs. iteration N. If they do, the cleanup removed something live — revert.
- Rationale: without this, `solution/triton/kernel.py` accumulates dead code and the agent's working context balloons with scratch notes it no longer needs.

## Environment

- No local GPU. All benchmarks and profiling run on Modal (NVIDIA B200, sm_100). Do not test locally.
- **Activate the pre-built conda env before any `modal` command**: `eval "$(conda shell.bash hook)" && conda activate fi-bench`. `scripts/bench.sh` does this for you; when you run `modal run -m bench.modal_profile` / `bench.modal_ncu` directly, you must activate it first or `modal` will not be on PATH. Do not `pip install` or create a new env.
- Pre-installed on Modal: `flashinfer-bench`, `modal`, `sol-execbench`. Do not reinstall.
- Kernel entry point: `run()` in `solution/triton/kernel.py` (DPS style — output tensor passed as the last parameter) unless the harness says otherwise.
- Always pass `-m` (module) flag to `modal run`. File-path syntax causes import failures.

## Workload hints (adjust per project)

- Short-sequence workloads are usually memory-bandwidth or latency bound. Fine-tuning block sizes rarely helps; look at data layout, fusion, and weight/activation footprint.
- Long-sequence workloads are usually compute-bound. Look at MMA utilization, async pipeline depth, and SMEM reuse.
- Routing/dispatch stages often dominate at the shortest sequence lengths — a change in the compute stage may be invisible until the routing stage is addressed.


## What not to spend iterations on

- Fine-tuning existing block sizes when they are already in the right neighborhood.
- Re-running autotune sweeps hoping for a different answer.
- Micro-optimizations that cannot be tied to a specific profiler finding.
