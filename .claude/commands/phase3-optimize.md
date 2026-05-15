---
description: Phase 3 — classify workloads, optimize per category under profile discipline, then integrate.
---

# Phase 3: Optimization

Your job in this phase is to reduce runtime while preserving correctness. Phase 3 has **three sub-phases**, in order: **3.A Classify**, **3.B Per-Category Optimization**, **3.C Integrate**. Do not skip, reorder, or merge them. Optimizing without classification leads to wins on one workload shape that regress another; integrating without per-category wins leads to a kernel that is average everywhere.

## Entry Check

Before doing anything:
1. Read `TASK.md`, `RULES.md`, `HINTS.md`, `HANDOFF.md`, `ITERATIONS.md`, `knowledge/INDEX.md`, `solution/autotune_notes.md`, `solution/fusion_notes.md`, `solution/subgraph_specs.json`, the workload list at `context/moe_workloads.jsonl` (or the project-specific workload file), and `bench/GUIDE.md` §"Script → Phase Map".
2. Verify `HANDOFF.md` shows Phase 2 complete. If not, stop and tell the user.
3. Run `bash scripts/bench.sh` once to confirm `CORRECT=True` on the current `solution/`. If correctness is broken, stop and report.

## Script selection (from bench/GUIDE.md)

- Correctness + timing → `bash scripts/bench.sh iter-N` (wraps `modal_bench.py`).
- Stage 1 profile (which stage dominates) → `PYTHONPATH=. modal run -m bench.modal_profile --workload-index <N>`, then `modal volume get flashinfer-trace profiles/ ./profiler_results/`.
- Stage 2 profile (deep-dive on one kernel) → `PYTHONPATH=. modal run -m bench.modal_ncu --workload-index <N>` (optionally `--kernel-name`, `--sections`).
- Full autotune sweep → `PYTHONPATH=. modal run -m bench.modal_autotune`. **3.C only**, and only when the algorithmic change justifies it per the freeze rule below. If your change renamed or added kernels, update `modal_autotune.py` first (see `bench/GUIDE.md` §"When to Update Which Script").

## Autotune Freeze (inherited from Phase 2)

Phase 2 already ran autotune and froze the winning configs into `solution/triton/kernel.py`. **Do not re-run autotune sweeps in Phase 3.** Re-tuning is allowed *only* when you have introduced a significant algorithmic change — specifically:

- a new kernel boundary (merging or splitting subgraphs from `subgraph_specs.json`),
- a fundamentally different data layout (e.g., swapping NCHW ↔ NHWC, changing weight packing),
- a new quantization scheme or dtype,
- switching from one algorithmic family to another (e.g., naive GEMM → split-K, grouped GEMM → persistent GEMM).

For anything short of that — block-size tweaks, stage-count tweaks, warp-count tweaks — use the existing frozen config. If you believe the frozen config is wrong for a workload, treat it as a bug in Phase 2's autotune search space, document it in `ITERATIONS.md`, and fix the search space once rather than sweeping every iteration.

---

## Sub-Phase 3.A — Workload Classification

Goal: partition the benchmark workloads into a small number of classes that share a bottleneck profile. Every subsequent optimization iteration is tagged to one class. This prevents wins on class X from silently regressing class Y.

**Procedure:**
1. Read every workload in `context/moe_workloads.jsonl` (or equivalent). Extract the distinguishing axes — for MoE the primary axis is `seq_len`; record all of them.
2. For each workload, run (or read, if a baseline run already produced them) the two-stage profile:
   - `PYTHONPATH=. modal run -m bench.modal_profile --workload-index <N>`
   - `PYTHONPATH=. modal run -m bench.modal_ncu --workload-index <N>` on the dominant kernel.
   If many workloads share an axis value range, profile only representative samples (e.g., one short, one medium, one long) — but justify the sampling in `workload_classes.json`.
3. Cluster workloads into classes by **dominant bottleneck**, not by shape alone. Common classes for MoE:
   - *routing/latency-bound short* — dispatch overhead and small-M GEMM dominate.
   - *memory-bandwidth-bound medium* — weight-load dominates, compute is underutilized.
   - *compute-bound long* — MMA pipeline saturation, SMEM reuse matters.
   Use whatever classes the profile data actually supports. Do not invent classes from shape ranges without profile evidence.
4. Record the classes and the workload-to-class mapping in `solution/workload_classes.json`:
   ```json
   [
     {
       "class_id": "short_routing_bound",
       "bottleneck": "<one sentence from profile>",
       "workload_indices": [1, 7, 14, 15],
       "axes_range": {"seq_len": [1, 16]},
       "representative_index": 1,
       "target_metric": "<e.g., routing-kernel latency (us)>"
     },
     ...
   ]
   ```
5. Commit: `git commit -m "[phase3-A] Workload classification"`.

**Hard rules for 3.A:**
- No code edits under `solution/triton/` in this sub-phase. Classification is measurement only.
- Every class must cite a specific profiler finding (stall reason, bandwidth cap, kernel name, etc.). Uncited classes are not valid.
- If two classes look identical after profiling, merge them.

---

## Sub-Phase 3.B — Per-Category Optimization

Goal: optimize each class independently against its *representative workload*. Track iterations per class. Never optimize against a workload you have not profiled.

### Iteration Protocol

Every modification to `solution/` followed by a benchmark run counts as one iteration. Number iterations sequentially across the whole phase (not per class) in `ITERATIONS.md`, and tag each row with its `class_id`. For each iteration, **all of the following must be completed before starting the next one**:

1. **Pick a target class and a target bottleneck from the profile.** The target must correspond to a specific bottleneck in that class's profile (stall reason, bandwidth cap, occupancy limit, etc.). Record `class_id` and target in `ITERATIONS.md` before editing code.
2. **Apply one change.** One technique per iteration. Do not bundle. Cite the knowledge file or skill the technique comes from.
3. **Measure on the representative workload for the target class.** Run `bash scripts/bench.sh iter-N` (label required). Run it **5 times** and record the median.
4. **Regression guard on other classes.** After the representative-workload run, run the bench across the representative workloads for *every other class* you have already optimized. If any previously-optimized class regresses beyond a pre-agreed tolerance (default: 3% on median runtime), treat the iteration as a regression and revert before continuing.
5. **Profile.** Two stages:
   - End-to-end: `PYTHONPATH=. modal run -m bench.modal_profile --workload-index <representative_index>` to find the new dominant stage for that class.
   - Kernel deep-dive: `PYTHONPATH=. modal run -m bench.modal_ncu --workload-index <representative_index>` on the dominant kernel.
   Always use the `-m` module flag with `modal run`.
6. **Update both iteration logs** — machine-readable *and* human-readable:
   - Append one JSON object to `solution/iterations.jsonl` matching `context/schemas/iteration_row.schema.json`. Minimum required fields: `iter`, `class_id`, `technique`, `source` (path to the knowledge file or skill), `status` ∈ {improved, no-change, regression, failed, reverted}. Recommended: `median_ms`, `baseline_ms`, `worst_regression_pct`, `bottleneck_next`, `commit` (fill after step 7 with the short hash).
   - Append a matching row to `ITERATIONS.md` for humans: iter number, `class_id`, title, median speedup on representative, worst regression across other classes, status, and one-line bottleneck observation for the next iteration.
   Run `python3 tools/validate_artifacts.py --quiet` — any schema error must be fixed before committing.
7. **Commit** — `git commit -m "[iter N][<class_id>] <short description>"`. Commit on every iteration, regardless of outcome, so the trajectory is replayable. After the commit lands, re-run `python3 -c "import json; ..."` to patch the `commit` field of the row you just wrote (or do it in the same commit by writing the row, committing, then amending — either pattern is fine as long as the final row has the hash).
8. **Cleanup gate.** After committing, if `N % 5 == 0`, run the Cleanup Cadence procedure below **before** opening iteration `N+1`. Do not skip. Do not fold cleanup into the next feature iteration.

### Cleanup Cadence

Every fifth iteration (N = 5, 10, 15, ...), perform a cleanup pass in a separate commit before the next feature iteration begins. This prevents `solution/triton/kernel.py` from accumulating dead branches, stale autotune configs, and debug prints that balloon the file and the agent's context.

**Trigger:** completed iteration number N (taken from the highest `[iter N]` commit) is divisible by 5, and no `[iter-N cleanup]` commit exists yet.

**Checklist:**
- Delete commented-out code, `# OLD:` / `# Removed:` markers, and dead branches that no live call site reaches.
- Drop autotune configs that have not won on any workload across the last 5 iterations. Verify against the most recent `modal_autotune` table in `solution/autotune_notes.md`; if a config's only winning workload has since been superseded, drop it.
- Remove `tl.device_print`, Python `print`, and scratch debug helpers introduced during the last 5 iterations.
- Collapse exploratory notes in `ITERATIONS.md` rows N−4..N into a single "Summary after iter N" block that retains the decisions and drops the scratch reasoning.
- Verify `git status` is clean against `solution/` apart from the cleanup diff, and that no orphaned files remain (stray `.py.bak`, leftover `scratch_*.py`, etc.).
- Re-run `bash scripts/bench.sh iter-N-cleanup`. Must still print `CORRECT=True` and must not regress the class medians recorded at iteration N by more than 1%. If it does, the cleanup removed something live — revert and investigate.

**Commit:** `git commit -m "[iter-N cleanup] Post-iteration-N cleanup pass"`. This commit must contain *only* cleanup changes, never a feature change. Opening iteration `N+1` before this commit exists is a protocol violation.

### Stall Rules

- If 5 consecutive iterations on the *same class* show no improvement (no-change or regression), stop editing that class. Re-profile its representative from scratch, re-read `knowledge/INDEX.md` *and* `context/INDEX.md` grouped by the observed bottleneck class, and write a plan in `ITERATIONS.md` before the next iteration on that class. Check `solution/iterations.jsonl` first — techniques already tagged `regression` or `no-change` for this `class_id` are the ones the plan should *not* repeat.

### Hard Rules for 3.B

- **One class per iteration.** Do not attempt to optimize two classes in a single code change.
- **Branch the code per class only if necessary.** Prefer a single kernel with shape-dependent dispatch over separate kernels per class; separate kernels are a 3.C decision, not a 3.B one.
- **Correctness is non-negotiable.** Any iteration that regresses correctness must be reverted in the same iteration.
- **No reward hacking.** Stream injection, process injection, timing patches, monkey-patching the benchmark, etc. are forbidden (see `RULES.md`).
- **No algorithmic regression to Phase 2 decisions** without writing the rationale to `HANDOFF.md` and asking the user.
- **No fine-tuning block sizes as a primary strategy.** Look for structural changes (data layout, fusion, reduced memory traffic, quantization) over block-size sweeps. If block sizes truly need to change per class, treat that as a 3.C integration decision.

### Exit condition for 3.B

Each class has either (a) a documented win vs. its Phase-2 baseline, or (b) a stall-rule-triggered "no viable further gain" entry in `ITERATIONS.md`. Do not proceed to 3.C before every class has reached one of these outcomes.

---

## Sub-Phase 3.C — Integration

Goal: produce a single `solution/triton/kernel.py` that dispatches to the right variant per class and does not regress any class. If 3.B produced separate code paths per class, this is where they are unified behind a dispatch mechanism.

**Procedure:**
1. Inventory the per-class wins from `ITERATIONS.md`. For each class, list the specific code changes (file, line range, technique).
2. Decide the dispatch mechanism, in order of preference:
   - **Single kernel, shape-keyed autotune** — if the per-class wins differ only in config. Extend the frozen autotune table; this counts as a justified autotune change per the freeze rule above.
   - **Single kernel, compile-time branch on `tl.constexpr`** — if the wins differ in control flow but not in memory layout.
   - **Multiple kernels, launcher-level dispatch** — if the wins differ in memory layout or algorithmic family. Choose the dispatch on axes already present in `subgraph_specs.json` (e.g., `seq_len`).
3. Implement the unified kernel. Update `solution/autotune_notes.md` with any new frozen entries (justified per the freeze rule).
4. Verification run: `bash scripts/bench.sh integrated` **5 times** across every representative workload. Record medians. Acceptance:
   - `CORRECT=True` on all workloads.
   - No class regresses beyond the tolerance set in 3.B vs. that class's best 3.B runtime.
   - Geometric-mean runtime across classes is ≤ the best 3.B per-class result weighted by workload frequency.
5. If any class regresses, do not ship the integration. Either refine the dispatch or keep the class-specific code path and document the duplication in `solution/fusion_notes.md`.
6. Commit: `git commit -m "[phase3-C] Integrated kernel across classes"`.

**Artifact updates:**
- `solution/workload_classes.json` — add a `best_runtime_ms` and `winning_technique` field per class.
- `solution/autotune_notes.md` — reflect the final frozen configs post-integration.
- `ITERATIONS.md` — append an "Integration" section summarizing the dispatch choice and per-class final runtimes.

---

## Exit

Phase 3 ends when:
- 3.C is complete and green, OR
- The user stops the run, OR
- Stall rules trigger a hard stop with no recovery path, OR
- A pre-agreed iteration budget is exhausted.

When ending, append a Phase 3 block to `HANDOFF.md`:

```markdown
## Phase 3 — ended (<date>)

- Classes identified: <list class_ids>
- Total iterations: <N>
- Per-class best runtime (median): <table class_id → runtime>
- Dispatch mechanism chosen in 3.C: <autotune-key | constexpr-branch | launcher-dispatch | none>
- Primary wins per class: <list, each citing a knowledge file>
- Deadends: <list>
- Current bottleneck per class (if work continues later): <one line per class>
```
