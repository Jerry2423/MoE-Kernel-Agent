---
description: Phase 2 — fuse, extract subgraph specs, implement correct kernels, verify, then freeze an autotuned baseline.
---

# Phase 2: Implementation

Phase 2 is **five serial sub-stages**. Each sub-stage has an explicit artifact. Do not start a sub-stage until the previous one's artifact exists and has been validated. Correctness — not performance — is the target of sub-stages 2.1–2.4. Sub-stage 2.5 freezes an autotuned baseline so Phase 3 can focus on algorithmic changes instead of config sweeps.

## Entry Check

Before doing anything:
1. Read `TASK.md`, `RULES.md`, `HANDOFF.md`, every file under `knowledge/`.
2. Verify `HANDOFF.md` shows Phase 1 complete. If not, stop.
3. Read `input/` — the reference PyTorch implementation and any starter skeleton.
4. **Do not read `HINTS.md`.** It is Phase-3 guidance and will bias you toward premature optimization.

Then create the solution scaffolding (once):
- If `solution/` does not exist: create `solution/` and `scripts/`. Copy `input/` contents into `solution/`.
- Generate `scripts/bench.sh` from `bench-wrapper.sh` by replacing `{{BENCH_COMMAND}}` with the correct invocation (see `bench/GUIDE.md` or `bench/kernelbench/GUIDE.md`).
- Create a new git branch: `impl/<kernel-name>`.

**Environment:** the pre-built `fi-bench` conda env has `modal`, `flashinfer-bench`, and all deps installed. `scripts/bench.sh` activates it automatically. Any direct `modal run ...` invocation from the shell must be preceded by `eval "$(conda shell.bash hook)" && conda activate fi-bench` in the same command (shell state does not persist between Bash tool calls). Do not `pip install` or create a new env.

---

## Stage 2.1 — Fusion Judgment (code-to-code)

Goal: produce a *fused* PyTorch reference that makes subgraph boundaries explicit, preserving original semantics and control flow. Operate directly on the PyTorch source — do not lower to IR.

**Procedure:**
1. Parse the reference forward pass. Extract operation sequences, data dependencies, and control-flow boundaries (branches, loops, shape-dependent dispatch).
2. Identify fusion opportunities. A group of ops is fusible if:
   - they share a producer/consumer chain without external users of intermediates,
   - they can be expressed in a single pass over their largest input,
   - fusion does not cross a control-flow boundary,
   - fusion does not alter numerics beyond what the reference already tolerates (if it does, document it).
3. Rewrite the reference as clean PyTorch with each fused group wrapped in a named `nn.Module` or function. Control flow remains intact at the outer level.
4. Validate: the fused module must produce outputs matching the original reference within the same tolerance the bench uses. Run this check locally with small synthetic inputs — not via Modal.

**Artifact:** `solution/fused_reference.py` — runnable PyTorch, one class/function per candidate subgraph, plus a `validate()` function that compares against the original reference on a fixed seed.

**Judgment log:** `solution/fusion_notes.md` — for each fusion decision, one bullet with: `<fused name>` ← `<op list>`, rationale (data reuse, eliminated round-trip, etc.), and any numerical caveats. For non-fusions, one bullet per rejected candidate with the reason.

---

## Stage 2.2 — Subgraph Boundary Inference

Goal: catalog the *distinct* subgraphs the kernel must implement, with typed specs. Two subgraphs with the same signature (ops + shapes + weights + dtype) collapse into one spec with a count.

**Procedure:**
1. Walk `solution/fused_reference.py`. For each fused function/module, emit a spec.
2. Infer input/output shapes symbolically where the reference uses them symbolically (e.g., `["B", "C_in", "H", "W"]`), concrete where the reference fixes them.
3. Enumerate weights with their shapes and whether they are pre-fused (e.g., BN folded into conv weights) or separate.
4. Record the source snippet verbatim so later stages can trace every kernel back to its PyTorch origin.
5. Deduplicate by stable signature (ops + shapes + dtype + weights). Aggregate a `count` field and a `where` field listing every call site.

**Artifact:** `solution/subgraph_specs.json` — JSON array. Each entry has:

```json
{
  "id": "<stable-slug>",
  "type": "<human-readable chain name>",
  "data_layout": "...",
  "dtype": "...",
  "ops": [{"op": "...", "...": "..."}],
  "input_shape": ["..."],
  "output_shape": ["..."],
  "weights_original": {"<name>": ["..."]},
  "weights_fused":   {"<name>": ["..."]} ,
  "count": <int>,
  "where": "<call-site description>",
  "source": {"module": "<class name>", "code": "<verbatim forward body>"}
}
```

Boundary rule: a subgraph ends wherever a control-flow change, a non-fusible op, or a layout/dtype transition occurs. Document any judgment call in `solution/fusion_notes.md`.

---

## Stage 2.3 — Kernel Implementation

Goal: one Triton kernel per unique spec. The simplest correct implementation that matches the spec.

**Procedure:**
1. For each entry in `subgraph_specs.json`, implement a `@triton.jit` kernel plus its launcher.
2. Match the bench harness entry point — `run()` in `solution/triton/kernel.py` (DPS style — output as last parameter) unless the harness specifies otherwise.
3. Hand-pick one config per kernel. Do not add `@triton.autotune` yet — autotuning happens in Stage 2.5 after correctness is proven.
4. Naive is good. A single loop is good. Fusion across subgraph boundaries is forbidden at this stage — that's a Phase-3 decision.

**Artifact:** `solution/triton/kernel.py` (plus any helper files) that exports `run(...)` as the bench harness expects.

---

## Stage 2.4 — Verification

Goal: prove correctness and prove the kernel does not cheat.

**Procedure — local (run first, before touching Modal):**
1. `python -c "import solution.triton.kernel"` to catch import/syntax errors.
2. `python -m py_compile solution/triton/kernel.py` on every `.py` under `solution/`.
3. **RULES.md cross-check — machine-readable.** Produce *two* files:
   - `solution/rules_check.md` — short narrative, one paragraph per rule, for humans.
   - `solution/rules_check.json` — structured per `context/schemas/rules_check.schema.json`. One entry per rule (1–5), each with `status: "pass"` and at least one `evidence` block citing `{file, line_start, line_end, quote}` in `solution/triton/kernel.py`.
   Then run the gate:
   ```
   python3 tools/check_rules.py
   ```
   Exit code 0 is required. If any rule is `fail`, or an evidence quote doesn't resolve to the cited line range, `check_rules.py` prints the offending `rule_id` to stderr — fix the kernel (not the evidence) and re-run. Do not touch Modal until this exits 0.

**Procedure — remote:**
4. `bash scripts/bench.sh baseline`. Expected: `CORRECT=True`. The wrapper also writes `trajectory/<stamp>_baseline/bench_result.json` matching `context/schemas/bench_result.schema.json` — this is the ground-truth artifact Phase 3 iterations diff against. If `CORRECT=False`, diagnose from the reference diff the harness prints. Fix and re-run.
5. Commit the correct-but-unoptimized baseline: `git commit -m "[baseline] Phase 2 correct kernel (pre-autotune)"`.

Do not proceed to Stage 2.5 until `python3 tools/check_rules.py` exits 0 and `CORRECT=True`.

---

## Stage 2.5 — Autotune Freeze

Goal: pick the best config per workload *once*, embed it, and declare configs frozen for Phase 3. Phase 3 must not re-run an autotune sweep unless it introduces a significant algorithmic change (see phase3-optimize.md).

**Procedure:**
1. Enumerate the workload set from `context/moe_workloads.jsonl` (or the project-specific workload file if the kernel family is not MoE). **Every** workload entry must be covered — do not sample. Extract the distinguishing axes (for MoE: `seq_len`, `local_expert_offset`) and record them in `solution/autotune_notes.md` as the workload table Phase 3 will consume.
2. Use `bench/modal_autotune.py` as the sweep driver — do not write an ad-hoc sweep script. Read `bench/GUIDE.md` §"`modal_autotune.py`" for its current contract.
3. **Update `modal_autotune.py` if and only if** your kernels in Stage 2.3 do not match its existing assumptions. Specifically:
   - If you added or renamed a kernel, update the dispatch, the SMEM estimator, and the min-`BN` constraint.
   - If you added a new tunable `tl.constexpr`, extend the config product and the SMEM estimator.
   - If MoE geometry constants (`H`, `I`, `E_LOCAL`, `E_GLOBAL`, `TOP_K`, `BLOCK_K`) differ from your kernel, align them — drift silently produces wrong "winners".
   - If `context/moe_workloads.jsonl` contains a `seq_len` not in `SEQ_LENS`, add it.
   Any change to this script must be committed separately with `[phase2.5] Update modal_autotune.py for <reason>` before running the sweep, and the reason must be recorded in `solution/autotune_notes.md` under a "Search-space deltas" section.
4. Justify the search space in `solution/autotune_notes.md` — one bullet per tunable axis and per range — before invoking the sweep. Do not cover the whole lattice.
5. Run `PYTHONPATH=. modal run -m bench.modal_autotune`. Capture the stdout log into `solution/autotune_notes.md` under a "Sweep results" section as a full (workload, kernel) → (winning config, measured runtime) table — one row per (workload, kernel) pair, no omissions.
6. Copy the winning configs into a dictionary at the top of `solution/triton/kernel.py` and wire `@triton.autotune(configs=[...], key=[...])` so the list contains *only* those winners (deduplicated across workloads that chose the same config). The key must reference shape parameters from `subgraph_specs.json`. This is the "freeze" — the production kernel has exactly the configs that won, nothing speculative.
7. Re-run `bash scripts/bench.sh autotuned` across the full workload set (one invocation per workload, or a single invocation iterating all of them if the harness supports it) to confirm:
   - `CORRECT=True` on **every** workload,
   - no workload regresses vs. the Stage 2.4 baseline.
   If any workload regresses, re-examine its row in the sweep table — the search space may have missed its winning region. Fix the search space in `modal_autotune.py`, re-sweep, and re-freeze.
8. Commit: `git commit -m "[baseline-autotuned] Phase 2 frozen autotune configs (all workloads)"`.

**Artifact:** `solution/autotune_notes.md` — must contain (a) the full workload table from step 1, (b) any "Search-space deltas" from step 3, (c) the justified search space from step 4, and (d) the full "Sweep results" table from step 5. Phase 3 reads all four to know the starting point and to classify workloads in sub-phase 3.A.

---

## Hard Rules

- **Serial stages.** No Stage-2.3 kernel before `subgraph_specs.json` exists. No Stage-2.5 autotune before `CORRECT=True`.
- **No fusion across subgraph boundaries in Stage 2.3.** Cross-subgraph fusion is a Phase-3 algorithmic decision.
- **No omission of reference components.** Every op in the reference must appear in a subgraph spec and in a kernel (see `RULES.md` rule 3).
- **No delegating compute to PyTorch** (see `RULES.md` rule 1). `rules_check.md` must cite line ranges.
- **No performance rewrites driven by hunches.** If a block shape is correct, keep it through Stage 2.4. Shape changes happen in Stage 2.5 via autotune, not by hand.
- **No reading `HINTS.md`.**
- **No silent algorithm changes.** If a spec cannot be implemented as a single kernel, stop and ask.

## Exit

Phase 2 is complete when:
- `solution/fused_reference.py`, `solution/fusion_notes.md`, `solution/subgraph_specs.json`, `solution/rules_check.md`, `solution/rules_check.json`, `solution/autotune_notes.md` all exist.
- `python3 tools/check_rules.py` exits 0.
- `python3 tools/validate_artifacts.py` exits 0 (confirms `subgraph_specs.json`, `rules_check.json`, and the latest `trajectory/*/bench_result.json` all validate).
- `bash scripts/bench.sh` prints `CORRECT=True` on the autotuned baseline.
- Both commits exist (`[baseline]` and `[baseline-autotuned]`).
- You append a Phase 2 block to `HANDOFF.md`:

```markdown
## Phase 2 — complete (<date>)

- Fused subgraphs: <N unique / M instances>
- Kernels implemented: <N>
- Correctness run (baseline): <path under trajectory/>
- Correctness run (autotuned): <path under trajectory/>
- Baseline median runtime: <value>
- Autotuned median runtime: <value>
- Autotune configs frozen: yes — see solution/autotune_notes.md
- Branch: impl/<kernel-name>

Note for Phase 3: autotune configs are frozen. Do not re-tune unless a
significant algorithmic change is introduced.
```

After writing the handoff block, stop **if you were invoked directly via `/phase2-implement`** — the user will start a fresh session for Phase 3. If you are running under `/run-all`, return control to the orchestrator (which advances `.claude/state/current_phase` and begins Phase 3); do not invoke `/phase3-optimize` yourself.
