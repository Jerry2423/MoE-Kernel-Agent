---
description: Phase 1 — build a structured knowledge base from context/ before writing any kernel code.
---

# Phase 1: Learning

Your only job in this phase is to produce a **structured knowledge base** under `knowledge/` by studying the reference materials in `context/` and the reusable capabilities in `.claude/skills/`. Do not write kernel code. Do not run benchmarks. Do not edit `solution/`.

## Entry Check

Before doing anything:
1. Read `TASK.md`, `RULES.md`, `HANDOFF.md`, `knowledge/README.md`, `knowledge/_template.md`.
2. Confirm `HANDOFF.md` shows no Phase 1 completion block. If Phase 1 is already complete, stop and tell the user.
3. List `context/` and identify which reference implementations apply to the target kernel family (read `input/` file names to determine the kernel family — e.g., MoE, attention, GEMM).

## Procedure

1. **Survey** — produce a short inventory of what's in `context/`: one bullet per file or subdirectory, noting what kernel pattern it demonstrates. Save as `knowledge/00_inventory.md`.

2. **Topic pass** — for each topic listed in `knowledge/README.md`, create one knowledge file using `_template.md`. Required frontmatter:
   ```yaml
   ---
   topic: <short slug>
   applies_to: <which kernel shapes / workloads>
   source: <which file(s) in context/ or which skill>
   confidence: <high | medium | low>
   ---
   ```
   Body sections (all required):
   - **Summary** (≤3 sentences).
   - **When to use** — shape thresholds, workload characteristics.
   - **Code pattern** — minimal Triton snippet or pseudocode showing the technique. No full kernels.
   - **Tradeoffs** — what it costs (register pressure, shared memory, compile time, etc.).
   - **Pitfalls** — concrete failure modes observed in `context/`.

3. **Cross-reference pass** — add a `knowledge/INDEX.md` that groups entries by bottleneck class (memory-bound, compute-bound, routing-bound, etc.) so Phase 3 can retrieve by symptom. **Seed this from `context/INDEX.md`**: copy its rows, then specialize each to reference the specific knowledge file you produced in step 2. Add new rows only for techniques `context/INDEX.md` doesn't already cover — do not delete or restate its entries.

4. **Open questions** — things you could not answer from `context/` go into `knowledge/OPEN_QUESTIONS.md`. Do not fabricate answers.

## Hard Rules

- **No edits to `solution/`, `input/`, `scripts/`, `bench/`.** If you touch these, you've left the phase.
- **No benchmarking or profiling.** No `bash scripts/bench.sh`, no `modal run`.
- **No free-form notes.** Everything goes through the template. Observations that don't fit become entries in `OPEN_QUESTIONS.md`.
- **Cite sources.** Every claim in a knowledge file must cite a file path in `context/` or a skill in `.claude/skills/`. Uncited claims are not allowed.
- **No invented techniques.** Only record techniques you can point to in `context/` or `.claude/skills/`.

## Exit

Phase 1 is complete when:
- Every required topic in `knowledge/README.md` has a corresponding file.
- `knowledge/INDEX.md` and `knowledge/OPEN_QUESTIONS.md` exist.
- You append a Phase 1 block to `HANDOFF.md` in this format:

```markdown
## Phase 1 — complete (<date>)

- Knowledge files: <count>
- Topics covered: <list of slugs>
- Open questions: <count>
- Recommended starting algorithm for Phase 2: <name + one-sentence rationale, citing a knowledge file>
```

After writing the handoff block, stop **if you were invoked directly via `/phase1-learn`** — the user will start a new session for Phase 2. If you are running under `/run-all`, return control to the orchestrator (which advances `.claude/state/current_phase` and begins Phase 2); do not invoke `/phase2-implement` yourself.
