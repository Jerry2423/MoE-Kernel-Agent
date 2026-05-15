---
description: Run all three phases (Learn → Implement → Optimize) end-to-end in one session.
---

# /run-all — End-to-end orchestrator

Run Phases 1, 2, and 3 sequentially in a single session. Use this when starting a fresh kernel from scratch. To resume in the middle of an existing run, invoke the per-phase commands (`/phase1-learn`, `/phase2-implement`, `/phase3-optimize`) directly instead — they remain available and unchanged.

This file does **not** re-implement the phases. It is a thin orchestrator that follows the existing phase prompts in order and flips `.claude/state/current_phase` between them so the `PreToolUse` hooks (`pre_edit.sh`, `pre_bash.sh`) keep gating writes to the correct directories at each step.

## Entry check

1. Read `TASK.md`, `RULES.md`, and `HANDOFF.md`.
2. If `HANDOFF.md` already contains a `## Phase N — complete` block for the run you are about to start, stop and tell the user to either:
   - clear/archive the existing artifacts and re-invoke `/run-all`, or
   - resume mid-pipeline with the matching per-phase command (`/phase2-implement` if Phase 1 is done, `/phase3-optimize` if Phase 2 is done).
3. Seed the phase state file:
   ```bash
   mkdir -p .claude/state && echo phase1 > .claude/state/current_phase
   ```
   (`user_prompt_submit.sh` also writes `phase1` on `/run-all`, but doing it explicitly here keeps the orchestrator self-contained.)

## Procedure

### Step 1 — Phase 1 (Learn)

Follow `.claude/commands/phase1-learn.md` to completion, including appending the Phase 1 block to `HANDOFF.md`. When that file's final paragraph tells you to stop, do **not** stop under `/run-all` — return control here and proceed to Step 2.

### Step 2 — transition to Phase 2

```bash
echo phase2-implement > .claude/state/current_phase
```

### Step 3 — Phase 2 (Implement)

Follow `.claude/commands/phase2-implement.md` end-to-end (Stages 2.1 → 2.2 → 2.3 → 2.4 → 2.5) and append the Phase 2 block to `HANDOFF.md`. When that file's final paragraph tells you to stop, do **not** stop under `/run-all` — return control here and proceed to Step 4.

### Step 4 — transition to Phase 3

```bash
echo phase3-optimize > .claude/state/current_phase
```

### Step 5 — Phase 3 (Optimize)

Follow `.claude/commands/phase3-optimize.md` (Sub-phases 3.A → 3.B → 3.C). Terminate per that file's existing exit rules — 3.C complete and green, stall rules triggered, or the iteration budget exhausted. Append the Phase 3 block to `HANDOFF.md`.

### Step 6 — final summary

Stop and summarize: final commit SHA, baseline-vs-final median runtime, the path of each new `HANDOFF.md` block, and any open issues.

## Failure handling

If any phase fails an exit criterion (Phase 2's `CORRECT=True` check, `tools/check_rules.py`, `tools/validate_artifacts.py`, a hook rejection, etc.), **stop immediately** and report to the user. Do **not** advance `.claude/state/current_phase` past the failing phase — leaving it on the current phase preserves the write-gating rails so the user can investigate and resume with the matching per-phase command.

Do not invoke `/phase2-implement` or `/phase3-optimize` as a recovery shortcut from inside `/run-all`; the per-phase commands are the user's resume path, not yours.
