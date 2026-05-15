# Knowledge Base

Structured notes produced by Phase 1 and consumed by Phases 2 and 3. Entries are populated by the Phase 1 agent; do not add entries manually outside of that phase.


## Required index files

- `00_inventory.md` — inventory of `context/` contents.
- `INDEX.md` — entries grouped by **bottleneck class** (memory-bound, compute-bound, latency-bound, routing-bound, etc.) so Phase 3 can retrieve by symptom.
- `OPEN_QUESTIONS.md` — anything Phase 1 could not resolve from `context/`.

## Entry format

Use `_template.md` verbatim. Do not add or rename frontmatter fields. Do not add top-level sections beyond those in the template. If a claim cannot be attributed to a file in `context/` or a skill under `.claude/skills/`, it does not belong in a knowledge file — put it in `OPEN_QUESTIONS.md`.
