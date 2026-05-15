# `context/` — Phase 1 input slot

Per-problem reference material the operator supplies before Phase 1 runs. Phase 1 reads everything here and synthesizes it into `knowledge/`. Phases 2 and 3 read `knowledge/`, not `context/` directly (except for the workload/target data and schemas listed below, which the bench harness loads at runtime).