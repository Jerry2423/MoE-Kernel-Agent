# Handoff Log

This file is the single source of truth for which phase the agent is in. Every phase appends its completion block here in order. Do not delete entries.

If this file contains no phase blocks, the current phase is **Phase 1 (Learning)**.

<!-- Phase blocks will be appended below this line -->

## Phase 1 — complete (2026-04-17)

- Knowledge files: 10 (`00_inventory.md`, `tiling.md`, `memory-hierarchy.md`, `async-pipelines.md`, `numerics.md`, `quantization.md`, `autotune-space.md`, `moe-data-layouts.md`, `INDEX.md`, `OPEN_QUESTIONS.md`).
- Topics covered: tiling, memory-hierarchy, async-pipelines, numerics, quantization, autotune-space, moe-data-layouts.
- Open questions: 10 (scheduler/launch contract, precision/tolerance details, `tl.dot_scaled` FP8 path, `warp_specialize` stability, routing edge cases, NCU preset availability).
- Recommended starting algorithm for Phase 2: **persistent M-grouped grouped-GEMM with gather-fused GEMM1 and scatter-add GEMM2 epilogue**, preceded by a separate fused no-aux routing + bitmatrix/histogram kernel. Rationale: this is the design the existing harness-authored code targets and the one with the most directly-applicable context. It gives a clean baseline before Phase 3 considers warp specialization or `tl.dot_scaled`.

## Phase 2 — complete (2026-04-18)

- Fused subgraphs: 4 unique / 4 instances (routing-dsv3-noaux, dispatch-local-sort, gemm1-fp8-swiglu, gemm2-fp8-weighted-scatter).
- Kernels implemented: 2 (`_gemm1_swiglu_kernel`, `_gemm2_scatter_kernel`). Routing + dispatch live in `run()` via primitive torch ops; no forbidden high-level APIs.
- Correctness run (baseline): `trajectory/20260418_004349_baseline/` — CORRECT=True on all 19 workloads.
- Correctness run (autotuned): `trajectory/20260418_005421_autotuned/` — CORRECT=True on all 19 workloads, no regression.
- Baseline median runtime: 3.504 ms mean across workloads (6.29x over torch_ref; SOTA_GEOMEAN_GAP 9.85x, total headroom 53.65 ms).
- Autotuned median runtime: 1.778 ms mean across workloads (11.73x over torch_ref; SOTA_GEOMEAN_GAP 5.19x, total headroom 20.86 ms). Largest absolute headroom remains on seq_len ∈ {901, 11948, 14107}.
- Autotune configs frozen: yes — see `solution/autotune_notes.md`. Deduplicated winners: GEMM1 {[16,64,3,4], [32,128,3,4], [128,64,3,4]}, GEMM2 {[16,256,3,8]}.
- Branch: `impl/moe-dsv3`. Commits: `[baseline]` (20bd060), `[phase2.5] Update modal_autotune.py…` (c6afb3f), `[baseline-autotuned]` (9b96c83). `tools/check_rules.py` and `tools/validate_artifacts.py` both exit 0.

Note for Phase 3: autotune configs are frozen. Do not re-tune unless a significant algorithmic change is introduced.

## Phase 3 — ended (2026-04-18)

- Classes identified: `short_latency_bound` (16 workloads, seq_len ≤ 80), `medium_mixed` (1 workload, seq_len = 901), `long_compute_bound` (2 workloads, seq_len ∈ {11948, 14107}).
- Total iterations: **16** (15 feature iters + 1 phase3-C integration). Iters 1, 3, 7, 9, 10, 12 improved; 13 no-change; 2, 4, 5, 6, 8, 11, 14, 15 reverted. Cleanup commits at N=5, 10, 15.
- Per-class best runtime (median across 5 integrated runs):

  | class | rep | Phase-2 rep ms | Phase-3 final ms | speedup vs Phase-2 |
  |:------|:---:|--------------:|------------------:|-------------------:|
  | short_latency_bound | seq=1 | 0.939 | **0.373** | 2.52× |
  | medium_mixed | seq=901 | 1.801 | **1.202** | 1.50× |
  | long_compute_bound | seq=14107 | 8.461 | **7.279** | 1.16× |
  | long_compute_bound | seq=11948 | 6.017 | **4.934** | 1.22× |

  Aggregate — mean 1.778 → **1.111 ms** (1.60× faster), speedup vs torch_ref 11.73× → **25.7×**, SOTA geomean gap 5.19× → **2.53×**, total SOTA headroom 20.86 ms → **8.33 ms (−59%)**.
- Dispatch mechanism chosen in 3.C: **none** (single kernel file, shape-keyed `_pick_cfg` already present from Phase 2; no new dispatch needed because every win is universal).
- Primary wins per class:
  - short_latency_bound:
    - iter 1 — fused Triton routing kernel replacing ~15 torch kernels.
    - iter 7 — drop `is_local.any()` CPU-GPU sync.
    - iter 9 — emit `-1`-sentinel local-expert id in routing.
    - iter 10 — two-pass Triton dispatch (`_dispatch_count_kernel` + `_dispatch_scatter_kernel`) replacing torch.argsort/bincount/index_select. Biggest single win.
    - iter 12 — fused fp32→bf16 output cast kernel (`knowledge/memory-hierarchy.md`).
  - medium_mixed: inherits every short-class win; additional gain from iter 3.
  - long_compute_bound: iter 3 — bf16 intermediate buffer between GEMM1 and GEMM2 (`knowledge/quantization.md`).
- Deadends (reverted, do not retry without new angle):
  - FP8 dot native (iter 2) — long-seq numerical failure; B200/Triton FP8 dot appears to mis-scale at K=7168.
  - Move per-block scales out of the K-loop (iter 4) — breaks Triton dequant+cast+dot pipeline fusion.
  - GROUP_M swizzle on GEMM2 (iter 5) — overhead > benefit at small max_tiles_m.
  - FP8 intermediate via separate quant kernel (iter 6) — Triton compile error (`FP8_MAX` constexpr / `tl.float8e4nv` cast path).
  - GEMM2 grid axis reorder (iter 8) — scheduler already handles locality.
  - bf16 atomic scatter directly to output (iter 11) — bf16 atomic_add on B200 hits a slow CAS-loop fallback.
  - Persistent GEMM2, strided tile_ids (iter 14) — L2 thrash from expert-boundary striding on long seqs.
  - Persistent GEMM2, block-partitioned (iter 15) — Triton compile error on nested control flow around `tl.dot`.
- Current bottleneck per class (if work continues later):
  - short_latency_bound — GEMMs now ~90% of rep time; only structural gains left (GEMM1+SwiGLU+GEMM2 cross-subgraph fusion, or tighter compile-time constant propagation via a compile-cache at `run()`).
  - medium_mixed — balanced GEMM1+GEMM2, bottleneck shifts with BM/BN tuning. Could benefit from a re-sweep now that dispatch overhead is gone.
  - long_compute_bound — GEMM2 (~4–5 ms at seq_len 14107); biggest uncaptured lever remains fp8 intermediate + per-128 block scale, but requires a working fp8 quant-and-dot path (both previous attempts stalled on Triton compile issues). An NCU deep-dive on `_gemm2_scatter_kernel` would likely identify a specific stall category to chase.
