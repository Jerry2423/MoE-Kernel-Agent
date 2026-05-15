# Autotune notes (Phase 2.5 freeze)

Frozen on 2026-04-18 against the Phase 2.3 baseline kernel. Phase 3 must
not re-run `modal_autotune.py` unless an algorithmic change (new operand
layout, new kernel, new tl.constexpr) is introduced and commit-prefixed
with `[phase2.5] Update modal_autotune.py for <reason>`.

## Workload table (from `context/moe_workloads.jsonl`)

All 19 workloads are covered — no sampling. Distinguishing axes are
`seq_len` and `local_expert_offset`. Phase 3 consumes this table when
classifying workloads in sub-phase 3.A.

| idx | seq_len | local_expert_offset | uuid prefix |
|----:|--------:|--------------------:|:------------|
|   1 |       1 |   32 | e05c6c03 |
|   0 |       7 |  192 | b8f4f012 |
|   7 |      14 |    0 | 8cba5890 |
|   6 |      15 |   32 | 2e69caee |
|   5 |      16 |  224 | a7c2bcfd |
|   2 |      32 |   32 | 6230e838 |
|  18 |      52 |  160 | f7d6ac7c |
|  17 |      53 |   32 | fc378037 |
|  16 |      54 |  128 | 76010cb4 |
|  15 |      55 |  128 | 81955b1e |
|  14 |      56 |   64 | 4822167c |
|  13 |      57 |   96 | 74d7ff04 |
|  12 |      58 |   64 | e626d3e6 |
|  11 |      59 |  160 | eedc63b2 |
|  10 |      62 |   96 | 5eadab1e |
|   3 |      80 |   96 | 8f1ff9f1 |
|   4 |     901 |   96 | 1a4c6ba1 |
|   9 |   11948 |  128 | 58a34f27 |
|   8 |   14107 |   32 | 5e8dc11c |

## Search-space deltas

Replaced the prior `modal_autotune.py` (which expected TMA-fused kernel
names `_gemm1_fp8_tma_swiglu_kernel` / `_gemm1_fp8_tma_gather_kernel` /
`_gemm2_fp8_fp8_tma_scatter_kernel`) with a version matching the
Phase 2.3 baseline:

- Dispatch on `gemm1` and `gemm2` kernel slugs only (no TMA/warp-spec
  variants; those are Phase-3 options).
- Dropped `loop_num_stages` and `WARP_SPEC` tunables (not present in
  baseline kernel).
- New SMEM estimator reflects operands:
  - `gemm1`: FP8 A + 2×FP8 W per stage, 2× fp32 accumulators
    (`ns × (BM·BK + 2·BN·BK) + 2·BM·BN·4 + 2048`).
  - `gemm2`: fp32 A + FP8 W per stage, 1× fp32 accumulator
    (`ns × (BM·BK·4 + BN·BK) + BM·BN·4 + 2048`).
- Prunes by `N_AX % BN == 0` (I=2048 for gemm1, H=7168 for gemm2) and
  `num_warps × 32 ≤ BM × BN`.
- Synthetic inputs use uniform per-expert token count
  (`tpe = seq_len × TOP_K × E_LOCAL / E_GLOBAL / E_LOCAL`) to reflect
  steady-state GEMM cost, not skew.

Committed separately as `[phase2.5] Update modal_autotune.py for new
kernel names (_gemm1_swiglu_kernel, _gemm2_scatter_kernel)`.

## Justified search space

| Axis | Range | Rationale |
|------|-------|-----------|
| `BLOCK_M` | {16, 32, 64, 128} | MoE per-expert M is tiny for short seq (1–10 tokens) up to ~440 for seq=14107. Need small BM for short seqs, large BM for long. Pruned to `BM ≤ max(16, tpe × 4)`. |
| `BLOCK_N` | {64, 128, 256} | Must divide `I=2048` (gemm1) or `H=7168` (gemm2). BN=256 gives H/256=28 tiles_n, good grid for long seqs. BN=64/128 allow BM=16 configs without SMEM blow-up. |
| `num_stages` | {2, 3, 4} | `knowledge/async-pipelines.md`: optimal is 3–5 on B200; 2 as lower bound, 4 as upper (SMEM pressure with bf16 operands). |
| `num_warps` | {4, 8} | `knowledge/tiling.md`: compact set covering Triton's default 4 + a 2× step for larger tiles. Below 4 is register-limited for tensor-core dots; above 8 exceeds common SM allocation. |

Total unpruned configs per (kernel, seq_len) ≈ 72; after SMEM/valid-BN/
warp-count pruning, ≈ 20 configs per run. 19 seq_lens × 2 kernels ×
~20 configs = ~760 timed measurements.

## Sweep results

Captured from `trajectory/autotune-sweep/sweep.log` /
`trajectory/autotune-sweep/winners.json`. All 19 workloads × 2 kernels
covered; no ALL-FAILED row.

### GEMM1 winners

| seq_len | BLOCK_M | BLOCK_N | num_stages | num_warps | median (ms) |
|--------:|--------:|--------:|-----------:|----------:|------------:|
|       1 |      16 |      64 |          3 |         4 |      0.2223 |
|       7 |      16 |      64 |          3 |         4 |      0.2181 |
|      14 |      16 |      64 |          3 |         4 |      0.2202 |
|      15 |      16 |      64 |          3 |         4 |      0.2201 |
|      16 |      16 |      64 |          3 |         4 |      0.2181 |
|      32 |      16 |      64 |          3 |         4 |      0.2201 |
|      52 |      16 |      64 |          3 |         4 |      0.2202 |
|      53 |      16 |      64 |          3 |         4 |      0.2181 |
|      54 |      16 |      64 |          3 |         4 |      0.2202 |
|      55 |      16 |      64 |          3 |         4 |      0.2202 |
|      56 |      16 |      64 |          3 |         4 |      0.2202 |
|      57 |      16 |      64 |          3 |         4 |      0.2202 |
|      58 |      16 |      64 |          3 |         4 |      0.2202 |
|      59 |      16 |      64 |          3 |         4 |      0.2202 |
|      62 |      16 |      64 |          3 |         4 |      0.2202 |
|      80 |      16 |      64 |          3 |         4 |      0.2202 |
|     901 |      32 |     128 |          3 |         4 |      0.3041 |
|   11948 |     128 |      64 |          3 |         4 |      1.7737 |
|   14107 |     128 |      64 |          3 |         4 |      2.2704 |

### GEMM2 winners

| seq_len | BLOCK_M | BLOCK_N | num_stages | num_warps | median (ms) |
|--------:|--------:|--------:|-----------:|----------:|------------:|
|       1 |      16 |     256 |          3 |         8 |      0.1689 |
|       7 |      16 |     256 |          3 |         8 |      0.1670 |
|      14 |      16 |     256 |          3 |         8 |      0.1689 |
|      15 |      16 |     256 |          3 |         8 |      0.1690 |
|      16 |      16 |     256 |          3 |         8 |      0.1672 |
|      32 |      16 |     256 |          3 |         8 |      0.1712 |
|      52 |      16 |     256 |          3 |         8 |      0.1731 |
|      53 |      16 |     256 |          3 |         8 |      0.1730 |
|      54 |      16 |     256 |          3 |         8 |      0.1731 |
|      55 |      16 |     256 |          3 |         8 |      0.1732 |
|      56 |      16 |     256 |          3 |         8 |      0.1732 |
|      57 |      16 |     256 |          3 |         8 |      0.1730 |
|      58 |      16 |     256 |          3 |         8 |      0.1733 |
|      59 |      16 |     256 |          3 |         8 |      0.1731 |
|      62 |      16 |     256 |          3 |         8 |      0.1732 |
|      80 |      16 |     256 |          3 |         8 |      0.1793 |
|     901 |      16 |     256 |          3 |         8 |      0.3288 |
|   11948 |      16 |     256 |          3 |         8 |      3.3537 |
|   14107 |      16 |     256 |          3 |         8 |      3.9332 |

## Deduplicated frozen config set (wired into `solution/triton/kernel.py`)

- **GEMM1** — 3 unique configs:
  1. `[16, 64, 3, 4]` — all seq_len ≤ 80 (16 workloads).
  2. `[32, 128, 3, 4]` — seq_len = 901 (1 workload).
  3. `[128, 64, 3, 4]` — seq_len ∈ {11948, 14107} (2 workloads).
- **GEMM2** — 1 unique config:
  1. `[16, 256, 3, 8]` — all 19 workloads.

`run()` picks the winner via a per-seq-len dict lookup
(`_GEMM1_BEST` / `_GEMM2_BEST`), falling back to `_GEMM1_DEFAULT =
[32, 128, 3, 4]` / `_GEMM2_DEFAULT = [16, 256, 3, 8]` for unseen T.

## Phase 3 starting point

- Workload groups observed from the winners:
  - **Latency-floor cluster** (seq_len ≤ 80): both kernels ~0.17–0.22 ms,
    dominated by kernel launch + routing + dispatch. GEMM itself is tiny.
    SOTA is 0.063–0.27 ms; we're ~1 ms end-to-end (8–20× SOTA). Primary
    opportunity: reduce per-iteration launch overhead, fuse routing/dispatch,
    avoid the fp32 intermediate round-trip.
  - **Medium cluster** (seq_len = 901): ~0.6 ms combined GEMM; we're ~3 ms
    end-to-end (~4.8× SOTA). Compute + bandwidth mix; candidate for
    tl.dot_scaled / tma / better pipeline overlap.
  - **Long cluster** (seq_len ∈ {11948, 14107}): dominated by GEMM2 (3.4–
    3.9 ms per kernel); we're ~15–18 ms end-to-end (~3.2× SOTA). Primary
    opportunity: reduce GEMM2 atomic-scatter overhead, keep intermediate in
    FP8 instead of fp32, unfuse activations.
