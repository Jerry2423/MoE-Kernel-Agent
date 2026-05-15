# Context INDEX — bottleneck → technique

Hand-authored retrieval map into `context/` and `.claude/skills/`. Phase 1 seeds `knowledge/INDEX.md` from this file (extend, don't rebuild). Phase 3 re-reads this file during stall recovery.

Columns: **bottleneck class** | **technique** | **one-line hook**.

## memory-bandwidth-bound

| technique | hook |
|-----------|------|
| TMA async load + multicast | descriptor-driven async loads free epilogue registers; multicast for replicated operands |
| 2D swizzle for L2 reuse | reorders tile visit order so adjacent programs share L2 lines |
| Shared-memory bank conflicts | align SMEM dtype, use TMA swizzle pattern |
| Cache eviction policies | keep activations L2-resident, evict weights on use |
| Memory loading / instr order | load early, issue far from use so MMA can overlap |

## compute-bound

| technique | hook |
|-----------|------|
| Warp specialization | producer/consumer warps overlap load↔MMA on Hopper/B200 |
| Software pipelining (multi-stage) | `num_stages≥3` + K-loop `tl.range(..., num_stages=...)` |
| Stage merging for large BLOCK_K | fold two K-tiles into one stage when BK≥256 |
| FP8 block-scaled `dot` | `tl.dot(..., lhs_scales, rhs_scales)` — per-block scaling free |
| Block-size heuristics | DeepGEMM-style occupancy/SMEM trade-off |
| Register pressure budget | keep BM*BN*accum ≤ 256 regs/thread to avoid spills |

## latency-bound (short seq)

| technique | hook |
|-----------|------|
| Persistent grid (fixed SMs) | avoid launch overhead; one CTA/SM loops over tiles |
| Atomic ops & zero-init pre-hook | pre-hook zeros output; atomics avoid race on scatter |

## small-M GEMM

| technique | hook |
|-----------|------|
| Split-K Triton pattern | split reduction across programs when M≪N |
| Split-K selection heuristic | decide split factor from BM, occupancy |

## profiling & measurement

| technique | hook |
|-----------|------|
| Stage 1 torch.profiler | `record_function` per stage — first thing to run per Phase 3 class |
| Stage 2 NCU deep-dive | pick NCU sections by bottleneck hypothesis, not by default |
| NCU canonical preset (25 metrics) | 5-section snapshot (SM util, mem BW, access patterns, occupancy, stalls); writes `*_metrics.json` for iteration logs |
| Autotune config pruning | prune by SMEM, n_valid, num_warps*32 ≤ BM*BN before sweeping |

## architecture & hardware constants

| technique | hook |
|-----------|------|
| DeepSeek-V3 fixed shapes | H=7168 I=2048 E_local=32 E_global=256 TOP_K=8 BLOCK_K=128 (problem spec / `input/torch_ref.py`) |
| SMEM budget (B200 = 228 KB/SM) | budget per kernel variant: see `bench/modal_autotune.py:SMEM_LIMIT` |
| B200 hardware reference | full spec sheet + derived roofline crossovers (fp16 ≈ 280 FLOPs/byte, fp8 ≈ 560) — see `hardware/b200.md` |
