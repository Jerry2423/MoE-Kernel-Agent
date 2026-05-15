# Knowledge INDEX — bottleneck → technique → knowledge file

Seeded from `context/INDEX.md`. Each row points at the corresponding Phase-1 knowledge file. Phase 3 re-reads this file during the stall-rule recovery (see `HINTS.md`).

Columns: **bottleneck class** | **technique** | **knowledge file** | **one-line hook**

## memory-bandwidth-bound

| technique | knowledge file | hook |
|-----------|----------------|------|
| TMA async load + multicast | [memory-hierarchy.md](memory-hierarchy.md) | descriptor-driven async loads free epilogue registers; multicast for replicated operands |
| 2D swizzle for L2 reuse | [tiling.md](tiling.md) | reorders tile visit order so adjacent programs share L2 lines |
| Shared-memory bank conflicts | [memory-hierarchy.md](memory-hierarchy.md) | align SMEM dtype, use TMA swizzle pattern |
| Cache eviction policies | [memory-hierarchy.md](memory-hierarchy.md) | keep activations L2-resident (`evict_last`), evict weights on use (`evict_first`) |
| Memory loading / instr order | [memory-hierarchy.md](memory-hierarchy.md), [autotune-space.md](autotune-space.md) | load early, issue far from use so MMA can overlap |

## compute-bound

| technique | knowledge file | hook |
|-----------|----------------|------|
| Warp specialization | [async-pipelines.md](async-pipelines.md) | producer/consumer warps overlap load↔MMA on Hopper/B200 |
| Software pipelining (multi-stage) | [async-pipelines.md](async-pipelines.md) | `num_stages ≥ 3` + K-loop `tl.range(..., num_stages=...)` |
| Stage merging for large BLOCK_K | [async-pipelines.md](async-pipelines.md) | fold two K-tiles into one stage when BK≥256 |
| FP8 block-scaled `dot` | [quantization.md](quantization.md), [numerics.md](numerics.md) | `tl.dot(..., lhs_scales, rhs_scales)` — per-block scaling free |
| Block-size heuristics | [tiling.md](tiling.md), [autotune-space.md](autotune-space.md) | DeepGEMM-style occupancy/SMEM trade-off |
| Register pressure budget | [memory-hierarchy.md](memory-hierarchy.md), [tiling.md](tiling.md) | keep BM*BN*accum ≤ 256 regs/thread to avoid spills |

## latency-bound (short seq)

| technique | knowledge file | hook |
|-----------|----------------|------|
| Persistent grid (fixed SMs) | [tiling.md](tiling.md) | avoid launch overhead; one CTA/SM loops over tiles |
| Atomic ops & zero-init pre-hook | [autotune-space.md](autotune-space.md) | pre-hook zeros output; atomics avoid race on scatter |

## profiling & measurement

| technique | knowledge file | hook |
|-----------|----------------|------|
| Stage 1 torch.profiler | (Phase-3 only, see `HINTS.md`) | `record_function` per stage — first thing to run per Phase 3 class |
| Stage 2 NCU deep-dive | (Phase-3 only, see `HINTS.md`) | pick NCU sections by bottleneck hypothesis, not by default |
| NCU canonical preset (25 metrics) | (Phase-3 only, see `HINTS.md`) | 5-section snapshot (SM util, mem BW, access, occupancy, stalls) |
| Autotune config pruning | [autotune-space.md](autotune-space.md) | prune by SMEM, n_valid, num_warps*32 ≤ BM*BN before sweeping |

## architecture & hardware constants

| technique | knowledge file | hook |
|-----------|----------------|------|
| DeepSeek-V3 fixed shapes | [moe-data-layouts.md](moe-data-layouts.md) | H=7168 I=2048 E_local=32 E_global=256 TOP_K=8 BLOCK_K=128 (problem spec / `input/torch_ref.py`) |
| SMEM budget (B200 = 228 KB/SM) | [tiling.md](tiling.md), [autotune-space.md](autotune-space.md) | budget per kernel variant: see `bench/modal_autotune.py:SMEM_LIMIT` |
| B200 hardware reference | [memory-hierarchy.md](memory-hierarchy.md), [numerics.md](numerics.md) | full spec sheet + derived roofline crossovers (fp16 ≈ 280 FLOPs/byte, fp8 ≈ 560) |

