---
topic: tiling
applies_to: GEMM1 (H=7168×2I=4096) and GEMM2 (I=2048×H=7168) in the MoE kernel; per-expert M varies from 1 to ~14k
source: .claude/skills/OPTIMIZATION_TECHNIQUES.md (§1), context/techniques/persistence.md, context/techniques/pid_swizzle.md, .claude/skills/optimization/tier1-block-tiling/SKILL.md, .claude/skills/kernels/gemm/SKILL.md
confidence: high
---

# Tiling — block shapes, loop orderings, SM occupancy

## Summary

Select `(BLOCK_M, BLOCK_N, BLOCK_K)` to feed B200 tensor cores while respecting the 228 KB SMEM / 256 regs/thread budget. Pair with a persistent grid sized to `NUM_SMS = 148` and a 2D super-group swizzle to reuse B tiles from L2 across `GROUP_M` neighbouring M-tiles. Per-expert M in MoE is ragged and often tiny, so small-M and routing-dominated configs matter as much as compute-bound ones.

## When to use

- Block shapes:
  - Compute-bound path (`M > 128`): `BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, NUM_STAGES=3`.
  - Memory/latency-bound (`M ≤ 64`): `BLOCK_M=64, BLOCK_N=128, NUM_STAGES=4`.
  - Skinny-N (`N ≤ 128`): shrink `BLOCK_N` to 64/128, reduce `GROUP_M` to 4.
  - Routing-dominant tiny-M: `BLOCK_M=16, BLOCK_N=256, NUM_STAGES=2`.
- Persistent grid when tiles per SM ≥ 4 (`persistence.md` §"When it applies") — that is practically always for our GEMM1 (E_local=32 experts × ≥1 M-block × 32 N-blocks).
- 2D swizzle with `GROUP_M=4–8` (`pid_swizzle.md` §"When to use") when B is streamed across many M-tiles; skip if per-expert M is so small that only one tile-row exists.

## Code pattern

```python
# persistent, swizzled tile iterator
NUM_SMS: tl.constexpr
start_pid = tl.program_id(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
for tile_id in range(start_pid, num_pid_m * num_pid_n, NUM_SMS):
    group_id   = tile_id // num_pid_in_group
    first_m    = group_id * GROUP_M
    gs_m       = min(num_pid_m - first_m, GROUP_M)
    pid_m      = first_m + ((tile_id % num_pid_in_group) % gs_m)
    pid_n      = (tile_id % num_pid_in_group) // gs_m
    # ... load A[pid_m], B[pid_n], accumulate in fp32 ...
```

## Tradeoffs

- Larger `BLOCK_M × BLOCK_N` ⇒ more compute per SMEM fill, but at FP8 an accumulator of 128×256×fp32 = 128 KB alone, which combined with 4-stage FP8 operand buffers exceeds 228 KB. Mitigate with epilogue subtiling.
- `GROUP_M` higher than 8 reduces parallelism at the last super-group and hurts ragged MoE; `GROUP_M=1` disables L2 reuse and costs ~10–15 % per `gemm/SKILL.md` and `OPTIMIZATION_TECHNIQUES.md §1`.
- Persistent grid forces grid-stride correctness for every tile index — off-by-one is the first thing to debug.
- Per Tier-1 skill: `num_warps` raises occupancy but expands register footprint; co-tune with `num_stages` to avoid spills (`tier1-block-tiling/SKILL.md`).

## Pitfalls

- **Wrapped M offset hint bug** ((problem spec)): never add `tl.multiple_of` / `tl.max_contiguous` to `offs_m % expert_len` on B200 when `expert_len % BLOCK_M != 0` — generates incorrect PTX.
- **Last super-group short**: `group_size = min(num_pid_m − first_m, GROUP_M)` — not clamping produces wrong `pid_m` for the tail (`pid_swizzle.md:36`).
- **Block-size-independent SMEM pressure from accumulator**: `smem = NUM_STAGES × (BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) · sizeof(dtype) + BLOCK_M·BLOCK_N · 4`; with FP8 operands the accumulator usually dominates.
- Very small per-expert M in MoE turns GEMM into essentially GEMV — big tiles idle most warps; switch to tiny `BLOCK_M` + split-K (see `autotune-space`).
