---
topic: async-pipelines
applies_to: GEMM1 and GEMM2 when per-expert M-block ≥ ~64; less effective for M ≤ 16 routing-dominant workloads
source: .claude/skills/OPTIMIZATION_TECHNIQUES.md (§4, §5), context/techniques/tma.md, .claude/skills/optimization/tier4-advanced-scheduling/SKILL.md
confidence: high
---

# Async pipelines — producer/consumer + software pipelining

## Summary

On Hopper/Blackwell we can overlap GMEM→SMEM async copies with tensor-core compute two ways: (1) multi-stage software pipelining with `num_stages` SMEM buffers, (2) warp specialization where producer warps issue TMA loads and consumer warps run MMAs. Both depend on TMA to free issue slots; both collapse to no-op benefit when the K-loop has only a few iterations.

## When to use

- **Multi-stage pipelining**: whenever the K-loop runs ≥ `NUM_STAGES` iterations; optimal `NUM_STAGES ≈ ceil(memory_latency / compute_time_per_tile)` ≈ 3–5 on B200.
- **Warp specialization**: FP8 GEMM with `BLOCK_M ≥ 128` and TMA in place (Tier-4 skill "when in play"). Not a first move for small kernels.
- **Stage merging**: double `BLOCK_K`, halve `NUM_STAGES` when SMEM budget is tight but pipeline depth must be preserved.

## Code pattern

```python
# multi-stage (OPTIMIZATION_TECHNIQUES.md §5)
for i in range(NUM_STAGES - 1):
    tl.async_copy(a_desc, smem_a[i], k=i)             # prologue

for k in range(K // BLOCK_K):
    slot = k % NUM_STAGES
    tl.wait_async_copy(smem_a[slot])
    acc += tl.dot(smem_a[slot], smem_b[slot])
    tl.async_copy(a_desc, smem_a[(k + NUM_STAGES - 1) % NUM_STAGES],
                  k=k + NUM_STAGES - 1)

# warp-specialized (OPTIMIZATION_TECHNIQUES.md §4)
for tile_id in tl.range(start, end, step, warp_specialize=True):
    tl.async_copy(a_desc, smem_a, ...)                # producer warps
    c += tl.dot(smem_a_prev, smem_b_prev)             # consumer warps
```

## Tradeoffs

- Each pipeline stage consumes `(BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) · sizeof(dtype)` SMEM; with FP8 and 4 stages the accumulator alone can blow the 228 KB budget. Fix via epilogue subtiling (`§4`).
- Warp specialization demands extra registers for producer state and careful barriers; deadlocks possible if producer/consumer counts are mis-set (Tier-4 skill "Risks").
- Stage merging increases register pressure per K-step (double BLOCK_K on the dot), which can force a tile-size reduction.
- Async copies from the same descriptor must not cross barriers incorrectly — see `wait_barrier(expected_bytes)` pattern.

## Pitfalls

- Setting `NUM_STAGES ≥ 10` without merging K-blocks ⇒ SMEM overflow.
- `range(K // BLOCK_K)` vs `tl.range(0, K // BLOCK_K, 1, num_stages=1)` — the Triton form sometimes avoids IR-level crashes and tightens asm (`triton_tricks.md §3`).
- Epilogue subtiling without reshape care gives wrong results: must respect `acc.reshape(BLOCK_M, 2, BLOCK_N // 2)` and `can_reorder=False` (`triton_tricks.md §5` table row).
- Turning on `warp_specialize=True` with only 2 warps/block makes it a no-op — need ≥4 warps per CTA.
- Pipeline depth beyond the K-loop iteration count wastes SMEM and warp slots; measure first.
