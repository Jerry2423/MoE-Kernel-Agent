---
topic: autotune-space
applies_to: both GEMM kernels; sweep dimensions pulled from DeepGEMM heuristics + Triton tricks
source: .claude/skills/OPTIMIZATION_TECHNIQUES.md, context/triton_tricks.md (§1, §4, §5), .claude/skills/optimization/tier1-block-tiling/SKILL.md, .claude/skills/kernels/gemm/SKILL.md, context/hardware/b200.md
confidence: medium
---

# Autotune space — shape of the search, not the values

## Summary

The search space is `(BLOCK_M, BLOCK_N, BLOCK_K) × num_warps × num_stages × GROUP_M × load_order × split_k × eviction_policy`, with an SMEM-budget pre-filter (`SMEM_LIMIT = 232448 B`, `b200.md` cross-ref). We sweep shapes, not magic numbers; the winning config is frozen after Phase 2.5 and only the algorithm can be changed in Phase 3.

## When to use

- New kernel or after an algorithmic change: sweep a **compact** grid (Tier-1 skill says 3–6 `(BM,BN,BK)` × 2–3 `num_warps` × 2–3 `num_stages`).
- When the current winner's SMEM/regs show headroom (NCU `smem_per_block`, `achieved_occupancy`): try one step up on `NUM_STAGES` or `BLOCK_N`.
- When routing-dominated workloads underperform: include tiny-M configs (`BM=16, BN=256, NS=2`).

## Code pattern

```python
# config pruner (triton_tricks.md §4)
def prune(configs, named_args):
    keep = []
    for c in configs:
        BM, BN, BK = c.kwargs['BLOCK_M'], c.kwargs['BLOCK_N'], c.kwargs['BLOCK_K']
        NS          = c.num_stages
        smem = NS*(BM*BK + BK*BN)*1 + BM*BN*4   # FP8 ops + fp32 accum
        if smem > 232448:          # B200 per-SM SMEM budget
            continue
        if c.num_warps * 32 > BM * BN:           # warps can't exceed tile size
            continue
        keep.append(c)
    return keep

@triton.autotune(configs=BASE_CONFIGS,
                 prune_configs_by={'early_config_prune': prune})
@triton.jit
def kernel(..., LOAD_ORDER: tl.constexpr, SPLIT_K: tl.constexpr):
    ...
    if LOAD_ORDER == 0: a = tl.load(a_ptr); b = tl.load(b_ptr)   # §1
    else:               b = tl.load(b_ptr); a = tl.load(a_ptr)
```

## Tradeoffs

- Larger sweeps find marginal wins but blow up compile + tuning wall time; the `phase2-implement` skill freezes winners in `solution/autotune_notes.md` to avoid re-sweeps in Phase 3.
- Pruning is worth it — unpruned sweeps waste minutes on obviously-broken configs (SMEM overflow, warp count mismatch).
- Pre-hook zero-init (`triton_tricks.md §5`) is free latency on scatter kernels that need a zeroed output but costs a hook per-config.

## Pitfalls

- **Do not re-sweep without an algorithmic change** (`HINTS.md "What not to spend iterations on"`, TASK.md Phase 3 "forbidden"): repeat sweeps almost always find the same winner.
- Forgetting to expose `load_order` / `eviction_policy` as tunable constexprs leaves a free knob unexplored.
- `num_stages=0` is invalid on HIP / old Triton; always start from 1 (Tier-1 skill "num_stages"). On B200 stick to 2–5.
- Atomic kernels must use `pre_hook=init_to_zero("output_ptr")` with a matching arg name — typos silently skip zeroing (`triton_tricks.md §5`).
- Memory-sync hint `sem='relaxed'` for `tl.atomic_add` is faster but only valid when downstream doesn't depend on acq-rel ordering (`triton_tricks.md §5` table).
