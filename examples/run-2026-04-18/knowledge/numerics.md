---
topic: numerics
applies_to: whole kernel — FP8 e4m3fn operands, fp32 accumulators, bf16 output, fp32 routing, 1e-20 weight-sum epsilon
source: .claude/skills/OPTIMIZATION_TECHNIQUES.md (§3), context/hardware/b200.md, input/torch_ref.py
confidence: high
---

# Numerics — dtypes, accumulator promotion, epilogue scaling

## Summary

Every tensor-core `dot` on B200 must accumulate in fp32 regardless of operand type (`gemm/SKILL.md` "Common Pitfalls 2", `OPTIMIZATION_TECHNIQUES.md §3`). For DSv3 the operand types are fixed: hidden states and weights in FP8 e4m3fn with per-128 block scales, logits/bias/intermediate SwiGLU in fp32, final output in bf16. Routing weight normalization uses `s` (pre-bias sigmoid) summed with `+1e-20` — both details fail atol if swapped.

## When to use

- Whenever building a dot kernel, start with `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` and cast at store (`gemm/SKILL.md`).
- FP8 block-scaled post-scaling epilogue when activation scale is per-row-128 and weight scale is per-col-128.
- `tl.dot_scaled(..., out_dtype=tl.float32)` when the tile shape aligns with Blackwell's 5D preshuffled scale layout.
- Output atol/rtol for this benchmark: `atol=1, rtol=0.3, required_matched_ratio=0.9` ((problem spec)) — generous but not free.

## Code pattern

```python
# FP8 × FP8 with fp32 epilogue post-scale
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k in range(0, K, BLOCK_K):
    a8 = tl.load(a_desc, ...)        # float8e4m3fn
    w8 = tl.load(w_desc, ...)
    acc += tl.dot(a8, w8, out_dtype=tl.float32)
a_scale = tl.load(a_scale_ptr + ...)    # [BLOCK_M // 128]
w_scale = tl.load(w_scale_ptr + ...)    # [BLOCK_N // 128]
out = acc * a_scale[:, None] * w_scale[None, :]
tl.store(c_ptr + ..., out.to(tl.bfloat16))

# SwiGLU (input/torch_ref.py:157–159)
silu = x2 / (1.0 + tl.exp(-x2))          # sigmoid form
c = silu * x1

# weight normalization (input/torch_ref.py:124–126) — note `s`, not `s_with_bias`
weights = s * mask
weights = (weights / (tl.sum(weights, 1, keep_dims=True) + 1e-20)) * routed_scaling_factor
```

## Tradeoffs

- FP8 raises the arithmetic-intensity crossover to ~560 FLOPs/byte (`b200.md` "Derived"), so small-M GEMMs become bandwidth-bound and bf16/fp16 can compete despite lower TFLOPs.
- Accumulating in fp16 halves register cost but costs ~3× relative error on K=7168 reductions — visible in rtol=0.3 only for pathological inputs, still rule-violating per `rules_check`.
- Keeping SwiGLU in fp32 between GEMM1 and GEMM2 adds 2 × intermediate tensor size vs quantizing to FP8 with a new block scale; the quantized path halves DRAM between stages but adds a reduction-max in the epilogue.
- UE8M0 packed scales (SM100 path) save 4× scale memory but require bit-packing.

## Pitfalls

- Applying weight scales before `tl.dot` instead of in the epilogue inflates register pressure and defeats the point of FP8 operands.
- Using `s_with_bias` for the weight normalization (instead of `s`) reorders normalization and bias — breaks bitwise match with `torch_ref.py:124–126`.
- Dropping the `+1e-20` floor on `weights_sum` gives NaNs when a token has zero selected mass (happens when `score_mask` zeroes a group full of `-inf`).
- Casting `acc` to bf16 before the scatter-add accumulation loses precision and conflicts with Rule 3 (required math must stay correct); accumulate bf16 outputs by casting only at the final store.
- `tl.dot_scaled` scale layout must be preshuffled to Blackwell's 5D form — forgetting the permute silently corrupts results.
