---
topic: moe-data-layouts
applies_to: DSv3 fused-MoE kernel — hidden states, weights, scales, routing outputs, output buffer
source: input/torch_ref.py, .claude/skills/OPTIMIZATION_TECHNIQUES.md (§3)
confidence: high
---

# MoE data layouts — what each tensor looks like and why

## Summary

Shapes are fixed by DSv3 ((problem spec), `torch_ref.py:41–63`): activations `[T, H=7168]` FP8 with scale `[H//128=56, T]` (transposed), GEMM1 weights `[E_local=32, 2I=4096, H]` FP8 with scale `[E,32,56]`, GEMM2 weights `[E_local, H, I=2048]` FP8 with scale `[E,56,16]`, routing outputs `[T, TOP_K=8]`, final output `[T, H]` bf16. The transpose in the activation scale layout matters: it lets a GEMM1 tile load one scale row for an M-column group in one coalesced vector.

## When to use

- Match these exact strides when writing any new kernel — the reference (`torch_ref.py:57–62`) shape-asserts them, and the bench harness allocates accordingly.
- Use `[expert, block_row, block_col]` scale order for weights so the per-expert offset is a single multiply ((problem spec)).
- Plan the routing outputs the bench expects: `topk_idx: [T,8] int32`, `topk_weights: [T,8] fp32`.
- Pad `sum_M_per_expert` up to `BLOCK_M` when using vLLM block-assignment.

## Code pattern

```python
# Hidden-state FP8 row load with transposed scale
# a_ptr:      float8e4m3fn, shape [T, H],          stride (H, 1)
# a_scale:    float32,      shape [H//128, T],     stride (T, 1)  <-- transposed
a_row  = tl.load(a_ptr      + offs_m[:, None] * H   + offs_k[None, :])
a_sc   = tl.load(a_scale_ptr + (offs_k // 128) * T + offs_m)       # [BLOCK_M]

# Weight FP8 and scale for expert `e` (GEMM1)
# W13:        [E, 2I, H], stride (2I*H, H, 1)
# W13_scale:  [E, 2I//128, H//128], stride (32*56, 56, 1)
w_off  = e * (2*I) * H + offs_n[:, None] * H + offs_k[None, :]
ws_off = e * 32 * 56 + (offs_n // 128)[:, None] * 56 + (offs_k // 128)[None, :]
```

## Tradeoffs

- Activation scale `[H//128, T]` is fast for GEMM1 (reads one scale row per K-block) but awkward for GEMM2 where the activation is the `[T, I]` SwiGLU intermediate (re-quantize on the fly).
- Weight layout `[E, N, K]` is **N-major** inside each expert (matches `torch_ref.py:59`); `[E, K, N]` would be faster for dot B-fragment loads but conflicts with the fixed contract.
- Padding M to `BLOCK_M` wastes FLOPs proportional to `E_local × BLOCK_M / 2` per skewed distribution — small-seq workloads suffer most.
- Output buffer is `[T, H]` — scatter GEMM2 requires atomics; a per-expert dense output avoids atomics at the cost of `E_local × T × H × 2B = 32 × T × 14.3 KB` staging (prohibitive past T ≈ 1k).

## Pitfalls

- Forgetting the transpose on `hidden_states_scale` (`torch_ref.py:58`): `[H//128, T]` not `[T, H//128]` — silent factor-of-T stride bug.
- GEMM1 weight is `[E, 2I, H]` — meaning `W13.t()` is implicit (`torch_ref.py:153`). Loading as `[E, H, 2I]` swaps K and N.
- GEMM2 weight is `[E, H, I]` and the reference does `W2.t()` (`torch_ref.py:162`) — easy to invert.
- Routing output tensors must be **fp32** (`topk_weights`) and **int32/int64** (`topk_indices`).
- vLLM `sorted_token_ids` uses padding sentinel ≥ `num_valid_tokens` for masking; storing 0 as the padding value silently accumulates wrong rows.
