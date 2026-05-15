---
topic: quantization
applies_to: hidden states + gemm1/gemm2 weights (FP8 e4m3fn + per-128 block scales); optional quantized GEMM1→GEMM2 intermediate
source: .claude/skills/OPTIMIZATION_TECHNIQUES.md (§3)
confidence: high
---

# Quantization — FP8 block-scale layout and scaling epilogues

## Summary

DSv3 uses FP8 e4m3fn with per-128 block scales in fp32 ((problem spec)). Activation scales live in `[H//128, T]` (transposed), GEMM1 weight scales in `[E, 2I//128, H//128]`, GEMM2 weight scales in `[E, H//128, I//128]`. A Blackwell-native path can use `tl.dot_scaled` with preshuffled scales; a portable path applies the scales in the fp32 epilogue after the dot.

## When to use

- Default portable path (works on both SM90 & SM100): accumulate FP8×FP8 to fp32, multiply by `a_scale[:, None] * w_scale[None, :]` at epilogue.
- Blackwell-optimized: `tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", out_dtype=fp32)` (`OPTIMIZATION_TECHNIQUES.md §3 B200`). Requires 5D scale preshuffle.
- Intermediate FP8 between GEMM1→GEMM2 only when GEMM1 output bandwidth is on the critical path — extra `block_max` reduction in epilogue.
- vLLM unified `QUANT_TYPE` epilogue when the kernel needs to handle FP8/INT8/INT4 without recompile — not our case but a readable template.

## Code pattern

```python
# per-block scale load, 1 scalar per 128-row × 128-col tile
a_scale = tl.load(a_scale_ptr + pid_m * stride_am_s
                  + tl.arange(0, BLOCK_M // 128))   # [BLOCK_M/128]
w_scale = tl.load(w_scale_ptr + expert_id * stride_we_s
                  + pid_n * stride_wn_s
                  + tl.arange(0, BLOCK_N // 128))   # [BLOCK_N/128]

# FP8 dot then broadcast-scale
acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
for k in range(...):
    acc += tl.dot(a8, w8, out_dtype=tl.float32)
out = acc * tl.broadcast_to(a_scale[:, None, None], (BM//128, 128, BN))
           * tl.broadcast_to(w_scale[None, None, :], (BM//128, 128, BN))
```

## Tradeoffs

- FP8 cuts weight footprint 2× vs bf16 → effective peak ≈ 4500 TF vs 2250 TF (`b200.md`); but AI crossover doubles to ~560 FLOPs/byte, so small-M kernels stay bandwidth-bound.
- `tl.dot_scaled` hides scale broadcast in hardware (0 extra FLOPs) but forces Blackwell-specific 5D preshuffle — compile-time only, but must be inverse-tested for correctness.
- Intermediate FP8 between GEMMs halves GEMM1→GEMM2 DRAM writes but costs one `block_max` reduction per tile and pollutes the epilogue register budget.
- Per-128 granularity means the full SwiGLU intermediate (`[T, I]`) carries `I/128 = 16` scales per token — small overhead.

## Pitfalls

- Loading the transposed activation scale with the wrong stride (`[H//128, T]` vs `[T, H//128]`) is easy to miss — see shape assertion in `torch_ref.py:58`.
- Scale pointers for GEMM1 weights are `[E, (2I)/128, H//128] = [E, 32, 56]`; ordering them as `[E, H//128, 2I/128]` corrupts every tile by a single transpose.
- Applying weight scales before the dot (to "pre-scale" FP8 ops) inflates operands to fp32 and removes all FP8 throughput gain.
- UE8M0 packed SM100 path needs explicit `(exp >> 23) & 0xFF` extraction; getting the bit positions wrong silently scales outputs by 2^k.
- Re-quantizing the GEMM1 output to FP8 requires a new `[M//128, N//128]` scale buffer plumbed to GEMM2; failing to thread it through is a classic Stage-2.3 bug.
