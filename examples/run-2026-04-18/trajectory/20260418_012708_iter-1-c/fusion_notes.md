# Fusion notes (Phase 2.1)

## Fused subgraphs

- **`routing`** ← `sigmoid(logits) + bias → reshape(N_GROUP, GROUP_SZ) → per-group top-2 → sum → top-k groups (mask) → broadcast mask → masked global top-k → one-hot build → weights = s * M`. Rationale: every op in the chain has a single consumer in the next op of the same per-token slice; no intermediate escapes the chain. The fused form matches `input/torch_ref.py:94-123` exactly (line-by-line).
- **`normalize_weights`** ← `weights / (sum+1e-20) × routed_scaling_factor`. Kept as a second function only to make it clear in `subgraph_specs.json` that the routing kernel's epilogue is numerically `s`-normalized (not `s_bias`-normalized); same chain with one reduction, no semantic change.
- **`dispatch`** ← `(topk_idx − offset) > mask → gather → stable argsort → bincount → cumsum`. Rationale: post-routing bookkeeping is data-only — no arithmetic operand flows into a compute kernel without going through dispatch first. All the downstream kernels consume `(tokens_sorted, gamma_sorted, hist, offs)` as an atom.
- **`gemm1_swiglu_per_expert`** ← `A_e @ W13_e.t() → split (X1, X2) → silu(X2) * X1`. Rationale: the `[Tk_e, 2I]` tensor out of the matmul has a single consumer (SwiGLU), and the SwiGLU collapses it to `[Tk_e, I]`. Fusing saves one round-trip to HBM of size `Tk × 2I × 4 bytes`. Numerically identical to the reference because SwiGLU uses the same sigmoid form (`x / (1 + exp(-x))`, `input/torch_ref.py:158`).
- **`gemm2_weighted_scatter_per_expert`** ← `C_e @ W2_e.t() → * gamma[:, None] → index_add_`. Rationale: the `[Tk_e, H]` matmul output is only multiplied by gamma before atomic-accumulation into the output; fusing the gamma multiply and the index_add is a pure epilogue collapse. We accumulate in fp32 rather than bf16 because bf16 `index_add_` (atomic on GPU) is lossy for chains with multiple accumulations per row; the final cast to bf16 happens once in `run()`.

## Non-fusions (rejected candidates)

- **`_dequant_A` + GEMM1** — fusing the per-token-row dequant scale into GEMM1's A-load is a pure Triton optimization (per-128-row scale applied inside the K-loop epilogue) that survives the reference. Left separate here because in the fused-reference PyTorch file the dequant is a single op that populates `A`; the Triton build will re-fuse it into the GEMM1 kernel. Equivalent in algebra.
- **Dequant of weights (`_dequant_W13`, `_dequant_W2`)** — same story: each weight dequant is a separate pass in the fused PyTorch, but the Triton kernel will load FP8 operands directly and scale per K-block in the dot accumulator. Fusion is algorithmic, not done at the PyTorch level, to keep `fused_reference.py` readable.
- **Routing ↔ Dispatch fusion** — possible in principle (the routing kernel could emit the sorted token list directly), but dispatch is bincount+argsort which is data-dependent and hard to express in a single PyTorch pass without loops. Kept separate; Phase 3 can revisit. No numerical impact.
- **GEMM1+SwiGLU ↔ GEMM2 fusion** — the two operate on different `(M, N)` shapes (`[Tk, I]` vs `[Tk, H]`), and their K-loops go different directions. A true fusion would stall at the SwiGLU write because GEMM2's K dimension is over the same `I` that GEMM1 just produced. Not worth chasing in Phase 2; this is the classic "fused MoE" target for Phase 3 if the intermediate memory traffic turns out to be the bottleneck (roofline check pending profile).

## Numerical caveats

- Routing normalization uses **`s` (pre-bias sigmoid)**, not `s_with_bias` (`input/torch_ref.py:124`). The fused version repeats this — violating it breaks atol immediately.
- `+1e-20` floor on the weight-sum denominator matches `input/torch_ref.py:125`. Removing it produces NaNs when a token selects no local-rank expert mass.
- The fp32 → bf16 final cast in `run()` is the only precision-losing step we inherit from the reference (`input/torch_ref.py:168`).
- `torch.topk(..., sorted=False)` is used throughout to match the reference exactly; the DeepSeek-V3 no-aux algorithm is tolerant of unordered top-k but the reference uses this flag explicitly, so we preserve it.
