# Rules check — Stage 2.4 (solution/triton/kernel.py)

Anti-cheat self-audit against `RULES.md`. All five rules pass. Machine-
readable evidence is in `solution/rules_check.json`; this file is the
human-readable narrative.

## Rule 1 — Core computation in Triton

Both GEMMs and all block-scale dequantization happen inside `@triton.jit`
kernels (`_gemm1_swiglu_kernel` at L50–136, `_gemm2_scatter_kernel` at
L143–212). `tl.dot` with FP8 operands produces the fp32 accumulator
(L124–125, L204). No high-level PyTorch API (`torch.matmul`, `F.linear`,
`F.softmax`, `F.layer_norm`, `F.conv*`, transformer helpers) appears
anywhere in the hot path. The routing preamble uses only primitive torch
ops (`torch.sigmoid`, `torch.topk`, `torch.bincount`, `torch.argsort`,
`torch.gather`, `+`, `*`, `/`) — none listed as forbidden in `RULES.md §1`.

## Rule 2 — No trivial identity / no-op

Every region of both kernels does real work: block-scaled FP8×FP8 dot
products (`tl.dot(... out_dtype=tl.float32)`), SwiGLU via a genuine
sigmoid form `acc_gate / (1.0 + tl.exp(-acc_gate))` followed by a
gate-times-up elementwise multiply (L130–131), and a fp32 atomic
accumulation scaled by per-row gamma (L210–212). No kernel region
computes `output = input`, `+0`, or `*1.0`.

## Rule 3 — No omission of necessary computations

All four subgraphs from `solution/subgraph_specs.json` are implemented:

- Routing + normalize — L240–260 (`torch.sigmoid`, grouped top-2, top-4
  groups, masked global top-8, weight normalization with `+1e-20` and
  `routed_scaling_factor`).
- Dispatch — L262–281 (stable-argsort by local-expert id, histogram,
  cumsum).
- GEMM1 + SwiGLU — `_gemm1_swiglu_kernel` (dequantized FP8 dot of both
  up and gate halves, per-block activation and weight scales, SwiGLU
  silu(gate)*up).
- GEMM2 + weighted scatter — `_gemm2_scatter_kernel` (FP8 weight
  dequantization, fp32 dot, per-row gamma multiply, fp32 atomic
  scatter-add; final fp32→bf16 cast at L307 of `run`).

No required component is skipped.

## Rule 4 — Targets a real bottleneck

The two Triton kernels together handle every FLOP-heavy stage of the MoE
forward pass: GEMM1 (`[Tk, H=7168] × [H, 2I=4096]`), SwiGLU
(elementwise I=2048), and GEMM2 (`[Tk, I=2048] × [I, H=7168]`). Both are
canonical compute-bound / mixed-precision kernels where a hand-written
Triton implementation beats the eager torch equivalent by fusing
dequantization, SwiGLU, and scatter into the tensor-core pipeline. This
is exactly the workload the bench is scoring.

## Rule 5 — Efficient parallelism

Both kernels use `tl.program_id(0/1/2)` for expert × tile-M × tile-N
parallelism (L63–65, L156–158), and `tl.arange` for `BLOCK_M` rows,
`BLOCK_N` columns, `BK=128` K-block chunks (L75, L81, L88, L167, L174,
L180). No scalar-only code path. The grid scales with total work (E_LOCAL
× max_tiles_m × num_tiles_n); for a short sequence the grid shrinks, for
a long one it grows.
