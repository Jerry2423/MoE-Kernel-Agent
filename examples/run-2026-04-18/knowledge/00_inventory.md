---
topic: inventory
applies_to: general
source: context/
confidence: high
---

# Context inventory (what each reference demonstrates)

## Top-level docs

- `context/INDEX.md` — **retrieval map** from bottleneck class → technique → source (file + line range). Seed for `knowledge/INDEX.md`.
- `context/triton_tricks.md` — Triton-level heuristics: load order, cache eviction policies, `tl.range` over `range`, config-pruning autotune, zero-init pre-hook, atomic memory-sync tuning.
- `context/config.toml` — harness config (not a technique).

## `context/techniques/`

- `persistence.md` — persistent programming style (launch ~#SMs blocks, grid-stride over tiles).
- `pid_swizzle.md` — super-group tile ordering for L2 reuse (`GROUP_SIZE_M = 4–8`).
- `tma.md` — on-host vs on-device TMA descriptors, swizzled SMEM, multicast.

## `context/hardware/`

- `b200.md` — B200 SXM constants: 148 SMs, 228 KB SMEM/SM, 256 regs/thread at 64 warps, 8 TB/s HBM3e, 4500 TF fp8, AI crossover fp16≈280 / fp8≈560 FLOPs/byte, 126 MB L2 (split across two dies).


## Workload / SOTA data

- `context/moe_workloads.jsonl` — per-workload input/shape specs.
- `context/sota_targets.jsonl` — per-UUID B200 SOTA latency (used for Phase-3 gap reporting).
- `context/ds_moe.json` — DeepSeek MoE architectural constants.

## Schemas (`context/schemas/*.schema.json`)

- `subgraph_spec.schema.json`, `rules_check.schema.json`, `workload_class.schema.json`, `iteration_row.schema.json`, `bench_result.schema.json` — Draft-07 JSON Schemas consumed by `tools/validate_artifacts.py`.

## Reusable skills (`.claude/skills/`)

- `OPTIMIZATION_TECHNIQUES.md` — synthesized guide to B200 kernel techniques: tile scheduling, TMA, FP8 block-scale GEMM, warp specialization, software pipelining, memory hierarchy, profiling, split-K.
- `kernels/{cross-entropy,flash-attention,gemm,rmsnorm,rotary-embedding,softmax}/SKILL.md` + templates.
- `optimization/tier{1-5}/SKILL.md` — tiered AutoKernel playbook (block-tiling, memory-access, compute-fusion, advanced-scheduling, arch-specific).
- `system/{benchmark,bottleneck-diagnosis,optimize-loop,profiling,verification}/SKILL.md` — harness-level procedures.
