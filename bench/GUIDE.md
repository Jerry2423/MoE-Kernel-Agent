# Modal Benchmark & Profiling Scripts

This is the canonical reference for every Modal-driven script under `bench/`. Each script maps to a specific phase and sub-phase. Use the table below to pick the right one — do not guess, and do not invoke `modal_common.py` directly.

## Prerequisites

- `modal setup` completed (one-time).
- `flashinfer-trace` volume populated with workload data.
- `flashinfer-bench`, `modal` packages available locally.
- Always pass `-m` (module) flag to `modal run`. File-path syntax (`modal run bench/modal_bench.py`) causes import failures.

---

## Script → Phase Map

| Script | Purpose | When to run | Typical phase |
|--------|---------|-------------|---------------|
| `modal_bench.py` | Correctness + end-to-end runtime on one or all workloads | Every iteration, every verification check | Phase 2.4 (verify), Phase 2.5 (freeze), Phase 3.B (per-iteration measure), Phase 3.C (integration check) |
| `modal_profile.py` | torch.profiler end-to-end trace; identifies which stage dominates | Before changing code in a new sub-phase, after any bottleneck shift | Phase 3.A (classify), Phase 3.B (find new target bottleneck) |
| `modal_ncu.py` | Nsight Compute deep-dive on one kernel | After `modal_profile.py` names a dominant kernel | Phase 3.A (per representative), Phase 3.B (after profile shifts) |
| `modal_autotune.py` | Full search-space sweep per (kernel, seq_len) on B200 | Phase 2.5 freeze; Phase 3.C re-freeze after a justified algorithmic change | Phase 2.5, Phase 3.C |
| `modal_common.py` | Shared image definitions, solution packing | Never invoked directly — imported by every other script | n/a |

`scripts/bench.sh` wraps `modal_bench.py` and adds trajectory capture. Always call `bash scripts/bench.sh <label>` from phase prompts; do not call `modal_bench.py` directly unless debugging the wrapper.

---

## Per-Script Reference

### `modal_bench.py`

Packs `solution/triton/kernel.py` into a `flashinfer_bench.Solution` and runs it on B200.

```
cd .. && PYTHONPATH=. modal run -m bench.modal_bench
PYTHONPATH=. modal run -m bench.modal_bench --workload-index 8
```

**Output format** (stdout, captured by `scripts/bench.sh` to `trajectory/*/output.txt`):
```
COMPILED: True
CORRECT: True
RUNTIME: 1.2340
REF_RUNTIME: 5.6780
SPEEDUP: 4.60x
```

**You must update this script when:**
- The bench needs a new CLI flag (e.g., `--workload-range`).
- The solution entry point moves off `solution/triton/kernel.py`.
- You change the output contract that `bench-wrapper.sh` / trajectory parsing depends on.

### `modal_profile.py` (Stage 1 profiling)

Wraps the full `run()` call in `torch.profiler` with `record_function` markers per stage, writes a Chrome JSON trace to the Modal volume.

```
PYTHONPATH=. modal run -m bench.modal_profile --workload-index 8
PYTHONPATH=. modal run -m bench.modal_profile --workload-index 1 --warmup 5 --active 3
```

**Fetch traces locally (required — they live on the volume, not local disk):**
```
modal volume get flashinfer-trace profiles/ ./profiler_results/
```

Then open in https://ui.perfetto.dev or `chrome://tracing`.

**Current `record_function` markers** (project-specific — expect to rename when the kernel family changes):
- `Routing` → routing kernel(s).
- `GEMM1+SwiGLU` → FP8×FP8 GEMM + SwiGLU.
- `GEMM2` → f32×FP8 GEMM + weighted scatter.

**You must update this script when:**
- Stages are added, removed, or renamed. The `record_function` names here are the labels Phase 3.A uses to reason about bottlenecks — keep them in sync with `subgraph_specs.json`.
- You need per-invocation NVTX markers that the current script does not emit.

### `modal_ncu.py` (Stage 2 profiling)

Runs Nsight Compute on a named kernel via `flashinfer_bench.agents.ncu`. Wraps the profiled call in an NVTX range so NCU skips warmup/JIT.

```
PYTHONPATH=. modal run -m bench.modal_ncu --workload-index 8
PYTHONPATH=. modal run -m bench.modal_ncu --workload-index 1 --set basic
PYTHONPATH=. modal run -m bench.modal_ncu --workload-index 10 --kernel-name "gemm1"
PYTHONPATH=. modal run -m bench.modal_ncu --workload-index 1 --sections "MemoryWorkloadAnalysis,Occupancy"
```

**Output:** text report printed to stdout; JSON saved to `./ncu_results/<solution>_w<N>.json`. No volume fetch needed.

**You must update this script when:**
- A new kernel name needs to be targetable via `--kernel-name`.
- The set of NCU sections you routinely want changes (e.g., Blackwell-specific metrics).

### `modal_autotune.py`

Full search-space sweep per (kernel, seq_len), producing the winning config per pair. The current script is hard-coded for DeepSeek-V3 MoE geometry (`H=7168`, `I=2048`, `E_LOCAL=32`, `E_GLOBAL=256`, `TOP_K=8`, `BLOCK_K=128`) and knows three kernel names: `gemm1_swiglu`, `gemm1_gather`, `gemm2`.

```
PYTHONPATH=. modal run -m bench.modal_autotune
```

The script prunes the search space with:
- SMEM estimators (`smem_gemm1_swiglu`, `smem_gemm1_gather`, `smem_gemm2`) vs. B200 SMEM limit 232448.
- `n_valid_for(seq_len)` to reject `BLOCK_M` values that waste tile rows.
- `num_warps * 32 > BM * BN` guard.
- `loop_num_stages ≤ num_stages + 1` guard.

**You must update this script when (very likely as kernels evolve):**
- A new kernel is added. Add its name to the dispatch, its SMEM estimator, and its min-`BN` constraint.
- MoE geometry changes (`H`, `I`, `E_LOCAL`, `E_GLOBAL`, `TOP_K`, `BLOCK_K`). These constants must match `solution/triton/kernel.py` — drift here silently produces wrong "winning" configs.
- A new tunable parameter is added to a kernel (e.g., a new `tl.constexpr`). Add it to the config product and the SMEM estimator.
- A new workload `seq_len` shows up in `context/moe_workloads.jsonl` that is not in `SEQ_LENS`. Add it.
- SMEM limit or other hardware constants change (new GPU target).

**Phase 2.5 uses this script to freeze** — every workload in `context/moe_workloads.jsonl` must be covered. The winning configs are copied into `solution/triton/kernel.py` and into `solution/autotune_notes.md`.

**Phase 3 must not invoke this script** except at sub-phase 3.C, and only then if the algorithmic change it is integrating justifies a re-freeze per the rules in `phase3-optimize.md`.

### `modal_common.py`

Infrastructure — not invoked directly. Provides `pack_solution_from_dir`, image definitions (`FLASHINFER_IMAGE`, `NCU_IMAGE`), and volume bindings. Every other script imports from it.

**You must update this file when:**
- Pinned package versions need to change across all images (keep them consistent).
- A new image variant is required (e.g., a different CUDA toolkit).
- `solution/` layout changes (current contract: `solution/triton/kernel.py` with `config.toml` selecting `language = "triton"`).

---

## Result Locations

| Tool | Where results land | How to access |
|------|--------------------|---------------|
| `modal_bench.py` | stdout | Printed live; `scripts/bench.sh` captures to `trajectory/<stamp>_<label>/output.txt` |
| `modal_profile.py` | Modal volume `/data/profiles/` | `modal volume get flashinfer-trace profiles/ ./profiler_results/` |
| `modal_ncu.py` | stdout + local file | Printed live; saved to `ncu_results/<name>_w<N>.json` |
| `modal_autotune.py` | stdout (winning configs); user writes them to `solution/autotune_notes.md` and the kernel |

---

## Workload Index Reference

Nineteen workloads in `context/moe_workloads.jsonl`, four coarse categories. Categories below are a *starting point* — Phase 3.A reclassifies by measured bottleneck, not by shape.

### Very short (seq_len ≤ 32)
| Index | seq_len | expert_offset | UUID prefix |
|-------|---------|---------------|-------------|
| 1 | 1   | 32  | e05c6c03 |
| 5 | 16  | 224 | a7c2bcfd |
| 6 | 15  | 32  | 2e69caee |
| 7 | 14  | 0   | 8cba5890 |
| 0 | 7   | 192 | b8f4f012 |
| 2 | 32  | 32  | 6230e838 |

### Short (50–80)
| Index | seq_len | expert_offset | UUID prefix |
|-------|---------|---------------|-------------|
| 3  | 80 | 96  | 8f1ff9f1 |
| 10 | 62 | 96  | 5eadab1e |
| 11 | 59 | 160 | eedc63b2 |
| 12 | 58 | 64  | e626d3e6 |
| 13 | 57 | 96  | 74d7ff04 |
| 14 | 56 | 64  | 4822167c |
| 15 | 55 | 128 | 81955b1e |
| 16 | 54 | 128 | 76010cb4 |
| 17 | 53 | 32  | fc378037 |
| 18 | 52 | 160 | f7d6ac7c |

### Long (~901)
| Index | seq_len | expert_offset | UUID prefix |
|-------|---------|---------------|-------------|
| 4 | 901 | 96 | 1a4c6ba1 |

### Very long (>901)
| Index | seq_len | expert_offset | UUID prefix |
|-------|---------|---------------|-------------|
| 8 | 14107 | 32  | 5e8dc11c |
| 9 | 11948 | 128 | 58a34f27 |

**Default profiling targets when just surveying:** index 1 (very-short end) and index 8 (very-long end). Phase 3.A chooses its own representatives per measured class.

---

## Solution Contract

- Entry point: `run()` in `solution/triton/kernel.py`.
- Calling convention: destination_passing_style — output tensor is the last parameter.
- `context/config.toml` selects `language = "triton"`; `context/ds_moe.json` holds the operation definition.
- Directory layout: `solution/triton/kernel.py` (not `solution/kernel.py`) because `config.toml` selects the `triton/` subdirectory.

Every Modal run re-packs the **latest** `solution/triton/kernel.py` fresh — there is no caching of kernel source.

---

## When to Update Which Script (checklist as kernels evolve)

Use this during Phase 2.3 and any Phase 3 sub-phase that introduces an algorithmic change.

| Change you are making | Update these scripts |
|-----------------------|----------------------|
| Add a new kernel name | `modal_autotune.py` (dispatch, SMEM estimator, min-BN), `modal_ncu.py` (`--kernel-name`), `modal_profile.py` (record_function label) |
| Rename a kernel or stage | Every script that references the old name by string (grep before editing) |
| Add a new tunable `tl.constexpr` | `modal_autotune.py` search-space product and SMEM estimator |
| Change MoE geometry (`H`, `I`, `E_LOCAL`, `E_GLOBAL`, `TOP_K`, `BLOCK_K`) | `modal_autotune.py` constants (must match `kernel.py`) |
| Add a new workload to `context/moe_workloads.jsonl` | `modal_autotune.py` `SEQ_LENS`, `bench/GUIDE.md` workload table, Phase 2.5 sweep |
| Change output tensor layout or add outputs | `modal_common.py` `pack_solution_from_dir` contract; verify `modal_bench.py` diff logic |
| Target a different GPU | `modal_common.py` image + `modal_autotune.py` `SMEM_LIMIT` |
| Change the solution file layout | `modal_common.py` and `scripts/bench.sh` |

Rule of thumb: if a change affects a string a Modal script matches on, or a numeric constant a Modal script uses to prune a config space, assume the script needs updating. Grep for the old value before trusting that it doesn't.

---

## How It Works

1. `modal_common.py` packs `solution/triton/kernel.py` into a `flashinfer_bench.Solution`.
2. The packed solution is sent to a Modal B200 container.
3. **Bench:** container loads workloads from the `flashinfer-trace` volume and runs FlashInfer eval.
4. **Profile:** container extracts kernel source, runs `torch.profiler`, writes the Chrome JSON trace to the volume.
5. **NCU:** container runs NCU via `flashinfer_bench.agents.ncu` and returns the text report directly.
6. **Autotune:** container iterates the generated config list for each (kernel, seq_len), records per-config medians, returns the best config per pair.
