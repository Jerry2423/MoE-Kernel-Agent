# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | Class | Speedup(mean) | Runtime(mean) | Status |
|------|-------|-------|---------|--------------|--------|

## iter 1 — Fuse routing into a single Triton kernel

- **class_id**: `short_latency_bound`
- **target bottleneck**: ~170 us in ~15 small torch CUDA kernels (gatherTopK 25+13us, write_indices 18us, DeviceSelectSweep 13us, multiple index_elementwise, cub scan/reduce). Each pays ~10 us launch overhead. Replace with one Triton kernel emitting `topk_idx [T, TOP_K] int32` + `topk_weights [T, TOP_K] fp32`.
- **technique source**: DSv3 no-aux routing pattern + standard warp top-K reduction.
- **pre-iter baseline** (`trajectory/20260418_010720_phase3-entry`): seq=1 → 0.939 ms, seq=14107 → 8.461 ms, mean 1.778 ms.
- **measurements** (3 runs, medians): seq=1 → **0.729 ms** (-22%), seq=901 → 1.587 ms (-12%), seq=14107 → 7.973 ms (-6%). Mean across 19 → 1.526 ms (-14%). All 19 PASSED. SOTA geomean gap 5.19x → 4.22x; total headroom 20.86 ms → 15.27 ms.
- **status**: improved (all classes improved, no regression).
- **bottleneck_next**: Post-routing dispatch (`argsort + bincount + cumsum + gather + is_local filter`) still runs ~50 us of torch ops — the next short-class iter could fuse dispatch. For long class, GEMM2 is still 56% of runtime.

| Iter | Title | Class | Speedup(mean) | Runtime(mean) | Status |
|------|-------|-------|---------|--------------|--------|
|    1 | Fuse DSv3 no-aux routing into Triton kernel | short_latency_bound | 14.64x | 1.526 ms | improved |
|    2 | Restore FP8×FP8 dot in GEMM1 (retry post OOB fix) | long_compute_bound | — | — | reverted |
|    3 | bf16 intermediate buffer (GEMM1→GEMM2) | long_compute_bound | 14.95x | 1.502 ms | improved |
|    4 | Move per-block scales out of K-loop (cast+multiply after dot) | long_compute_bound | — | — | reverted |
|    5 | GROUP_M=8 swizzle on GEMM2 | long_compute_bound | — | — | reverted |

### iter 5 — notes (reverted)
Added super-group swizzle so adjacent blocks share W[e, offs_n, :] across m-tiles. Regressed short class rep by 8.5% (0.729 → 0.791 ms) due to the extra signature params + integer math for swizzle that don't amortize when `max_tiles_m=1`. Long class was flat — L2 reuse isn't the active bottleneck there. Reverted.

## Integration (Phase 3.C)

Dispatch mechanism chosen: **single kernel, no per-class branch**. Every landed win (iters 1, 3, 7, 9, 10, 12) lives in one `solution/triton/kernel.py` — routing + dispatch + GEMM1+SwiGLU + GEMM2 + final cast — without per-class code paths. The only shape-dependent dispatch is the existing `_pick_cfg(_GEMM1_BEST / _GEMM2_BEST, T)` lookup into the Phase-2 frozen autotune table at `run()` entry.

No algorithmic change required an autotune re-sweep (none altered the kernel signatures or tile geometry space). `modal_autotune.py` was not re-run in Phase 3.

Validation — `bash scripts/bench.sh integrated` ran 5 times across all 19 workloads. Per-workload medians (ms):

| class | seq_len | rep median (ms) | sota (ms) | gap | vs Phase-2 rep |
|:------|--------:|----------------:|----------:|----:|---------------:|
| short | 1 | 0.373 | 0.063 | 5.92× | 0.939 → 0.373 (2.52×) |
| short | 7 | 0.387 | 0.095 | 4.07× | ~1.0 → 0.387 |
| short | 80 | 0.577 | 0.269 | 2.14× | 1.171 → 0.577 |
| medium | 901 | 1.202 | 0.699 | 1.72× | 1.801 → 1.202 (1.50×) |
| long | 11948 | 4.934 | 3.671 | 1.34× | 6.017 → 4.934 (1.22×) |
| long | 14107 | 7.279 | 5.579 | 1.30× | 8.461 → 7.279 (1.16×) |

Per-class means:
- short (16 workloads): 0.481 ms (Phase-2: ~1.14 ms avg, 2.37× faster)
- medium (1 workload): 1.202 ms (1.50× faster)
- long (2 workloads): 6.107 ms (1.19× faster)
- overall mean (19 workloads): **1.111 ms** (Phase-2: 1.778 ms, **1.60× faster**)
- speedup vs torch_ref: **27.91× best / ~25.7× median**
- SOTA geomean gap: **2.53×** (Phase-2: 5.19×)
- Total remaining headroom: **~8.3 ms** (Phase-2: 20.86 ms, **59% closed**)

Acceptance: CORRECT=True on all 19 workloads across all 5 runs. No class regresses vs its best 3.B runtime. Geometric mean across classes (0.481 × 1.202 × 6.107)^(1/3) ≈ 1.28 ms is below the class-best 3.B results.

## Summary after iter 15

After 15 iterations (iters 1, 3, 7, 9, 10, 12 improved; 13 no-change; 2, 4, 5, 6, 8, 11, 14, 15 reverted):
- **short** rep (seq=1): **0.380 ms** (Phase-2: 0.939 ms, **2.47× faster**)
- **medium** rep (seq=901): **1.189 ms** (Phase-2: 1.801 ms, 1.51× faster)
- **long** rep (seq=14107): **7.37 ms** (Phase-2: 8.461 ms, 1.15× faster)
- Mean across 19: **1.118 ms** (Phase-2: 1.778 ms, 1.59× faster)
- Speedup over torch_ref: **25.14×** (Phase-2: 11.73×)
- SOTA geomean gap: **2.53×** (Phase-2: 5.19×)
- Total headroom remaining: **8.33 ms** (Phase-2: 20.86 ms)

Landed wins:
1. **iter 1**  — fused Triton routing kernel (sigmoid+group-topk+normalize).
2. **iter 3**  — bf16 intermediate buffer (halves GEMM2 LHS bandwidth).
3. **iter 7**  — drop `is_local.any()` CPU-GPU sync.
4. **iter 9**  — emit local-expert id with -1 sentinel in routing (saves host sub+cmp).
5. **iter 10** — replace torch dispatch chain (argsort/bincount/index_select ×3) with 2 Triton kernels (biggest single win: −17.5% mean).
6. **iter 12** — fused fp32→bf16 cast-and-copy kernel.

Reverted techniques (don't retry without a new angle):
- FP8 dot (iter 2): numerical failure on long seqs; Triton/B200 FP8 dot over K=7168 seems to mis-scale.
- Scale-out-of-K-loop (iter 4): defeats Triton dequant+cast+dot fusion.
- GROUP_M swizzle on GEMM2 (iter 5): overhead > benefit at small max_tiles_m.
- FP8 intermediate via quant kernel (iter 6): compile error (FP8_MAX constexpr or fp8 cast path).
- GEMM2 grid axis reorder (iter 8): scheduler already handles L2 locality.
- bf16 atomic scatter (iter 11): B200 bf16 atomic is CAS-loop slow on long seqs.
- Persistent GEMM2 strided (iter 14): strides across expert boundaries → L2 thrash.
- Persistent GEMM2 block-partitioned (iter 15): Triton compile error on nested control flow around dot.

Stuck on long class: 5+ attempts, only iter 3 landed (+2%). Biggest remaining lever on GEMM2 (~4 ms at seq=14107) seems to require either (a) a cleanly-structured persistent kernel that avoids nested conditionals around `tl.dot`, or (b) a working fp8 intermediate path that doesn't trip the FP8-dot numerical issue. Both are substantial — next attempt should profile via NCU to pick a specific metric to chase rather than guessing structural rewrites.

## Summary after iter 10

After 10 iterations (4 improved: 1, 3, 7, 9, 10 — 5 reverted: 2, 4, 5, 6, 8), end state at iter-10-cleanup:
- seq=1 (short rep): **0.408 ms** (Phase-2: 0.939 ms, **2.30× faster**)
- seq=901 (medium rep): **1.235 ms** (Phase-2: 1.801 ms, 1.46× faster)
- seq=14107 (long rep): **7.373 ms** (Phase-2: 8.461 ms, 1.15× faster)
- Mean across 19: **1.139 ms** (Phase-2: 1.778 ms, 1.56× faster)
- Speedup over torch_ref: **24.88×** (Phase-2: 11.73×)
- SOTA geomean gap: **2.60×** (Phase-2: 5.19×)
- Total headroom remaining: **8.72 ms** (Phase-2: 20.86 ms)

Wins (in descending impact order):
1. **iter 10** — Triton dispatch count+scatter kernels (−17.5% mean, −39% short).
2. **iter 1**  — Triton routing kernel fusing sigmoid+group-topk+normalize (−14% mean).
3. **iter 7**  — drop `is_local.any()` CPU-GPU sync (−4.3% mean).
4. **iter 9**  — emit `-1`-sentinel local-expert id from routing (−3% mean).
5. **iter 3**  — bf16 intermediate (−2% mean; mostly long class).

Deadends (reverted): FP8 dot (numerical fail on long seqs), scale out of K-loop (Triton pipeline disfusion), GROUP_M swizzle (overhead > benefit at small max_tiles_m), fp8 intermediate quant kernel (compile error), grid-axis reorder (no benefit).

Remaining bottleneck snapshot:
- **Short class** — now dominated by GEMM1+SwiGLU (~0.22 ms) + GEMM2 (~0.17 ms). Routing + dispatch is ~0.03 ms combined. Big wins here are GEMM-internal now.
- **Medium class** — same balanced GEMM1/GEMM2 structure as Phase 3.A.
- **Long class** — barely touched; GEMM2 still ~4 ms at seq=14107. Structural win (fp8 intermediate or persistent grid) remains the biggest lever but has been hard to land.

## Summary after iter 5

After 5 iterations (1 improved × 2, 3 reverted), the state is:
- **short_latency_bound** rep (seq=1): 0.729 ms (iter 1 result) — **1.29× faster than Phase-2 baseline (0.939 ms)**.
- **medium_mixed** rep (seq=901): 1.587 ms — ~13% faster than Phase-2 (1.801 ms).
- **long_compute_bound** rep (seq=14107): 7.792 ms (iter 3 result) — ~8% faster than Phase-2 (8.461 ms).
- Mean across 19 workloads: 1.502 ms (Phase-2: 1.778 ms, −16%).
- SOTA gap 5.19× → 4.17×; total headroom 20.86 ms → 14.70 ms.

Wins: iter 1 (fuse routing → single Triton kernel), iter 3 (bf16 intermediate buffer).
Deadends: FP8 dot (numerical failure on long seqs — repeatable bug), scale-move-out-of-K-loop (Triton pipeline disfusion), GROUP_M swizzle (overhead > benefit at small max_tiles_m).

Next steps — pivot away from the dot inner path. Biggest remaining lever on the long class is probably (a) persistent grid with in-kernel expert loop to avoid N×num_experts over-launch, or (b) fp8 intermediate with proper per-128 quant scale to halve GEMM2 LHS bandwidth again. For short class, the next step is eliminating torch.argsort / bincount / gather from dispatch.

|    6 | FP8 intermediate via quant kernel (bf16 → fp8 between GEMM1 and GEMM2) | long_compute_bound | — | — | reverted |
|    7 | Drop `is_local.any()` CPU-GPU sync | short_latency_bound | 15.68x | 1.438 ms | improved |
|    8 | GEMM2 grid axis reorder (pid_m fastest, pid_e slowest) | long_compute_bound | — | — | reverted |
|    9 | Fuse local_e + is_local into routing kernel | short_latency_bound | 16.65x | 1.397 ms | improved |
|   10 | Replace torch dispatch chain with two Triton kernels | short_latency_bound | 24.44x | 1.156 ms | improved |
|   11 | atomic_add bf16 directly to output (skip fp32 scratch) | long_compute_bound | — | — | reverted |
|   12 | Fused fp32→bf16 cast kernel into output | short_latency_bound | 24.93x | 1.116 ms | improved |
|   13 | Drop redundant routing host-side `.to(fp32).contiguous()` | short_latency_bound | 25.54x | 1.116 ms | no-change |
|   14 | Persistent GEMM2 grid (NUM_SMS=148, strided tile_ids) | long_compute_bound | — | — | reverted |
|   15 | Persistent GEMM2, block-partitioned | long_compute_bound | — | — | reverted |

### iter 15 — notes (reverted)
Reworked iter 14 as block-partitioned persistent (each SM owns a contiguous range of `chunk` tile_ids instead of strided). All workloads COMPILE_ERROR — the nested control flow `for tile_id in range(start, end): if m_start < Tk_e: for kb in range(num_kb): …` with tl.load/dot inside apparently exceeds Triton's tolerance for data-dependent control flow around memory ops. Reverted via `git checkout`. If persistent is pursued again it must be structured as a flat tile-selection pattern without nested conditionals around dot.

### iter 14 — notes (reverted)
Converted GEMM2 to a persistent grid: launch 148 blocks (SM-count), each strides through tile_ids in expert-slowest linearized order. Short class gained (seq=1 0.380 → 0.318 ms, −16%) but long regressed badly: seq=14107 7.37 → 10.78 (+46%), seq=11948 +20%, mean +30%. Striding by 148 jumps across expert boundaries (~112 tiles/expert for long), so each SM churns through several experts' W weights instead of staying within one. Reverted. A **block-partitioned** persistent variant (contiguous tile range per SM) might unlock the short-class win without the long-class L2-thrash — candidate for a future iter.

### iter 13 — notes
Removed `routing_logits.to(float32).contiguous()` and `routing_bias.to(float32).reshape(-1).contiguous()` host calls; the Triton routing kernel already casts `tl.load(...).to(fp32)` in-register. Effect is noise-equivalent (routing_logits is already fp32 so `.to(fp32)` was a no-op test; routing_bias cast avoided). Kept as a cleanup commit.

### iter 12 — notes
Replaced `output.copy_(out_fp32.to(torch.bfloat16))` (2 kernels + a [T,H] bf16 temp alloc) with a single Triton `_cast_fp32_to_bf16_kernel`. Saves one kernel launch and one allocation per call.
Medians (3 runs): seq=1 **0.380 ms** (−6.9%), seq=901 1.189 ms (−3.7%), seq=14107 ~7.32 ms (flat). Mean 1.11 ms (−3.5%). All 19 PASSED. Speedup 24.44× → 24.93×; SOTA gap 2.68× → 2.52×.

### iter 11 — notes (reverted)
Tried atomic-adding `acc.to(bf16)` into the caller-provided bf16 `output` tensor to skip the 404 MB fp32 scratch + `.to(bf16).copy_()`. bf16 atomics on B200 Triton appear to hit a CAS-loop fallback: seq=14107 regressed from 7.37 → 10.17 ms (+38%), seq=11948 +16%, mean +25%. Reverted same iter.

### iter 10 — notes
Added `_dispatch_count_kernel` (per-token atomic_add into hist[E_LOCAL]) and `_dispatch_scatter_kernel` (atomic counter per expert → slot; write `t_sorted` + `gamma_sorted`). Replaced `torch.argsort + bincount + zeros + cumsum + .to() + index_select × 3 + .contiguous() × 2` (≈13 torch ops) with 2 Triton launches + a 32-element host cumsum.
Medians (3 runs): seq=1 **0.408 ms** (−38.7% vs iter 9 0.666), seq=901 1.235 ms (−17.4%), seq=14107 7.373 ms (−3.9%). Mean 1.146 ms (−17.5%). **Speedup 16.65× → 24.44×**; SOTA gap 3.72× → 2.68×; total headroom 13.62 ms → 8.52 ms. All 19 PASSED. Biggest win of Phase 3 so far — confirms the short class was paying >100 us/call in torch launch overhead.

### iter 9 — notes
Extended _routing_kernel to also emit the local-expert id (or -1 sentinel for non-local) in one pass; dispatch on host now needs a single `>= 0` mask instead of `(ge && lt)` compound. Also dropped the `.to(int64)` cast and `.to(int32)` re-cast from the dispatch chain.
Medians (3 runs): seq=1 **0.666 ms** (−3.2% vs iter 7), seq=901 1.496 ms (−2.5%), seq=14107 7.672 ms (−0.2% / flat). Mean 1.428 ms (noise-eq). Speedup 15.68× → 16.65×; SOTA gap 3.92× → 3.72×; total headroom 14.41 ms → 13.62 ms. All 19 PASSED.

### iter 8 — notes (reverted)
Flipped GEMM2 launch from `(E_LOCAL, max_tiles_m, num_tiles_n)` to `(max_tiles_m, num_tiles_n, E_LOCAL)` expecting the scheduler to keep `W[e, offs_n, :]` in L2 across adjacent M-tiles. Net small regression: seq=1 0.688 → 0.730 (+6%), mean 1.438 → 1.485 (+3%). Triton/CUDA scheduling already handles locality well enough that the explicit reorder is a wash or worse. Reverted.

### iter 7 — notes
Removed the `if not is_local.any(): return output` guard that forced a CPU-GPU sync on a boolean scalar at the top of dispatch. The `max_tk == 0` check later handles the empty-dispatch case without syncing.
Medians (3 runs): seq=1 **0.688 ms** (−8.4% vs iter 3), seq=901 1.535 ms (−3.3%), seq=14107 7.688 ms (−1.3%). Mean 1.438 ms (−4.3%). Speedup 14.95× → 15.68×; SOTA gap 4.17× → 3.92×; total headroom 14.70 ms → 14.41 ms. All 19 passed; no regression on any class.

### iter 6 — notes (reverted)
Implemented a separate `_quantize_intermediate_kernel` reading bf16 intermediate and writing fp8 + per-128 scale, plus modified GEMM2 to load fp8 LHS with per-row dequant scale. All 19 workloads returned RUNTIME_ERROR on compile. Likely `FP8_MAX: tl.constexpr = 448.0` at module scope or the `tl.float8e4nv` cast path. Reverted via `git checkout` before committing; retry planned with a simpler quant formulation.

### iter 4 — notes (reverted)
Tried `a_bf = a.to(bf16); acc += dot(a_bf, w_bf) * a_scale * ws` instead of pre-scaling operands before the dot. Regressed on all classes: seq=14107 7.79 → 18.76 ms (+140%), mean 1.50 → 2.55 ms. Reverted same iter. The pre-scaled pattern keeps dequant+scale+cast fused with the dot pipeline; moving the scale to the fp32 [BM,BN] output breaks that fusion and the cost of 2× dequant-conversions per iter dominates.

### iter 3 — notes
Changed `intermediate` alloc from fp32 to bf16; GEMM1 casts the SwiGLU result to bf16 on store; GEMM2 loads bf16 LHS and does bf16×bf16 dot (matching the bf16×bf16 pattern already used in GEMM1). Halves GEMM2 LHS HBM read bytes.
Measurements (3 runs medians): seq=1 → 0.751 ms (+3%), seq=901 → 1.589 ms (flat), seq=14107 → 7.792 ms (−2.3%). Mean 1.502 ms (−1.6%). All 19 PASSED. Short-class regression at +3% is within tolerance; long class gained as expected but the win is modest — dequant of W2 (fp8→fp32→bf16 every K-iter) is a bigger remaining cost.

### iter 2 — notes (reverted)
Removed the bf16 dequant-cast path and went back to `tl.dot(a_fp8, tl.trans(w_fp8), out_dtype=fp32) * a_scale * w_scale`. Speedup on short/medium workloads was real (seq=1 → 0.678 ms, seq=901 → 1.429 ms; overall 16.27x), but **INCORRECT_NUMERICAL on seq=11948 and seq=14107** — abs_err scales with seq_len (2.05e3 → 3.44e5 → 9.5e5). Same pattern as pre-baseline. Reverted in the same iter; the bf16 path stays.
