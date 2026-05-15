"""Autotune sweep for the DeepSeek-V3 MoE kernel (Phase 2.3 baseline).

Sweeps two kernels exported by `solution/triton/kernel.py`:
  - `_gemm1_swiglu_kernel`  — FP8 dequant → bf16 dot × 2 (up+gate) → SwiGLU, per-expert.
  - `_gemm2_scatter_kernel` — fp32 × FP8 dequant → per-row gamma × atomic scatter.

Tunable axes per kernel:
  BLOCK_M, BLOCK_N, num_stages, num_warps.

Pruning:
  - SMEM estimate vs. B200 `SMEM_LIMIT = 232448`.
  - `num_warps * 32 ≤ BLOCK_M * BLOCK_N`.
  - For GEMM1: `I % BLOCK_N == 0`. For GEMM2: `H % BLOCK_N == 0`.
  - Skip BLOCK_M larger than 4× average per-expert token count for the workload.

Uses uniform-routing synthetic inputs (each expert gets `~T/E_local` tokens)
so the sweep reflects the GEMM cost structure rather than any skew bias.

Usage:
    PYTHONPATH=. modal run -m bench.modal_autotune
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import modal

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

_PROJECT_ROOT = BENCH_DIR.parent

app   = modal.App("flashinfer-autotune-v2")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy")
)

# ---------------------------------------------------------------------------
# DeepSeek-V3 MoE geometry (must match solution/triton/kernel.py)
# ---------------------------------------------------------------------------
H        = 7168
I        = 2048
E_LOCAL  = 32
E_GLOBAL = 256
TOP_K    = 8
BLOCK_K  = 128

SMEM_LIMIT = 232448

# All workload seq_lens (must cover every row in context/moe_workloads.jsonl).
SEQ_LENS = [1, 7, 14, 15, 16, 32, 52, 53, 54, 55, 56, 57, 58, 59, 62,
            80, 901, 11948, 14107]

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
BLOCK_M_OPTIONS   = [16, 32, 64, 128]
BLOCK_N_OPTIONS   = [64, 128, 256]
NUM_STAGES_OPTS   = [2, 3, 4]
NUM_WARPS_OPTS    = [4, 8]


def smem_gemm1(BM, BN, ns):
    """Rough SMEM estimate for `_gemm1_swiglu_kernel` (per-stage A+W_up+W_gate + fp32 accs)."""
    per_stage = BM * BLOCK_K + 2 * BN * BLOCK_K   # fp8, 1B each
    acc       = 2 * BM * BN * 4                   # fp32 acc_up + acc_gate
    return ns * per_stage + acc + 2048


def smem_gemm2(BM, BN, ns):
    """Rough SMEM estimate for `_gemm2_scatter_kernel` (fp32 A + fp8 W per stage + fp32 acc)."""
    per_stage = BM * BLOCK_K * 4 + BN * BLOCK_K   # fp32 A + fp8 W
    acc       = BM * BN * 4
    return ns * per_stage + acc + 2048


def tokens_per_expert(seq_len):
    """Expected tokens per local expert under uniform routing: T * TOP_K * E_LOCAL / E_GLOBAL."""
    return max(1, (seq_len * TOP_K * E_LOCAL) // E_GLOBAL // E_LOCAL or 1)


def generate_configs(kernel_name, seq_len):
    smem_fn = smem_gemm1 if kernel_name == "gemm1" else smem_gemm2
    N_AX    = I if kernel_name == "gemm1" else H
    tpe     = tokens_per_expert(seq_len)

    configs, seen = [], set()
    for BM in BLOCK_M_OPTIONS:
        if tpe > 0 and BM > max(16, tpe * 4):
            continue
        for BN in BLOCK_N_OPTIONS:
            if N_AX % BN != 0:
                continue
            for ns in NUM_STAGES_OPTS:
                if smem_fn(BM, BN, ns) > SMEM_LIMIT:
                    continue
                for nw in NUM_WARPS_OPTS:
                    if nw * 32 > BM * BN:
                        continue
                    key = (BM, BN, ns, nw)
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append({
                        'BLOCK_M': BM, 'BLOCK_N': BN,
                        'num_stages': ns, 'num_warps': nw,
                    })
    return configs


# ---------------------------------------------------------------------------
# Remote benchmark function — one (kernel, seq_len) pair per call
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="B200:1", timeout=1800)
def tune_kernel(kernel_source: str, kernel_name: str, seq_len: int, configs: list):
    import importlib.util, torch
    import triton.testing as tt

    # Load the kernel module from source text
    kpath = "/tmp/kernel.py"
    with open(kpath, "w") as f:
        f.write(kernel_source)
    spec = importlib.util.spec_from_file_location("kernel", kpath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["kernel"] = mod
    spec.loader.exec_module(mod)

    device = "cuda"
    T      = max(seq_len, 1)
    tpe    = tokens_per_expert(seq_len)
    sum_Tk = max(tpe * E_LOCAL, 1)

    hist   = torch.full((E_LOCAL,), tpe, dtype=torch.int32, device=device)
    offs   = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    offs[1:] = hist.cumsum(0).to(torch.int64)
    tokens = (torch.arange(sum_Tk, device=device) % T).to(torch.int32)
    gamma  = torch.ones(sum_Tk, dtype=torch.float32, device=device)

    def cuda_reset():
        try: torch.cuda.synchronize()
        except Exception: pass
        torch.cuda.empty_cache()

    results = []

    if kernel_name == "gemm1":
        A   = torch.randn(T, H, device=device).to(torch.float8_e4m3fn)
        Asc = torch.rand(H // BLOCK_K, T, dtype=torch.float32, device=device) * 0.1
        W   = torch.randn(E_LOCAL, 2 * I, H, device=device).to(torch.float8_e4m3fn)
        Wsc = torch.rand(E_LOCAL, (2 * I) // BLOCK_K, H // BLOCK_K,
                         dtype=torch.float32, device=device) * 0.1
        C   = torch.empty(sum_Tk, I, dtype=torch.float32, device=device)

        for cfg in configs:
            BM, BN = cfg['BLOCK_M'], cfg['BLOCK_N']
            ns, nw = cfg['num_stages'], cfg['num_warps']
            try:
                max_tiles_m = (tpe + BM - 1) // BM
                num_tiles_n = I // BN
                grid = (E_LOCAL, max_tiles_m, num_tiles_n)

                def run(_BM=BM, _BN=BN, _ns=ns, _nw=nw, _grid=grid):
                    mod._gemm1_swiglu_kernel[_grid](
                        A, A.stride(0), A.stride(1),
                        Asc, Asc.stride(0), Asc.stride(1),
                        W, W.stride(0), W.stride(1), W.stride(2),
                        Wsc, Wsc.stride(0), Wsc.stride(1), Wsc.stride(2),
                        C, C.stride(0), C.stride(1),
                        tokens, offs, hist,
                        H_SZ=H, I_SZ=I,
                        BLOCK_M=_BM, BLOCK_N=_BN, BK=BLOCK_K,
                        num_warps=_nw, num_stages=_ns,
                    )

                torch.cuda.synchronize()
                ms = tt.do_bench(run, warmup=5, rep=40, return_mode="median")
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  [g1 seq={seq_len} BM={BM} BN={BN} ns={ns} nw={nw}] FAILED: {e}")
                cuda_reset()
                ms = float("inf")
            results.append((cfg, ms))

    else:  # gemm2
        Cin = torch.randn(sum_Tk, I, dtype=torch.float32, device=device)
        W   = torch.randn(E_LOCAL, H, I, device=device).to(torch.float8_e4m3fn)
        Wsc = torch.rand(E_LOCAL, H // BLOCK_K, I // BLOCK_K,
                         dtype=torch.float32, device=device) * 0.1
        O   = torch.zeros(T, H, dtype=torch.float32, device=device)

        for cfg in configs:
            BM, BN = cfg['BLOCK_M'], cfg['BLOCK_N']
            ns, nw = cfg['num_stages'], cfg['num_warps']
            try:
                max_tiles_m = (tpe + BM - 1) // BM
                num_tiles_n = H // BN
                grid = (E_LOCAL, max_tiles_m, num_tiles_n)

                def run(_BM=BM, _BN=BN, _ns=ns, _nw=nw, _grid=grid, _O=O):
                    _O.zero_()
                    mod._gemm2_scatter_kernel[_grid](
                        Cin, Cin.stride(0), Cin.stride(1),
                        W, W.stride(0), W.stride(1), W.stride(2),
                        Wsc, Wsc.stride(0), Wsc.stride(1), Wsc.stride(2),
                        _O, _O.stride(0), _O.stride(1),
                        gamma, tokens, offs, hist,
                        H_SZ=H, I_SZ=I,
                        BLOCK_M=_BM, BLOCK_N=_BN, BK=BLOCK_K,
                        num_warps=_nw, num_stages=_ns,
                    )

                torch.cuda.synchronize()
                ms = tt.do_bench(run, warmup=5, rep=40, return_mode="median")
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  [g2 seq={seq_len} BM={BM} BN={BN} ns={ns} nw={nw}] FAILED: {e}")
                cuda_reset()
                ms = float("inf")
            results.append((cfg, ms))

    results.sort(key=lambda x: x[1])
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(json_out: str = "", label: str = ""):
    kernel_source = (_PROJECT_ROOT / "solution" / "triton" / "kernel.py").read_text()

    tasks, task_info = [], []

    print(f"{'seq_len':>8}  {'kernel':>6}  {'tpe':>6}  {'configs':>7}")
    print("-" * 40)
    for sl in SEQ_LENS:
        tpe = tokens_per_expert(sl)
        g1  = generate_configs("gemm1", sl)
        g2  = generate_configs("gemm2", sl)
        print(f"  {sl:>6}  {'gemm1':>6}  {tpe:>6}  {len(g1):>7}")
        print(f"  {sl:>6}  {'gemm2':>6}  {tpe:>6}  {len(g2):>7}")
        tasks.append((kernel_source, "gemm1", sl, g1))
        task_info.append(("gemm1", sl))
        tasks.append((kernel_source, "gemm2", sl, g2))
        task_info.append(("gemm2", sl))

    print(f"\nDispatching {len(tasks)} jobs on B200...")

    all_results = list(tune_kernel.starmap(tasks))

    gemm1_best, gemm2_best = {}, {}
    sweep_rows = []

    for (kname, sl), ranked in zip(task_info, all_results):
        top = [r for r in ranked if r[1] < float("inf")]
        if not top:
            print(f"  {kname:>6} seq={sl:>6}: ALL FAILED")
            sweep_rows.append({"kernel": kname, "seq_len": sl, "winner": None, "top_k": []})
            continue
        best_cfg, best_ms = top[0]
        entry = [best_cfg['BLOCK_M'], best_cfg['BLOCK_N'],
                 best_cfg['num_stages'], best_cfg['num_warps']]
        if kname == "gemm1":
            gemm1_best[sl] = entry
        else:
            gemm2_best[sl] = entry

        print(f"  {kname:>6} seq={sl:>6}  "
              f"BM={entry[0]:3d} BN={entry[1]:3d} ns={entry[2]} nw={entry[3]:2d}  "
              f"{best_ms:.4f} ms")

        top_k_rows = []
        for cfg, ms in top[:3]:
            e = [cfg['BLOCK_M'], cfg['BLOCK_N'], cfg['num_stages'], cfg['num_warps']]
            mark = " <-" if e == entry else ""
            print(f"      BM={e[0]:3d} BN={e[1]:3d} ns={e[2]} nw={e[3]:2d}  {ms:.4f} ms{mark}")
            top_k_rows.append({"config": dict(cfg), "ms": ms})

        sweep_rows.append({
            "kernel": kname, "seq_len": sl,
            "winner": {"BLOCK_M": entry[0], "BLOCK_N": entry[1],
                       "num_stages": entry[2], "num_warps": entry[3], "ms": best_ms},
            "top_k": top_k_rows,
        })

    print("\n" + "=" * 62)
    print("Paste into solution/triton/kernel.py:")
    print("=" * 62)
    g1 = json.dumps({str(k): v for k, v in sorted(gemm1_best.items())}, indent=4)
    g2 = json.dumps({str(k): v for k, v in sorted(gemm2_best.items())}, indent=4)
    print(f"_GEMM1_BEST = {g1}\n")
    print(f"_GEMM2_BEST = {g2}\n")

    if json_out:
        import datetime
        document = {
            "schema_version": 1,
            "label": label or "",
            "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "moe_geometry": {
                "H": H, "I": I, "E_LOCAL": E_LOCAL, "E_GLOBAL": E_GLOBAL,
                "TOP_K": TOP_K, "BLOCK_K": BLOCK_K,
            },
            "seq_lens": list(SEQ_LENS),
            "winners": {
                "gemm1": {str(k): v for k, v in sorted(gemm1_best.items())},
                "gemm2": {str(k): v for k, v in sorted(gemm2_best.items())},
            },
            "sweep": sweep_rows,
        }
        with open(json_out, "w") as f:
            json.dump(document, f, indent=2)
        print(f"[autotune] wrote JSON to {json_out}")
