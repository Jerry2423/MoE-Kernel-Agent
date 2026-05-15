"""DeepSeek-V3 MoE Triton kernel.

Phase 2.3 baseline (naive correct). Two Triton kernels carry the compute:
  * ``_gemm1_swiglu_kernel`` — FP8×FP8 block-scaled GEMM1 fused with SwiGLU,
    produces the ``[sum_Tk, I=2048]`` fp32 intermediate.
  * ``_gemm2_scatter_kernel`` — fp32 × FP8 block-scaled GEMM2 with gamma-
    weighted atomic scatter-add into an fp32 ``[T, H]`` accumulator.

Routing and dispatch run in PyTorch primitives (sigmoid / topk / bincount
/ argsort) — none of the forbidden high-level APIs (matmul, layer_norm,
softmax, conv, linear, transformer). Every mathematically-nontrivial op
(both GEMMs, both block-scaled dequants, SwiGLU, weighted scatter) is
expressed in Triton.

Config is hand-picked for Stage 2.3; Stage 2.5 replaces these numbers with
an autotuned freeze.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# DeepSeek-V3 fixed geometry (must match bench/modal_autotune.py)
# ---------------------------------------------------------------------------
H        = 7168
I_DIM    = 2048
E_LOCAL  = 32
E_GLOBAL = 256
TOP_K    = 8
N_GROUP  = 8
GROUP_SZ = E_GLOBAL // N_GROUP    # 32
TOPK_GRP = 4
BLOCK_K  = 128


# ---------------------------------------------------------------------------
# Frozen autotune winners — Phase 2.5 (solution/autotune_notes.md).
# Format: {seq_len_str: [BLOCK_M, BLOCK_N, num_stages, num_warps]}.
# Phase 3 must not re-sweep without an algorithmic change.
# ---------------------------------------------------------------------------

_GEMM1_BEST = {
    "1": [16, 64, 3, 4], "7": [16, 64, 3, 4], "14": [16, 64, 3, 4],
    "15": [16, 64, 3, 4], "16": [16, 64, 3, 4], "32": [16, 64, 3, 4],
    "52": [16, 64, 3, 4], "53": [16, 64, 3, 4], "54": [16, 64, 3, 4],
    "55": [16, 64, 3, 4], "56": [16, 64, 3, 4], "57": [16, 64, 3, 4],
    "58": [16, 64, 3, 4], "59": [16, 64, 3, 4], "62": [16, 64, 3, 4],
    "80": [16, 64, 3, 4],
    "901": [32, 128, 3, 4],
    "11948": [128, 64, 3, 4], "14107": [128, 64, 3, 4],
}
# Default fallback for unseen seq_lens — picks the "middle" config
_GEMM1_DEFAULT = [32, 128, 3, 4]

_GEMM2_BEST = {
    "1": [16, 256, 3, 8], "7": [16, 256, 3, 8], "14": [16, 256, 3, 8],
    "15": [16, 256, 3, 8], "16": [16, 256, 3, 8], "32": [16, 256, 3, 8],
    "52": [16, 256, 3, 8], "53": [16, 256, 3, 8], "54": [16, 256, 3, 8],
    "55": [16, 256, 3, 8], "56": [16, 256, 3, 8], "57": [16, 256, 3, 8],
    "58": [16, 256, 3, 8], "59": [16, 256, 3, 8], "62": [16, 256, 3, 8],
    "80": [16, 256, 3, 8], "901": [16, 256, 3, 8],
    "11948": [16, 256, 3, 8], "14107": [16, 256, 3, 8],
}
_GEMM2_DEFAULT = [16, 256, 3, 8]


def _pick_cfg(best_dict, default_cfg, T):
    return best_dict.get(str(T), default_cfg)


# ---------------------------------------------------------------------------
# GEMM1 + SwiGLU — block-scaled FP8 matmul fused with silu(gate) * up
#
# Each program handles one (expert, m_tile, n_tile) triple. The n_tile axis
# is over the *intermediate* columns (`[0, I)`), not the 2I GEMM1 output —
# each program loads both the "up" (`W[:, 0..I)`) and "gate" (`W[:, I..2I)`)
# halves at the same intermediate column offset, accumulates them as two
# independent fp32 accumulators, and writes `silu(gate) * up` into the
# `[sum_Tk, I]` fp32 intermediate.
# ---------------------------------------------------------------------------

@triton.jit
def _gemm1_swiglu_kernel(
    A_ptr,   stride_a_t,  stride_a_h,
    As_ptr,  stride_as_kb, stride_as_t,
    W_ptr,   stride_w_e,  stride_w_n,  stride_w_h,
    Ws_ptr,  stride_ws_e, stride_ws_n, stride_ws_h,
    C_ptr,   stride_c_m,  stride_c_n,
    tokens_ptr, offs_ptr, hist_ptr,
    H_SZ:   tl.constexpr,
    I_SZ:   tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BK:     tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    Tk_e = tl.load(hist_ptr + pid_e)
    m_start = pid_m * BLOCK_M
    if m_start >= Tk_e:
        return

    expert_start = tl.load(offs_ptr + pid_e)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < Tk_e
    offs_m_safe = tl.where(m_mask, offs_m, 0)

    toks = tl.load(tokens_ptr + expert_start + offs_m_safe, mask=m_mask, other=0).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < I_SZ
    offs_n_safe = tl.where(n_mask, offs_n, 0)

    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BK)
    num_kb = H_SZ // BK    # = 56

    for kb in range(0, num_kb):
        k_off = kb * BK

        a_ptrs = A_ptr + toks[:, None] * stride_a_t + (k_off + offs_k)[None, :] * stride_a_h
        a_tile = tl.load(a_ptrs, mask=m_mask[:, None])

        a_scale = tl.load(As_ptr + kb * stride_as_kb + toks * stride_as_t,
                          mask=m_mask, other=0.0)

        w_up_ptrs = (W_ptr
                     + pid_e * stride_w_e
                     + offs_n_safe[:, None] * stride_w_n
                     + (k_off + offs_k)[None, :] * stride_w_h)
        w_up = tl.load(w_up_ptrs, mask=n_mask[:, None])

        w_gate_ptrs = (W_ptr
                       + pid_e * stride_w_e
                       + (offs_n_safe + I_SZ)[:, None] * stride_w_n
                       + (k_off + offs_k)[None, :] * stride_w_h)
        w_gate = tl.load(w_gate_ptrs, mask=n_mask[:, None])

        ws_up_ptrs = (Ws_ptr
                      + pid_e * stride_ws_e
                      + (offs_n_safe // 128) * stride_ws_n
                      + kb * stride_ws_h)
        ws_up = tl.load(ws_up_ptrs, mask=n_mask, other=0.0)

        ws_gate_ptrs = (Ws_ptr
                        + pid_e * stride_ws_e
                        + ((offs_n_safe + I_SZ) // 128) * stride_ws_n
                        + kb * stride_ws_h)
        ws_gate = tl.load(ws_gate_ptrs, mask=n_mask, other=0.0)

        a_bf   = (a_tile.to(tl.float32) * a_scale[:, None]).to(tl.bfloat16)
        w_up_bf   = (w_up.to(tl.float32)   * ws_up[:, None]).to(tl.bfloat16)
        w_gate_bf = (w_gate.to(tl.float32) * ws_gate[:, None]).to(tl.bfloat16)

        acc_up   += tl.dot(a_bf, tl.trans(w_up_bf),   out_dtype=tl.float32)
        acc_gate += tl.dot(a_bf, tl.trans(w_gate_bf), out_dtype=tl.float32)

    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
    result    = silu_gate * acc_up

    c_ptrs = (C_ptr
              + (expert_start + offs_m_safe)[:, None] * stride_c_m
              + offs_n_safe[None, :] * stride_c_n)
    tl.store(c_ptrs, result.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Fused DeepSeek-V3 no-aux routing — one program per token, emits
# topk_idx [T, TOP_K] int32 and topk_weights [T, TOP_K] fp32 in a single
# launch. Replaces ~15 small torch CUDA kernels (sigmoid, topk×3, scatter,
# masked_fill, sum, ...) that paid ~10 us launch overhead each on short
# workloads.
# ---------------------------------------------------------------------------

@triton.jit
def _routing_kernel(
    logits_ptr,        # [T, E_GLOBAL] float32
    bias_ptr,          # [E_GLOBAL]    float32 (pre-converted host-side)
    topk_idx_ptr,      # [T, TOP_K]    int32 (output) — local expert id, or -1 if non-local
    topk_weights_ptr,  # [T, TOP_K]    float32 (output)
    T,
    routed_scaling_factor,
    local_expert_offset,
    E_GLOBAL:  tl.constexpr,
    N_GROUP:   tl.constexpr,
    GROUP_SZ:  tl.constexpr,
    TOPK_GRP:  tl.constexpr,
    TOP_K_C:   tl.constexpr,
    E_LOCAL_C: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return

    offs_e  = tl.arange(0, E_GLOBAL)
    offs_g  = tl.arange(0, N_GROUP)
    offs_tk = tl.arange(0, TOP_K_C)

    logits = tl.load(logits_ptr + t * E_GLOBAL + offs_e).to(tl.float32)
    bias   = tl.load(bias_ptr + offs_e).to(tl.float32)
    s      = 1.0 / (1.0 + tl.exp(-logits))
    s_bias = s + bias

    # Group top-2 sum
    sb_grouped = tl.reshape(s_bias, (N_GROUP, GROUP_SZ))
    max1       = tl.max(sb_grouped, axis=1)                           # [N_GROUP]
    sb_masked  = tl.where(sb_grouped == max1[:, None], -float("inf"), sb_grouped)
    max2       = tl.max(sb_masked, axis=1)                            # [N_GROUP]
    group_score = max1 + max2

    # Top-TOPK_GRP groups via iterative argmax
    gs = group_score
    grp_kept = tl.zeros((N_GROUP,), dtype=tl.int32)
    for _kg in tl.static_range(TOPK_GRP):
        gm = tl.argmax(gs, axis=0)
        sel = (offs_g == gm).to(tl.int32)
        grp_kept = grp_kept + sel
        gs = tl.where(offs_g == gm, -float("inf"), gs)

    # Broadcast group keep mask to [E_GLOBAL]
    grp_kept_2d = tl.broadcast_to(grp_kept[:, None], (N_GROUP, GROUP_SZ))
    e_kept      = tl.reshape(grp_kept_2d, (E_GLOBAL,))

    # Mask scores
    sp = tl.where(e_kept > 0, s_bias, -float("inf"))

    # Top-TOP_K experts via iterative argmax; accumulate s-at-topk for normalization
    topk_idxs = tl.zeros((TOP_K_C,), dtype=tl.int32)
    topk_s    = tl.zeros((TOP_K_C,), dtype=tl.float32)
    for k in tl.static_range(TOP_K_C):
        idx = tl.argmax(sp, axis=0)
        is_idx = offs_e == idx
        s_val = tl.sum(tl.where(is_idx, s, 0.0), axis=0)
        topk_idxs = tl.where(offs_tk == k, idx.to(tl.int32), topk_idxs)
        topk_s    = tl.where(offs_tk == k, s_val,             topk_s)
        sp = tl.where(is_idx, -float("inf"), sp)

    denom  = tl.sum(topk_s, axis=0) + 1e-20
    topk_w = (topk_s / denom) * routed_scaling_factor

    # iter 9: emit local-expert id with -1 sentinel for non-local in one pass,
    # saves the subtract + compound-compare dispatch ops on the short path.
    local_e  = topk_idxs - local_expert_offset
    in_local = (local_e >= 0) & (local_e < E_LOCAL_C)
    out_idx  = tl.where(in_local, local_e, -1).to(tl.int32)

    tl.store(topk_idx_ptr     + t * TOP_K_C + offs_tk, out_idx)
    tl.store(topk_weights_ptr + t * TOP_K_C + offs_tk, topk_w)


# ---------------------------------------------------------------------------
# Elementwise fp32 -> bf16 cast-and-copy into the caller output tensor.
# Replaces `output.copy_(out_fp32.to(torch.bfloat16))` which allocates a full
# temp bf16 tensor + runs two kernels; this is one kernel, no alloc.
# ---------------------------------------------------------------------------

@triton.jit
def _cast_fp32_to_bf16_kernel(
    src_ptr,    # fp32 [total]
    dst_ptr,    # bf16 [total]
    total,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v    = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, v.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Dispatch — two-pass scatter of (token, expert, gamma) into expert-contiguous
# arrays without torch.argsort / bincount / index_select.
#
#   Pass 1 (_dispatch_count_kernel):  per-token counts via atomic_add(hist[e])
#   Host-side cumsum:                 hist → offs[E_LOCAL+1]
#   Pass 2 (_dispatch_scatter_kernel): per-token slot via atomic_add(counter[e])
#                                       writes t_sorted[slot], gamma_sorted[slot].
# ---------------------------------------------------------------------------

@triton.jit
def _dispatch_count_kernel(
    topk_local_e_ptr,   # [T, TOP_K] int32 (local-expert id, or -1)
    hist_ptr,           # [E_LOCAL]  int32 (atomic accum, pre-zeroed)
    T,
    TOP_K_C: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    for k in tl.static_range(TOP_K_C):
        e = tl.load(topk_local_e_ptr + t * TOP_K_C + k)
        if e >= 0:
            tl.atomic_add(hist_ptr + e, 1)


@triton.jit
def _dispatch_scatter_kernel(
    topk_local_e_ptr,   # [T, TOP_K] int32
    topk_weights_ptr,   # [T, TOP_K] fp32
    offs_ptr,           # [E_LOCAL+1] int64  (input)
    counter_ptr,        # [E_LOCAL]   int32  (atomic slot, pre-zeroed)
    t_sorted_ptr,       # [sum_Tk]    int32  (output)
    gamma_sorted_ptr,   # [sum_Tk]    fp32   (output)
    T,
    TOP_K_C: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return
    for k in tl.static_range(TOP_K_C):
        e = tl.load(topk_local_e_ptr + t * TOP_K_C + k)
        g = tl.load(topk_weights_ptr + t * TOP_K_C + k)
        if e >= 0:
            local_slot = tl.atomic_add(counter_ptr + e, 1)
            base       = tl.load(offs_ptr + e)
            slot       = base + local_slot.to(tl.int64)
            tl.store(t_sorted_ptr     + slot, t)
            tl.store(gamma_sorted_ptr + slot, g)


# ---------------------------------------------------------------------------
# GEMM2 + weighted scatter-add
# ---------------------------------------------------------------------------

@triton.jit
def _gemm2_scatter_kernel(
    C_ptr,   stride_c_m,  stride_c_k,
    W_ptr,   stride_w_e,  stride_w_n,  stride_w_k,
    Ws_ptr,  stride_ws_e, stride_ws_n, stride_ws_k,
    Out_ptr, stride_o_t,  stride_o_h,
    gamma_ptr, tokens_ptr, offs_ptr, hist_ptr,
    H_SZ:   tl.constexpr,
    I_SZ:   tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BK:     tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    Tk_e = tl.load(hist_ptr + pid_e)
    m_start = pid_m * BLOCK_M
    if m_start >= Tk_e:
        return

    expert_start = tl.load(offs_ptr + pid_e)

    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < Tk_e
    offs_m_safe = tl.where(m_mask, offs_m, 0)

    toks  = tl.load(tokens_ptr + expert_start + offs_m_safe, mask=m_mask, other=0).to(tl.int64)
    gamma = tl.load(gamma_ptr  + expert_start + offs_m_safe, mask=m_mask, other=0.0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < H_SZ
    offs_n_safe = tl.where(n_mask, offs_n, 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BK)
    num_kb = I_SZ // BK    # = 16

    for kb in range(0, num_kb):
        k_off = kb * BK

        c_ptrs = (C_ptr
                  + (expert_start + offs_m_safe)[:, None] * stride_c_m
                  + (k_off + offs_k)[None, :] * stride_c_k)
        c_tile = tl.load(c_ptrs, mask=m_mask[:, None])               # [BM, BK] bf16

        w_ptrs = (W_ptr
                  + pid_e * stride_w_e
                  + offs_n_safe[:, None] * stride_w_n
                  + (k_off + offs_k)[None, :] * stride_w_k)
        w_tile = tl.load(w_ptrs, mask=n_mask[:, None])

        ws_ptrs = (Ws_ptr
                   + pid_e * stride_ws_e
                   + (offs_n_safe // 128) * stride_ws_n
                   + kb * stride_ws_k)
        ws = tl.load(ws_ptrs, mask=n_mask, other=0.0)

        w_bf = (w_tile.to(tl.float32) * ws[:, None]).to(tl.bfloat16)
        acc += tl.dot(c_tile, tl.trans(w_bf), out_dtype=tl.float32)

    acc = acc * gamma[:, None]

    out_ptrs = (Out_ptr
                + toks[:, None] * stride_o_t
                + offs_n_safe[None, :] * stride_o_h)
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Host-side entry point: run()  (destination-passing style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    routing_logits:      torch.Tensor,   # [T, E_GLOBAL]              float32
    routing_bias:        torch.Tensor,   # [E_GLOBAL]                 bfloat16
    hidden_states:       torch.Tensor,   # [T, H]                     float8_e4m3fn
    hidden_states_scale: torch.Tensor,   # [H/128, T]                 float32
    gemm1_weights:       torch.Tensor,   # [E_LOCAL, 2I, H]           float8_e4m3fn
    gemm1_weights_scale: torch.Tensor,   # [E_LOCAL, 2I/128, H/128]   float32
    gemm2_weights:       torch.Tensor,   # [E_LOCAL, H, I]            float8_e4m3fn
    gemm2_weights_scale: torch.Tensor,   # [E_LOCAL, H/128, I/128]    float32
    local_expert_offset: int,
    routed_scaling_factor: float,
    output:              torch.Tensor,   # [T, H]                     bfloat16 (DPS out)
) -> torch.Tensor:
    device = output.device
    T = int(routing_logits.shape[0])
    output.zero_()

    # --- Routing (fused Triton kernel — iter 1) ----------------------------
    logits_f32 = routing_logits.to(torch.float32).contiguous()
    bias_f32   = routing_bias.to(torch.float32).reshape(-1).contiguous()
    topk_idx_i32     = torch.empty(T, TOP_K, dtype=torch.int32,   device=device)
    topk_weights_f32 = torch.empty(T, TOP_K, dtype=torch.float32, device=device)
    _routing_kernel[(T,)](
        logits_f32, bias_f32, topk_idx_i32, topk_weights_f32,
        T, float(routed_scaling_factor), int(local_expert_offset),
        E_GLOBAL=E_GLOBAL, N_GROUP=N_GROUP, GROUP_SZ=GROUP_SZ,
        TOPK_GRP=TOPK_GRP, TOP_K_C=TOP_K, E_LOCAL_C=E_LOCAL,
        num_warps=4,
    )

    # --- Dispatch (two Triton kernels; no argsort / bincount / index_select) ---
    # iter 10: Pass 1 counts hist via atomic_add; host does cumsum (32 elements);
    # Pass 2 scatters each (t, e, gamma) to offs[e] + slot via atomic counter.
    hist    = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    _dispatch_count_kernel[(T,)](
        topk_idx_i32, hist,
        T, TOP_K_C=TOP_K, num_warps=1,
    )
    offs       = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    offs[1:]   = hist.cumsum(0).to(torch.int64)

    # Upper-bound buffer size is T * TOP_K (we only touch offs[E_LOCAL] slots).
    buf_sz       = T * TOP_K
    t_sorted     = torch.empty(buf_sz, dtype=torch.int32, device=device)
    gamma_sorted = torch.empty(buf_sz, dtype=torch.float32, device=device)
    counter      = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    _dispatch_scatter_kernel[(T,)](
        topk_idx_i32, topk_weights_f32,
        offs, counter, t_sorted, gamma_sorted,
        T, TOP_K_C=TOP_K, num_warps=1,
    )
    sum_Tk = int(offs[E_LOCAL].item())

    # --- Allocate intermediates --------------------------------------------
    # bf16 intermediate (halves GEMM2 LHS HBM reads; bf16 exp range matches fp32).
    intermediate = torch.empty(sum_Tk, I_DIM, dtype=torch.bfloat16, device=device)
    out_fp32     = torch.zeros(T, H, dtype=torch.float32, device=device)

    # --- Block sizes (Phase 2.5 frozen autotune winners per seq_len T) -----
    BM1, BN1, NS1, NW1 = _pick_cfg(_GEMM1_BEST, _GEMM1_DEFAULT, T)
    BM2, BN2, NS2, NW2 = _pick_cfg(_GEMM2_BEST, _GEMM2_DEFAULT, T)

    max_tk = int(hist.max().item())
    if max_tk == 0:
        return output

    max_tiles_m_g1 = (max_tk + BM1 - 1) // BM1
    num_tiles_n_g1 = I_DIM // BN1
    _gemm1_swiglu_kernel[(E_LOCAL, max_tiles_m_g1, num_tiles_n_g1)](
        hidden_states,       hidden_states.stride(0),       hidden_states.stride(1),
        hidden_states_scale, hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights,       gemm1_weights.stride(0),       gemm1_weights.stride(1),       gemm1_weights.stride(2),
        gemm1_weights_scale, gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        intermediate,        intermediate.stride(0),        intermediate.stride(1),
        t_sorted, offs, hist,
        H_SZ=H, I_SZ=I_DIM,
        BLOCK_M=BM1, BLOCK_N=BN1, BK=BLOCK_K,
        num_warps=NW1, num_stages=NS1,
    )

    max_tiles_m_g2 = (max_tk + BM2 - 1) // BM2
    num_tiles_n_g2 = H // BN2
    _gemm2_scatter_kernel[(E_LOCAL, max_tiles_m_g2, num_tiles_n_g2)](
        intermediate,        intermediate.stride(0),        intermediate.stride(1),
        gemm2_weights,       gemm2_weights.stride(0),       gemm2_weights.stride(1),       gemm2_weights.stride(2),
        gemm2_weights_scale, gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        out_fp32,            out_fp32.stride(0),            out_fp32.stride(1),
        gamma_sorted, t_sorted, offs, hist,
        H_SZ=H, I_SZ=I_DIM,
        BLOCK_M=BM2, BLOCK_N=BN2, BK=BLOCK_K,
        num_warps=NW2, num_stages=NS2,
    )
    # iter 12: single-pass fp32->bf16 cast into the caller's output tensor.
    total = T * H
    CAST_BLOCK = 4096
    _cast_fp32_to_bf16_kernel[((total + CAST_BLOCK - 1) // CAST_BLOCK,)](
        out_fp32, output, total,
        BLOCK=CAST_BLOCK, num_warps=4,
    )
    return output
