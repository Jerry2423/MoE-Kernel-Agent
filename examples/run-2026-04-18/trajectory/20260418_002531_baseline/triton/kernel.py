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

        dot_up   = tl.dot(a_tile, tl.trans(w_up),   out_dtype=tl.float32)
        dot_gate = tl.dot(a_tile, tl.trans(w_gate), out_dtype=tl.float32)

        acc_up   += dot_up   * a_scale[:, None] * ws_up[None, :]
        acc_gate += dot_gate * a_scale[:, None] * ws_gate[None, :]

    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
    result    = silu_gate * acc_up

    c_ptrs = (C_ptr
              + (expert_start + offs_m)[:, None] * stride_c_m
              + offs_n[None, :] * stride_c_n)
    tl.store(c_ptrs, result, mask=m_mask[:, None] & n_mask[None, :])


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
        c_tile = tl.load(c_ptrs, mask=m_mask[:, None], other=0.0)

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

        w_f32 = w_tile.to(tl.float32) * ws[:, None]
        dot   = tl.dot(c_tile, tl.trans(w_f32), out_dtype=tl.float32)
        acc  += dot

    acc = acc * gamma[:, None]

    out_ptrs = (Out_ptr
                + toks[:, None] * stride_o_t
                + offs_n[None, :] * stride_o_h)
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :], sem="relaxed")


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

    # --- Routing (primitive torch ops only) ---------------------------------
    logits = routing_logits.to(torch.float32)
    bias   = routing_bias.to(torch.float32).reshape(-1)
    s      = torch.sigmoid(logits)
    s_bias = s + bias

    s_grouped = s_bias.view(T, N_GROUP, GROUP_SZ)
    top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
    group_score = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_score, k=TOPK_GRP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_score)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, GROUP_SZ).reshape(T, E_GLOBAL)

    neg_inf        = torch.finfo(torch.float32).min
    scores_pruned  = s_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx    = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor

    # --- Dispatch (expert-contiguous) ---------------------------------------
    local_experts = topk_idx - int(local_expert_offset)
    is_local      = (local_experts >= 0) & (local_experts < E_LOCAL)
    if not is_local.any():
        return output

    t_indices = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K)[is_local]
    e_indices = local_experts[is_local].to(torch.int32)
    gamma_all = torch.gather(weights, 1, topk_idx)[is_local].to(torch.float32)

    sort_idx     = torch.argsort(e_indices, stable=True)
    t_sorted     = t_indices[sort_idx].to(torch.int32).contiguous()
    e_sorted     = e_indices[sort_idx]
    gamma_sorted = gamma_all[sort_idx].contiguous()

    sum_Tk = int(t_sorted.shape[0])

    hist = torch.bincount(e_sorted.to(torch.int64), minlength=E_LOCAL).to(torch.int32)
    offs = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    offs[1:] = hist.cumsum(0).to(torch.int64)

    # --- Allocate intermediates --------------------------------------------
    intermediate = torch.empty(sum_Tk, I_DIM, dtype=torch.float32, device=device)
    out_fp32     = torch.zeros(T, H, dtype=torch.float32, device=device)

    # --- Block sizes (Stage 2.3 hand-picked; Stage 2.5 autotune freeze) ----
    BM1, BN1 = 64, 128
    BM2, BN2 = 64, 128

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
        num_warps=4, num_stages=2,
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
        num_warps=4, num_stages=2,
    )

    output.copy_(out_fp32.to(torch.bfloat16))
    return output
