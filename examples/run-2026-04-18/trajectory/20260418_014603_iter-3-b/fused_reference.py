"""Fused PyTorch reference for DeepSeek-V3 MoE — Stage 2.1 artifact.

Rewrites `input/torch_ref.py` so the boundaries of the four fusible subgraphs
are explicit, while preserving the exact numerical semantics of the original
reference. The four subgraphs are:

    1. Routing            — sigmoid + group top-2 sum + top-k groups + global top-k + weight normalization.
    2. Dispatch           — build per-local-expert contiguous token lists + per-row routing weight (gamma).
    3. GEMM1 + SwiGLU     — FP8 block-scaled GEMM → split → silu(gate) * up, one pass per (token, intermediate-column).
    4. GEMM2 + WeightedScatter — FP8 block-scaled GEMM → multiply by gamma → atomic/scatter-add into the token's row.

Each subgraph is a named callable. The outer `run()` composes them in the
same order as `input/torch_ref.py`. `validate()` confirms this fused form
produces outputs that match the unfused reference under the same tolerance
the bench harness uses.
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# DeepSeek-V3 fixed geometry
# ---------------------------------------------------------------------------
H        = 7168
I_DIM    = 2048
E_LOCAL  = 32
E_GLOBAL = 256
TOP_K    = 8
N_GROUP  = 8
GROUP_SZ = E_GLOBAL // N_GROUP    # 32
TOPK_GRP = 4
BLOCK    = 128


# ---------------------------------------------------------------------------
# Subgraph 1: Routing (fused no-aux top-k)
# ---------------------------------------------------------------------------

def routing(
    routing_logits: torch.Tensor,      # [T, E_GLOBAL] float32
    routing_bias:   torch.Tensor,      # [E_GLOBAL]   bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (topk_idx [T, TOP_K] int64, weights [T, E_GLOBAL] float32).

    Fusion rationale: sigmoid → +bias → group-top-2-sum → top-k-groups →
    masked global-top-k → normalize-from-s is a single-token single-pass
    reduction chain; no intermediate escapes the chain.
    """
    T = routing_logits.shape[0]
    logits = routing_logits.to(torch.float32)
    bias   = routing_bias.to(torch.float32).reshape(-1)

    s      = torch.sigmoid(logits)                                        # [T, E_GLOBAL]
    s_bias = s + bias

    s_grouped = s_bias.view(T, N_GROUP, GROUP_SZ)
    top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
    group_score = top2_vals.sum(dim=2)                                    # [T, N_GROUP]

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
    return topk_idx, weights


def normalize_weights(
    weights: torch.Tensor,             # [T, E_GLOBAL]
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Normalize along E_GLOBAL with an epsilon floor; scales by routed_scaling_factor.

    Kept as a distinct function so Phase 2.2 can record it as part of the
    routing chain; in the Triton build this will be the routing kernel's
    epilogue.
    """
    denom = weights.sum(dim=1, keepdim=True) + 1e-20
    return (weights / denom) * routed_scaling_factor


# ---------------------------------------------------------------------------
# Subgraph 2: Dispatch (per-local-expert contiguous indexing)
# ---------------------------------------------------------------------------

def dispatch(
    topk_idx: torch.Tensor,             # [T, TOP_K]        int64
    weights:  torch.Tensor,             # [T, E_GLOBAL]     float32
    local_expert_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce `(tokens_sorted, gamma_sorted, hist, offs)`.

    Fusion rationale: given the routing result, building expert-contiguous
    dispatch lists is a single gather-sort chain with no compute operands —
    all subsequent subgraphs read its output.
    """
    T, device = topk_idx.shape[0], topk_idx.device
    local_experts = topk_idx - int(local_expert_offset)                    # [T, TOP_K]
    is_local = (local_experts >= 0) & (local_experts < E_LOCAL)

    # (token, expert, gamma) flat lists
    t_indices = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K)[is_local]
    e_indices = local_experts[is_local].to(torch.int32)
    gamma_all = torch.gather(weights, 1, topk_idx)[is_local].to(torch.float32)

    if t_indices.numel() == 0:
        empty_i32 = torch.zeros(0, dtype=torch.int32, device=device)
        empty_f32 = torch.zeros(0, dtype=torch.float32, device=device)
        hist      = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
        offs      = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
        return empty_i32, empty_f32, hist, offs

    sort_idx = torch.argsort(e_indices, stable=True)
    t_sorted     = t_indices[sort_idx].to(torch.int32).contiguous()
    e_sorted     = e_indices[sort_idx]
    gamma_sorted = gamma_all[sort_idx].contiguous()

    hist = torch.bincount(e_sorted.to(torch.int64), minlength=E_LOCAL).to(torch.int32)
    offs = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    offs[1:] = hist.cumsum(0).to(torch.int64)
    return t_sorted, gamma_sorted, hist, offs


# ---------------------------------------------------------------------------
# Subgraph 3: GEMM1 + SwiGLU (per-expert FP8 block-scale → fp32 intermediate)
# ---------------------------------------------------------------------------

def gemm1_swiglu_per_expert(
    A_fp32_scaled: torch.Tensor,       # [T, H]                           float32 (already dequantized)
    W13_fp32:      torch.Tensor,       # [E_LOCAL, 2I, H]                 float32 (already dequantized)
    tokens_sorted: torch.Tensor,       # [sum_Tk]                         int32
    offs:          torch.Tensor,       # [E_LOCAL+1]                      int64
) -> torch.Tensor:
    """Return intermediate [sum_Tk, I] float32 — GEMM1 → split → silu(gate) * up.

    Fusion rationale: GEMM1 output [Tk, 2I] is consumed by SwiGLU only; the
    intermediate [Tk, 2I] never escapes the chain, so collapsing to [Tk, I]
    in the same kernel is strictly a memory win. Subgraph boundary holds at
    the fp32-intermediate output because GEMM2 is a separate expert-agnostic
    reduction over I.
    """
    device    = A_fp32_scaled.device
    sum_Tk    = tokens_sorted.shape[0]
    intermediate = torch.empty(sum_Tk, I_DIM, dtype=torch.float32, device=device)

    for e in range(E_LOCAL):
        s, en = int(offs[e].item()), int(offs[e + 1].item())
        if en == s:
            continue
        toks = tokens_sorted[s:en].to(torch.int64)
        A_e  = A_fp32_scaled.index_select(0, toks)                         # [Tk_e, H]
        G1_e = A_e @ W13_fp32[e].t()                                       # [Tk_e, 2I]
        X1   = G1_e[:, :I_DIM]
        X2   = G1_e[:, I_DIM:]
        silu = X2 / (1.0 + torch.exp(-X2))
        intermediate[s:en] = silu * X1
    return intermediate


# ---------------------------------------------------------------------------
# Subgraph 4: GEMM2 + WeightedScatter (per-expert → atomic accumulator)
# ---------------------------------------------------------------------------

def gemm2_weighted_scatter_per_expert(
    intermediate:  torch.Tensor,       # [sum_Tk, I]                      float32
    W2_fp32:       torch.Tensor,       # [E_LOCAL, H, I]                  float32 (already dequantized)
    tokens_sorted: torch.Tensor,       # [sum_Tk]                         int32
    gamma_sorted:  torch.Tensor,       # [sum_Tk]                         float32
    offs:          torch.Tensor,       # [E_LOCAL+1]                      int64
    T: int,
) -> torch.Tensor:
    """Return fp32 [T, H] accumulator — GEMM2 → multiply by gamma → scatter-add.

    Fusion rationale: the gamma scale is per-row and only consumed by the
    scatter epilogue; no op between GEMM2 and the scatter writes the
    unmultiplied intermediate, so the chain is tight. We accumulate in fp32
    to avoid bf16-atomic precision loss; caller casts to bf16 at the end.
    """
    device = intermediate.device
    out_fp32 = torch.zeros(T, H, dtype=torch.float32, device=device)

    for e in range(E_LOCAL):
        s, en = int(offs[e].item()), int(offs[e + 1].item())
        if en == s:
            continue
        toks  = tokens_sorted[s:en].to(torch.int64)
        C_e   = intermediate[s:en]                                         # [Tk_e, I]
        O_e   = C_e @ W2_fp32[e].t()                                       # [Tk_e, H]
        gamma = gamma_sorted[s:en].unsqueeze(1)
        out_fp32.index_add_(0, toks, O_e * gamma)
    return out_fp32


# ---------------------------------------------------------------------------
# Outer composition (mirrors torch_ref.run order)
# ---------------------------------------------------------------------------

def _dequant_A(hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor) -> torch.Tensor:
    """Float hidden_states = fp8 * per-128-row scale (scale layout is [H/128, T])."""
    T = hidden_states.shape[0]
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)                        # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()                        # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    return A_fp32 * A_scale_expanded


def _dequant_W13(W_fp8: torch.Tensor, W_scale: torch.Tensor) -> torch.Tensor:
    W = W_fp8.to(torch.float32)
    S = W_scale.to(torch.float32)
    S = torch.repeat_interleave(S, BLOCK, dim=1)
    S = torch.repeat_interleave(S, BLOCK, dim=2)
    return W * S


def _dequant_W2(W_fp8: torch.Tensor, W_scale: torch.Tensor) -> torch.Tensor:
    W = W_fp8.to(torch.float32)
    S = W_scale.to(torch.float32)
    S = torch.repeat_interleave(S, BLOCK, dim=1)
    S = torch.repeat_interleave(S, BLOCK, dim=2)
    return W * S


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Fused-reference composition. Returns [T, H] bfloat16."""
    T = routing_logits.shape[0]

    # --- Dequantize (same as torch_ref) ---
    A   = _dequant_A(hidden_states, hidden_states_scale)
    W13 = _dequant_W13(gemm1_weights, gemm1_weights_scale)
    W2  = _dequant_W2(gemm2_weights, gemm2_weights_scale)

    # --- Subgraph 1: Routing ---
    topk_idx, weights_raw = routing(routing_logits, routing_bias)
    weights = normalize_weights(weights_raw, routed_scaling_factor)

    # --- Subgraph 2: Dispatch ---
    t_sorted, gamma_sorted, hist, offs = dispatch(topk_idx, weights, local_expert_offset)

    if t_sorted.numel() == 0:
        return torch.zeros(T, H, dtype=torch.bfloat16, device=A.device)

    # --- Subgraph 3: GEMM1 + SwiGLU ---
    intermediate = gemm1_swiglu_per_expert(A, W13, t_sorted, offs)

    # --- Subgraph 4: GEMM2 + WeightedScatter ---
    out_fp32 = gemm2_weighted_scatter_per_expert(
        intermediate, W2, t_sorted, gamma_sorted, offs, T,
    )

    return out_fp32.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Local validation against input/torch_ref.py
# ---------------------------------------------------------------------------

def validate(seed: int = 0, T: int = 64, device: str = "cpu") -> dict:
    """Compare fused output against input/torch_ref on small synthetic inputs.

    Runs on CPU by default to avoid Modal round-trips during Phase 2.1.
    Tolerance matches the bench harness (atol=1, rtol=0.3).
    """
    import sys, pathlib
    # Import the original reference — add project root to path.
    project_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from input.torch_ref import run as ref_run           # type: ignore

    torch.manual_seed(seed)
    device = torch.device(device)

    routing_logits      = torch.randn(T, E_GLOBAL, dtype=torch.float32, device=device)
    routing_bias        = torch.randn(E_GLOBAL,    dtype=torch.bfloat16, device=device) * 0.01
    hidden_states       = torch.randn(T, H, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand(H // BLOCK, T, dtype=torch.float32, device=device) * 0.1
    gemm1_weights       = torch.randn(E_LOCAL, 2 * I_DIM, H, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand(E_LOCAL, (2 * I_DIM) // BLOCK, H // BLOCK, dtype=torch.float32, device=device) * 0.1
    gemm2_weights       = torch.randn(E_LOCAL, H, I_DIM, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.rand(E_LOCAL, H // BLOCK, I_DIM // BLOCK, dtype=torch.float32, device=device) * 0.1

    kwargs = dict(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        local_expert_offset=32,
        routed_scaling_factor=2.5,
    )

    expected = ref_run(**kwargs)                         # bf16
    actual   = run(**kwargs)                             # bf16

    diff = (expected.to(torch.float32) - actual.to(torch.float32)).abs()
    max_abs = float(diff.max().item())
    # Per-element match (atol=1, rtol=0.3 per bench/modal_bench.py)
    atol, rtol = 1.0, 0.3
    match = (diff <= (atol + rtol * expected.to(torch.float32).abs())).float().mean().item()

    return {
        "shape": tuple(actual.shape),
        "max_abs_err": max_abs,
        "match_ratio": float(match),
        "passes": max_abs < 1.0 or match >= 0.9,
    }


if __name__ == "__main__":
    print(validate())
