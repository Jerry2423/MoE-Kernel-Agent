"""
End-to-end torch.profiler profiling of the MoE kernel on Modal B200.

Stage 1 of the two-stage profiling workflow:
  Stage 1 (this script) — torch.profiler: identifies overall bottleneck stages
  Stage 2 (modal_ncu.py) — NCU: deep-dives into the bottleneck kernel

Traces are annotated with record_function markers for each stage:
  "Routing"        — routing kernel(s) → sorted tokens / weights
  "GEMM1+SwiGLU"  — FP8×FP8 GEMM + SwiGLU activation
  "GEMM2"         — float32×FP8 GEMM + weighted scatter to output

Output: Chrome JSON traces written to the Modal volume at:
  /data/profiles/profile_kernel_{uuid8}_T{seq_len}.json

Fetch results locally after the run:
  modal volume get flashinfer-trace profiles/ ./profiler_results/
  (then open in https://ui.perfetto.dev or chrome://tracing)

Usage:
    modal run bench/modal_profile.py
    modal run bench/modal_profile.py --workload-index 8
    modal run bench/modal_profile.py --workload-index 1 --warmup 5 --active 3
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import modal
from modal_common import (
    AKO_ROOT,
    FLASHINFER_IMAGE,
    TRACE_SET_PATH,
    pack_solution_from_dir,
    trace_volume,
)

from flashinfer_bench import Solution, TraceSet

app = modal.App("ako-moe-profiler")

PROFILE_DIR = f"{TRACE_SET_PATH}/profiles"


# --------------------------------------------------------------------------- #
#  Remote helpers                                                              #
# --------------------------------------------------------------------------- #

def _extract_callable(solution: Solution):
    """Write solution sources to a tempdir and import the entry-point callable."""
    import importlib, importlib.util, sys, tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.mkdtemp())
    for src in solution.sources:
        dest = tmp_dir / src.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.content)

    sys.path.insert(0, str(tmp_dir))

    entry_path, func_name = solution.spec.entry_point.split("::")
    # e.g. "kernel.py::run"  →  module "kernel"
    module_name = entry_path.replace("/", ".").removesuffix(".py")
    mod = importlib.import_module(module_name)
    return getattr(mod, func_name)


def _load_workload_inputs(workload, dataset_root: Path, device: str = "cuda"):
    """Load / generate all inputs for a single workload (mirrors torch_ref.py logic)."""
    import torch
    from safetensors.torch import load_file

    H, I, E_local, BLOCK = 7168, 2048, 32, 128
    T    = workload.axes["seq_len"]
    specs = workload.inputs

    _sf_cache: dict = {}

    def _st(spec) -> torch.Tensor:
        if spec.type == "safetensors":
            path = str(dataset_root / spec.path)
            if path not in _sf_cache:
                _sf_cache[path] = load_file(path)
            return _sf_cache[path][spec.tensor_key].to(device)
        raise ValueError(f"Unexpected input type: {spec.type}")

    routing_logits       = _st(specs["routing_logits"])
    routing_bias         = _st(specs["routing_bias"]).reshape(-1)
    hidden_states        = (torch.randn(T, H, device=device) * 0.01).to(torch.float8_e4m3fn)
    hidden_states_scale  = torch.ones(H // BLOCK, T, device=device) * 0.01
    gemm1_weights        = (torch.randn(E_local, 2 * I, H, device=device) * 0.01).to(torch.float8_e4m3fn)
    gemm1_weights_scale  = torch.ones(E_local, (2 * I) // BLOCK, H // BLOCK, device=device) * 0.01
    gemm2_weights        = (torch.randn(E_local, H, I, device=device) * 0.01).to(torch.float8_e4m3fn)
    gemm2_weights_scale  = torch.ones(E_local, H // BLOCK, I // BLOCK, device=device) * 0.01
    local_expert_offset  = int(specs["local_expert_offset"].value)
    routed_scaling_factor = float(specs["routed_scaling_factor"].value)

    out = torch.zeros(T, H, dtype=torch.bfloat16, device=device)

    return (
        routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor, out,
    )


def _profile_one(label: str, run_fn, inputs: tuple, output_path: str,
                 warmup: int = 3, active: int = 2) -> None:
    """Warm up run_fn then export a Chrome JSON trace via torch.profiler."""
    import torch

    print(f"    [{label}] warmup ({warmup} iters)…", flush=True)
    for _ in range(warmup):
        run_fn(*inputs)
    torch.cuda.synchronize()

    print(f"    [{label}] profiling ({active} iters)…", flush=True)
    schedule = torch.profiler.schedule(wait=1, warmup=warmup, active=active, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        schedule=schedule,
    ) as prof:
        for _ in range(1 + warmup + active + 1):
            run_fn(*inputs)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(output_path)
    print(f"    [{label}] → {output_path}", flush=True)


# --------------------------------------------------------------------------- #
#  Modal remote function                                                       #
# --------------------------------------------------------------------------- #

@app.function(
    image=FLASHINFER_IMAGE,
    gpu="B200:1",
    timeout=7200,
    volumes={TRACE_SET_PATH: trace_volume},
)
def profile_workload(solution: Solution, workload_index: int = 0,
                     warmup: int = 3, active: int = 2) -> list[str]:
    """Profile MoE kernel with torch.profiler on Modal B200, write Chrome JSON to volume."""
    import os, torch

    dataset_root = Path(TRACE_SET_PATH)
    os.makedirs(PROFILE_DIR, exist_ok=True)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition_name = solution.definition
    if definition_name not in trace_set.definitions:
        raise ValueError(f"Definition '{definition_name}' not found in trace set")

    workloads = trace_set.workloads.get(definition_name, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{definition_name}'")

    workload_index = min(workload_index, len(workloads) - 1)
    trace = workloads[workload_index]
    wl    = trace.workload
    uuid8   = wl.uuid[:8]
    seq_len = wl.axes["seq_len"]
    offset  = wl.inputs["local_expert_offset"].value
    print(f"\nWorkload [{workload_index}]: uuid={uuid8} seq_len={seq_len} offset={offset}")

    # Build callable from packed solution
    run_fn = _extract_callable(solution)

    # Generate inputs
    inputs = _load_workload_inputs(wl, dataset_root)

    # Profile
    output_path = f"{PROFILE_DIR}/profile_kernel_{uuid8}_T{seq_len}.json"
    _profile_one(
        label=f"kernel uuid={uuid8} T={seq_len}",
        run_fn=run_fn,
        inputs=inputs,
        output_path=output_path,
        warmup=warmup,
        active=active,
    )

    del inputs
    torch.cuda.empty_cache()

    trace_volume.commit()
    return [output_path]


# --------------------------------------------------------------------------- #
#  Local entrypoint                                                            #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(workload_index: int = 0, warmup: int = 3, active: int = 2):
    """Pack solution and run torch.profiler on Modal B200."""
    print("Packing solution from source files...")
    solution = pack_solution_from_dir()
    print(f"Loaded: {solution.name} ({solution.definition})")

    print(f"\nRunning torch.profiler (workload_index={workload_index}) on Modal B200...")
    saved_paths = profile_workload.remote(
        solution,
        workload_index=workload_index,
        warmup=warmup,
        active=active,
    )

    print(f"\nTrace(s) written to Modal volume:")
    for p in saved_paths:
        print(f"  {p}")

    print(
        "\n── Fetch results locally ──────────────────────────────────────────\n"
        "  modal volume get flashinfer-trace profiles/ ./profiler_results/\n"
        "\nThen open in https://ui.perfetto.dev or chrome://tracing\n"
        "──────────────────────────────────────────────────────────────────"
    )
