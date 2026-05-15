"""
NCU profiling of MoE kernel on Modal B200.

Uses flashinfer_bench's NCU wrapper which:
  - Builds the solution via the standard build pipeline
  - Allocates inputs with correct dtypes/shapes from the workload definition
  - Handles DPS output allocation automatically
  - Wraps the profiled call in an NVTX range so NCU skips warmup/JIT runs

Usage:
    modal run -m bench.modal_ncu
    modal run -m bench.modal_ncu --workload-index 1
    modal run -m bench.modal_ncu --workload-index 8 --set basic
    modal run -m bench.modal_ncu --workload-index 10 --kernel-name "gemm1"
    modal run -m bench.modal_ncu --workload-index 1 --sections "MemoryWorkloadAnalysis,Occupancy"
    modal run -m bench.modal_ncu --workload-index 1 --preset ako   # canonical 25-metric snapshot

The ``--preset ako`` flag activates the metric list in ``bench/ncu_metrics.py``
(lifted from KernelAgent ``metric_schema.py``). The extracted values are
written to ``ncu_results/<name>_w<N>_metrics.json`` alongside the full text
report.
"""

import sys
from pathlib import Path

# Ensure bench/ is importable
BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import modal
from modal_common import (
    AKO_ROOT,
    NCU_IMAGE,
    TRACE_SET_PATH,
    pack_solution_from_dir,
    trace_volume,
)

from flashinfer_bench import Solution, TraceSet

app = modal.App("ako-flashinfer-ncu")


@app.function(
    image=NCU_IMAGE,
    gpu="B200:1",
    timeout=600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_ncu(
    solution: Solution,
    workload_index: int = 0,
    ncu_set: str = "detailed",
    kernel_name: str = "",
    sections: str = "",
    metrics: str = "",
) -> dict:
    """Run NCU profiling on Modal B200 and return the report."""
    import glob

    from flashinfer_bench.agents.ncu import flashinfer_bench_run_ncu

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    definition_name = solution.definition
    if definition_name not in trace_set.definitions:
        return {"error": f"Definition '{definition_name}' not found in trace set."}

    workloads = trace_set.workloads.get(definition_name, [])
    if not workloads:
        return {"error": f"No workloads found for definition '{definition_name}'."}

    workload_index = min(workload_index, len(workloads) - 1)
    trace = workloads[workload_index]

    # Find the NCU binary installed by nsight-compute
    ncu_candidates = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    ncu_path = ncu_candidates[-1] if ncu_candidates else "ncu"

    seq_len = trace.workload.axes.get("seq_len", "?")
    print(f"Profiling workload [{workload_index}]: {trace.workload.uuid} (T={seq_len})")
    print(f"NCU binary: {ncu_path}, set: {ncu_set}")
    if kernel_name:
        print(f"Kernel filter: {kernel_name}")
    if sections:
        print(f"Extra sections: {sections}")
    if metrics:
        metric_count = len([m for m in metrics.split(",") if m.strip()])
        print(f"Metrics preset: {metric_count} keys (truncated set — overrides sections)")

    # Build kwargs for flashinfer_bench_run_ncu
    ncu_kwargs = {}
    if kernel_name:
        ncu_kwargs["kernel_name"] = kernel_name
    if sections:
        ncu_kwargs["sections"] = [s.strip() for s in sections.split(",")]
    if metrics:
        ncu_kwargs["metrics"] = [m.strip() for m in metrics.split(",") if m.strip()]

    output = flashinfer_bench_run_ncu(
        solution=solution,
        workload=trace.workload,
        set=ncu_set,
        page="details",
        ncu_path=ncu_path,
        trace_set_path=TRACE_SET_PATH,
        timeout=480,
        max_lines=None,
        **ncu_kwargs,
    )

    return {
        "solution_name": solution.name,
        "definition": solution.definition,
        "workload_uuid": trace.workload.uuid,
        "workload_index": workload_index,
        "seq_len": seq_len,
        "ncu_set": ncu_set,
        "kernel_name": kernel_name or "(all)",
        "report": output,
    }


@app.local_entrypoint()
def main(
    workload_index: int = 0,
    set: str = "detailed",
    kernel_name: str = "",
    sections: str = "",
    preset: str = "",
):
    """Pack solution and run NCU profiling on Modal B200.

    Args:
        workload_index: Which workload to profile (0-based).
        set: NCU section set (basic, detailed, full, source).
        kernel_name: Regex filter for kernel name (e.g. "gemm1", "gemm2", "routing").
        sections: Comma-separated extra NCU sections (e.g. "MemoryWorkloadAnalysis,Occupancy").
        preset: Named metric preset. Supported: "ako" — the 25-metric snapshot
                defined in bench/ncu_metrics.py (5 sections × ~25 metrics).
                When set, writes a companion *_metrics.json next to the text
                report with parsed numeric values.
    """
    import json

    from bench.ncu_metrics import ALL_METRIC_KEYS, parse_report, summarize

    metrics_csv = ""
    if preset:
        if preset == "ako":
            metrics_csv = ",".join(ALL_METRIC_KEYS)
            print(f"[preset=ako] requesting {len(ALL_METRIC_KEYS)} NCU metrics")
        else:
            raise SystemExit(f"unknown --preset '{preset}'. Supported: ako")

    print("Packing solution from source files...")
    solution = pack_solution_from_dir()
    print(f"Loaded: {solution.name} ({solution.definition})")

    print(f"\nRunning NCU (set={set}, workload_index={workload_index}) on Modal B200...")
    if kernel_name:
        print(f"Kernel filter: {kernel_name}")
    result = run_ncu.remote(
        solution,
        workload_index=workload_index,
        ncu_set=set,
        kernel_name=kernel_name,
        sections=sections,
        metrics=metrics_csv,
    )

    # Save result to local JSON file
    out_dir = AKO_ROOT / "ncu_results"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_k-{kernel_name}" if kernel_name else ""
    out_path = out_dir / f"{solution.name}_w{workload_index}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {out_path}")

    # Preset post-processing: extract canonical metrics and summarize.
    if preset == "ako" and "report" in result:
        metrics_values = parse_report(result["report"])
        metrics_path = out_dir / f"{solution.name}_w{workload_index}{suffix}_metrics.json"
        metrics_payload = {
            "schema_version": 1,
            "preset": "ako",
            "solution": result.get("solution_name"),
            "workload_index": result.get("workload_index"),
            "workload_uuid": result.get("workload_uuid"),
            "seq_len": result.get("seq_len"),
            "kernel_name": result.get("kernel_name"),
            "values": metrics_values,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        print(f"Parsed metrics saved to: {metrics_path}")
        print("\n" + summarize(metrics_values))

    print("\n" + "=" * 80)
    print("NCU PROFILING REPORT")
    print("=" * 80)
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        seq_len = result.get("seq_len", "?")
        print(f"Workload: [{result['workload_index']}] T={seq_len}")
        print(f"Kernel filter: {result['kernel_name']}")
        print("-" * 80)
        print(result["report"])
