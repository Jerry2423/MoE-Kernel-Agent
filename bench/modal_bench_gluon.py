"""
FlashInfer-Bench Modal Cloud Benchmark with Triton Source Build (Gluon).

Same as modal_bench.py but uses a Triton source-built image that includes
Gluon APIs for Blackwell SM100 hardware primitives.

First run builds Triton from source (~15 min). Subsequent runs use cache.

Usage:
    PYTHONPATH=. modal run -m bench.modal_bench_gluon
"""

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import modal
from modal_common import (
    AKO_ROOT,
    _make_triton_source_image,
    TRACE_SET_PATH,
    pack_solution_from_dir,
    trace_volume,
)

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

# Import the print function from the standard bench module
from modal_bench import print_results_ako

GLUON_IMAGE = _make_triton_source_image()
app = modal.App("ako-flashinfer-bench-gluon")


@app.function(image=GLUON_IMAGE, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark_gluon(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 with Triton source build (Gluon support)."""
    if config is None:
        config = BenchmarkConfig(
            warmup_runs=3, iterations=100, num_trials=5,
            atol=1.0, rtol=0.3, required_matched_ratio=0.9,
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    results["_kernel_info"] = {"has_tlx": False, "gemm1": "unknown", "gemm2": "unknown"}
    return results


@app.local_entrypoint()
def main(workload_index: int = -1):
    """Pack solution and run benchmark on Modal B200 with Gluon/Triton source."""
    print("Packing solution from source files...")
    solution = pack_solution_from_dir()
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200 (Gluon / Triton source build)...")
    print("First run builds Triton from source (~15 min). Subsequent runs use cache.")
    results = run_benchmark_gluon.remote(solution)

    if not results:
        print("COMPILED: False")
        print("CORRECT: False")
        print("No results returned!")
        sys.exit(1)

    print_results_ako(results)
