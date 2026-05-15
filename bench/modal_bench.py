"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Packs the solution from solution/triton/kernel.py and runs benchmarks
on NVIDIA B200 GPUs via Modal. Outputs structured lines.

Usage:
    python bench/modal_bench.py
    python bench/modal_bench.py --workload-index 8
"""

import sys
from pathlib import Path

# Ensure bench/ is importable
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

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("ako-flashinfer-bench")


@app.function(image=FLASHINFER_IMAGE, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:  # noqa: E302
    """Run benchmark on Modal B200 and return results."""
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

    # Read TLX kernel status if available
    kernel_info = {"has_tlx": False, "gemm1": "unknown", "gemm2": "unknown"}
    try:
        import json, tempfile, os
        ki_path = os.path.join(tempfile.gettempdir(), "_kernel_info.json")
        if os.path.exists(ki_path):
            with open(ki_path) as f:
                kernel_info = json.load(f)
    except Exception:
        pass
    results["_kernel_info"] = kernel_info

    return results


def load_sota_targets() -> dict:
    """Load SOTA targets from context/sota_targets.jsonl.

    Returns {uuid: {"seq_len": int, "sota_ms": float, ...}}.
    Empty dict if the file is missing (targets are optional).
    """
    import json
    sota_path = AKO_ROOT / "context" / "sota_targets.jsonl"
    if not sota_path.exists():
        return {}
    targets = {}
    with open(sota_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            targets[entry["uuid"]] = entry
    return targets


def build_structured_result(results: dict, sota_targets: dict, label: str) -> dict:
    """Assemble bench_result.schema.json-shaped dict. Does not mutate `results`."""
    import datetime
    kernel_info = results.get("_kernel_info", {}) or {}
    sota_targets = sota_targets or {}

    rows = []
    latencies, ref_latencies, speedups, sota_pairs = [], [], [], []
    all_passed = True

    for def_name, traces in results.items():
        if def_name == "_kernel_info":
            continue
        for workload_uuid, result in traces.items():
            status = result.get("status", "unknown")
            if status.lower() != "passed":
                all_passed = False
            sota_entry = sota_targets.get(workload_uuid) or {}
            latency = result.get("latency_ms")
            if latency is not None:
                latencies.append(latency)
                if sota_entry.get("sota_ms") is not None:
                    sota_pairs.append((latency, sota_entry["sota_ms"]))
            if result.get("reference_latency_ms") is not None:
                ref_latencies.append(result["reference_latency_ms"])
            if result.get("speedup_factor") is not None:
                speedups.append(result["speedup_factor"])
            sota_ms = sota_entry.get("sota_ms")
            rows.append({
                "uuid": workload_uuid,
                "seq_len": sota_entry.get("seq_len"),
                "status": status,
                "latency_ms": latency,
                "reference_latency_ms": result.get("reference_latency_ms"),
                "speedup_factor": result.get("speedup_factor"),
                "max_abs_error": result.get("max_abs_error"),
                "max_rel_error": result.get("max_rel_error"),
                "sota_ms": sota_ms,
                "sota_gap": (latency / sota_ms) if (latency is not None and sota_ms) else None,
            })

    out = {
        "schema_version": 1,
        "label": label or "",
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "compiled": True,
        "correct": all_passed,
        "kernel_info": kernel_info,
        "workloads": rows,
    }
    if latencies:
        out["runtime_ms_mean"] = sum(latencies) / len(latencies)
    if ref_latencies:
        out["ref_runtime_ms_mean"] = sum(ref_latencies) / len(ref_latencies)
    if speedups:
        out["speedup_mean"] = sum(speedups) / len(speedups)
    if sota_pairs:
        import math
        gaps = [m / s for m, s in sota_pairs]
        out["sota_geomean_gap"] = math.exp(sum(math.log(g) for g in gaps) / len(gaps))
        out["sota_total_headroom_ms"] = sum(max(0.0, m - s) for m, s in sota_pairs)
        out["sota_converged_count"] = sum(1 for g in gaps if g <= 1.10)
        out["sota_eligible_count"] = len(sota_pairs)
    return out


def print_results_ako(results: dict, sota_targets: dict = None):
    """Print results in structured format + detailed per-workload info."""
    kernel_info = results.pop("_kernel_info", None)
    sota_targets = sota_targets or {}

    all_passed = True
    latencies = []
    ref_latencies = []
    speedups = []
    # For SOTA comparison: pairs of (measured_ms, sota_ms) for workloads with a target
    sota_pairs = []

    for def_name, traces in results.items():
        for workload_uuid, result in traces.items():
            status = result.get("status", "unknown")
            if status.lower() != "passed":
                all_passed = False

            if result.get("latency_ms") is not None:
                latencies.append(result["latency_ms"])
                sota_entry = sota_targets.get(workload_uuid)
                if sota_entry is not None:
                    sota_pairs.append((result["latency_ms"], sota_entry["sota_ms"]))
            if result.get("reference_latency_ms") is not None:
                ref_latencies.append(result["reference_latency_ms"])
            if result.get("speedup_factor") is not None:
                speedups.append(result["speedup_factor"])

    # --- Structured output ---
    print("COMPILED: True")
    print(f"CORRECT: {all_passed}")

    if latencies:
        mean_runtime = sum(latencies) / len(latencies)
        print(f"RUNTIME: {mean_runtime:.4f}")
    if ref_latencies:
        mean_ref = sum(ref_latencies) / len(ref_latencies)
        print(f"REF_RUNTIME: {mean_ref:.4f}")
    if speedups:
        mean_speedup = sum(speedups) / len(speedups)
        print(f"SPEEDUP: {mean_speedup:.2f}x")

    # --- SOTA gap (aggregate) ---
    if sota_pairs:
        # Geometric mean of per-workload gap: fair across tiny and huge workloads.
        # Arithmetic mean of absolute headroom_ms: shows total time the agent could still save.
        import math
        gaps = [m / s for m, s in sota_pairs]
        geomean_gap = math.exp(sum(math.log(g) for g in gaps) / len(gaps))
        total_headroom = sum(max(0.0, m - s) for m, s in sota_pairs)
        converged = sum(1 for g in gaps if g <= 1.10)
        print(f"SOTA_GEOMEAN_GAP: {geomean_gap:.3f}x")
        print(f"SOTA_TOTAL_HEADROOM_MS: {total_headroom:.3f}")
        print(f"SOTA_CONVERGED: {converged}/{len(sota_pairs)} (within 1.10x)")

    # --- Detailed per-workload output ---
    if kernel_info:
        has_tlx = kernel_info.get("has_tlx", False)
        g1 = kernel_info.get("gemm1", "?")
        g2 = kernel_info.get("gemm2", "?")
        print(f"\n[Kernel] TLX installed: {has_tlx} | GEMM1: {g1} | GEMM2: {g2}")

    for def_name, traces in results.items():
        print(f"\n--- {def_name} ---")
        # Sort by seq_len (ascending) if SOTA entries known, else insertion order.
        items = list(traces.items())
        if sota_targets:
            items.sort(key=lambda kv: sota_targets.get(kv[0], {}).get("seq_len", 10**9))
        for workload_uuid, result in items:
            status = result.get("status")
            sota_entry = sota_targets.get(workload_uuid)
            seq_tag = f"seq={sota_entry['seq_len']}" if sota_entry else ""
            prefix = f"  [{seq_tag}] " if seq_tag else "  "
            line = f"{prefix}Workload {workload_uuid[:8]}...: {status}"

            if result.get("latency_ms") is not None:
                line += f" | {result['latency_ms']:.3f} ms"
            if sota_entry is not None and result.get("latency_ms") is not None:
                gap = result["latency_ms"] / sota_entry["sota_ms"]
                line += f" | sota={sota_entry['sota_ms']:.3f} ms | gap={gap:.2f}x"
            if result.get("speedup_factor") is not None:
                line += f" | {result['speedup_factor']:.2f}x speedup"
            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                line += f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"

            print(line)


@app.local_entrypoint()
def main(workload_index: int = -1, json_out: str = "", label: str = ""):
    """Pack solution and run benchmark on Modal B200.

    Args:
        workload_index: -1 to run all workloads; otherwise a specific index.
        json_out:       optional path; if set, a bench_result.schema.json-shaped
                        document is written there in addition to stdout.
        label:          label string threaded into the structured output (e.g.
                        'iter-7', 'baseline'). No effect on stdout.
    """
    import json

    print("Packing solution from source files...")
    solution = pack_solution_from_dir()
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("COMPILED: False")
        print("CORRECT: False")
        print("No results returned!")
        if json_out:
            failure = {
                "schema_version": 1,
                "label": label or "",
                "compiled": False,
                "correct": False,
                "workloads": [],
            }
            with open(json_out, "w") as f:
                json.dump(failure, f, indent=2)
        sys.exit(1)

    sota_targets = load_sota_targets()

    # Build structured result before print_results_ako mutates `results`.
    if json_out:
        structured = build_structured_result(results, sota_targets, label)
        with open(json_out, "w") as f:
            json.dump(structured, f, indent=2)
        print(f"\n[bench] wrote JSON to {json_out}")

    print_results_ako(results, sota_targets=sota_targets)

    # Exit code based on correctness
    kernel_info = results.pop("_kernel_info", None)
    all_passed = all(
        r.get("status", "").lower() == "passed"
        for traces in results.values()
        for r in traces.values()
    )
    sys.exit(0 if all_passed else 1)
