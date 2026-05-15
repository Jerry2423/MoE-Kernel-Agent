"""
Shared infrastructure for Modal-based benchmarking and profiling.

Provides:
- pack_solution_from_dir(): Pack kernel source into a flashinfer_bench Solution
- Modal image definitions for benchmark and NCU containers
- Common constants (volume, paths)
"""

from pathlib import Path

import modal

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution
from flashinfer_bench.agents import pack_solution_from_files

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root (parent of bench/)
AKO_ROOT = Path(__file__).resolve().parent.parent

# Modal volume for FlashInfer trace data
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# ---------------------------------------------------------------------------
# Modal images
# ---------------------------------------------------------------------------

FLASHINFER_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

def _make_triton_source_image():
    """Triton source build for Gluon API access (tcgen05_mma, TMA gather/scatter,
    warp specialization, multi-buffer pipelining on Blackwell SM100).
    Built on the same debian base as FLASHINFER_IMAGE, replaces pip triton
    with a source build that includes Gluon APIs.
    Call this function only when Gluon is needed — building from source takes ~15 min."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "gcc", "g++", "zlib1g-dev")
        .pip_install("flashinfer-bench", "torch", "triton", "numpy", "safetensors")
        .pip_install("setuptools>=40.8.0", "wheel", "cmake>=3.20,<4.0", "ninja>=1.11.1", "pybind11>=2.13.1", "lit", "pydantic")
        .run_commands(
            "git clone --depth=1 https://github.com/triton-lang/triton.git /triton",
            "cd /triton && pip install -r python/requirements.txt &&  MAX_JOBS=4 pip install -e .",
        )
    )

NCU_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "safetensors", "nvtx")
    .apt_install("wget", "gnupg")
    .run_commands(
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub"
        " | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg]"
        " https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /'"
        " > /etc/apt/sources.list.d/cuda.list",
        "apt-get update && apt-get install -y nsight-compute-2026.1.0",
    )
)

# ---------------------------------------------------------------------------
# Solution packing
# ---------------------------------------------------------------------------


def load_config(ako_root: Path = None) -> dict:
    """Load configuration from context/config.toml."""
    if ako_root is None:
        ako_root = AKO_ROOT
    config_path = ako_root / "context" / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution_from_dir(ako_root: Path = None, output_path: Path = None) -> Solution:
    """Pack solution source files into a flashinfer_bench Solution object.

    Reads config from context/config.toml and source from solution/<language>/.
    """
    if ako_root is None:
        ako_root = AKO_ROOT

    config = load_config(ako_root)
    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language
    if language == "triton":
        source_dir = ako_root / "solution" / "triton"
    elif language == "cuda":
        source_dir = ako_root / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    dps = build_config.get("destination_passing_style", True)
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
    )

    # Optionally write to disk
    if output_path is not None:
        output_path.write_text(solution.model_dump_json(indent=2))
        print(f"Solution packed to: {output_path}")

    return solution
