"""Canonical NCU metric preset.

Lifted (with attribution) from
``KernelAgent/kernel_perf_agent/kernel_opt/diagnose_prompt/metric_schema.py``
(Apache-2.0, Meta Platforms Inc.). Extended for B200 with stall-metric notes.

Provides:

* ``SECTION_MAP`` — ordered dict of section name → list of
  ``(display_label, metric_key, unit)`` tuples.
* ``ALL_METRIC_KEYS`` — flat list of every NCU metric key, suitable for
  passing to ``ncu --metrics``.
* ``parse_report(text)`` — pull numeric values for every known metric out of
  an NCU text report.
* ``summarize(values)`` — return a one-line-per-section summary string.

The set is intentionally small (~25 metrics) so the NCU run stays fast and the
resulting dict is something Phase-3 iterations can attach wholesale to
``iterations.jsonl`` if we opt into that later.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

MetricDef = Tuple[str, str, str]  # (display_label, ncu_key, unit_suffix)

SECTION_MAP: Dict[str, List[MetricDef]] = {
    "SM & Compute Utilization": [
        ("SM Cycles Active", "sm__cycles_active.avg", ""),
        ("Warp Active", "sm__warps_active.avg.pct_of_peak_sustained_active", "%"),
        ("Total Instructions Executed", "sm__inst_executed.sum", ""),
        ("Tensor Core Utilization",
         "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active", "%"),
        ("Tensor Core Pipeline Active",
         "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", "%"),
    ],
    "Memory Bandwidth & Cache": [
        ("DRAM Throughput",
         "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%"),
        ("DRAM Bandwidth", "dram__bytes.sum.per_second", " bytes/sec"),
        ("GPU DRAM Throughput",
         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "%"),
        ("DRAM Bytes Read", "dram__bytes_read.sum", " bytes"),
        ("DRAM Bytes Write", "dram__bytes_write.sum", " bytes"),
        ("L1 Cache Hit Rate", "l1tex__t_sector_hit_rate.pct", "%"),
        ("L1 Throughput",
         "l1tex__throughput.avg.pct_of_peak_sustained_active", "%"),
        ("L2 Cache Hit Rate", "lts__t_sector_hit_rate.pct", "%"),
        ("L2 Throughput",
         "lts__throughput.avg.pct_of_peak_sustained_active", "%"),
    ],
    "Memory Access Patterns": [
        ("Memory Coalescing",
         "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", "%"),
        ("Branch Uniformity",
         "smsp__sass_average_branch_targets_threads_uniform.pct", "%"),
    ],
    "Occupancy & Resources": [
        ("Occupancy Limited By Blocks",   "launch__occupancy_limit_blocks", ""),
        ("Occupancy Limited By Registers", "launch__occupancy_limit_registers", ""),
        ("Occupancy Limited By Shared Memory",
         "launch__occupancy_limit_shared_mem", ""),
        ("Registers per Thread", "launch__registers_per_thread", ""),
        ("Shared Memory per Block",
         "launch__shared_mem_per_block_allocated", " bytes"),
    ],
    "Stall Metrics (Warp Issue Stalls)": [
        ("Short Scoreboard Stalls",
         "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", "%"),
        ("Long Scoreboard Stalls",
         "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "%"),
        ("Barrier Stalls",
         "smsp__warp_issue_stalled_barrier_per_warp_active.pct", "%"),
        ("Branch Resolving Stalls",
         "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct", "%"),
    ],
}

ALL_METRIC_KEYS: List[str] = [k for entries in SECTION_MAP.values() for _, k, _ in entries]

# Index every metric key to its (section, label, unit) for fast lookup.
_KEY_TO_META: Dict[str, Tuple[str, str, str]] = {
    key: (section, label, unit)
    for section, entries in SECTION_MAP.items()
    for label, key, unit in entries
}


# --- Parsing --------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
)


def parse_report(text: str) -> Dict[str, Optional[float]]:
    """Extract numeric values for known metric keys from an ``ncu`` text report.

    NCU prints one metric per line, e.g.::

        sm__warps_active.avg.pct_of_peak_sustained_active              %        72.41
        dram__throughput.avg.pct_of_peak_sustained_elapsed             %        48.12

    Lines may have thousand separators and/or K/M/G suffixes. Returns a dict
    mapping every key in ``ALL_METRIC_KEYS`` to a float (or ``None`` if not
    present in the report).
    """
    values: Dict[str, Optional[float]] = {k: None for k in ALL_METRIC_KEYS}
    if not text:
        return values

    for line in text.splitlines():
        stripped = line.strip()
        # Each metric line starts with a key token.
        tok_end = 0
        while tok_end < len(stripped) and not stripped[tok_end].isspace():
            tok_end += 1
        key = stripped[:tok_end]
        if key not in values:
            continue
        rest = stripped[tok_end:]
        val = _extract_number(rest)
        if val is not None:
            values[key] = val
    return values


def _extract_number(segment: str) -> Optional[float]:
    """Pull the numeric value out of the trailing portion of an NCU line.

    NCU sometimes prints a unit token before the value (``Kbyte``, ``cycle``,
    ``%``), sometimes after. We take the last number on the line, after
    stripping commas, and handle K/M/G scale suffixes attached to the number.
    """
    matches = _NUMBER_RE.findall(segment)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        base = float(raw)
    except ValueError:
        return None
    # Look for a scale suffix immediately after the matched number.
    start = segment.rfind(matches[-1])
    tail = segment[start + len(matches[-1]):].lstrip()
    suffix = tail[:1]
    scale = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}.get(suffix)
    if scale:
        base *= scale
    return base


def summarize(values: Dict[str, Optional[float]]) -> str:
    """Return a compact, one-line-per-section summary string."""
    lines: List[str] = []
    for section, entries in SECTION_MAP.items():
        cells = []
        for label, key, unit in entries:
            v = values.get(key)
            if v is None:
                cells.append(f"{label}=?")
            elif unit == "%":
                cells.append(f"{label}={v:.1f}%")
            elif unit.strip().endswith("bytes/sec"):
                cells.append(f"{label}={v / 1e9:.2f}GB/s")
            elif unit.strip().endswith("bytes"):
                cells.append(f"{label}={v / 1e6:.1f}MB")
            else:
                cells.append(f"{label}={v:g}")
        lines.append(f"[{section}] " + " | ".join(cells))
    return "\n".join(lines)


def metric_section(key: str) -> Optional[str]:
    """Return the section name that owns ``key``, or ``None`` if unknown."""
    meta = _KEY_TO_META.get(key)
    return meta[0] if meta else None
