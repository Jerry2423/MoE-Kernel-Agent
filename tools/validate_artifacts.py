#!/usr/bin/env python3
"""Validate every structured artifact in the repo against its JSON Schema.

Exit code is the number of artifacts that failed (clamped to 255). Exit 0
means every artifact that exists passes its schema. Artifacts that don't
exist yet are skipped silently (the harness is designed so artifacts appear
over time).

Usage:
    python3 tools/validate_artifacts.py          # full report
    python3 tools/validate_artifacts.py --quiet  # only print on failure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = PROJECT_ROOT / "context" / "schemas"

# (artifact_path, schema_name, per_line)
ARTIFACTS: list[tuple[Path, str, bool]] = [
    (PROJECT_ROOT / "solution" / "subgraph_specs.json", "subgraph_spec", False),
    (PROJECT_ROOT / "solution" / "workload_classes.json", "workload_class", False),
    (PROJECT_ROOT / "solution" / "rules_check.json", "rules_check", False),
    (PROJECT_ROOT / "solution" / "iterations.jsonl", "iteration_row", True),
]


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text())


def _validator(schema: dict):
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None
    return jsonschema.Draft7Validator(schema)


def _validate_doc(doc: object, schema: dict, validator) -> list[str]:
    if validator is None:
        # Minimal structural fallback: top-level shape check only.
        if schema.get("type") == "array" and not isinstance(doc, list):
            return ["expected a JSON array at the top level"]
        if schema.get("type") == "object" and not isinstance(doc, dict):
            return ["expected a JSON object at the top level"]
        return []
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(doc)]


def _scan_trajectory(quiet: bool) -> tuple[int, int]:
    traj_root = PROJECT_ROOT / "trajectory"
    if not traj_root.exists():
        return 0, 0
    schema = _load_schema("bench_result")
    validator = _validator(schema)
    checked = 0
    failures = 0
    for bench_json in sorted(traj_root.glob("*/bench_result.json")):
        checked += 1
        try:
            doc = json.loads(bench_json.read_text())
        except json.JSONDecodeError as e:
            failures += 1
            print(f"[validate] {bench_json.relative_to(PROJECT_ROOT)}: invalid JSON — {e}", file=sys.stderr)
            continue
        errs = _validate_doc(doc, schema, validator)
        if errs:
            failures += 1
            print(f"[validate] {bench_json.relative_to(PROJECT_ROOT)}: {len(errs)} schema error(s)", file=sys.stderr)
            if not quiet:
                for e in errs[:5]:
                    print(f"    - {e}", file=sys.stderr)
    return checked, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="only print on failure")
    args = ap.parse_args()

    total_checked = 0
    total_failed = 0

    for path, schema_name, per_line in ARTIFACTS:
        if not path.exists():
            continue
        schema_path = SCHEMAS_DIR / f"{schema_name}.schema.json"
        if not schema_path.exists():
            print(f"[validate] missing schema: {schema_path}", file=sys.stderr)
            total_failed += 1
            continue
        schema = _load_schema(schema_name)
        validator = _validator(schema)
        rel = path.relative_to(PROJECT_ROOT)
        if per_line:
            # JSONL: validate each non-empty line against the schema.
            line_failures = 0
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                total_checked += 1
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    line_failures += 1
                    print(f"[validate] {rel}:{lineno} invalid JSON — {e}", file=sys.stderr)
                    continue
                errs = _validate_doc(doc, schema, validator)
                if errs:
                    line_failures += 1
                    print(f"[validate] {rel}:{lineno} {len(errs)} schema error(s)", file=sys.stderr)
                    if not args.quiet:
                        for e in errs[:3]:
                            print(f"    - {e}", file=sys.stderr)
            total_failed += line_failures
            if not args.quiet and line_failures == 0:
                print(f"[validate] {rel} OK (jsonl)")
        else:
            total_checked += 1
            try:
                doc = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                total_failed += 1
                print(f"[validate] {rel}: invalid JSON — {e}", file=sys.stderr)
                continue
            errs = _validate_doc(doc, schema, validator)
            if errs:
                total_failed += 1
                print(f"[validate] {rel}: {len(errs)} schema error(s)", file=sys.stderr)
                if not args.quiet:
                    for e in errs[:5]:
                        print(f"    - {e}", file=sys.stderr)
            elif not args.quiet:
                print(f"[validate] {rel} OK")

    traj_checked, traj_failed = _scan_trajectory(args.quiet)
    total_checked += traj_checked
    total_failed += traj_failed

    if not args.quiet:
        print(f"[validate] {total_checked - total_failed}/{total_checked} artifacts pass")
    return min(255, total_failed)


if __name__ == "__main__":
    raise SystemExit(main())
