#!/usr/bin/env python3
"""Gate for Stage 2.4 exit: validate solution/rules_check.json.

Exit 0 iff:
  - solution/rules_check.json exists and validates against
    context/schemas/rules_check.schema.json.
  - Every rule has status == "pass".
  - Every evidence block points at a real file+line-range.

Usage:
    python3 tools/check_rules.py [--rules-check PATH] [--schema PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES_CHECK = PROJECT_ROOT / "solution" / "rules_check.json"
DEFAULT_SCHEMA = PROJECT_ROOT / "context" / "schemas" / "rules_check.schema.json"


def _load_json(path: Path, label: str) -> object:
    if not path.exists():
        print(f"[check_rules] {label} not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"[check_rules] {label} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)


def _validate_against_schema(doc: object, schema: dict) -> list[str]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        # Fall back to a minimal structural check if jsonschema isn't installed.
        errs: list[str] = []
        if not isinstance(doc, dict):
            return ["top-level document must be an object"]
        for key in ("schema_version", "kernel_file", "rules"):
            if key not in doc:
                errs.append(f"missing required key: {key}")
        if isinstance(doc.get("rules"), list) and len(doc["rules"]) < 5:
            errs.append("rules array must have at least 5 entries")
        return errs

    validator = jsonschema.Draft7Validator(schema)
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(doc)]


def _check_evidence(evidence: dict) -> list[str]:
    errs = []
    file_path = evidence.get("file")
    if not file_path:
        return ["evidence.file missing"]
    abs_path = (PROJECT_ROOT / file_path).resolve()
    if not abs_path.exists():
        errs.append(f"evidence file does not exist: {file_path}")
        return errs
    try:
        text = abs_path.read_text().splitlines()
    except Exception as e:
        errs.append(f"cannot read evidence file {file_path}: {e}")
        return errs
    start = evidence.get("line_start", 0)
    end = evidence.get("line_end", 0)
    if start < 1 or end < start:
        errs.append(f"bad line range in {file_path}: [{start}, {end}]")
    if end > len(text):
        errs.append(f"line_end {end} > {len(text)} lines in {file_path}")
    quote = evidence.get("quote")
    if quote and start >= 1 and end <= len(text):
        snippet = "\n".join(text[start - 1:end])
        if quote.strip() and quote.strip() not in snippet:
            errs.append(f"quote not found in {file_path}:{start}-{end}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules-check", type=Path, default=DEFAULT_RULES_CHECK)
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = ap.parse_args()

    schema = _load_json(args.schema, "schema")
    doc = _load_json(args.rules_check, "rules_check")

    schema_errs = _validate_against_schema(doc, schema)  # type: ignore[arg-type]
    if schema_errs:
        print("[check_rules] schema validation failed:", file=sys.stderr)
        for e in schema_errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    rules = doc["rules"] if isinstance(doc, dict) else []
    seen_ids = set()
    failed = False
    for rule in rules:
        rid = rule.get("rule_id")
        if rid in seen_ids:
            print(f"[check_rules] rule {rid} appears more than once", file=sys.stderr)
            failed = True
        seen_ids.add(rid)
        if rule.get("status") != "pass":
            print(f"[check_rules] rule {rid} status={rule.get('status')!r}: {rule.get('notes', '')}",
                  file=sys.stderr)
            failed = True
            continue
        for ev in rule.get("evidence", []):
            for msg in _check_evidence(ev):
                print(f"[check_rules] rule {rid}: {msg}", file=sys.stderr)
                failed = True

    required = {1, 2, 3, 4, 5}
    missing = required - seen_ids
    if missing:
        print(f"[check_rules] missing rule entries: {sorted(missing)}", file=sys.stderr)
        failed = True

    if failed:
        return 1
    try:
        rel = args.rules_check.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.rules_check
    print(f"[check_rules] OK — all 5 rules pass with valid evidence ({rel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
