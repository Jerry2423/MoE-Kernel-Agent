#!/usr/bin/env bash
# PreToolUse hook for Edit / Write / NotebookEdit.
# Rejects (exit 2) writes to directories that belong to a different phase.
#
# Phase → forbidden write targets:
#   phase1: solution/, scripts/, bench/, input/
#   phase2: knowledge/, context/
#   phase3: knowledge/, context/, and bench/modal_autotune.py unless
#           .claude/state/autotune_change_reason.txt exists (freeze rule).
set -eu

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
STATE_FILE="$PROJECT_DIR/.claude/state/current_phase"
AUTOTUNE_REASON="$PROJECT_DIR/.claude/state/autotune_change_reason.txt"

# If we don't know the phase, don't gate — user may be working outside a phase.
[ -f "$STATE_FILE" ] || exit 0
phase="$(tr -d '[:space:]' < "$STATE_FILE")"
[ -n "$phase" ] || exit 0

payload="$(cat)"
path="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"
[ -z "$path" ] && exit 0

# Normalize to a path relative to PROJECT_DIR for matching.
case "$path" in
  "$PROJECT_DIR"/*) rel="${path#$PROJECT_DIR/}" ;;
  /*)               rel="$path" ;;
  *)                rel="$path" ;;
esac

reject() {
  printf '[pre_edit.sh] %s\n' "$1" >&2
  exit 2
}

case "$phase" in
  phase1)
    case "$rel" in
      solution/*|input/*|bench/*|scripts/*)
        reject "Phase 1 is read-only except for knowledge/. Edit '$rel' belongs to Phase 2 or later. See .claude/commands/phase1-learn.md."
        ;;
    esac
    ;;
  phase2)
    case "$rel" in
      knowledge/*|context/*)
        reject "Phase 2 must not modify '$rel'. knowledge/ is Phase 1's output; context/ is read-only reference material. See .claude/commands/phase2-implement.md."
        ;;
    esac
    ;;
  phase3)
    case "$rel" in
      knowledge/*|context/*)
        reject "Phase 3 must not modify '$rel'. Open a new /phase1-learn session if the knowledge base needs expansion."
        ;;
      bench/modal_autotune.py)
        if [ ! -s "$AUTOTUNE_REASON" ]; then
          reject "Phase 3 freeze rule: write a non-empty .claude/state/autotune_change_reason.txt first (see phase3-optimize.md §\"Autotune Freeze\") before editing bench/modal_autotune.py."
        fi
        ;;
    esac
    ;;
esac

exit 0
