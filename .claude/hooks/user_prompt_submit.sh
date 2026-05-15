#!/usr/bin/env bash
# UserPromptSubmit hook: when the user invokes /phase1-learn, /phase2-implement,
# /phase3-optimize, or /run-all, record the current phase in
# .claude/state/current_phase. PreToolUse hooks read this to enforce
# phase-gated writes. /run-all is the auto-chaining orchestrator and seeds
# phase1; it then rewrites this file mid-session as it advances phases.
set -eu

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
STATE_DIR="$PROJECT_DIR/.claude/state"
STATE_FILE="$STATE_DIR/current_phase"
mkdir -p "$STATE_DIR"

payload="$(cat)"
prompt="$(printf '%s' "$payload" | jq -r '.prompt // empty')"

detect_phase() {
  case "$1" in
    *"/run-all"*)          echo phase1 ;;
    *"/phase1-learn"*)     echo phase1 ;;
    *"/phase2-implement"*) echo phase2 ;;
    *"/phase3-optimize"*)  echo phase3 ;;
    *) echo "" ;;
  esac
}

phase="$(detect_phase "$prompt")"
if [ -n "$phase" ]; then
  printf '%s\n' "$phase" > "$STATE_FILE"
fi

exit 0
