#!/usr/bin/env bash
# Stop hook: non-blocking warnings.
#  - If the session wrote phase artifacts but HANDOFF.md is unchanged, warn.
#  - Run tools/validate_artifacts.py and surface any schema failures.
# Always exits 0 so the session can close.
set -eu

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
STATE_FILE="$PROJECT_DIR/.claude/state/current_phase"
phase="none"
[ -f "$STATE_FILE" ] && phase="$(tr -d '[:space:]' < "$STATE_FILE")"

# Artifact → phase map for the HANDOFF drift check.
phase_artifacts_dir() {
  case "$1" in
    phase1) echo "knowledge" ;;
    phase2) echo "solution" ;;
    phase3) echo "solution" ;;
    *)      echo "" ;;
  esac
}

if command -v git >/dev/null 2>&1 && [ -d "$PROJECT_DIR/.git" ]; then
  artifact_dir="$(phase_artifacts_dir "$phase")"
  if [ -n "$artifact_dir" ]; then
    if git -C "$PROJECT_DIR" status --porcelain "$artifact_dir" 2>/dev/null | grep -q .; then
      if ! git -C "$PROJECT_DIR" status --porcelain HANDOFF.md 2>/dev/null | grep -q .; then
        printf '[stop.sh] %s/ has unstaged changes but HANDOFF.md was not updated. Add the %s block per .claude/commands/%s*.md before handing off.\n' \
          "$artifact_dir" "$phase" "$phase" >&2
      fi
    fi
  fi
fi

if [ -f "$PROJECT_DIR/tools/validate_artifacts.py" ]; then
  if ! python3 "$PROJECT_DIR/tools/validate_artifacts.py" --quiet 2>/dev/null; then
    printf '[stop.sh] tools/validate_artifacts.py reports schema errors. Run it directly for details.\n' >&2
  fi
fi

exit 0
