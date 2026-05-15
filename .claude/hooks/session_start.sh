#!/usr/bin/env bash
# SessionStart hook: ensure .claude/state/ exists. Phase detection happens in
# user_prompt_submit.sh so it reacts to the actual /phaseN-* invocation.
set -eu
STATE_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/state"
mkdir -p "$STATE_DIR"
exit 0
