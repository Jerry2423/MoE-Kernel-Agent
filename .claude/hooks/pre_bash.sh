#!/usr/bin/env bash
# PreToolUse hook for Bash. Rejects (exit 2) on policy violations.
#
# Rules:
#   1. `modal run` must use `-m bench.modal_*` (module form).
#   2. `modal run` must be prefixed by
#      `eval "$(conda shell.bash hook)" && conda activate fi-bench`
#      in the same command so modal is on PATH.
#   3. `git commit` inside solution/ phases must use an approved prefix.
#   4. Destructive ops (git reset --hard, git push --force, rm -rf of critical
#      dirs) are refused. This mirrors settings.json deny rules as
#      defense-in-depth.
set -eu

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
STATE_FILE="$PROJECT_DIR/.claude/state/current_phase"
phase="none"
[ -f "$STATE_FILE" ] && phase="$(tr -d '[:space:]' < "$STATE_FILE")"

payload="$(cat)"
cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty')"
[ -z "$cmd" ] && exit 0

reject() {
  printf '[pre_bash.sh] %s\n' "$1" >&2
  exit 2
}

# --- Destructive-op guard (defense in depth vs settings.json) ----------------
case "$cmd" in
  *"git reset --hard"*)        reject "'git reset --hard' is denied. Use 'git revert' or ask the user." ;;
  *"git push --force"*|*"git push -f "*|*"git push -f"|*" -f "*"git push"*)
                               reject "'git push --force' is denied. Resolve the divergence instead." ;;
  *"rm -rf solution"*|*"rm -rf ./solution"*|*"rm -rf /data/jxu/kernel_agent/solution"*)
                               reject "'rm -rf solution' is denied. Work in a branch or delete individual files." ;;
  *"rm -rf knowledge"*|*"rm -rf ./knowledge"*)
                               reject "'rm -rf knowledge' is denied." ;;
  *"rm -rf context"*|*"rm -rf ./context"*)
                               reject "'rm -rf context' is denied." ;;
  *"rm -rf bench"*|*"rm -rf ./bench"*)
                               reject "'rm -rf bench' is denied." ;;
esac

# --- Modal invocation guards -------------------------------------------------
if printf '%s' "$cmd" | grep -qE '(^|[^a-zA-Z_])modal run([[:space:]]|$)'; then
  # Module-form check
  if ! printf '%s' "$cmd" | grep -qE 'modal run[[:space:]]+(-m[[:space:]]+|--module[[:space:]]+)?(-m[[:space:]]+)?bench\.modal_[A-Za-z0-9_]+'; then
    # Allow only when -m bench.modal_* is clearly present
    if ! printf '%s' "$cmd" | grep -qE 'modal run[[:space:]]+-m[[:space:]]+bench\.modal_[A-Za-z0-9_]+'; then
      reject "modal run must use module form: 'modal run -m bench.modal_<name>' (not 'modal run bench/modal_<name>.py'). See bench/GUIDE.md."
    fi
  fi
  # Conda activation check (must be in the same command because shell state does not persist)
  if ! printf '%s' "$cmd" | grep -qE 'conda[[:space:]]+activate[[:space:]]+fi-bench'; then
    reject "Prefix modal runs with: eval \"\$(conda shell.bash hook)\" && conda activate fi-bench && <modal run ...>. scripts/bench.sh does this automatically."
  fi
fi

# --- Commit-prefix guard (Phase 2/3 writes under solution/) ------------------
if [ "$phase" = "phase2" ] || [ "$phase" = "phase3" ]; then
  if printf '%s' "$cmd" | grep -qE '^[[:space:]]*git[[:space:]]+commit([[:space:]]|$)'; then
    # Extract the message passed via -m, --message, or here-doc.
    msg="$(printf '%s' "$cmd" | sed -nE "s/.*-m[[:space:]]+['\"]?([^'\"]*).*/\\1/p" | head -n1)"
    if [ -z "$msg" ]; then
      msg="$(printf '%s' "$cmd" | sed -nE "s/.*--message[[:space:]]+['\"]?([^'\"]*).*/\\1/p" | head -n1)"
    fi
    # Heredoc (git commit -m "$(cat <<'EOF' ... EOF)") is an approved pattern;
    # we can't parse its contents here, so we accept it and trust the commit-msg
    # hook to enforce format. Skip the check if message appears empty.
    if [ -n "$msg" ]; then
      case "$msg" in
        "[baseline]"*|"[baseline-autotuned]"*|"[phase3-A]"*|"[phase3-C]"*|"[phase2.5]"*|\
        "[iter "*"]"*|"[iter-"*"]"*)
          : ;;
        *)
          reject "commit message must start with one of: [baseline], [baseline-autotuned], [phase3-A], [phase3-C], [phase2.5], [iter N][<class>], [iter-N cleanup]. Got: $msg"
          ;;
      esac
    fi
  fi
fi

exit 0
