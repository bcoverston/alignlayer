#!/usr/bin/env bash
# AlignLayer deploy script
#
# Usage:
#   ./scripts/deploy.sh openclaw   — sync hook to deploy host + restart gateway
#   ./scripts/deploy.sh claudecode  — install PreToolUse/PostToolUse hooks locally
#   ./scripts/deploy.sh all         — both
#
# Requirements:
#   openclaw target: ssh access to ALIGNLAYER_DEPLOY_HOST, jq, rsync, docker
#   claudecode target: jq
#
# Configuration (via .env or environment):
#   ALIGNLAYER_DEPLOY_HOST  — SSH host for OpenClaw deploy (default: pi)
#   ALIGNLAYER_DEPLOY_PATH  — OpenClaw install path on remote (default: /opt/openclaw)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

DEPLOY_HOST="${ALIGNLAYER_DEPLOY_HOST:-pi}"
DEPLOY_PATH="${ALIGNLAYER_DEPLOY_PATH:-/opt/openclaw}"

# ── OpenClaw ─────────────────────────────────────────────────────────────────

deploy_openclaw() {
  echo "→ OpenClaw: syncing hook files to $DEPLOY_HOST..."
  rsync -av --delete \
    "$REPO_ROOT/src/openclaw-plugin/" \
    "$DEPLOY_HOST:$DEPLOY_PATH/data/hooks/alignlayer/"

  echo "→ OpenClaw: syncing plugin files to $DEPLOY_HOST..."
  ssh "$DEPLOY_HOST" "mkdir -p $DEPLOY_PATH/data/extensions/alignlayer"
  rsync -av \
    "$REPO_ROOT/src/openclaw-plugin/openclaw.plugin.json" \
    "$REPO_ROOT/src/openclaw-plugin/index.ts" \
    "$DEPLOY_HOST:$DEPLOY_PATH/data/extensions/alignlayer/"

  echo "→ OpenClaw: enabling hook + plugin in openclaw.json..."
  ssh "$DEPLOY_HOST" DEPLOY_PATH="$DEPLOY_PATH" bash <<'REMOTE'
    set -euo pipefail
    CONFIG="$DEPLOY_PATH/data/openclaw.json"

    # Idempotent: set or update the alignlayer entry.
    # Uses python3 — jq is not available on the deploy host.
    python3 - "$CONFIG" <<'PY'
import json, sys, os

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

d.setdefault("hooks", {}) \
 .setdefault("internal", {}) \
 .setdefault("entries", {})["alignlayer"] = {
    "enabled": True,
    "env": {
        "ALIGNLAYER_TRACES_DIR": "/home/node/.openclaw/traces"
    }
}
# Plugin auto-discovered from extensions dir — no explicit entry needed.
# Remove any stale entry that may have been added previously.
d.get("plugins", {}).get("entries", {}).pop("alignlayer", None)

tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(d, f, indent=4)
os.replace(tmp, path)
print("  openclaw.json updated")
PY
REMOTE

  echo "→ OpenClaw: restarting gateway..."
  ssh "$DEPLOY_HOST" "cd $DEPLOY_PATH && docker compose restart"

  echo "→ OpenClaw: waiting for gateway to come up..."
  sleep 5

  echo "→ OpenClaw: checking hook registration..."
  ssh "$DEPLOY_HOST" "docker logs --tail 40 openclaw 2>&1 | grep -i alignlayer || echo '  (no alignlayer log lines yet — check for errors below)'"
  ssh "$DEPLOY_HOST" "docker logs --tail 20 openclaw 2>&1 | grep -i 'error\|warn\|hook' | tail -10 || true"

  echo "✓ OpenClaw deploy complete. Traces → $DEPLOY_PATH/data/traces/"
}

# ── Claude Code ───────────────────────────────────────────────────────────────

deploy_claudecode() {
  SETTINGS="$HOME/.claude/settings.json"
  HOOK_CMD="npx tsx $REPO_ROOT/src/claudecode-hook/hook.ts"

  echo "→ Claude Code: patching $SETTINGS..."

  # Merge PreToolUse and PostToolUse hooks into existing settings.
  # Idempotent: checks for existing alignlayer entry before adding.
  jq --arg cmd "$HOOK_CMD" '
    # Helper: add hook entry to an event array if not already present (matched by command)
    def add_hook(event; entry):
      if (.hooks[event] // []) | any(
        (.hooks // []) | any(.command == entry.hooks[0].command)
      )
      then .
      else .hooks[event] = ((.hooks[event] // []) + [entry])
      end;

    add_hook("PreToolUse"; {
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": $cmd,
        "timeout": 10,
        "statusMessage": "AlignLayer scoring..."
      }]
    }) |
    add_hook("PostToolUse"; {
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": $cmd,
        "timeout": 5,
        "async": true
      }]
    })
  ' "$SETTINGS" > "$SETTINGS.tmp" && mv "$SETTINGS.tmp" "$SETTINGS"

  echo "→ Claude Code: verifying hook is runnable..."
  echo '{"session_id":"deploy-test","hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"echo hello"},"tool_use_id":"test-001","transcript_path":"","cwd":"","permission_mode":"default"}' \
    | npx --prefix "$REPO_ROOT" tsx "$REPO_ROOT/src/claudecode-hook/hook.ts" \
    | jq .

  echo "✓ Claude Code deploy complete. Traces → ~/.alignlayer/traces/"
}

# ── Entrypoint ────────────────────────────────────────────────────────────────

case "${1:-all}" in
  openclaw)   deploy_openclaw ;;
  claudecode) deploy_claudecode ;;
  all)        deploy_openclaw && deploy_claudecode ;;
  *)
    echo "Usage: $0 openclaw | claudecode | all"
    exit 1
    ;;
esac
