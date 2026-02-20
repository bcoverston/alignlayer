#!/usr/bin/env bash
# AlignLayer deploy script
#
# Usage:
#   ./scripts/deploy.sh openclaw   — sync hook to pi + restart gateway
#   ./scripts/deploy.sh claudecode  — install PreToolUse/PostToolUse hooks locally
#   ./scripts/deploy.sh all         — both
#
# Requirements:
#   openclaw target: ssh access to 'pi', jq, rsync, docker
#   claudecode target: jq

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── OpenClaw ─────────────────────────────────────────────────────────────────

deploy_openclaw() {
  echo "→ OpenClaw: syncing hook files to pi..."
  rsync -av --delete \
    "$REPO_ROOT/src/openclaw-plugin/" \
    pi:/opt/openclaw/data/hooks/alignlayer/

  echo "→ OpenClaw: syncing plugin files to pi..."
  ssh pi "mkdir -p /opt/openclaw/data/extensions/alignlayer"
  rsync -av \
    "$REPO_ROOT/src/openclaw-plugin/openclaw.plugin.json" \
    "$REPO_ROOT/src/openclaw-plugin/index.ts" \
    pi:/opt/openclaw/data/extensions/alignlayer/

  echo "→ OpenClaw: enabling hook + plugin in openclaw.json..."
  ssh pi bash <<'REMOTE'
    set -euo pipefail
    CONFIG=/opt/openclaw/data/openclaw.json

    # Idempotent: set or update the alignlayer entry.
    # Uses python3 — jq is not available on the pi host.
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
  ssh pi "cd /opt/openclaw && docker compose restart"

  echo "→ OpenClaw: waiting for gateway to come up..."
  sleep 5

  echo "→ OpenClaw: checking hook registration..."
  ssh pi "docker logs --tail 40 openclaw 2>&1 | grep -i alignlayer || echo '  (no alignlayer log lines yet — check for errors below)'"
  ssh pi "docker logs --tail 20 openclaw 2>&1 | grep -i 'error\|warn\|hook' | tail -10 || true"

  echo "✓ OpenClaw deploy complete. Traces → /opt/openclaw/data/traces/"
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
