# AlignLayer — OpenClaw Integration Guide

This guide is written for an agent performing the installation. Follow each step in order.

---

## What This Does

AlignLayer installs as an OpenClaw hook. Once active it:

1. Intercepts every tool call before execution via the agent-events bus
2. Scores the proposed action using a heuristic risk engine (blast radius × plan confidence)
3. Logs `before_tool_call` and `after_tool_call` traces as JSONL for training data collection
4. Appends high-risk calls to a `pending-interrupts.jsonl` queue and logs to stderr

**Phase 0 limitation**: the hook is observational — it cannot synchronously block a tool call.
High-risk actions are flagged and logged but not prevented. Hard blocking is Phase 1.

---

## Prerequisites

- OpenClaw gateway installed and running
- Node.js 18+ (OpenClaw's runtime)
- `git` available
- Write access to the OpenClaw hooks directory (typically `/opt/openclaw/data/hooks/` or
  `/home/node/.openclaw/hooks/`)

Find the correct hooks path:
```sh
openclaw config get hooksDir 2>/dev/null \
  || ls /opt/openclaw/data/hooks/ 2>/dev/null \
  || ls /home/node/.openclaw/hooks/ 2>/dev/null
```

---

## Installation

### 1. Clone the repo (or pull latest)

```sh
git clone https://github.com/bcoverston/alignlayer.git /tmp/alignlayer
# or if already cloned:
# git -C /path/to/alignlayer pull
```

### 2. Copy the hook to OpenClaw's hooks directory

```sh
# Adjust HOOKS_DIR to match your OpenClaw install
HOOKS_DIR=/opt/openclaw/data/hooks
# or: HOOKS_DIR=/home/node/.openclaw/hooks

cp -r /tmp/alignlayer/src/openclaw-plugin "$HOOKS_DIR/alignlayer"
```

Verify:
```sh
ls "$HOOKS_DIR/alignlayer"
# Expected: HOOK.md  handler.ts  index.ts  openclaw.plugin.json  scorer.ts
```

### 3. Enable the hook

```sh
openclaw hooks enable alignlayer
```

### 4. Restart the gateway

```sh
openclaw restart
# or: systemctl restart openclaw  (if running as a service)
```

### 5. Verify the hook is active

After restart, the startup log should contain:
```
[alignlayer] subscribed to agent-events bus
```

If you see instead:
```
[alignlayer] could not resolve agent-events bus — tool interception disabled
```
see **Troubleshooting** below.

---

## Configuration

Traces are written to `/home/node/.openclaw/traces/` by default.

To override, add an `env` block to the hook entry in `openclaw.json`:

```json
"hooks": {
  "internal": {
    "entries": {
      "alignlayer": {
        "enabled": true,
        "env": {
          "ALIGNLAYER_TRACES_DIR": "/your/preferred/traces/path"
        }
      }
    }
  }
}
```

---

## Verifying Traces

Run an agent session, then check the traces directory:

```sh
TRACES_DIR=/home/node/.openclaw/traces
ls -lh "$TRACES_DIR"
# Expected: alignlayer-YYYY-MM-DD.jsonl

# View the last few trace entries
tail -n 5 "$TRACES_DIR/alignlayer-$(date +%Y-%m-%d).jsonl" | python3 -m json.tool
```

A well-formed trace entry looks like:
```json
{
  "session_id": "agent-abc:session-xyz",
  "turn_id": "run-uuid",
  "timestamp": "2026-02-22T10:00:00.000Z",
  "event": "before_tool_call",
  "tool": "bash",
  "args": { "command": "git status" },
  "risk_score": 0.25,
  "blast_radius": 0.25,
  "plan_confidence": 1.0,
  "decision": "allow",
  "human_outcome": null
}
```

Check for flagged high-risk calls:
```sh
cat "$TRACES_DIR/pending-interrupts.jsonl" 2>/dev/null | python3 -m json.tool
```

---

## Risk Threshold

The default interrupt threshold is `0.55`. Actions scoring at or above this are flagged.

| Risk range | Decision | Typical commands |
|---|---|---|
| 0.0 – 0.24 | allow | `ls`, `git status`, `cat file` |
| 0.25 – 0.54 | allow | `git add`, `npm install`, `curl GET` |
| 0.55+ | interrupt | `git push`, `rm -rf`, `kubectl delete`, `terraform destroy` |

Adjust `RISK_THRESHOLD` in `handler.ts` if the default is too conservative or too permissive
for your use case.

---

## Troubleshooting

### "could not resolve agent-events bus"

The hook runner loaded `agent-events.js` in a separate module context from the gateway.
`handler.ts` tries several resolution paths. To debug:

```sh
# Find where agent-events.js lives in your install
find /opt/openclaw -name "agent-events.js" 2>/dev/null
find /usr/local/lib/node_modules/openclaw -name "agent-events.js" 2>/dev/null
```

If found at a path not in `handler.ts`, add it to the `candidates` array in
`resolveOnAgentEvent()` and reinstall.

### No traces appearing

1. Confirm the hook is enabled: `openclaw hooks list`
2. Confirm the gateway restarted after installation
3. Check gateway stderr for `[alignlayer]` log lines
4. Verify the traces directory is writable by the OpenClaw process

### TypeScript compilation errors

OpenClaw's hook runner transpiles TypeScript internally. If it requires pre-compiled JS:

```sh
cd "$HOOKS_DIR/alignlayer"
npx tsc --target ES2020 --module commonjs --outDir . handler.ts
```

---

## What Gets Collected

Traces are stored locally in `ALIGNLAYER_TRACES_DIR`. They contain:
- The tool name and arguments for every intercepted call
- Risk scores and interrupt decisions
- Session/run correlation IDs

Traces are used as training data for the AlignLayer ML model. They do not leave the machine
unless you explicitly move them. `data/traces/` is gitignored.

---

## Next Steps

Once traces are accumulating:
- Review `pending-interrupts.jsonl` to tune the risk threshold
- Feed traces into the ML training pipeline: `model/pairs.py` → `model/siamese.py`
- Phase 1: wire the exec-approvals socket for actual tool blocking
