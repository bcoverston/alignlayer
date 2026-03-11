---
name: alignlayer
description: "Runtime preference alignment — scores tool calls, surfaces interrupts, captures training traces"
homepage: https://github.com/bcoverston/alignlayer
metadata:
  openclaw:
    emoji: "🎯"
    events:
      - "gateway:startup"
      - "agent:bootstrap"
    requires:
      config: []
---

# AlignLayer Hook

Intercepts agent tool calls via the agent-events bus, scores proposed actions on two heuristic axes
(blast radius × plan confidence), logs interrupts for high-risk actions, and captures before/after
traces as JSONL for future ML training.

## How It Works

Tool-level events are not delivered via the `HookEvent` system. This hook subscribes to the
`onAgentEvent()` bus at `gateway:startup` and listens for `stream: "tool"` events.

At `agent:bootstrap` it captures `agentId` and `sessionId` for trace correlation.

## Installation

```sh
cp -r alignlayer/src/openclaw-plugin /opt/openclaw/data/hooks/alignlayer
openclaw hooks enable alignlayer
```

Restart the gateway to apply.

## Configuration

Set `ALIGNLAYER_TRACES_DIR` in the hook's `env` block (openclaw.json) to override the default
trace output path (`/home/node/.openclaw/traces`).

```json
"hooks": {
  "internal": {
    "entries": {
      "alignlayer": {
        "enabled": true,
        "env": {
          "ALIGNLAYER_TRACES_DIR": "/home/node/.openclaw/traces"
        }
      }
    }
  }
}
```

## Trace Output

Traces are written as JSONL to `$ALIGNLAYER_TRACES_DIR/alignlayer-YYYY-MM-DD.jsonl`.

High-risk tool calls (decision: `interrupt`) are also appended to `pending-interrupts.jsonl`
in the same directory and logged to stderr.

## Known Limitations (Phase 0)

- **No hard blocking**: the agent-events bus is observational. Actual tool call blocking requires
  integration with the exec-approvals socket (Phase 1).
- **Module isolation bug**: if `onAgentEvent` cannot be resolved, a warning is surfaced and
  tool interception is disabled for that session. Check `/opt/openclaw` install paths in handler.ts.
- **human_outcome always null**: approval annotation requires Phase 1 socket integration.
