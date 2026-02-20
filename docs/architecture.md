# AlignLayer — Architecture

## Overview

AlignLayer is a runtime preference alignment layer for agentic systems. It sits between an agent's
decision loop and execution, scoring proposed actions before they run.

```
Agent decision loop
        ↓
   before_tool_call   ← AlignLayer scores proposed action
        ↓
   risk = blast_radius × (1 − plan_confidence)
        ↓
   risk < threshold   → allow (trace logged)
   risk ≥ threshold   → interrupt (trace + pending-interrupts log)
        ↓
   after_tool_call    ← capture outcome
        ↓
   human correction?  ← training signal (Phase 1+)
```

## Risk Model

Risk is two-dimensional. The axes are independent — high confidence in a plan does not make a
destructive action safe, but it does reduce the probability that the action is a mistake.

### Blast Radius

Rule-based. Derived from tool name and arguments.

**Signal categories:**

| Category | Tokens | Contribution |
|---|---|---|
| Exec surface | `exec`, `bash`, `shell`, `run`, `computer` | +0.25 |
| Irreversibility | `push`, `send`, `deploy`, `drop`, `delete`, `rm`, `truncate`, `overwrite`, `destroy`, `nuke`, `reset`, `purge`, `wipe`, `revoke`, `terminate`, `kill` | +0.50 |
| Boundary crossing | `curl`, `wget`, `fetch`, `http://`, `https://`, `upload`, `email`, `smtp`, `webhook`, `s3://`, `gs://`, `azure` | +0.25 |

For exec-family tools, the command string (`args.command`, `args.cmd`, `args.input`) is included
in the token search alongside the tool name and serialised args.

Score is additive, capped at 1.0.

**Examples:**

| Tool + Args | blast_radius |
|---|---|
| `read { path: "README.md" }` | 0.0 |
| `bash { command: "ls -la" }` | 0.25 |
| `bash { command: "git push origin main" }` | 0.75 |
| `bash { command: "curl https://api.example.com/data" }` | 0.50 |
| `bash { command: "rm -rf /tmp/work" }` | 0.75 |

### Plan Confidence

Positional proxy. Within a run (one agent invocation), early tool calls are more likely to be
executing a known plan. Late calls may reflect plan drift.

```
plan_confidence = max(0.0, 1.0 − tool_call_index / MAX_CONFIDENT_CALLS)
```

`MAX_CONFIDENT_CALLS = 6` — confidence reaches 0 on the 6th call.

This is a blunt instrument. Phase 3 replaces it with a learned signal.

### Combined Score

```
risk = blast_radius × (1 − plan_confidence)
```

Threshold: `0.55`. Below → allow. At or above → interrupt.

**Calibration note**: the multiplicative formula means high-blast-radius actions on the first
call of a run score 0.0 (plan_confidence = 1.0). This is intentional for Phase 0 — we're
collecting data, not being maximally conservative. Adjust `RISK_THRESHOLD` or the formula
for stricter enforcement.

## Trace Schema

Every intercepted tool call produces two JSONL entries: `before_tool_call` and `after_tool_call`.

```jsonc
{
  "session_id": "<agentId>:<sessionId>",  // from agent:bootstrap
  "turn_id": "<runId>",                   // agent run UUID
  "timestamp": "2026-02-19T12:00:00.000Z",
  "event": "before_tool_call",
  "tool": "bash",
  "args": { "command": "git push origin main" },
  "risk_score": 0.75,
  "blast_radius": 0.75,
  "plan_confidence": 0.0,
  "decision": "interrupt",
  "human_outcome": null                   // populated in Phase 1
}
```

`human_outcome` values: `"approved"` | `"denied"` | `"allow-always"` | `null`

Trace files: `$ALIGNLAYER_TRACES_DIR/alignlayer-YYYY-MM-DD.jsonl`
Interrupt queue: `$ALIGNLAYER_TRACES_DIR/pending-interrupts.jsonl`

## OpenClaw Integration

### Hook vs Plugin

AlignLayer is implemented as an OpenClaw **hook** (deployed to `data/hooks/alignlayer/`), not a
plugin. Hooks can subscribe to gateway lifecycle events; plugins expose commands and tools.

### Deployment Path

```
pi: /opt/openclaw/data/hooks/alignlayer/   (maps to /home/node/.openclaw/hooks/alignlayer/)
```

### Hook Events

| Event | Purpose |
|---|---|
| `gateway:startup` | Subscribe to agent-events bus |
| `agent:bootstrap` | Capture `agentId` + `sessionId` for trace correlation |

### Agent-Events Bus

Tool calls are delivered via an internal event bus, not the `HookEvent` system.

```typescript
// Stream: "tool", phase: "start"  — before execution
{ name, phase: "start", toolCallId, args }

// Stream: "tool", phase: "result" — after execution
{ name, phase: "result", toolCallId, result, isError? }

// Stream: "lifecycle", phase: "start"/"end"/"error"
{ phase, messageChannel? }
```

**Known issue**: the hook runner may load `agent-events.js` in a separate module context from the
gateway, causing `onAgentEvent` listeners to register against a dead emitter. `handler.ts` checks
`globalThis.__openclawAgentEvents` first (reliable when it works), then falls back to dynamic
imports of known install paths.

### Approval Integration (Phase 1)

The exec-approvals socket at `/home/node/.openclaw/exec-approvals.sock` is the mechanism for
runtime tool blocking. Protocol is undocumented; integration is deferred to Phase 1.

Phase 0 writes interrupt annotations to `pending-interrupts.jsonl` and logs to stderr.

## Roadmap

| Phase | Deliverable |
|---|---|
| 0 (current) | Heuristic scorer + trace capture. No ML. |
| 1 | Pair generation from traces. Synthetic corpus bootstrap. |
| 2 | Siamese network trained on pairs (contrastive loss). |
| 3 | ML-backed scorer replaces heuristics. Approval socket integration. |
| 4 | Triplet loss. Per-user preference adaptation. |
| 5 | Claude Code + LangChain/CrewAI adapters. |
