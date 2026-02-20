# OpenClaw — Minimal PR for Blocking Approval Workflow

## Problem

The AlignLayer hook can observe and score every tool call via the agent-events bus, but it cannot
block execution. When a high-risk action is detected, we can log it — we cannot stop it.

The exec-approvals mechanism already exists (ExecApprovalManager, `exec.approval.requested`
broadcast, `/approve` UX), but it only fires for `exec`/`bash` tools when the security policy
is configured to ask. There is no plugin hook that fires between "approval registered" and
"awaiting decision."

## Goal

Add a plugin hook that fires after an exec approval request is registered, giving AlignLayer a
chance to (a) be notified, (b) surface its own interrupt to the user, and (c) resolve the
approval decision via the gateway WS protocol.

## Chosen Approach: `exec_approval_requested` notification hook

This is Option 1 (lower-risk, non-blocking hook) from the architecture analysis.

A plugin subscribes to `exec_approval_requested`. When it fires:
1. Plugin receives the approval ID and full exec context
2. Plugin calls `exec.approval.resolve` on the gateway WS (it already has `operator.approvals`
   scope if configured) with `allow-once | allow-always | deny`
3. The in-flight approval promise in the gateway resolves and the agent continues

The hook is fire-and-forget from the gateway's perspective. The gateway does not wait on it.
The plugin drives resolution via the existing gateway WS protocol — no new IPC surface needed.

**Why not Option 2 (defer/resolve callback)?**
A synchronous `defer` hook that holds the agent's tool execution loop is fragile — if the plugin
hangs, the agent hangs. The notification approach has zero risk to gateway stability.

---

## Files to Change

```
src/
├── plugins/
│   ├── types.ts          (+) new hook event type + result type
│   └── hooks.ts          (+) wire into plugin runner
└── bash-tools/
    ├── exec-host-gateway.ts   (+) fire hook after approval registered
    └── exec-host-node.ts      (+) fire hook after approval registered
```

Total: ~40 lines of new code across 4 files. No existing behaviour changes.

---

## Diff Sketch

### `plugins/types.ts`

```typescript
// Add after existing hook types:

export interface PluginHookExecApprovalRequestedEvent {
  /** Approval ID — pass to exec.approval.resolve to unblock the pending decision. */
  id: string;
  command: string;
  cwd: string | null;
  /** "gateway" = exec runs in container; "node" = exec runs on paired companion */
  host: "gateway" | "node";
  agentId: string | null;
  sessionKey: string | null;
  /** Unix ms — after this the gateway auto-expires the approval */
  expiresAtMs: number;
}

// Add to PluginHooks interface:
export interface PluginHooks {
  // ... existing hooks ...
  exec_approval_requested?: (
    event: PluginHookExecApprovalRequestedEvent
  ) => void | Promise<void>;
}
```

### `plugins/hooks.ts`

```typescript
// Add to the hook runner (mirrors pattern of existing before_tool_call dispatch):

export async function runExecApprovalRequestedHook(
  plugins: LoadedPlugin[],
  event: PluginHookExecApprovalRequestedEvent
): Promise<void> {
  for (const plugin of plugins) {
    if (typeof plugin.hooks?.exec_approval_requested === "function") {
      try {
        await plugin.hooks.exec_approval_requested(event);
      } catch (err) {
        console.error(`[openclaw] plugin ${plugin.id} exec_approval_requested error:`, err);
      }
    }
  }
}
```

### `bash-tools/exec-host-gateway.ts`

```typescript
// In the path where requestExecApprovalDecision() is called:

// Existing:
const decision = await requestExecApprovalDecision({ id, command, cwd, ... });

// Add before the await:
void runExecApprovalRequestedHook(loadedPlugins, {
  id,
  command,
  cwd: cwd ?? null,
  host: "gateway",
  agentId: agentId ?? null,
  sessionKey: sessionKey ?? null,
  expiresAtMs: Date.now() + APPROVAL_TIMEOUT_MS,
});
// Then continue to await decision as before.
```

### `bash-tools/exec-host-node.ts`

```typescript
// Same pattern — fire after approval ID is registered, before awaiting decision.
void runExecApprovalRequestedHook(loadedPlugins, {
  id,
  command,
  cwd: cwd ?? null,
  host: "node",
  agentId: agentId ?? null,
  sessionKey: sessionKey ?? null,
  expiresAtMs: Date.now() + APPROVAL_TIMEOUT_MS,
});
```

---

## AlignLayer Integration (post-PR)

Once this hook exists, `src/openclaw-plugin/handler.ts` registers for it at gateway startup:

```typescript
// In the gateway:startup handler, after subscribing to onAgentEvent:

plugin.hooks.exec_approval_requested = async (event) => {
  const { id, command, cwd, agentId, sessionKey } = event;

  // Re-score using the actual command string (richer signal than tool name alone)
  const riskScore = score("exec", { command }, 0);

  // Write interrupt trace
  appendTrace({ ..., decision: "interrupt", ... });

  // Surface interrupt to user via gateway WS (requires operator.approvals scope)
  // Implementation: use the existing gateway WS client to call exec.approval.resolve
  // Gateway WS message: { type: "exec.approval.resolve", id, decision: "deny" | "allow-once" }

  // For now (Phase 0.2): log the pending interrupt. Human resolves via /approve as usual.
  // Phase 3: ML scorer drives the resolution automatically when confidence is high.
};
```

---

## What This Unlocks

| Capability | Before PR | After PR |
|---|---|---|
| See exec tool calls | Yes (agent-events bus) | Yes |
| Score risk | Yes | Yes |
| Block exec calls | No | Yes (resolve via gateway WS) |
| Block non-exec tools | No | No (before_tool_call plugin hook handles these) |
| Surface custom interrupt UX | No | Yes (plugin fires before user sees /approve prompt) |

**Non-exec tools** (read, write, web_search, etc.) are already interceptable via the existing
`before_tool_call` plugin hook with `block: true`. The PR closes the gap specifically for exec
tool calls where the approval path bypasses `before_tool_call`.

---

## Submission Notes

- Target: `openclaw` core repo (not open-source; submit as PR to owner)
- Zero breaking changes — the hook is additive, no existing call sites change
- Safe to land behind a feature flag (`plugins.enableExecApprovalHook: true`) if preferred
- Pairs naturally with the AlignLayer plugin as a reference implementation
