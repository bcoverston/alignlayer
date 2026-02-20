# OpenClaw PR — `exec_approval_requested` plugin hook

## Problem

AlignLayer can score every non-exec tool call via `before_tool_call` (and block with `{ block: true }`).
Exec tool calls bypass this path: they go through `processGatewayAllowlist` /
`executeNodeHostCommand`, which register an approval and fire-and-forget `requestExecApprovalDecision`.
By the time any plugin sees the result, the decision is already made.

There is no hook point between "approval registered" and "awaiting human decision" where a plugin
can inject its own scoring, enrich the approval prompt, or auto-resolve based on policy.

## Solution

Add an `exec_approval_requested` void hook that fires immediately after the approval ID is
registered, before the gateway awaits the human decision. The hook receives the approval ID and
full exec context. A plugin can then call `exec.approval.resolve` on the gateway WS to
auto-resolve — or simply observe and let the human decide as usual.

This is purely additive. No existing behavior changes. The hook is fire-and-forget from the
gateway's perspective (same pattern as `after_tool_call`).

---

## Diff

### `src/plugins/types.ts`

Add after the last existing hook event/result block (before `PluginHookHandlerMap`):

```typescript
// exec_approval_requested hook
export type PluginHookExecApprovalRequestedEvent = {
  /** Approval ID — pass to exec.approval.resolve to auto-resolve the pending decision. */
  id: string;
  command: string;
  cwd: string | null;
  /** "gateway" = exec runs in container; "node" = exec runs on paired companion */
  host: "gateway" | "node";
  agentId: string | null;
  sessionKey: string | null;
  /** Unix ms — after this the gateway auto-expires the approval request */
  expiresAtMs: number;
};
```

In `PluginHookName`:

```diff
   | "session_start"
   | "session_end"
   | "gateway_start"
-  | "gateway_stop";
+  | "gateway_stop"
+  | "exec_approval_requested";
```

In `PluginHookHandlerMap` (after `gateway_stop` entry):

```typescript
  exec_approval_requested: (
    event: PluginHookExecApprovalRequestedEvent,
    ctx: PluginHookGatewayContext,
  ) => Promise<void> | void;
```

---

### `src/plugins/hooks.ts`

Add after `runGatewayStop`, before the return object:

```typescript
  /**
   * Run exec_approval_requested hook.
   * Fires after an exec approval is registered, before the human decides.
   * Plugins may call exec.approval.resolve to auto-resolve.
   * Runs in parallel (fire-and-forget).
   */
  async function runExecApprovalRequested(
    event: PluginHookExecApprovalRequestedEvent,
    ctx: PluginHookGatewayContext,
  ): Promise<void> {
    return runVoidHook("exec_approval_requested", event, ctx);
  }
```

In the return object:

```diff
     // Gateway hooks
     runGatewayStart,
     runGatewayStop,
+    runExecApprovalRequested,
     // Utility
```

Also add to the import at the top:

```diff
-import type { ... PluginHookGatewayStopEvent ... } from "./types.js";
+import type { ... PluginHookGatewayStopEvent, PluginHookExecApprovalRequestedEvent ... } from "./types.js";
```

---

### `src/agents/bash-tools.exec-host-gateway.ts`

In `processGatewayAllowlist`, inside the `if (requiresAsk)` block, immediately after
`expiresAtMs` is defined and before the `void (async () => {` IIFE:

```diff
+  // Notify plugins — they may observe or auto-resolve via exec.approval.resolve.
+  void hooks.runExecApprovalRequested(
+    {
+      id: approvalId,
+      command: params.command,
+      cwd: params.workdir ?? null,
+      host: "gateway",
+      agentId: params.agentId ?? null,
+      sessionKey: params.sessionKey ?? null,
+      expiresAtMs,
+    },
+    gatewayCtx,
+  );
+
   void (async () => {
     let decision: string | null = null;
```

`hooks` and `gatewayCtx` are already available in the call site (same pattern used for
`before_tool_call`). Check the exact parameter names against the call site — `gatewayCtx`
may be named `ctx` or constructed inline.

---

### `src/agents/bash-tools.exec-host-node.ts`

In `executeNodeHostCommand`, inside the `if (requiresAsk)` block, same position:

```diff
+  void hooks.runExecApprovalRequested(
+    {
+      id: approvalId,
+      command: params.command,
+      cwd: params.workdir ?? null,
+      host: "node",
+      agentId: params.agentId ?? null,
+      sessionKey: params.sessionKey ?? null,
+      expiresAtMs,
+    },
+    gatewayCtx,
+  );
+
   void (async () => {
     let decision: string | null = null;
```

---

## Size

~45 lines of new code across 4 files. No deletions. No behavior changes to existing paths.

---

## AlignLayer integration (post-merge)

Once the hook exists, `src/openclaw-plugin/index.ts` registers:

```typescript
api.on("exec_approval_requested", async (event, ctx) => {
  const { id, command, agentId, sessionKey, expiresAtMs } = event;

  // Score using actual command string (richer than tool name alone)
  const br = blastRadius("exec", { command });
  const pc = planConfidence(nextCallIndex());
  const risk = br * (1 - pc);

  appendTrace({
    runtime: "openclaw",
    session_id: sessionKey ?? "",
    turn_id: "",
    timestamp: new Date().toISOString(),
    event: "before_tool_call",
    tool: "exec",
    args: { command },
    risk_score: risk,
    blast_radius: br,
    plan_confidence: pc,
    decision: risk >= RISK_THRESHOLD ? "interrupt" : "allow",
    human_outcome: null,
  });

  // Phase 0.2: observe only — human resolves via /approve as usual.
  // Phase 3: call exec.approval.resolve when ML confidence is high.
  // gateway WS: { type: "exec.approval.resolve", id, decision: "allow-once" | "deny" }
});
```

---

## What this closes

| Capability                        | Today                    | Post-PR               |
|-----------------------------------|--------------------------|-----------------------|
| Score non-exec tool calls         | ✓ `before_tool_call`     | ✓                     |
| Block non-exec tool calls         | ✓ `{ block: true }`      | ✓                     |
| Observe exec tool calls           | ✗ bus broken             | ✓ `exec_approval_requested` |
| Auto-resolve exec approvals       | ✗                        | ✓ via `exec.approval.resolve` |
| Enrich approval prompt            | ✗                        | ✓ (plugin fires before user sees prompt) |

---

## Submission notes

- Zero breaking changes — purely additive hook
- Safe behind feature flag (`plugins.enableExecApprovalHook: true`) if preferred
- AlignLayer plugin is the reference implementation
- The `exec.approval.resolve` WS message format already exists — no new gateway protocol needed
