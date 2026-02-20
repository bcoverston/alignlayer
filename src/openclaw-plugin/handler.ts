/**
 * AlignLayer — OpenClaw hook handler.
 *
 * Lifecycle:
 *   gateway:startup    → subscribe to the agent-events bus
 *   agent:bootstrap    → capture session context
 *   (agent-events bus) → intercept tool calls, score risk, write traces
 *
 * Note on blocking:
 *   The agent-events bus is observational — there is no native intercept
 *   mechanism that can synchronously block a tool call from a hook.
 *   Phase 0 delivers: scoring + JSONL trace capture + interrupt annotations.
 *   Hard blocking via the exec-approvals socket is Phase 1.
 */

import fs from "fs";
import path from "path";
import { score } from "./scorer.js";

// ---------------------------------------------------------------------------
// OpenClaw types (inferred from runtime; no published @types/openclaw package)
// ---------------------------------------------------------------------------

interface HookEvent {
  type: string;
  action: string;
  sessionKey: string;
  timestamp: Date;
  /** Push strings here to surface messages to the user in the active channel. */
  messages: string[];
  context: {
    workspaceDir?: string;
    cfg?: Record<string, unknown>;
    agentId?: string;
    sessionId?: string;
    [key: string]: unknown;
  };
}

interface AgentEventPayload {
  runId: string;
  seq: number;
  /** "lifecycle" | "tool" | "assistant" | "exec" */
  stream: string;
  /** Unix milliseconds */
  ts: number;
  data: Record<string, unknown>;
  sessionKey?: string;
}

// ---------------------------------------------------------------------------
// Trace schema
// ---------------------------------------------------------------------------

interface TraceEntry {
  session_id: string;
  turn_id: string;
  timestamp: string;
  event: "before_tool_call" | "after_tool_call";
  tool: string;
  args: Record<string, unknown>;
  risk_score: number | null;
  blast_radius: number | null;
  plan_confidence: number | null;
  decision: "allow" | "interrupt" | null;
  /** Populated when a human acts on an interrupt. Null until Phase 1 approval integration. */
  human_outcome: "approved" | "denied" | "allow-always" | null;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface RunState {
  toolCallCount: number;
}

const sessionContext = { agentId: "", sessionId: "" };
const runs = new Map<string, RunState>();

// toolCallId → pending before-trace awaiting result annotation
const pendingCalls = new Map<string, TraceEntry>();

// ---------------------------------------------------------------------------
// Trace I/O
// ---------------------------------------------------------------------------

function tracesDir(): string {
  return (
    process.env["ALIGNLAYER_TRACES_DIR"] ?? "/home/node/.openclaw/traces"
  );
}

function appendTrace(entry: TraceEntry): void {
  const dir = tracesDir();
  fs.mkdirSync(dir, { recursive: true });
  const date = new Date().toISOString().slice(0, 10);
  const file = path.join(dir, `alignlayer-${date}.jsonl`);
  fs.appendFileSync(file, JSON.stringify(entry) + "\n", "utf8");
}

// ---------------------------------------------------------------------------
// Agent-events bus resolution
//
// Known issue: OpenClaw's hook runner may load agent-events.js in a separate
// module context from the gateway, causing onAgentEvent to register against a
// dead emitter. The globalThis check is the reliable path when it works;
// dynamic imports are a fallback that may silently no-op.
// See: data/hooks/mission-control README (known isolation bug).
// ---------------------------------------------------------------------------

type OnAgentEvent = (cb: (event: AgentEventPayload) => void) => void;

async function resolveOnAgentEvent(): Promise<OnAgentEvent | null> {
  const g = globalThis as Record<string, unknown>;
  const globalBus = g["__openclawAgentEvents"] as
    | { onAgentEvent?: OnAgentEvent }
    | undefined;
  if (globalBus?.onAgentEvent) return globalBus.onAgentEvent;

  const candidates = [
    "/usr/local/lib/node_modules/openclaw/dist/infra/agent-events.js",
    "/opt/homebrew/lib/node_modules/openclaw/dist/infra/agent-events.js",
    `${process.env["HOME"] ?? ""}/.npm-global/lib/node_modules/openclaw/dist/infra/agent-events.js`,
  ];

  for (const p of candidates) {
    try {
      const mod = (await import(p)) as { onAgentEvent?: OnAgentEvent };
      if (mod.onAgentEvent) return mod.onAgentEvent;
    } catch {
      // path not present — try next
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Event handling
// ---------------------------------------------------------------------------

function handleToolStart(payload: AgentEventPayload): void {
  const { runId, ts, sessionKey, data } = payload;

  const run = runs.get(runId) ?? { toolCallCount: 0 };
  runs.set(runId, run);

  const toolName = String(data["name"] ?? "unknown");
  const toolCallId = String(data["toolCallId"] ?? "");
  const args = (data["args"] as Record<string, unknown>) ?? {};

  const riskScore = score(toolName, args, run.toolCallCount);
  run.toolCallCount++;

  const sessionId =
    sessionContext.sessionId ||
    (sessionKey ? sessionKey.split(":")[1] ?? "" : "");
  const agentId =
    sessionContext.agentId ||
    (sessionKey ? sessionKey.split(":")[0] ?? "" : "");

  const entry: TraceEntry = {
    session_id: `${agentId}:${sessionId}`,
    turn_id: runId,
    timestamp: new Date(ts).toISOString(),
    event: "before_tool_call",
    tool: toolName,
    args,
    risk_score: riskScore.risk,
    blast_radius: riskScore.blastRadius,
    plan_confidence: riskScore.planConfidence,
    decision: riskScore.decision,
    human_outcome: null,
  };

  if (toolCallId) pendingCalls.set(toolCallId, entry);
  appendTrace(entry);

  if (riskScore.decision === "interrupt") {
    const interruptFile = path.join(tracesDir(), "pending-interrupts.jsonl");
    fs.appendFileSync(interruptFile, JSON.stringify(entry) + "\n", "utf8");

    // Surface a message in the active channel if possible.
    // (This path only runs if we're inside a HookEvent context — we're not,
    // since we're on the agent-events bus. Logged to file only for Phase 0.)
    console.warn(
      `[alignlayer] INTERRUPT — ${toolName} risk=${riskScore.risk.toFixed(2)} ` +
        `(blast=${riskScore.blastRadius.toFixed(2)}, confidence=${riskScore.planConfidence.toFixed(2)})`
    );
  }
}

function handleToolResult(payload: AgentEventPayload): void {
  const { ts, data } = payload;

  const toolCallId = String(data["toolCallId"] ?? "");
  const toolName = String(data["name"] ?? "unknown");
  const isError = Boolean(data["isError"]);

  const sessionId = sessionContext.sessionId;
  const agentId = sessionContext.agentId;

  const pending = pendingCalls.get(toolCallId);
  const entry: TraceEntry = {
    session_id: `${agentId}:${sessionId}`,
    turn_id: payload.runId,
    timestamp: new Date(ts).toISOString(),
    event: "after_tool_call",
    tool: toolName,
    // Carry forward args + risk scores from the paired before-trace if available.
    args: pending?.args ?? {},
    risk_score: pending?.risk_score ?? null,
    blast_radius: pending?.blast_radius ?? null,
    plan_confidence: pending?.plan_confidence ?? null,
    decision: pending?.decision ?? null,
    human_outcome: null,
  };

  pendingCalls.delete(toolCallId);
  appendTrace(entry);

  if (isError) {
    console.warn(`[alignlayer] tool error — ${toolName}`);
  }
}

function handleLifecycle(payload: AgentEventPayload): void {
  const { runId, data } = payload;

  if (data["phase"] === "start") {
    runs.set(runId, { toolCallCount: 0 });
  } else if (data["phase"] === "end" || data["phase"] === "error") {
    // Keep run state briefly for any late-arriving tool results, then clean up.
    setTimeout(() => runs.delete(runId), 5_000);
  }
}

// ---------------------------------------------------------------------------
// Hook entry point
// ---------------------------------------------------------------------------

export default async function handler(event: HookEvent): Promise<void> {
  const { type, action, context } = event;

  if (type === "agent" && action === "bootstrap") {
    if (context.agentId) sessionContext.agentId = context.agentId;
    if (context.sessionId) sessionContext.sessionId = context.sessionId;
    return;
  }

  if (type === "gateway" && action === "startup") {
    const onAgentEvent = await resolveOnAgentEvent();

    if (!onAgentEvent) {
      event.messages.push(
        "[alignlayer] WARNING: could not resolve agent-events bus — tool interception disabled. " +
          "Check OpenClaw install path."
      );
      return;
    }

    onAgentEvent((payload: AgentEventPayload) => {
      try {
        if (payload.stream === "lifecycle") {
          handleLifecycle(payload);
        } else if (payload.stream === "tool") {
          const phase = payload.data["phase"];
          if (phase === "start") handleToolStart(payload);
          else if (phase === "result") handleToolResult(payload);
        }
      } catch (err) {
        console.error("[alignlayer] handler error:", err);
      }
    });

    event.messages.push(
      `[alignlayer] active — traces → ${tracesDir()}`
    );
  }
}
