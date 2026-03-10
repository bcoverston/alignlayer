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

// ---------------------------------------------------------------------------
// ML scoring server (async enrichment — advisory, not blocking)
// ---------------------------------------------------------------------------

const ML_SERVER_URL = process.env["ALIGNLAYER_ML_URL"] ?? "http://localhost:8000";
const ML_TIMEOUT_MS = 200;

interface MlScoreResult {
  command: string;
  risk: number;
  tier: number;
  decision: string;
}

async function fetchMlScore(command: string): Promise<MlScoreResult | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ML_TIMEOUT_MS);
  try {
    const res = await fetch(`${ML_SERVER_URL}/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
      signal: controller.signal,
    });
    if (!res.ok) return null;
    return (await res.json()) as MlScoreResult;
  } catch {
    return null;  // timeout or connection refused — ML is advisory
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Scorer (inlined — OpenClaw's hook runtime does not resolve relative imports)
// Keep in sync with scorer.ts
// ---------------------------------------------------------------------------

const IRREVERSIBILITY_TOKENS = [
  "push", "send", "deploy", "drop", "delete", "rm", "truncate",
  "overwrite", "destroy", "nuke", "reset", "purge", "wipe",
  "revoke", "terminate", "kill",
];

const BOUNDARY_TOKENS = [
  "curl", "wget", "fetch", "http://", "https://", "upload",
  "email", "smtp", "webhook", "s3://", "gs://", "azure",
];

const EXEC_TOOLS = new Set(["exec", "bash", "shell", "run", "computer"]);

const FORCE_FLAGS     = new Set(["-f", "--force", "--hard", "--no-backup", "--overwrite", "--delete"]);
const RECURSIVE_FLAGS = new Set(["-r", "-R", "--recursive", "--all", "-A", "--all-namespaces"]);
const DRY_RUN_FLAGS   = new Set(["--dry-run", "-n", "--simulate", "--check", "--preview", "--no-act"]);
const INTERACTIVE_FLAGS = new Set(["-i", "--interactive", "--confirm", "--prompt"]);

const RISK_THRESHOLD = 0.55;
const MAX_CONFIDENT_CALLS = 6;

function expandFlag(f: string): string[] {
  if (f.startsWith("-") && !f.startsWith("--") && f.length > 2)
    return f.slice(1).split("").map((c) => `-${c}`);
  return [f];
}
function normalizeFlag(f: string): string { const eq = f.indexOf("="); return eq === -1 ? f : f.slice(0, eq); }
function tokenizeCommand(cmd: string): { command: string; subcommand: string; flags: string[] } {
  const parts = cmd.trim().split(/\s+/).filter(Boolean);
  const command = (parts[0] ?? "").toLowerCase();
  const rest = parts.slice(1);
  const flags = rest.filter((p) => p.startsWith("-")).flatMap(expandFlag).map(normalizeFlag);
  const positional = rest.filter((p) => !p.startsWith("-"));
  return { command, subcommand: (positional[0] ?? "").toLowerCase(), flags };
}
function flagModifier(flags: string[]): number {
  let mod = 0;
  if (flags.some((f) => FORCE_FLAGS.has(f)))       mod += 0.2;
  if (flags.some((f) => RECURSIVE_FLAGS.has(f)))   mod += 0.1;
  if (flags.some((f) => DRY_RUN_FLAGS.has(f)))     mod -= 0.4;
  if (flags.some((f) => INTERACTIVE_FLAGS.has(f))) mod -= 0.2;
  return mod;
}

function blastRadius(toolName: string, args: Record<string, unknown>): number {
  const name = toolName.toLowerCase();
  const cmdStr = EXEC_TOOLS.has(name)
    ? String(args["command"] ?? args["cmd"] ?? args["input"] ?? "")
    : "";
  let s = 0;
  if (EXEC_TOOLS.has(name)) s += 0.25;
  const EXFIL_EXEC_RE = /eval\s+"\$\(curl|curl\s+.*\|\s*(ba)?sh|wget\s+.*\|\s*(ba)?sh|curl\s+.*\|\s*python|base64\s+-d\s*\|\s*(ba)?sh|\/dev\/tcp\//i;
  if (cmdStr) {
    // Exfil / remote code execution floor
    if (EXFIL_EXEC_RE.test(cmdStr)) s = Math.max(s, 0.95);
    const segments = cmdStr.split(/&&|\|\||;/).map((seg) => seg.trim()).filter(Boolean);
    let maxIrr = 0;
    for (const seg of segments) {
      const { command, subcommand, flags } = tokenizeCommand(seg);

      // Dry-run cap: segments with dry-run flags contribute nothing.
      if (flags.some((f) => DRY_RUN_FLAGS.has(f))) continue;
      if (/(?:^|\s)terraform\s+plan(?:\s|$)/.test(seg) ||
          /(?:^|\s)make\s+-[a-zA-Z]*n[a-zA-Z]*(?:\s|$)/.test(seg)) continue;

      if (IRREVERSIBILITY_TOKENS.some((t) => `${command} ${subcommand}`.includes(t)))
        maxIrr = Math.max(maxIrr, 0.5 + flagModifier(flags));
    }
    s += maxIrr;
  } else {
    if (IRREVERSIBILITY_TOKENS.some((t) => `${name} ${JSON.stringify(args)}`.toLowerCase().includes(t))) s += 0.5;
  }
  if (BOUNDARY_TOKENS.some((t) => `${name} ${cmdStr} ${JSON.stringify(args)}`.toLowerCase().includes(t))) s += 0.25;
  return Math.max(0, Math.min(1.0, s));
}

function planConfidence(toolCallIndexInRun: number): number {
  return Math.max(0.0, 1.0 - toolCallIndexInRun / MAX_CONFIDENT_CALLS);
}

function score(
  toolName: string,
  args: Record<string, unknown>,
  toolCallIndexInRun: number
): { blastRadius: number; planConfidence: number; risk: number; decision: "allow" | "interrupt" } {
  const br = blastRadius(toolName, args);
  const pc = planConfidence(toolCallIndexInRun);
  const rawRisk = br * (1 - pc);
  const risk = br >= 0.75 ? Math.max(0.60, rawRisk) : rawRisk;
  return { blastRadius: br, planConfidence: pc, risk, decision: risk >= RISK_THRESHOLD ? "interrupt" : "allow" };
}

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
  /** ML server enrichment (async, advisory). Null if ML server unavailable. */
  ml_risk?: number | null;
  ml_tier?: number | null;
  ml_decision?: string | null;
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

  // Resolve relative to the gateway's own entry point (mirrors mission-control approach).
  const candidates: string[] = [];
  const mainPath = process.argv[1];
  if (mainPath) {
    const mainDir = path.dirname(mainPath);
    candidates.push(path.join(mainDir, "infra", "agent-events.js"));
    candidates.push(path.join(mainDir, "..", "dist", "infra", "agent-events.js"));
  }
  candidates.push(
    "/usr/local/lib/node_modules/openclaw/dist/infra/agent-events.js",
    "/opt/homebrew/lib/node_modules/openclaw/dist/infra/agent-events.js",
    `${process.env["HOME"] ?? ""}/.npm-global/lib/node_modules/openclaw/dist/infra/agent-events.js`
  );

  for (const p of candidates) {
    try {
      if (!fs.existsSync(p)) continue;
      const mod = (await import(`file://${p}`)) as { onAgentEvent?: OnAgentEvent };
      if (mod.onAgentEvent) return mod.onAgentEvent;
    } catch {
      // try next
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
    args: pending?.args ?? {},
    risk_score: pending?.risk_score ?? null,
    blast_radius: pending?.blast_radius ?? null,
    plan_confidence: pending?.plan_confidence ?? null,
    decision: pending?.decision ?? null,
    human_outcome: null,
    ml_risk: null,
    ml_tier: null,
    ml_decision: null,
  };

  pendingCalls.delete(toolCallId);

  // Extract command for ML enrichment (exec tools only).
  const args = pending?.args ?? {};
  const cmdStr = String(args["command"] ?? args["cmd"] ?? args["input"] ?? "");

  if (cmdStr) {
    // Fire-and-forget: ML enrichment is advisory, not blocking.
    fetchMlScore(cmdStr).then((ml) => {
      if (ml) {
        entry.ml_risk = ml.risk;
        entry.ml_tier = ml.tier;
        entry.ml_decision = ml.decision;
      }
      appendTrace(entry);
    }).catch(() => {
      appendTrace(entry);
    });
  } else {
    appendTrace(entry);
  }

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
      console.error("[alignlayer] could not resolve agent-events bus — tool interception disabled");
      return;
    }

    console.log("[alignlayer] subscribed to agent-events bus");
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
