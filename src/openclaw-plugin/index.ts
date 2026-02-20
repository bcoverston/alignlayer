/**
 * AlignLayer — OpenClaw plugin.
 *
 * Intercepts every tool call via the plugin before_tool_call hook (in-process,
 * synchronous, can block). Scores risk, writes JSONL traces.
 *
 * Distinct from handler.ts (the lifecycle hook) which attempts agent-events
 * bus subscription — that path is broken by the module isolation bug.
 * This plugin is the reliable data collection path for Phase 0.
 *
 * Deployment: data/extensions/alignlayer/
 * Enable via openclaw.json: plugins.entries.alignlayer.enabled = true
 */

import fs from "fs";
import path from "path";

// ---------------------------------------------------------------------------
// Scorer (inlined — same logic as scorer.ts and claudecode-hook/hook.ts)
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
const RISK_THRESHOLD = 0.55;
const MAX_CONFIDENT_CALLS = 6;

function blastRadius(toolName: string, args: Record<string, unknown>): number {
  const name = toolName.toLowerCase();
  const cmdStr = EXEC_TOOLS.has(name)
    ? String(args["command"] ?? args["cmd"] ?? args["input"] ?? "")
    : "";
  const searchable = `${name} ${cmdStr} ${JSON.stringify(args)}`.toLowerCase();

  let s = 0;
  if (EXEC_TOOLS.has(name)) s += 0.25;
  if (IRREVERSIBILITY_TOKENS.some((t) => searchable.includes(t))) s += 0.5;
  if (BOUNDARY_TOKENS.some((t) => searchable.includes(t))) s += 0.25;
  return Math.min(1.0, s);
}

function planConfidence(toolCallIndex: number): number {
  return Math.max(0, 1 - toolCallIndex / MAX_CONFIDENT_CALLS);
}

// ---------------------------------------------------------------------------
// Per-session call counter
//
// The before_tool_call event doesn't carry session info. We track a global
// rolling counter per minute to approximate plan_confidence. Not perfect —
// Phase 3 will have proper session context from the ML model.
// ---------------------------------------------------------------------------

let globalCallCount = 0;
let windowStart = Date.now();
const WINDOW_MS = 60_000; // reset counter every minute

function nextCallIndex(): number {
  const now = Date.now();
  if (now - windowStart > WINDOW_MS) {
    globalCallCount = 0;
    windowStart = now;
  }
  return globalCallCount++;
}

// ---------------------------------------------------------------------------
// Trace I/O
// ---------------------------------------------------------------------------

interface TraceEntry {
  runtime: "openclaw";
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
  human_outcome: null;
}

function tracesDir(): string {
  return process.env["ALIGNLAYER_TRACES_DIR"] ?? "/home/node/.openclaw/traces";
}

function appendTrace(entry: TraceEntry): void {
  const dir = tracesDir();
  fs.mkdirSync(dir, { recursive: true });
  const date = new Date().toISOString().slice(0, 10);
  const file = path.join(dir, `alignlayer-${date}.jsonl`);
  fs.appendFileSync(file, JSON.stringify(entry) + "\n", "utf8");
}

// ---------------------------------------------------------------------------
// Plugin hooks
// ---------------------------------------------------------------------------

interface BeforeToolCallEvent {
  toolName: string;
  params: Record<string, unknown>;
}

interface BeforeToolCallResult {
  params?: Record<string, unknown>;
  block?: boolean;
  blockReason?: string;
}

export const hooks = {
  before_tool_call: (
    event: BeforeToolCallEvent
  ): BeforeToolCallResult => {
    const { toolName, params } = event;
    const callIdx = nextCallIndex();

    const br = blastRadius(toolName, params);
    const pc = planConfidence(callIdx);
    const risk = br * (1 - pc);
    const decision: "allow" | "interrupt" =
      risk >= RISK_THRESHOLD ? "interrupt" : "allow";

    appendTrace({
      runtime: "openclaw",
      session_id: "",   // not available in before_tool_call event; annotated in Phase 1
      turn_id: "",      // same
      timestamp: new Date().toISOString(),
      event: "before_tool_call",
      tool: toolName,
      args: params,
      risk_score: risk,
      blast_radius: br,
      plan_confidence: pc,
      decision,
      human_outcome: null,
    });

    if (decision === "interrupt") {
      const interruptFile = path.join(tracesDir(), "pending-interrupts.jsonl");
      const entry = { timestamp: new Date().toISOString(), tool: toolName, risk, blast_radius: br, plan_confidence: pc };
      fs.appendFileSync(interruptFile, JSON.stringify(entry) + "\n", "utf8");
    }

    // Phase 0: observational — never block.
    // Phase 3: return { block: true, blockReason: "..." } for decision === "interrupt".
    return {};
  },
};

// ---------------------------------------------------------------------------
// Plugin register (no commands or tools for now)
// ---------------------------------------------------------------------------

export default function register(api: any): void {
  const log = api.logger ?? console;
  log.info?.(`[alignlayer] plugin loaded — traces → ${tracesDir()}`);

  api.on("before_tool_call", hooks.before_tool_call);
}
