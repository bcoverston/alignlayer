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
  if (cmdStr) {
    const segments = cmdStr.split(/&&|\|\||;/).map((seg) => seg.trim()).filter(Boolean);
    let maxIrr = 0;
    for (const seg of segments) {
      const { command, subcommand, flags } = tokenizeCommand(seg);
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
    const rawRisk = br * (1 - pc);
    const risk = br >= 0.75 ? Math.max(0.60, rawRisk) : rawRisk;
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

      return {
        block: true,
        blockReason:
          `AlignLayer: high-risk action blocked (risk=${risk.toFixed(2)}, ` +
          `blast=${br.toFixed(2)}, confidence=${pc.toFixed(2)}). ` +
          `Tool: ${toolName}. ` +
          `Review the proposed action and re-invoke if appropriate.`,
      };
    }

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
