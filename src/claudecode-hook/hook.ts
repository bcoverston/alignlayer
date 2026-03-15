/**
 * AlignLayer — Claude Code hook.
 *
 * Handles PreToolUse and PostToolUse events. Run as a command hook in
 * ~/.claude/settings.json (see bottom of file for config snippet).
 *
 * Phase 0: observational — always returns permissionDecision: "allow".
 * Phase 3: flip to "deny" / "ask" for decision === "interrupt". Zero hook changes required.
 *
 * Invocation:
 *   npx tsx /path/to/alignlayer/src/claudecode-hook/hook.ts
 */

import fs from "fs";
import http from "http";
import path from "path";

// ---------------------------------------------------------------------------
// Scorer (inlined — hook must be self-contained for reliable invocation)
// Keep in sync with src/openclaw-plugin/scorer.ts
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

const EXEC_TOOLS = new Set(["bash", "exec", "shell", "run", "computer"]);

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

function planConfidence(toolCallIndexInSession: number): number {
  return Math.max(0, 1 - toolCallIndexInSession / MAX_CONFIDENT_CALLS);
}

// ---------------------------------------------------------------------------
// Claude Code hook payload types
// ---------------------------------------------------------------------------

interface PreToolUsePayload {
  session_id: string;
  transcript_path: string;
  cwd: string;
  permission_mode: string;
  hook_event_name: "PreToolUse";
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_use_id: string;
}

interface PostToolUsePayload {
  session_id: string;
  transcript_path: string;
  cwd: string;
  permission_mode: string;
  hook_event_name: "PostToolUse";
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_response: string;
  tool_use_id: string;
}

type HookPayload = PreToolUsePayload | PostToolUsePayload;

// ---------------------------------------------------------------------------
// Trace schema
// ---------------------------------------------------------------------------

interface TraceEntry {
  runtime: "claudecode";
  session_id: string;
  /** tool_use_id — closest available proxy for turn boundary in Claude Code. */
  turn_id: string;
  timestamp: string;
  event: "before_tool_call" | "after_tool_call";
  tool: string;
  args: Record<string, unknown>;
  risk_score: number | null;
  blast_radius: number | null;
  plan_confidence: number | null;
  decision: "allow" | "interrupt" | null;
  /** Populated in Phase 3 when hook returns "deny"/"ask" and user decides. */
  human_outcome: "approved" | "denied" | "allow-always" | null;
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

function tracesDir(): string {
  return (
    process.env["ALIGNLAYER_TRACES_DIR"] ??
    path.join(process.env["HOME"] ?? "~", ".alignlayer", "traces")
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
// Per-session tool call counter
//
// Claude Code hook invocations are stateless subprocesses — no shared memory
// with the main process. We persist a counter to maintain plan_confidence
// across calls within a session.
//
// Stored at: $ALIGNLAYER_TRACES_DIR/sessions/<session_id>.json
// ---------------------------------------------------------------------------

interface SessionState {
  toolCallCount: number;
}

function getAndIncrementToolIndex(sessionId: string): number {
  const dir = path.join(tracesDir(), "sessions");
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${sessionId}.json`);

  let state: SessionState = { toolCallCount: 0 };
  try {
    state = JSON.parse(fs.readFileSync(file, "utf8")) as SessionState;
  } catch {
    // first call in session — start from 0
  }

  const idx = state.toolCallCount;
  fs.writeFileSync(file, JSON.stringify({ toolCallCount: idx + 1 }), "utf8");
  return idx;
}

// ---------------------------------------------------------------------------
// ML model scoring via HTTP (localhost scoring server)
// ---------------------------------------------------------------------------

const ML_SERVER_URL = process.env["ALIGNLAYER_URL"] ?? "http://localhost:8000";

interface MLScoreResult {
  risk: number;
  tier: number;
  decision: "allow" | "interrupt";
}

function mlScore(command: string): Promise<MLScoreResult | null> {
  return new Promise((resolve) => {
    const body = JSON.stringify({ command });
    const url = new URL(`${ML_SERVER_URL}/score`);
    const req = http.request(
      {
        hostname: url.hostname,
        port: url.port || 8000,
        path: url.pathname,
        method: "POST",
        headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
        timeout: 2000,
      },
      (res) => {
        let data = "";
        res.on("data", (chunk: Buffer) => { data += chunk; });
        res.on("end", () => {
          try {
            const parsed = JSON.parse(data) as MLScoreResult;
            resolve(parsed);
          } catch { resolve(null); }
        });
      }
    );
    req.on("error", () => resolve(null));
    req.on("timeout", () => { req.destroy(); resolve(null); });
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const raw = fs.readFileSync(0, "utf8"); // fd 0 = stdin
const payload = JSON.parse(raw) as HookPayload;
const { session_id, hook_event_name, tool_name, tool_input, tool_use_id } =
  payload;

if (hook_event_name === "PreToolUse") {
  const toolIdx = getAndIncrementToolIndex(session_id);

  // TS heuristic (fallback / non-Bash tools)
  const br = blastRadius(tool_name, tool_input);
  const pc = planConfidence(toolIdx);
  const tsRawRisk = br * (1 - pc);
  const tsRisk = br >= 0.75 ? Math.max(0.60, tsRawRisk) : tsRawRisk;

  // For Bash commands, try ML model scoring via HTTP
  const cmdStr = EXEC_TOOLS.has(tool_name.toLowerCase())
    ? String(tool_input["command"] ?? tool_input["cmd"] ?? tool_input["input"] ?? "")
    : "";

  const score = async () => {
    let risk = tsRisk;
    let source = "ts_heuristic";
    let mlTier: number | null = null;

    if (cmdStr.length > 0 && cmdStr.length <= 2000) {
      const ml = await mlScore(cmdStr);
      if (ml) {
        risk = ml.risk;
        source = "ml_model";
        mlTier = ml.tier;
      }
    }

    const decision: "allow" | "interrupt" =
      risk >= RISK_THRESHOLD ? "interrupt" : "allow";

    appendTrace({
      runtime: "claudecode",
      session_id,
      turn_id: tool_use_id,
      timestamp: new Date().toISOString(),
      event: "before_tool_call",
      tool: tool_name,
      args: tool_input ?? {},
      risk_score: risk,
      blast_radius: br,
      plan_confidence: pc,
      decision,
      human_outcome: null,
    });

    const reason = source === "ml_model"
      ? `alignlayer[ml]: risk=${risk.toFixed(2)} tier=T${mlTier ?? "?"} (ts_fallback=${tsRisk.toFixed(2)})`
      : `alignlayer[ts]: risk=${risk.toFixed(2)} (blast=${br.toFixed(2)}, conf=${pc.toFixed(2)})`;

    // Phase 0: always allow (observational).
    process.stdout.write(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          permissionDecision: "allow",
          permissionDecisionReason: reason,
        },
      }) + "\n"
    );
  };

  score();
} else if (hook_event_name === "PostToolUse") {
  appendTrace({
    runtime: "claudecode",
    session_id,
    turn_id: tool_use_id,
    timestamp: new Date().toISOString(),
    event: "after_tool_call",
    tool: tool_name,
    args: tool_input ?? {},
    risk_score: null,
    blast_radius: null,
    plan_confidence: null,
    decision: null,
    human_outcome: null,
  });
  // PostToolUse: no output required.
}

/*
 * ─── Installation ─────────────────────────────────────────────────────────
 *
 * Add to ~/.claude/settings.json:
 *
 * {
 *   "hooks": {
 *     "PreToolUse": [
 *       {
 *         "matcher": ".*",
 *         "hooks": [
 *           {
 *             "type": "command",
 *             "command": "npx tsx /path/to/alignlayer/src/claudecode-hook/hook.ts",
 *             "timeout": 10,
 *             "statusMessage": "AlignLayer scoring..."
 *           }
 *         ]
 *       }
 *     ],
 *     "PostToolUse": [
 *       {
 *         "matcher": ".*",
 *         "hooks": [
 *           {
 *             "type": "command",
 *             "command": "npx tsx /path/to/alignlayer/src/claudecode-hook/hook.ts",
 *             "timeout": 5,
 *             "async": true
 *           }
 *         ]
 *       }
 *     ]
 *   }
 * }
 *
 * Set ALIGNLAYER_TRACES_DIR env var to override default trace location
 * (~/.alignlayer/traces).
 */
