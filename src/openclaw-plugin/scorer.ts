/**
 * Heuristic risk engine.
 *
 * Two independent axes → combined risk score → allow | interrupt
 *
 *   blast_radius    — how bad if this action is wrong (irreversibility + scope)
 *   plan_confidence — how likely the agent's current plan is still valid (positional proxy)
 *
 *   risk = blast_radius × (1 − plan_confidence)
 */

// Tokens that signal irreversible or high-consequence actions.
// Matched against the command name + first positional (subcommand) only —
// not against flags, to avoid false positives from --delete-branch, etc.
const IRREVERSIBILITY_TOKENS = [
  "push",
  "send",
  "deploy",
  "drop",
  "delete",
  "rm",
  "truncate",
  "overwrite",
  "destroy",
  "nuke",
  "reset",
  "purge",
  "wipe",
  "revoke",
  "terminate",
  "kill",
];

// Tokens that signal crossing a trust boundary (external / networked / cross-repo).
// Checked against the full command string — URLs and endpoints appear anywhere.
const BOUNDARY_TOKENS = [
  "curl",
  "wget",
  "fetch",
  "http://",
  "https://",
  "upload",
  "email",
  "smtp",
  "webhook",
  "s3://",
  "gs://",
  "azure",
];

// Tool names treated as high-capability execution surfaces (can do anything).
const EXEC_TOOLS = new Set(["exec", "bash", "shell", "run", "computer"]);

// Flags that amplify blast radius when an irreversibility token is already matched.
const FORCE_FLAGS     = new Set(["-f", "--force", "--hard", "--no-backup", "--overwrite", "--delete"]);
const RECURSIVE_FLAGS = new Set(["-r", "-R", "--recursive", "--all", "-A", "--all-namespaces"]);
// Flags that reduce blast radius — action is tentative or user-gated.
const DRY_RUN_FLAGS     = new Set(["--dry-run", "-n", "--simulate", "--check", "--preview", "--no-act"]);
const INTERACTIVE_FLAGS = new Set(["-i", "--interactive", "--confirm", "--prompt"]);

// Threshold above which a risk score routes to interrupt rather than allow.
export const RISK_THRESHOLD = 0.55;

// After this many tool calls in a run, plan_confidence reaches 0.
const MAX_CONFIDENT_CALLS = 6;

export interface RiskScore {
  blastRadius: number;
  planConfidence: number;
  risk: number;
  decision: "allow" | "interrupt";
}

// ---------------------------------------------------------------------------
// Command tokenizer
// ---------------------------------------------------------------------------

interface Tokens {
  /** Lowercased base command (e.g. "git", "rm", "kubectl"). */
  command: string;
  /** First positional arg — often the subcommand (e.g. "push", "delete"). */
  subcommand: string;
  /** Normalized flags: combined short flags expanded, =value suffixes stripped. */
  flags: string[];
}

/** Expand combined short flags: -rf → ["-r", "-f"] */
function expandFlag(f: string): string[] {
  if (f.startsWith("-") && !f.startsWith("--") && f.length > 2) {
    return f.slice(1).split("").map((c) => `-${c}`);
  }
  return [f];
}

/** Strip =value suffix for long flags: --dry-run=client → --dry-run */
function normalizeFlag(f: string): string {
  const eq = f.indexOf("=");
  return eq === -1 ? f : f.slice(0, eq);
}

function tokenizeCommand(cmd: string): Tokens {
  const parts = cmd.trim().split(/\s+/).filter(Boolean);
  const command = (parts[0] ?? "").toLowerCase();
  const rest = parts.slice(1);
  const rawFlags = rest.filter((p) => p.startsWith("-"));
  const positional = rest.filter((p) => !p.startsWith("-"));
  const flags = rawFlags.flatMap(expandFlag).map(normalizeFlag);
  return { command, subcommand: (positional[0] ?? "").toLowerCase(), flags };
}

/**
 * Flag-based modifier applied when an irreversibility token is matched.
 * Amplifiers push the score up; safety valves (dry-run, interactive) pull it down.
 */
function flagModifier(flags: string[]): number {
  let mod = 0;
  if (flags.some((f) => FORCE_FLAGS.has(f)))       mod += 0.2;
  if (flags.some((f) => RECURSIVE_FLAGS.has(f)))   mod += 0.1;
  if (flags.some((f) => DRY_RUN_FLAGS.has(f)))     mod -= 0.4;
  if (flags.some((f) => INTERACTIVE_FLAGS.has(f))) mod -= 0.2;
  return mod;
}

// ---------------------------------------------------------------------------
// Blast radius
// ---------------------------------------------------------------------------

/**
 * Estimate blast radius from tool name and serialised args.
 *
 * For exec-class tools the command string is tokenized so that flags are
 * scored separately from the command/subcommand — `rm -i` scores lower than
 * `rm -rf`, and `git push --dry-run` does not trigger at all.
 *
 * Score is additive across signal categories, clamped to [0, 1].
 */
export function blastRadius(
  toolName: string,
  args: Record<string, unknown>
): number {
  const name = toolName.toLowerCase();
  const cmdStr = EXEC_TOOLS.has(name)
    ? String(args["command"] ?? args["cmd"] ?? args["input"] ?? "")
    : "";

  let s = 0;
  if (EXEC_TOOLS.has(name)) s += 0.25; // baseline: exec surface

  if (cmdStr) {
    // Split on shell chain operators — score each segment, take the max.
    // Prevents `cd /tmp && rm -rf /` from masking the dangerous second command.
    const segments = cmdStr.split(/&&|\|\||;/).map((seg) => seg.trim()).filter(Boolean);
    let maxIrreversibility = 0;
    for (const seg of segments) {
      const { command, subcommand, flags } = tokenizeCommand(seg);
      const cmdSpace = `${command} ${subcommand}`;
      if (IRREVERSIBILITY_TOKENS.some((t) => cmdSpace.includes(t))) {
        maxIrreversibility = Math.max(maxIrreversibility, 0.5 + flagModifier(flags));
      }
    }
    s += maxIrreversibility;
  } else {
    // Non-exec tool: check serialized args for irreversibility signals.
    const searchable = `${name} ${JSON.stringify(args)}`.toLowerCase();
    if (IRREVERSIBILITY_TOKENS.some((t) => searchable.includes(t))) s += 0.5;
  }

  // Boundary tokens: URLs and endpoints can appear anywhere in the command.
  const fullStr = `${name} ${cmdStr} ${JSON.stringify(args)}`.toLowerCase();
  if (BOUNDARY_TOKENS.some((t) => fullStr.includes(t))) s += 0.25;

  return Math.max(0, Math.min(1.0, s));
}

/**
 * Estimate plan confidence from position within the current agent run.
 *
 * Early tool calls → agent is executing a known plan → high confidence.
 * Later calls → plan may have drifted → lower confidence.
 *
 * Returns 1.0 for the first call, decays linearly to 0.0 at MAX_CONFIDENT_CALLS.
 */
export function planConfidence(toolCallIndexInRun: number): number {
  return Math.max(0.0, 1.0 - toolCallIndexInRun / MAX_CONFIDENT_CALLS);
}

/**
 * Compute full risk score for a proposed tool call.
 *
 * @param toolName - tool name as reported by the agent runtime
 * @param args - tool arguments
 * @param toolCallIndexInRun - 0-based index of this call within the current run
 */
// Blast radius above which an action is considered unconditionally high-consequence.
// At this threshold, risk carries a floor regardless of plan confidence —
// `rm -rf /` on call 0 should not score 0.0 just because the agent is "confident".
const BLAST_FLOOR_THRESHOLD = 0.75;
const BLAST_FLOOR_RISK      = 0.60; // above RISK_THRESHOLD — always interrupts

export function score(
  toolName: string,
  args: Record<string, unknown>,
  toolCallIndexInRun: number
): RiskScore {
  const br = blastRadius(toolName, args);
  const pc = planConfidence(toolCallIndexInRun);
  const rawRisk = br * (1 - pc);
  // High-consequence actions carry a minimum risk regardless of plan confidence.
  const risk = br >= BLAST_FLOOR_THRESHOLD ? Math.max(BLAST_FLOOR_RISK, rawRisk) : rawRisk;

  return {
    blastRadius: br,
    planConfidence: pc,
    risk,
    decision: risk >= RISK_THRESHOLD ? "interrupt" : "allow",
  };
}
