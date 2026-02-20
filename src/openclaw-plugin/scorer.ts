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

/**
 * Estimate blast radius from tool name and serialised args.
 *
 * Score is additive across signal categories, capped at 1.0.
 */
export function blastRadius(
  toolName: string,
  args: Record<string, unknown>
): number {
  const name = toolName.toLowerCase();

  // For exec tools, the real signal is in the command string.
  const cmdStr = EXEC_TOOLS.has(name)
    ? String(args["command"] ?? args["cmd"] ?? args["input"] ?? "")
    : "";

  const searchable = `${name} ${cmdStr} ${JSON.stringify(args)}`.toLowerCase();

  let score = 0.0;

  if (EXEC_TOOLS.has(name)) score += 0.25; // baseline: can execute arbitrary code

  const irreversible = IRREVERSIBILITY_TOKENS.some((t) =>
    searchable.includes(t)
  );
  if (irreversible) score += 0.5;

  const crossesBoundary = BOUNDARY_TOKENS.some((t) => searchable.includes(t));
  if (crossesBoundary) score += 0.25;

  return Math.min(1.0, score);
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
export function score(
  toolName: string,
  args: Record<string, unknown>,
  toolCallIndexInRun: number
): RiskScore {
  const br = blastRadius(toolName, args);
  const pc = planConfidence(toolCallIndexInRun);
  const risk = br * (1 - pc);

  return {
    blastRadius: br,
    planConfidence: pc,
    risk,
    decision: risk >= RISK_THRESHOLD ? "interrupt" : "allow",
  };
}
