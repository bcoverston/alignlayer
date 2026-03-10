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
const EXEC_TOOLS = new Set(["exec", "bash", "shell", "run", "computer", "node", "repl"]);

// ---------------------------------------------------------------------------
// CLI verb table
//
// Deterministic (tool, verb-pattern) → blast floor or cap, checked before the
// generic heuristic. Floors guarantee a minimum blast for known-dangerous verbs;
// caps prevent over-firing on known-safe read ops.
// ---------------------------------------------------------------------------

interface VerbEntry {
  tool: string;         // exact match on base command
  verb: RegExp;         // matched against first positional arg (subcommand)
  blast: number;        // blast radius value to apply
  cap: boolean;         // true → cap (max), false → floor (min)
}

const CLI_VERB_TABLE: VerbEntry[] = [
  // Terraform
  { tool: "terraform", verb: /^destroy$/,                                           blast: 0.90, cap: false },
  { tool: "terraform", verb: /^apply$/,                                             blast: 0.80, cap: false },
  { tool: "terraform", verb: /^plan$/,                                              blast: 0.05, cap: true  }, // dry-run
  { tool: "terraform", verb: /^(show|output|validate|fmt|graph|version|workspace|state|providers)$/, blast: 0.05, cap: true },

  // Redis — verb is the Redis command (first non-flag positional after the tool)
  { tool: "redis-cli", verb: /^(flushdb|flushall)$/i,                              blast: 0.50, cap: false },
  { tool: "redis-cli", verb: /^(del|unlink|expire|persist|rename|move)$/i,         blast: 0.40, cap: false },
  { tool: "redis-cli", verb: /^(get|keys|scan|info|dbsize|ttl|type|llen|smembers|hgetall|zrange|config\s+get|ping|echo|debug\s+object)$/i, blast: 0.05, cap: true },

  // AWS — verb matched against the full remaining command (service + operation)
  { tool: "aws", verb: /terminate-instances|stop-instances|delete-|deregister-|remove-|revoke-|disable-|detach-/, blast: 0.65, cap: false },
  { tool: "aws", verb: /describe-|list-|get-|show-|read-/,                         blast: 0.05, cap: true },

  // kubectl
  { tool: "kubectl", verb: /^delete$/,                                              blast: 0.65, cap: false },
  { tool: "kubectl", verb: /^apply$/,                                               blast: 0.60, cap: false },
  { tool: "kubectl", verb: /^(get|describe|logs?|top|explain|diff|version|cluster-info|auth|rollout\s+status|config\s+view)$/, blast: 0.05, cap: true },

  // git
  { tool: "git", verb: /^push$/,                                                   blast: 0.65, cap: false },

  // aws write ops not covered by the terminate/delete pattern
  { tool: "aws",  verb: /sns\s+publish|sqs\s+send-message|s3\s+(cp|mv|sync)\b|ec2\s+run-instances/, blast: 0.60, cap: false },

  // find -delete operates recursively at scale
  { tool: "find", verb: /-delete\b/,                                                blast: 0.80, cap: false },

  // system power / service management
  { tool: "reboot",    verb: /.*/,                                                  blast: 0.82, cap: false },
  { tool: "shutdown",  verb: /.*/,                                                  blast: 0.82, cap: false },
  { tool: "poweroff",  verb: /.*/,                                                  blast: 0.82, cap: false },
  { tool: "systemctl", verb: /^(stop|disable|halt|poweroff|reboot)\b/,             blast: 0.75, cap: false },

  // npm / yarn / pnpm publish
  { tool: "npm",  verb: /^(publish|unpublish|deprecate)$/,                         blast: 0.70, cap: false },
  { tool: "yarn", verb: /^publish$/,                                                blast: 0.70, cap: false },
  { tool: "pnpm", verb: /^publish$/,                                                blast: 0.70, cap: false },
];

// ---------------------------------------------------------------------------
// Opaque execution detection
//
// Commands that delegate to an interpreter inline hide their real intent.
// Flag these for elevated blast radius as an alignment signal — the agent
// should rarely need to wrap logic in `python3 -c` for routine tasks.
// ---------------------------------------------------------------------------

// Remote code execution / exfiltration: fetch+execute or credential theft.
// These are a step beyond opaque exec — floor at T-2 blast territory.
const EXFIL_EXEC_RE = /eval\s+"\$\(curl|curl\s+.*\|\s*(ba)?sh|wget\s+.*\|\s*(ba)?sh|curl\s+.*\|\s*python|base64\s+-d\s*\|\s*(ba)?sh|curl\s+.*-d\s+.*@\/etc\/(passwd|shadow)|nc\s+-[a-z]*e\s+\/bin\/(ba)?sh|\/dev\/tcp\/|urllib\.request\.urlopen\(|requests\.get\(.*\)\.text.*exec\(/i;

/** Minimum blast radius for remote-code-exec / exfil patterns (T-2 territory). */
const EXFIL_EXEC_BLAST_FLOOR = 0.95;

const OPAQUE_EXEC_RE = /(?:python3?|node|perl|ruby|bash|sh)\s+-[ce]\s+|(?<![a-z])eval\s+[\$'"`(]|\|\s*(?:bash|sh)\b|base64\s+-d\s*\|/i;

/** Minimum blast radius for opaque interpreter invocations (T2 territory). */
const OPAQUE_EXEC_BLAST_FLOOR = 0.45;

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

// ---------------------------------------------------------------------------
// CLI verb table lookup
// ---------------------------------------------------------------------------

/**
 * Return a blast radius value from the CLI_VERB_TABLE if the command matches,
 * or null if no entry applies.
 *
 * Caps and floors are both returned here; the caller decides how to apply them.
 */
function lookupVerbTable(
  command: string,
  fullCmd: string
): { blast: number; cap: boolean } | null {
  // The "verb" for multi-word CLIs (e.g. aws, kubectl) is everything after the
  // base command — we match the verb pattern against that whole remainder.
  const remainder = fullCmd.trim().slice(command.length).trim().toLowerCase();
  const firstPositional = remainder.split(/\s+/).find((p) => !p.startsWith("-")) ?? "";

  for (const entry of CLI_VERB_TABLE) {
    if (entry.tool !== command) continue;
    const target = ["aws", "kubectl", "git", "find", "systemctl"].includes(entry.tool)
      ? remainder  // match against full remainder for multi-word verbs
      : firstPositional;
    if (entry.verb.test(target)) return { blast: entry.blast, cap: entry.cap };
  }
  return null;
}

/**
 * Extract the command string embedded in an SSH invocation, if present.
 * e.g. `ssh -i key.pem user@host 'rm -rf /tmp/x'` → `rm -rf /tmp/x`
 */
function extractSshInner(cmd: string): string | null {
  // Capture single- or double-quoted trailing arg after the host
  const m = cmd.match(/^ssh\b.*?\s+(?:[^\s'"]+)\s+['"](.+)['"]\s*$/s);
  return m ? m[1] : null;
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
    // Exfil / remote code execution: fetch+execute patterns — highest blast floor.
    if (EXFIL_EXEC_RE.test(cmdStr)) {
      s = Math.max(s, EXFIL_EXEC_BLAST_FLOOR);
    }

    // Opaque execution: interpreter wrapping hides intent — apply a blast floor.
    if (OPAQUE_EXEC_RE.test(cmdStr)) {
      s = Math.max(s, OPAQUE_EXEC_BLAST_FLOOR);
    }

    // Split on shell chain operators — score each segment, take the max.
    // Prevents `cd /tmp && rm -rf /` from masking a dangerous second command.
    const segments = cmdStr.split(/&&|\|\||;/).map((seg) => seg.trim()).filter(Boolean);
    let maxSegBlast = 0;

    for (const seg of segments) {
      const { command, subcommand, flags } = tokenizeCommand(seg);

      // Dry-run cap (highest priority — overrides verb table floors).
      // If any dry-run flag is present, this segment contributes nothing.
      if (flags.some((f) => DRY_RUN_FLAGS.has(f))) {
        continue;
      }
      // Subcommand-level dry-run patterns (e.g. "terraform plan")
      if (/(?:^|\s)terraform\s+plan(?:\s|$)/.test(seg) ||
          /(?:^|\s)make\s+-[a-zA-Z]*n[a-zA-Z]*(?:\s|$)/.test(seg) ||
          /(?:^|\s)helm\s+\S+\s+.*?--dry-run/.test(seg)) {
        continue;
      }

      // SSH with a quoted inner command: score the inner command instead of the
      // ssh invocation itself, so `ssh host 'rm -rf /tmp'` reflects the inner op.
      if (command === "ssh") {
        const inner = extractSshInner(seg);
        if (inner) {
          const innerScore = blastRadius("bash", { command: inner });
          maxSegBlast = Math.max(maxSegBlast, innerScore - 0.25); // subtract exec baseline already counted
          continue;
        }
      }

      // CLI verb table: check for known (tool, verb) pairs before generic tokens.
      const verbMatch = lookupVerbTable(command, seg);
      if (verbMatch !== null) {
        if (verbMatch.cap) {
          // Known-safe: cap this segment's contribution
          maxSegBlast = Math.max(maxSegBlast, 0); // contributes nothing beyond baseline
          continue;
        } else {
          // Known-risky: floor at the table value, still apply flag modifiers
          const floored = Math.max(verbMatch.blast, verbMatch.blast + flagModifier(flags));
          maxSegBlast = Math.max(maxSegBlast, floored - 0.25); // subtract exec baseline
          continue;
        }
      }

      // Generic irreversibility token scan (fallback for unknown CLIs)
      const cmdSpace = `${command} ${subcommand}`;
      if (IRREVERSIBILITY_TOKENS.some((t) => cmdSpace.includes(t))) {
        maxSegBlast = Math.max(maxSegBlast, 0.5 + flagModifier(flags));
      }
    }

    s += maxSegBlast;
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
