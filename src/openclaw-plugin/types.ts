/**
 * Shared type definitions for AlignLayer OpenClaw plugin.
 */

/**
 * Risk scoring result from heuristic engine.
 */
export interface RiskScore {
  blastRadius: number;
  planConfidence: number;
  risk: number;
  decision: "allow" | "interrupt";
}

/**
 * CLI verb table entry: deterministic (tool, verb-pattern) → blast radius.
 */
export interface VerbEntry {
  tool: string;         // exact match on base command
  verb: RegExp;         // matched against first positional arg (subcommand)
  blast: number;        // blast radius value to apply
  cap: boolean;         // true → cap (max), false → floor (min)
}

/**
 * Tokenized command structure.
 */
export interface Tokens {
  /** Lowercased base command (e.g. "git", "rm", "kubectl"). */
  command: string;
  /** First positional arg — often the subcommand (e.g. "push", "delete"). */
  subcommand: string;
  /** Normalized flags: combined short flags expanded, =value suffixes stripped. */
  flags: string[];
}

/**
 * OpenClaw hook event (inferred from runtime).
 */
export interface HookEvent {
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

/**
 * Agent event payload from the agent-events bus.
 */
export interface AgentEventPayload {
  runId: string;
  seq: number;
  /** "lifecycle" | "tool" | "assistant" | "exec" */
  stream: string;
  /** Unix milliseconds */
  ts: number;
  data: Record<string, unknown>;
  sessionKey?: string;
}

/**
 * Trace entry written to JSONL log for compliance and analysis.
 */
export interface TraceEntry {
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

/**
 * ML score result from async enrichment server.
 */
export interface MlScoreResult {
  command: string;
  risk: number;
  tier: number;
  decision: string;
}

/**
 * Run state tracking for plan confidence decay.
 */
export interface RunState {
  toolCallCount: number;
}

/**
 * Before-tool-call hook event.
 */
export interface BeforeToolCallEvent {
  toolName: string;
  params: Record<string, unknown>;
}

/**
 * Before-tool-call hook result.
 */
export interface BeforeToolCallResult {
  params?: Record<string, unknown>;
  block?: boolean;
  blockReason?: string;
}

/**
 * Plugin trace entry (simplified for index.ts plugin context).
 */
export interface PluginTraceEntry {
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
