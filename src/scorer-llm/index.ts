/**
 * AlignLayer — LLM risk scorer.
 *
 * Phase 3 component. Inactive in Phase 0/1 — the heuristic scorer in
 * scorer.ts is the only active scorer. This module is wired in at Phase 3
 * by replacing the heuristic call with scoreCombined().
 *
 * ## Design
 *
 * The LLM scorer is a second opinion, not a replacement for the tokenizer.
 * It is invoked only when the heuristic score falls in the uncertainty band
 * [BAND_LO, BAND_HI] — outside that range the heuristic result is
 * unambiguous and the LLM call is skipped.
 *
 * When invoked, the LLM returns blast_radius + reasoning. The reasoning is
 * surfaced to the user on interrupt (explains *why*) and stored in the trace
 * as a label for future training.
 *
 * ## Integration points (Phase 3)
 *
 * Replace blastRadius() calls in:
 *   src/openclaw-plugin/index.ts     before_tool_call hook
 *   src/claudecode-hook/hook.ts      PreToolUse hook
 *
 * Both hooks already receive planConfidence — pass it through to scoreCombined().
 *
 * ## LLM choice
 *
 * Haiku by default — fast and cheap for a latency-sensitive hook path.
 * Override via ALIGNLAYER_SCORER_MODEL env var.
 */

import Anthropic from "@anthropic-ai/sdk";
import { SYSTEM_PROMPT, buildUserPrompt } from "./prompt.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LLMScore {
  /** Estimated blast radius from the LLM. */
  blastRadius: number;
  /** Combined risk = blastRadius × (1 − planConfidence). */
  risk: number;
  /** One-sentence explanation of the primary risk factor. */
  reasoning: string;
  /** Key signals that drove the score (for trace annotation and UX). */
  flags: string[];
  /** Model used. */
  model: string;
}

export interface ScorerOptions {
  /**
   * Heuristic scores in [bandLo, bandHi] are ambiguous enough to warrant
   * an LLM call. Outside this range the heuristic result is returned as-is.
   * Default: [0.3, 0.7]
   */
  bandLo?: number;
  bandHi?: number;
  /** Override the model. Default: ALIGNLAYER_SCORER_MODEL or claude-haiku-4-5-20251001 */
  model?: string;
  /** Anthropic API key. Default: ANTHROPIC_API_KEY env var. */
  apiKey?: string;
  /** Abort signal for timeout control. */
  signal?: AbortSignal;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const DEFAULT_BAND_LO = 0.3;
const DEFAULT_BAND_HI = 0.7;
const DEFAULT_MODEL = "claude-haiku-4-5-20251001";

interface LLMResponse {
  blast_radius: number;
  risk: number;
  reasoning: string;
  flags: string[];
}

function clamp(v: number): number {
  return Math.max(0, Math.min(1, v));
}

async function callLLM(
  toolName: string,
  args: Record<string, unknown>,
  planConfidence: number,
  model: string,
  apiKey?: string,
  signal?: AbortSignal
): Promise<LLMResponse> {
  const client = new Anthropic({ apiKey });

  const message = await client.messages.create(
    {
      model,
      max_tokens: 256,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: buildUserPrompt(toolName, args, planConfidence),
        },
      ],
    },
    { signal }
  );

  const text =
    message.content[0]?.type === "text" ? message.content[0].text : "";

  // Strip markdown code fences if the model wraps the JSON.
  const jsonText = text.replace(/^```(?:json)?\n?/i, "").replace(/\n?```$/i, "").trim();

  const parsed = JSON.parse(jsonText) as Partial<LLMResponse>;

  return {
    blast_radius: clamp(Number(parsed.blast_radius ?? 0)),
    risk: clamp(Number(parsed.risk ?? 0)),
    reasoning: String(parsed.reasoning ?? ""),
    flags: Array.isArray(parsed.flags) ? parsed.flags.map(String) : [],
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Score a tool call using the LLM, bypassing the uncertainty band filter.
 *
 * Use this for testing or for contexts where you always want an LLM opinion.
 * In production, prefer scoreCombined() which applies the band filter.
 */
export async function scoreLLM(
  toolName: string,
  args: Record<string, unknown>,
  planConfidence: number,
  options: ScorerOptions = {}
): Promise<LLMScore> {
  const model =
    options.model ?? process.env["ALIGNLAYER_SCORER_MODEL"] ?? DEFAULT_MODEL;

  const raw = await callLLM(
    toolName,
    args,
    planConfidence,
    model,
    options.apiKey,
    options.signal
  );

  return {
    blastRadius: raw.blast_radius,
    risk: raw.risk,
    reasoning: raw.reasoning,
    flags: raw.flags,
    model,
  };
}

/**
 * Combined scorer: heuristic fast-path + LLM for uncertain mid-band scores.
 *
 * @param toolName       - tool name as reported by the agent runtime
 * @param args           - tool arguments
 * @param heuristicScore - blast radius already computed by the tokenizer scorer
 * @param planConfidence - plan_confidence already computed (positional proxy)
 * @param options        - model, band thresholds, api key
 *
 * @returns LLMScore when LLM was invoked, null when heuristic was unambiguous.
 *          A null return means the caller should use the heuristic result directly.
 */
export async function scoreCombined(
  toolName: string,
  args: Record<string, unknown>,
  heuristicScore: number,
  planConfidence: number,
  options: ScorerOptions = {}
): Promise<LLMScore | null> {
  const bandLo = options.bandLo ?? DEFAULT_BAND_LO;
  const bandHi = options.bandHi ?? DEFAULT_BAND_HI;

  // Outside the uncertainty band the heuristic result is clear enough.
  if (heuristicScore < bandLo || heuristicScore > bandHi) return null;

  return scoreLLM(toolName, args, planConfidence, options);
}
