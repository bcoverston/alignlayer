/**
 * AlignLayer — LLM risk scorer.
 *
 * Supports two backends:
 *   anthropic  — Anthropic API (ANTHROPIC_API_KEY)
 *   ollama     — local Ollama server (default: http://localhost:11434)
 *
 * Two models in practice:
 *   Training   — qwen2.5-coder:32b via Ollama (offline, batch, ComfyUI off)
 *   Arbiter    — qwen2.5-coder:14b via Ollama (online, synchronous hook path)
 *
 * Score cache:
 *   Runtime cache keyed by sha256(tool + JSON(args)).
 *   Persisted to ALIGNLAYER_SCORER_CACHE_PATH (default: ~/.alignlayer/scorer-cache.jsonl).
 *   Prevents re-scoring identical commands across hook invocations.
 *
 * Integration:
 *   scoreCombined() is the main entry point. It applies the uncertainty band
 *   filter and returns null when the heuristic result is unambiguous.
 *   Wire into src/openclaw-plugin/index.ts and src/claudecode-hook/hook.ts
 *   by replacing the heuristic-only path with scoreCombined() at Phase 3.
 */

import crypto from "crypto";
import fs from "fs";
import path from "path";
import { SYSTEM_PROMPT, buildUserPrompt } from "./prompt.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Backend = "anthropic" | "ollama";

export interface LLMScore {
  blastRadius: number;
  risk: number;
  reasoning: string;
  flags: string[];
  model: string;
  backend: Backend;
  cached: boolean;
}

export interface ScorerOptions {
  /**
   * LLM backend. Default: "ollama" if ALIGNLAYER_OLLAMA_URL is set or Ollama
   * is reachable on localhost; otherwise "anthropic".
   */
  backend?: Backend;
  /** Ollama base URL. Default: ALIGNLAYER_OLLAMA_URL or http://localhost:11434 */
  ollamaUrl?: string;
  /** Model name. Default: ALIGNLAYER_SCORER_MODEL or backend-specific default. */
  model?: string;
  /** Anthropic API key (anthropic backend only). Default: ANTHROPIC_API_KEY. */
  apiKey?: string;
  /**
   * Heuristic scores in [bandLo, bandHi] trigger an LLM call.
   * Outside this range the heuristic result is unambiguous. Default: [0.35, 0.75]
   */
  bandLo?: number;
  bandHi?: number;
  /** Request timeout in ms. Default: 8000. */
  timeoutMs?: number;
  /** Path to the persistent score cache JSONL file. */
  cachePath?: string;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_OLLAMA_URL       = "http://localhost:11434";
const DEFAULT_OLLAMA_MODEL     = "qwen2.5-coder:14b";   // arbiter model
const DEFAULT_ANTHROPIC_MODEL  = "claude-haiku-4-5-20251001";
const DEFAULT_BAND_LO          = 0.35;
const DEFAULT_BAND_HI          = 0.75;
const DEFAULT_TIMEOUT_MS       = 8_000;

function defaultCachePath(): string {
  return (
    process.env["ALIGNLAYER_SCORER_CACHE_PATH"] ??
    path.join(process.env["HOME"] ?? "~", ".alignlayer", "scorer-cache.jsonl")
  );
}

// ---------------------------------------------------------------------------
// Score cache
// ---------------------------------------------------------------------------

interface CacheEntry {
  key: string;
  tool: string;
  blastRadius: number;
  risk: number;
  reasoning: string;
  flags: string[];
  model: string;
  backend: Backend;
  cachedAt: string;
}

// In-process cache — avoids disk reads on repeated calls within the same process.
const memCache = new Map<string, CacheEntry>();
let cacheLoaded = false;

function cacheKey(tool: string, args: Record<string, unknown>): string {
  const payload = `${tool}:${JSON.stringify(args)}`;
  return crypto.createHash("sha256").update(payload).digest("hex").slice(0, 16);
}

function loadCache(cachePath: string): void {
  if (cacheLoaded) return;
  cacheLoaded = true;
  try {
    if (!fs.existsSync(cachePath)) return;
    for (const line of fs.readFileSync(cachePath, "utf8").split("\n")) {
      if (!line.trim()) continue;
      const entry = JSON.parse(line) as CacheEntry;
      memCache.set(entry.key, entry);
    }
  } catch {
    // Cache corrupt or missing — start fresh.
  }
}

function saveToCache(cachePath: string, entry: CacheEntry): void {
  memCache.set(entry.key, entry);
  try {
    fs.mkdirSync(path.dirname(cachePath), { recursive: true });
    fs.appendFileSync(cachePath, JSON.stringify(entry) + "\n", "utf8");
  } catch {
    // Non-fatal — cache miss on next run.
  }
}

// ---------------------------------------------------------------------------
// Ollama backend
// ---------------------------------------------------------------------------

interface OllamaResponse {
  blastRadius: number;
  blast_radius?: number; // accept snake_case from model output
  risk: number;
  reasoning: string;
  flags: string[];
}

async function callOllama(
  tool: string,
  args: Record<string, unknown>,
  planConfidence: number,
  baseUrl: string,
  model: string,
  timeoutMs: number
): Promise<OllamaResponse> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user",   content: buildUserPrompt(tool, args, planConfidence) },
        ],
        temperature: 0,
        max_tokens: 256,
        stream: false,
      }),
    });

    if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
    const data = await res.json() as { choices: Array<{ message: { content: string } }> };
    const text = data.choices[0]?.message?.content ?? "";
    const json = text.replace(/^```(?:json)?\n?/i, "").replace(/\n?```$/i, "").trim();
    const parsed = JSON.parse(json) as Partial<OllamaResponse>;

    return {
      blastRadius: Math.max(0, Math.min(1, Number(parsed.blastRadius ?? parsed.blast_radius ?? 0.5))),
      risk:        Math.max(0, Math.min(1, Number(parsed.risk ?? 0.5))),
      reasoning:   String(parsed.reasoning ?? ""),
      flags:       Array.isArray(parsed.flags) ? parsed.flags.map(String) : [],
    };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Anthropic backend
// ---------------------------------------------------------------------------

async function callAnthropic(
  tool: string,
  args: Record<string, unknown>,
  planConfidence: number,
  model: string,
  apiKey: string | undefined,
  timeoutMs: number
): Promise<OllamaResponse> {
  // Dynamic import keeps the Anthropic SDK optional — Ollama users don't need it.
  const { default: Anthropic } = await import("@anthropic-ai/sdk") as { default: typeof import("@anthropic-ai/sdk").default };
  const client = new Anthropic({ apiKey });
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const msg = await client.messages.create(
      {
        model,
        max_tokens: 256,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: buildUserPrompt(tool, args, planConfidence) }],
      },
      { signal: controller.signal }
    );
    const text = msg.content[0]?.type === "text" ? msg.content[0].text : "";
    const json = text.replace(/^```(?:json)?\n?/i, "").replace(/\n?```$/i, "").trim();
    const parsed = JSON.parse(json) as Partial<OllamaResponse>;

    return {
      blastRadius: Math.max(0, Math.min(1, Number(parsed.blastRadius ?? parsed.blast_radius ?? 0.5))),
      risk:        Math.max(0, Math.min(1, Number(parsed.risk ?? 0.5))),
      reasoning:   String(parsed.reasoning ?? ""),
      flags:       Array.isArray(parsed.flags) ? parsed.flags.map(String) : [],
    };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Score a tool call using the configured LLM backend.
 * Checks the persistent cache first; only calls the model on a cache miss.
 */
export async function scoreLLM(
  tool: string,
  args: Record<string, unknown>,
  planConfidence: number,
  options: ScorerOptions = {}
): Promise<LLMScore> {
  const cachePath = options.cachePath ?? defaultCachePath();
  loadCache(cachePath);

  const key = cacheKey(tool, args);
  const hit = memCache.get(key);
  if (hit) {
    return {
      blastRadius: hit.blastRadius,
      risk:        hit.risk,
      reasoning:   hit.reasoning,
      flags:       hit.flags,
      model:       hit.model,
      backend:     hit.backend,
      cached:      true,
    };
  }

  const backend  = options.backend ?? "ollama";
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  let raw: OllamaResponse;
  let model: string;

  if (backend === "ollama") {
    const baseUrl = options.ollamaUrl ?? process.env["ALIGNLAYER_OLLAMA_URL"] ?? DEFAULT_OLLAMA_URL;
    model = options.model ?? process.env["ALIGNLAYER_SCORER_MODEL"] ?? DEFAULT_OLLAMA_MODEL;
    raw = await callOllama(tool, args, planConfidence, baseUrl, model, timeoutMs);
  } else {
    model = options.model ?? process.env["ALIGNLAYER_SCORER_MODEL"] ?? DEFAULT_ANTHROPIC_MODEL;
    raw = await callAnthropic(tool, args, planConfidence, model, options.apiKey, timeoutMs);
  }

  const entry: CacheEntry = {
    key,
    tool,
    blastRadius: raw.blastRadius,
    risk:        raw.risk,
    reasoning:   raw.reasoning,
    flags:       raw.flags,
    model,
    backend,
    cachedAt:    new Date().toISOString(),
  };
  saveToCache(cachePath, entry);

  return { ...raw, model, backend, cached: false };
}

/**
 * Combined scorer: heuristic fast-path + LLM for uncertain mid-band scores.
 *
 * Returns LLMScore when the LLM was invoked, null when the heuristic score
 * was outside the uncertainty band (caller should use the heuristic result).
 */
export async function scoreCombined(
  tool: string,
  args: Record<string, unknown>,
  heuristicBlastRadius: number,
  planConfidence: number,
  options: ScorerOptions = {}
): Promise<LLMScore | null> {
  const bandLo = options.bandLo ?? DEFAULT_BAND_LO;
  const bandHi = options.bandHi ?? DEFAULT_BAND_HI;

  if (heuristicBlastRadius < bandLo || heuristicBlastRadius > bandHi) return null;

  return scoreLLM(tool, args, planConfidence, options);
}
