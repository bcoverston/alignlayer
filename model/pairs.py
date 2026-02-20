"""
AlignLayer — Phase 1 synthetic training pair generator.

Workflow
--------
1. Score a catalog of concrete actions via LLM (cached to scores.jsonl).
2. Emit labeled (action_A, action_B) pairs for Siamese network training.

Pair labels
-----------
  similar   (label=0) : |risk_A - risk_B| < SIMILAR_THRESHOLD
  dissimilar (label=1): |risk_A - risk_B| > DISSIMILAR_THRESHOLD
  Pairs in the ambiguous band are dropped from the corpus.

Usage
-----
  model/.venv/bin/python model/pairs.py [--output PATH] [--scores-cache PATH] [--dry-run]

Output JSONL schema
-------------------
  {
    "id": "pair-0001",
    "action_a": { "id": str, "tool": str, "args": dict, "text": str,
                  "risk": float, "tier": int, "reasoning": str },
    "action_b": { ... },
    "delta": float,
    "label": 0 | 1,
    "similar": bool,
    "generated_at": str
  }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIMILAR_THRESHOLD     = 0.15   # |delta| < this → label 0  (same risk tier)
DISSIMILAR_THRESHOLD  = 0.35   # |delta| > this → label 1  (clearly different)
# Pairs with delta in [SIMILAR_THRESHOLD, DISSIMILAR_THRESHOLD] are dropped.

DEFAULT_OUTPUT        = "data/synthetic/pairs-v0.jsonl"
DEFAULT_SCORES_CACHE  = "data/synthetic/scores-cache.jsonl"
# Ollama is the default backend for training (runs locally, no API cost).
# The 32B model is used when ComfyUI is not running (full 24GB available).
# Set ALIGNLAYER_BACKEND=anthropic to use the Anthropic API instead.
BACKEND               = os.environ.get("ALIGNLAYER_BACKEND", "ollama")
OLLAMA_URL            = os.environ.get("ALIGNLAYER_OLLAMA_URL", "http://localhost:11434")
OLLAMA_TRAINING_MODEL = os.environ.get("ALIGNLAYER_SCORER_MODEL", "qwen2.5-coder:32b")
ANTHROPIC_MODEL       = "claude-haiku-4-5-20251001"

MAX_RETRIES           = 3
RETRY_DELAY_S         = 2.0
RATE_LIMIT_DELAY_S    = 0.3    # between LLM calls (Ollama is local, can be tighter)

# ---------------------------------------------------------------------------
# Action catalog
#
# Each action is a (id, tool, args) tuple. The catalog spans five risk tiers:
#
#   Tier 0  (risk ~0.0–0.15)  — reads, status checks
#   Tier 1  (risk ~0.15–0.35) — local writes, installs, compilations
#   Tier 2  (risk ~0.35–0.55) — local destructive, reversible remote
#   Tier 3  (risk ~0.55–0.75) — external writes, pushes
#   Tier 4  (risk ~0.75–1.0)  — irreversible external, cascading destructive
#
# Weights reflect empirical command frequency (higher = more common in the
# wild). Used to bias pair sampling toward the realistic distribution.
# ---------------------------------------------------------------------------

CATALOG: list[dict[str, Any]] = [
    # ── Tier 0 — reads, status ───────────────────────────────────────────
    {"id": "ls_tmp",        "tool": "bash",   "args": {"command": "ls -la /tmp"},                                          "weight": 1.0},
    {"id": "cat_readme",    "tool": "bash",   "args": {"command": "cat README.md"},                                        "weight": 0.9},
    {"id": "grep_src",      "tool": "bash",   "args": {"command": "grep -r 'TODO' src/"},                                  "weight": 0.8},
    {"id": "git_status",    "tool": "bash",   "args": {"command": "git status"},                                           "weight": 1.0},
    {"id": "git_log",       "tool": "bash",   "args": {"command": "git log --oneline -10"},                                "weight": 0.9},
    {"id": "git_diff",      "tool": "bash",   "args": {"command": "git diff HEAD"},                                        "weight": 0.9},
    {"id": "docker_ps",     "tool": "bash",   "args": {"command": "docker ps"},                                            "weight": 0.8},
    {"id": "kubectl_get",   "tool": "bash",   "args": {"command": "kubectl get pods -n default"},                          "weight": 0.7},
    {"id": "ps_aux",        "tool": "bash",   "args": {"command": "ps aux | grep python"},                                 "weight": 0.8},
    {"id": "env",           "tool": "bash",   "args": {"command": "env | grep PATH"},                                      "weight": 0.7},
    {"id": "read_config",   "tool": "read",   "args": {"file_path": "/app/config.json"},                                   "weight": 0.9},
    {"id": "df_h",          "tool": "bash",   "args": {"command": "df -h"},                                                "weight": 0.6},
    {"id": "curl_get",      "tool": "bash",   "args": {"command": "curl -s https://api.example.com/health"},               "weight": 0.7},

    # ── Tier 1 — local writes, installs ──────────────────────────────────
    {"id": "write_config",  "tool": "write",  "args": {"file_path": "/app/config.json", "content": "{\"debug\":true}"},    "weight": 0.8},
    {"id": "git_add",       "tool": "bash",   "args": {"command": "git add src/main.py"},                                  "weight": 0.9},
    {"id": "git_commit",    "tool": "bash",   "args": {"command": "git commit -m 'fix: update config'"},                   "weight": 0.8},
    {"id": "mkdir",         "tool": "bash",   "args": {"command": "mkdir -p /tmp/build"},                                  "weight": 0.7},
    {"id": "cp_backup",     "tool": "bash",   "args": {"command": "cp config.json config.json.bak"},                       "weight": 0.6},
    {"id": "npm_install",   "tool": "bash",   "args": {"command": "npm install"},                                          "weight": 0.8},
    {"id": "pip_install",   "tool": "bash",   "args": {"command": "pip install -r requirements.txt"},                      "weight": 0.7},
    {"id": "make_build",    "tool": "bash",   "args": {"command": "make build"},                                           "weight": 0.6},
    {"id": "docker_build",  "tool": "bash",   "args": {"command": "docker build -t myapp:dev ."},                          "weight": 0.7},
    {"id": "git_push_dry",  "tool": "bash",   "args": {"command": "git push --dry-run origin main"},                       "weight": 0.5},

    # ── Tier 2 — local destructive, reversible remote ────────────────────
    {"id": "rm_interactive","tool": "bash",   "args": {"command": "rm -i old_file.txt"},                                   "weight": 0.5},
    {"id": "rm_rf_tmp",     "tool": "bash",   "args": {"command": "rm -rf /tmp/build"},                                    "weight": 0.6},
    {"id": "git_reset_soft","tool": "bash",   "args": {"command": "git reset HEAD~1"},                                     "weight": 0.4},
    {"id": "kill_proc",     "tool": "bash",   "args": {"command": "kill -9 12345"},                                        "weight": 0.3},
    {"id": "docker_rm",     "tool": "bash",   "args": {"command": "docker rm -f my-container"},                            "weight": 0.5},
    {"id": "truncate_log",  "tool": "bash",   "args": {"command": "truncate -s 0 app.log"},                                "weight": 0.3},
    {"id": "kubectl_apply", "tool": "bash",   "args": {"command": "kubectl apply -f deployment.yaml"},                     "weight": 0.5},

    # ── Tier 3 — external writes, pushes ─────────────────────────────────
    {"id": "git_push",      "tool": "bash",   "args": {"command": "git push origin main"},                                 "weight": 0.7},
    {"id": "git_push_force","tool": "bash",   "args": {"command": "git push --force origin main"},                         "weight": 0.3},
    {"id": "docker_push",   "tool": "bash",   "args": {"command": "docker push myrepo/myapp:latest"},                      "weight": 0.4},
    {"id": "npm_publish",   "tool": "bash",   "args": {"command": "npm publish"},                                          "weight": 0.3},
    {"id": "curl_post",     "tool": "bash",   "args": {"command": "curl -X POST https://api.example.com/deploy -d '{}'"},  "weight": 0.4},
    {"id": "ssh_restart",   "tool": "bash",   "args": {"command": "ssh user@prod.example.com 'systemctl restart app'"},    "weight": 0.3},
    {"id": "s3_sync",       "tool": "bash",   "args": {"command": "aws s3 sync ./dist s3://mybucket/"},                    "weight": 0.4},

    # ── Tier 4 — irreversible, cascading ─────────────────────────────────
    {"id": "rm_rf_root",    "tool": "bash",   "args": {"command": "rm -rf /"},                                             "weight": 0.01},
    {"id": "drop_table",    "tool": "bash",   "args": {"command": "psql -c 'DROP TABLE users;'"},                          "weight": 0.1},
    {"id": "git_push_fa",   "tool": "bash",   "args": {"command": "git push --force --all"},                               "weight": 0.1},
    {"id": "s3_delete_all", "tool": "bash",   "args": {"command": "aws s3 rm s3://backups/ --recursive"},                  "weight": 0.1},
    {"id": "kubectl_del_ns","tool": "bash",   "args": {"command": "kubectl delete namespace production"},                  "weight": 0.1},
    {"id": "dd_wipe",       "tool": "bash",   "args": {"command": "dd if=/dev/zero of=/dev/sda"},                          "weight": 0.01},
    {"id": "send_email_all","tool": "bash",   "args": {"command": "sendmail -t all-staff@company.com < breach-notice.txt"},"weight": 0.05},
]

# ---------------------------------------------------------------------------
# Action text — canonical string representation fed to the embedder
# ---------------------------------------------------------------------------

def action_text(tool: str, args: dict[str, Any]) -> str:
    if tool == "bash":
        return args.get("command", "")
    if tool in ("read", "write", "edit"):
        path = args.get("file_path", "")
        return f"{tool} {path}"
    return f"{tool}: {json.dumps(args, separators=(',', ':'))}"


# ---------------------------------------------------------------------------
# LLM scoring — individual action
# ---------------------------------------------------------------------------

SCORE_SYSTEM = """\
You are generating training data for a risk classification model.

Score the risk of executing this action on a scale of 0.0 to 1.0.

  0.0  = completely safe, fully reversible, no external effect
  1.0  = catastrophic, irreversible, broad external impact

Scoring factors:
  - Reversibility  : can this be undone? (rm -rf cannot; git commit can)
  - Scope          : local file vs remote system vs broad infrastructure
  - Safety valves  : --dry-run, --interactive, staging vs production
  - Amplifiers     : --force, --recursive, --all, production/main targets

Respond with valid JSON only. No prose outside the JSON object.

{
  "risk": <float 0.0–1.0>,
  "tier": <integer 0–4>,
  "reasoning": "<one concise sentence explaining the primary risk factor>",
  "flags": ["<signal 1>", "<signal 2>"]
}

Tiers:
  0 = read-only / status  (risk 0.0–0.15)
  1 = local writes        (risk 0.15–0.35)
  2 = local destructive   (risk 0.35–0.55)
  3 = external writes     (risk 0.55–0.75)
  4 = irreversible        (risk 0.75–1.0)
"""


def _call_ollama(prompt: str, model: str) -> str:
    """Call Ollama's OpenAI-compatible chat endpoint."""
    import urllib.request
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SCORE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def score_action(
    client: "anthropic.Anthropic | None",
    tool: str,
    args: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        text = action_text(tool, args)
        stub_risk = 0.1 if "ls" in text or "cat" in text or "status" in text else \
                    0.3 if "install" in text or "build" in text else \
                    0.5 if "rm -i" in text or "reset" in text else \
                    0.7 if "push" in text or "publish" in text else \
                    0.9
        return {"risk": stub_risk, "tier": int(stub_risk * 4), "reasoning": "(dry-run stub)", "flags": []}

    prompt = f"Tool: {tool}\nArgs: {json.dumps(args, indent=2)}\n\nScore this action."

    for attempt in range(MAX_RETRIES):
        try:
            if BACKEND == "ollama":
                raw = _call_ollama(prompt, OLLAMA_TRAINING_MODEL)
            else:
                assert client is not None
                msg = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=256,
                    system=SCORE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text if msg.content else ""

            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            return {
                "risk":      float(parsed.get("risk", 0.5)),
                "tier":      int(parsed.get("tier", 2)),
                "reasoning": str(parsed.get("reasoning", "")),
                "flags":     list(parsed.get("flags", [])),
            }
        except (json.JSONDecodeError, KeyError) as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_S * (attempt + 1))
            else:
                raise RuntimeError(f"Failed to score {tool}:{args} after {MAX_RETRIES} attempts") from exc

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Scores cache — persist per-action LLM scores so reruns are cheap
# ---------------------------------------------------------------------------

def load_scores_cache(path: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        cache[entry["id"]] = entry
    return cache


def save_score(path: str, entry: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def make_pairs(
    scored: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    all_combos = list(combinations(scored, 2))
    rng.shuffle(all_combos)

    for i, (a, b) in enumerate(all_combos):
        delta = abs(a["risk"] - b["risk"])
        if delta < SIMILAR_THRESHOLD:
            label, similar = 0, True
        elif delta > DISSIMILAR_THRESHOLD:
            label, similar = 1, False
        else:
            continue  # ambiguous band — drop

        pairs.append({
            "id":       f"pair-{i:04d}",
            "action_a": {k: a[k] for k in ("id", "tool", "args", "text", "risk", "tier", "reasoning")},
            "action_b": {k: b[k] for k in ("id", "tool", "args", "text", "risk", "tier", "reasoning")},
            "delta":    round(delta, 4),
            "label":    label,
            "similar":  similar,
            "generated_at": now,
        })

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate AlignLayer training pairs")
    ap.add_argument("--output",       default=DEFAULT_OUTPUT,       help="Output JSONL path")
    ap.add_argument("--scores-cache", default=DEFAULT_SCORES_CACHE, help="Per-action score cache JSONL")
    ap.add_argument("--seed",         type=int, default=42,          help="RNG seed for reproducibility")
    ap.add_argument("--dry-run",      action="store_true",           help="Use stub scores — no API calls")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if BACKEND == "ollama":
        client = None
        print(f"Backend: Ollama ({OLLAMA_URL}, model={OLLAMA_TRAINING_MODEL})")
    else:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        print(f"Backend: Anthropic (model={ANTHROPIC_MODEL})")

    # Step 1 — score every catalog action (cache hits skip LLM call)
    cache = load_scores_cache(args.scores_cache)
    scored: list[dict[str, Any]] = []

    print(f"Scoring {len(CATALOG)} catalog actions (cache={args.scores_cache})...")
    for action in CATALOG:
        action_id = action["id"]
        if action_id in cache:
            scored.append(cache[action_id])
            print(f"  [cached] {action_id}: risk={cache[action_id]['risk']:.2f}")
            continue

        print(f"  [scoring] {action_id}...", end=" ", flush=True)
        result = score_action(client, action["tool"], action["args"], dry_run=args.dry_run)
        entry = {
            "id":        action_id,
            "tool":      action["tool"],
            "args":      action["args"],
            "text":      action_text(action["tool"], action["args"]),
            "weight":    action["weight"],
            **result,
        }
        print(f"risk={entry['risk']:.2f} tier={entry['tier']} — {entry['reasoning'][:60]}")
        save_score(args.scores_cache, entry)
        cache[action_id] = entry
        scored.append(entry)

        if not args.dry_run:
            time.sleep(RATE_LIMIT_DELAY_S)

    # Step 2 — generate pairs
    print(f"\nGenerating pairs from {len(scored)} scored actions...")
    pairs = make_pairs(scored, rng=rng)

    similar_count    = sum(1 for p in pairs if p["similar"])
    dissimilar_count = sum(1 for p in pairs if not p["similar"])
    print(f"  {len(pairs)} pairs total — {similar_count} similar, {dissimilar_count} dissimilar")

    # Step 3 — write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(pairs)} pairs → {args.output}")

    # Step 4 — print risk distribution summary
    risks = sorted(s["risk"] for s in scored)
    print(f"\nRisk distribution across catalog:")
    print(f"  min={risks[0]:.2f}  p25={risks[len(risks)//4]:.2f}  "
          f"median={risks[len(risks)//2]:.2f}  p75={risks[3*len(risks)//4]:.2f}  max={risks[-1]:.2f}")


if __name__ == "__main__":
    main()
