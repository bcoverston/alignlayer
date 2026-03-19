#!/usr/bin/env python3
"""
Final ingest: merge ML scores, Haiku scores, and Sonnet tiebreaks into corpus.

Reads:
  /tmp/ingest_ml_scored.jsonl      — ML model scores (all 24K)
  /tmp/score_results/batch_*.json  — Haiku scores (uncertain zone)
  /tmp/sonnet_results/sonnet_*.json — Sonnet tiebreaks (disagreements)

For each command, final score is determined by:
  1. If Sonnet tiebreak exists → use Sonnet's tier/risk
  2. If ML + Haiku agree → average their risk, use agreed tier
  3. If ML only (confident zone) → use ML score directly

Outputs new corpus entries to scores-cache.jsonl.
"""

import json
import glob
import hashlib
from collections import Counter
from pathlib import Path

ML_SCORED = "/tmp/ingest_ml_scored.jsonl"
HAIKU_DIR = "/tmp/score_results"
SONNET_DIR = "/tmp/sonnet_results"
CORPUS = "data/synthetic/scores-cache.jsonl"

TIER_BOUNDARIES = [0.18, 0.38, 0.55, 0.74]
THRESHOLD = 0.55


def risk_to_tier(risk):
    if risk < TIER_BOUNDARIES[0]: return 0
    if risk < TIER_BOUNDARIES[1]: return 1
    if risk < TIER_BOUNDARIES[2]: return 2
    if risk < TIER_BOUNDARIES[3]: return 3
    return 4


def entry_id(cmd):
    return hashlib.md5(cmd.encode()).hexdigest()[:16]


def load_haiku():
    llm = {}
    for fp in sorted(glob.glob(f"{HAIKU_DIR}/batch_*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
            for r in data.get("results", []):
                cmd = r.get("cmd", "")
                if cmd:
                    llm[cmd] = {"tier": r.get("tier"), "risk": r.get("risk")}
        except (json.JSONDecodeError, KeyError):
            pass
    return llm


def load_sonnet():
    sonnet = {}
    for fp in sorted(glob.glob(f"{SONNET_DIR}/sonnet_*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
            for r in data.get("results", []):
                cmd = r.get("cmd", "")
                if cmd:
                    sonnet[cmd] = {
                        "tier": r.get("tier"),
                        "risk": r.get("risk"),
                        "picked": r.get("picked", "sonnet"),
                    }
        except (json.JSONDecodeError, KeyError):
            pass
    return sonnet


def load_existing_corpus():
    existing = set()
    if Path(CORPUS).exists():
        for line in open(CORPUS):
            try:
                existing.add(json.loads(line)["text"].strip())
            except (json.JSONDecodeError, KeyError):
                pass
    return existing


def main():
    # Load all scores
    ml_entries = []
    with open(ML_SCORED) as f:
        for line in f:
            ml_entries.append(json.loads(line))

    haiku = load_haiku()
    sonnet = load_sonnet()
    existing = load_existing_corpus()

    print(f"ML scored:     {len(ml_entries)}")
    print(f"Haiku scored:  {len(haiku)}")
    print(f"Sonnet scored: {len(sonnet)}")
    print(f"Existing corpus: {len(existing)}")

    # Determine final score for each command
    new_entries = []
    stats = Counter()

    for e in ml_entries:
        cmd = e["text"]
        if cmd in existing:
            stats["skip_existing"] += 1
            continue

        ml_risk = e["ml_risk"]
        ml_tier = e["ml_tier"]
        source_tag = e.get("source", "unknown")

        # Check Sonnet tiebreak first
        if cmd in sonnet:
            s = sonnet[cmd]
            final_risk = s["risk"]
            final_tier = s["tier"]
            scorer = f"sonnet-tiebreak-{source_tag}"
            stats["sonnet"] += 1
        elif cmd in haiku:
            h = haiku[cmd]
            h_tier = h.get("tier")
            h_risk = h.get("risk", ml_risk)
            if h_tier == ml_tier or abs((h_risk or 0) - ml_risk) < 0.15:
                # Agreement — average
                final_risk = round((ml_risk + (h_risk or ml_risk)) / 2, 4)
                final_tier = ml_tier
                scorer = f"ml+haiku-agree-{source_tag}"
                stats["agree"] += 1
            else:
                # Disagreement without Sonnet tiebreak — use Haiku (LLM understands semantics)
                final_risk = h_risk or ml_risk
                final_tier = h_tier if h_tier is not None else ml_tier
                scorer = f"haiku-override-{source_tag}"
                stats["haiku_override"] += 1
        else:
            # ML only (confident zone)
            final_risk = ml_risk
            final_tier = ml_tier
            scorer = f"ml-confident-{source_tag}"
            stats["ml_only"] += 1

        # Sanitize
        if final_tier is None:
            final_tier = risk_to_tier(final_risk)
        if final_risk is None:
            continue

        new_entries.append({
            "id": entry_id(cmd),
            "tool": "bash",
            "args": {"command": cmd},
            "text": cmd,
            "risk": round(float(final_risk), 4),
            "tier": int(final_tier),
            "expected_tier": int(final_tier),
            "reasoning": "",
            "flags": [],
            "scorer": scorer,
        })

    print(f"\nScoring resolution:")
    for k, v in stats.most_common():
        print(f"  {k}: {v}")

    print(f"\nNew entries to add: {len(new_entries)}")

    # Tier distribution of new entries
    tier_dist = Counter(e["tier"] for e in new_entries)
    print(f"Tier distribution:")
    for t in sorted(tier_dist.keys()):
        print(f"  T{t}: {tier_dist[t]}")

    # Write to corpus
    if new_entries:
        with open(CORPUS, "a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nAppended {len(new_entries)} entries to {CORPUS}")
        total = sum(1 for _ in open(CORPUS))
        print(f"Total corpus size: {total}")
    else:
        print("\nNo new entries to add.")


if __name__ == "__main__":
    main()
