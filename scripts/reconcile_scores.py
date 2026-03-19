#!/usr/bin/env python3
"""
Reconcile ML model scores with LLM agent scores.

Reads:
  /tmp/ingest_ml_scored.jsonl     — ML model scores for all candidates
  /tmp/score_results/batch_*.json — LLM agent scores for uncertain zone

Outputs:
  /tmp/reconciled.jsonl           — All commands with final tier/risk
  /tmp/disagreements.jsonl        — Commands where ML and LLM disagree (for Sonnet tiebreak)

Disagreement = tier differs by 2+ levels, OR one says allow and other says interrupt.
"""

import json
import glob
from collections import Counter
from pathlib import Path

ML_SCORED = "/tmp/ingest_ml_scored.jsonl"
RESULT_DIR = "/tmp/score_results"
OUTPUT = "/tmp/reconciled.jsonl"
DISAGREE = "/tmp/disagreements.jsonl"
SONNET_DIR = Path("/tmp/sonnet_batches")

THRESHOLD = 0.55


def load_ml_scores():
    ml = {}
    with open(ML_SCORED) as f:
        for line in f:
            e = json.loads(line)
            ml[e["text"]] = e
    return ml


def load_llm_scores():
    llm = {}
    for fp in sorted(glob.glob(f"{RESULT_DIR}/batch_*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARN: Could not parse {fp}")
            continue

        results = data.get("results", [])
        for r in results:
            cmd = r.get("cmd", "")
            if cmd:
                llm[cmd] = {"tier": r.get("tier"), "risk": r.get("risk")}
    return llm


def is_disagreement(ml_tier, llm_tier, ml_risk, llm_risk):
    """Significant disagreement: tier differs by 2+, or cross the interrupt threshold."""
    if ml_tier is None or llm_tier is None:
        return False
    # Handle T-2 as tier 5 for distance calc
    def effective(t):
        return 5 if t == -2 else (0 if t == -1 else t)

    tier_dist = abs(effective(ml_tier) - effective(llm_tier))
    if tier_dist >= 2:
        return True

    # One allows, other interrupts
    ml_decision = "interrupt" if ml_risk >= THRESHOLD else "allow"
    llm_decision = "interrupt" if (llm_risk or 0) >= THRESHOLD else "allow"
    if ml_decision != llm_decision:
        return True

    return False


def main():
    ml_scores = load_ml_scores()
    llm_scores = load_llm_scores()

    print(f"ML scored:  {len(ml_scores)}")
    print(f"LLM scored: {len(llm_scores)}")

    reconciled = []
    disagreements = []
    agree_count = 0
    disagree_count = 0

    for cmd, ml in ml_scores.items():
        entry = {**ml}
        llm = llm_scores.get(cmd)

        if llm:
            entry["llm_tier"] = llm["tier"]
            entry["llm_risk"] = llm["risk"]

            if is_disagreement(ml["ml_tier"], llm["tier"], ml["ml_risk"], llm["risk"]):
                disagree_count += 1
                # Average for now, Sonnet will tiebreak
                entry["status"] = "disagree"
                disagreements.append(entry)
            else:
                agree_count += 1
                # Agreement: average the two scores
                entry["final_risk"] = round((ml["ml_risk"] + (llm["risk"] or ml["ml_risk"])) / 2, 4)
                entry["final_tier"] = ml["ml_tier"]  # trust ML tier when they agree
                entry["status"] = "agree"
        else:
            # Not sent to LLM (confident zone) — trust ML
            entry["final_risk"] = ml["ml_risk"]
            entry["final_tier"] = ml["ml_tier"]
            entry["status"] = "ml_only"

        reconciled.append(entry)

    # Write outputs
    with open(OUTPUT, "w") as f:
        for e in reconciled:
            f.write(json.dumps(e) + "\n")

    with open(DISAGREE, "w") as f:
        for e in disagreements:
            f.write(json.dumps(e) + "\n")

    # Stats
    statuses = Counter(e["status"] for e in reconciled)
    print(f"\nReconciliation:")
    print(f"  ML only (confident):     {statuses.get('ml_only', 0)}")
    print(f"  ML + LLM agree:          {agree_count}")
    print(f"  ML + LLM disagree:       {disagree_count}")
    print(f"\nWrote {len(reconciled)} to {OUTPUT}")
    print(f"Wrote {disagree_count} disagreements to {DISAGREE}")

    if disagreements:
        # Prepare Sonnet batches
        SONNET_DIR.mkdir(exist_ok=True)
        batch_size = 100
        for i in range(0, len(disagreements), batch_size):
            batch = disagreements[i : i + batch_size]
            batch_id = f"sonnet_{i // batch_size:04d}"
            payload = {
                "batch_id": batch_id,
                "commands": [
                    {
                        "cmd": e["text"],
                        "ml_tier": e["ml_tier"],
                        "ml_risk": round(e["ml_risk"], 3),
                        "llm_tier": e.get("llm_tier"),
                        "llm_risk": round(e.get("llm_risk", 0), 3),
                    }
                    for e in batch
                ],
            }
            with open(SONNET_DIR / f"{batch_id}.json", "w") as f:
                json.dump(payload, f)

        n_batches = (len(disagreements) + batch_size - 1) // batch_size
        print(f"Wrote {n_batches} Sonnet tiebreak batches to {SONNET_DIR}/")

    # Show sample disagreements
    if disagreements:
        print(f"\nSample disagreements:")
        for e in disagreements[:10]:
            print(f"  ML=T{e['ml_tier']} r={e['ml_risk']:.3f}  LLM=T{e.get('llm_tier','?')} r={e.get('llm_risk',0):.3f}  {e['text'][:60]}")


if __name__ == "__main__":
    main()
