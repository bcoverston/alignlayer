#!/usr/bin/env python3
"""
Harvest human feedback signals into corpus corrections for RiskHead retraining.

Signal sources:
  1. Dashboard feedback (data/traces/feedback.jsonl) — thumbs up/down
  2. Hook traces (~/.alignlayer/traces/) — approve/deny on Phase 1 "ask" prompts

Outputs:
  - data/feedback_corrections.jsonl — new/updated corpus entries
  - Optionally appends to scores-cache.jsonl with --apply

Usage:
  python scripts/harvest_feedback.py                    # preview corrections
  python scripts/harvest_feedback.py --apply            # write to corpus
  python scripts/harvest_feedback.py --retrain          # apply + retrain RiskHead
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = ROOT / "data" / "traces" / "feedback.jsonl"
HOOK_TRACES_DIR = Path(os.environ.get(
    "ALIGNLAYER_TRACES_DIR",
    Path.home() / ".alignlayer" / "traces",
))
CORPUS_FILE = ROOT / "data" / "synthetic" / "scores-cache.jsonl"
CORRECTIONS_FILE = ROOT / "data" / "feedback_corrections.jsonl"

# Risk adjustments for feedback signals
# "incorrect" + "interrupt" decision → model over-flagged (FP) → lower risk
# "incorrect" + "allow" decision → model under-flagged (FN) → raise risk
FP_RISK_ADJUSTMENT = -0.20  # nudge toward allow
FN_RISK_ADJUSTMENT = +0.25  # nudge toward interrupt

# Tier boundaries (must match siamese.py _risk_to_tier)
TIER_BOUNDARIES = [0.18, 0.38, 0.55, 0.74]


def risk_to_tier(risk: float) -> int:
    if risk < TIER_BOUNDARIES[0]:
        return 0
    if risk < TIER_BOUNDARIES[1]:
        return 1
    if risk < TIER_BOUNDARIES[2]:
        return 2
    if risk < TIER_BOUNDARIES[3]:
        return 3
    return 4


def entry_id(cmd: str) -> str:
    return hashlib.md5(cmd.encode()).hexdigest()[:16]


def load_corpus_commands() -> set[str]:
    """Load existing corpus commands to avoid duplicates."""
    cmds: set[str] = set()
    if CORPUS_FILE.exists():
        for line in open(CORPUS_FILE):
            try:
                cmds.add(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return cmds


def harvest_dashboard_feedback() -> list[dict]:
    """Read dashboard feedback.jsonl and generate corrections for 'incorrect' judgments."""
    corrections = []
    if not FEEDBACK_FILE.exists():
        return corrections

    for line in open(FEEDBACK_FILE):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if entry.get("judgment") != "incorrect":
            continue

        cmd = entry["command"]
        original_risk = entry["risk"]
        original_decision = entry.get("decision", "allow")

        # Determine correction direction
        if original_decision == "interrupt":
            # FP: model said interrupt, human says wrong → lower risk
            corrected_risk = max(0.0, original_risk + FP_RISK_ADJUSTMENT)
            correction_type = "fp_dashboard"
        else:
            # FN: model said allow, human says wrong → raise risk
            corrected_risk = min(1.0, original_risk + FN_RISK_ADJUSTMENT)
            correction_type = "fn_dashboard"

        corrections.append({
            "id": entry_id(cmd),
            "tool": "bash",
            "args": {"command": cmd},
            "text": cmd,
            "risk": round(corrected_risk, 4),
            "tier": risk_to_tier(corrected_risk),
            "expected_tier": risk_to_tier(corrected_risk),
            "reasoning": f"human feedback: {correction_type} (original_risk={original_risk:.3f})",
            "flags": ["human-feedback"],
            "scorer": "human-feedback",
            "original_risk": original_risk,
            "correction_type": correction_type,
        })

    return corrections


def harvest_hook_traces() -> list[dict]:
    """Read hook traces and generate corrections from approve/deny outcomes."""
    corrections = []
    if not HOOK_TRACES_DIR.exists():
        return corrections

    # Collect before_tool_call entries that were "asked" (interrupt decision from ML)
    asked: dict[str, dict] = {}  # turn_id → entry
    outcomes: dict[str, str] = {}  # turn_id → human_outcome

    for trace_file in sorted(HOOK_TRACES_DIR.glob("alignlayer-*.jsonl")):
        for line in open(trace_file):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            turn_id = entry.get("turn_id")
            if not turn_id:
                continue

            if entry.get("event") == "before_tool_call":
                # Only interested in ML-scored interrupt decisions
                if entry.get("source") == "ml_model" and entry.get("decision") == "interrupt":
                    asked[turn_id] = entry
            elif entry.get("event") == "after_tool_call":
                if entry.get("human_outcome"):
                    outcomes[turn_id] = entry["human_outcome"]

    for turn_id, entry in asked.items():
        outcome = outcomes.get(turn_id)
        if not outcome:
            continue

        cmd = entry.get("args", {}).get("command", "")
        if not cmd:
            continue

        original_risk = entry.get("risk_score", 0.55)

        if outcome == "approved":
            # User approved a command we flagged → potential FP
            corrected_risk = max(0.0, original_risk + FP_RISK_ADJUSTMENT)
            correction_type = "fp_hook_approved"
        elif outcome == "denied":
            # User denied → confirmed dangerous, boost confidence
            corrected_risk = min(1.0, original_risk + 0.05)  # small boost, already high
            correction_type = "fn_hook_denied"
        else:
            continue

        corrections.append({
            "id": entry_id(cmd),
            "tool": "bash",
            "args": {"command": cmd},
            "text": cmd,
            "risk": round(corrected_risk, 4),
            "tier": risk_to_tier(corrected_risk),
            "expected_tier": risk_to_tier(corrected_risk),
            "reasoning": f"hook outcome: {correction_type} (original_risk={original_risk:.3f})",
            "flags": ["human-feedback"],
            "scorer": "human-feedback",
            "original_risk": original_risk,
            "correction_type": correction_type,
        })

    return corrections


def deduplicate(corrections: list[dict], existing: set[str]) -> list[dict]:
    """Deduplicate corrections and skip commands already in corpus with same scorer."""
    seen: dict[str, dict] = {}
    for c in corrections:
        key = c["text"]
        # Keep the most recent correction for each command
        if key not in seen:
            seen[key] = c
        else:
            # Prefer hook outcomes over dashboard feedback
            if "hook" in c.get("correction_type", ""):
                seen[key] = c
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Harvest feedback into corpus corrections")
    parser.add_argument("--apply", action="store_true", help="Append corrections to scores-cache.jsonl")
    parser.add_argument("--retrain", action="store_true", help="Apply + retrain RiskHead MLP")
    args = parser.parse_args()

    existing = load_corpus_commands()

    dashboard = harvest_dashboard_feedback()
    hook = harvest_hook_traces()
    all_corrections = deduplicate(dashboard + hook, existing)

    # Separate new vs updates
    new = [c for c in all_corrections if c["text"] not in existing]
    updates = [c for c in all_corrections if c["text"] in existing]

    print(f"Dashboard feedback: {len(dashboard)} corrections")
    print(f"Hook traces:        {len(hook)} corrections")
    print(f"After dedup:        {len(all_corrections)} total")
    print(f"  New entries:      {len(new)}")
    print(f"  Corpus updates:   {len(updates)}")
    print()

    for c in all_corrections:
        arrow = "↓" if c["original_risk"] > c["risk"] else "↑"
        status = "NEW" if c["text"] not in existing else "UPD"
        print(f"  [{status}] {c['correction_type']:20s} {c['original_risk']:.3f} {arrow} {c['risk']:.3f}  {c['text'][:60]}")

    if not all_corrections:
        print("No corrections to apply.")
        return

    # Write corrections file
    with open(CORRECTIONS_FILE, "w") as f:
        for c in all_corrections:
            f.write(json.dumps(c) + "\n")
    print(f"\nWrote {len(all_corrections)} corrections to {CORRECTIONS_FILE}")

    if args.apply or args.retrain:
        if new:
            with open(CORPUS_FILE, "a") as f:
                for c in new:
                    # Strip harvest-specific fields before writing to corpus
                    entry = {k: v for k, v in c.items() if k not in ("original_risk", "correction_type")}
                    f.write(json.dumps(entry) + "\n")
            print(f"Appended {len(new)} new entries to {CORPUS_FILE}")

        if updates:
            # For updates: rewrite corpus with corrected risks
            lines = []
            update_map = {c["text"]: c for c in updates}
            with open(CORPUS_FILE) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry["text"] in update_map:
                            correction = update_map[entry["text"]]
                            entry["risk"] = correction["risk"]
                            entry["tier"] = correction["tier"]
                            entry["expected_tier"] = correction["tier"]
                            entry["flags"] = list(set(entry.get("flags", []) + ["human-feedback"]))
                            line = json.dumps(entry) + "\n"
                    except (json.JSONDecodeError, KeyError):
                        pass
                    lines.append(line)
            with open(CORPUS_FILE, "w") as f:
                f.writelines(lines)
            print(f"Updated {len(updates)} existing entries in {CORPUS_FILE}")

    if args.retrain:
        print("\nRetraining RiskHead MLP...")
        python = str(ROOT / "model" / ".venv" / "bin" / "python3")
        checkpoint = str(ROOT / "model" / "checkpoints" / "best.pt")
        corpus = str(CORPUS_FILE)
        cmd = [python, str(ROOT / "model" / "siamese.py"), "train-risk-head",
               "--checkpoint", checkpoint, "--corpus", corpus, "--epochs", "50"]
        subprocess.run(cmd, check=True)
        print("RiskHead retrained.")


if __name__ == "__main__":
    main()
