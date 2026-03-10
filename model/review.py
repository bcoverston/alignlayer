"""
AlignLayer — Human review CLI for pending interrupts.

Reads pending-interrupts.jsonl, presents each for human decision,
writes approved/denied annotations to corrections.jsonl in contrastive
pair format compatible with pairs.py ingestion.

Usage:
    python model/review.py --pending data/traces/pending-interrupts.jsonl
    python model/review.py --pending data/traces/pending-interrupts.jsonl --out data/traces/corrections.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def load_interrupts(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def tier_label(tier: int) -> str:
    labels = {
        -2: "T-2 (exfil/attack)",
        -1: "T-1 (dry-run)",
         0: "T0  (read-only)",
         1: "T1  (local write)",
         2: "T2  (local destructive)",
         3: "T3  (external write)",
         4: "T4  (catastrophic)",
    }
    return labels.get(tier, f"T{tier}")


def format_entry(entry: dict, idx: int, total: int) -> str:
    tool = entry.get("tool", "unknown")
    args = entry.get("args", {})
    cmd = args.get("command", args.get("cmd", args.get("input", "")))

    risk = entry.get("risk_score")
    blast = entry.get("blast_radius")
    tier = entry.get("decision", "unknown")

    ml_risk = entry.get("ml_risk")
    ml_tier = entry.get("ml_tier")

    lines = [
        f"\n[{idx}/{total}] {tool}: {cmd}",
        f"      Heuristic: risk={risk:.2f} blast={blast:.2f} → {tier}" if risk is not None else "",
    ]
    if ml_risk is not None:
        lines.append(f"      ML:        risk={ml_risk:.2f} tier={tier_label(ml_tier)}" if ml_tier is not None else f"      ML:        risk={ml_risk:.2f}")

    return "\n".join(l for l in lines if l)


def to_correction(entry: dict, decision: str) -> dict:
    """Convert an interrupt entry + human decision to a correction record."""
    args = entry.get("args", {})
    cmd = str(args.get("command", args.get("cmd", args.get("input", ""))))

    heuristic_risk = entry.get("risk_score", 0.5)

    if decision == "approve":
        # Human says this was fine — actual risk is lower than heuristic thought
        corrected_risk = max(0.0, heuristic_risk - 0.3)
        corrected_tier = 1  # demote to local write territory
    else:
        # Human confirms danger — actual risk is at or above heuristic
        corrected_risk = min(1.0, heuristic_risk + 0.1)
        corrected_tier = 4  # confirm high risk

    return {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
        "risk": round(corrected_risk, 4),
        "tier": corrected_tier,
        "scorer": "human-review",
        "source": "review-cli",
        "original_risk": round(heuristic_risk, 4),
        "human_decision": decision,
    }


def review(pending_path: str, output_path: str) -> None:
    entries = load_interrupts(pending_path)
    if not entries:
        print("No pending interrupts found.")
        return

    print(f"Loaded {len(entries)} pending interrupt(s) from {pending_path}")
    corrections: list[dict] = []
    reviewed = 0

    for i, entry in enumerate(entries, 1):
        print(format_entry(entry, i, len(entries)))
        while True:
            try:
                choice = input("      → [a]pprove  [d]eny  [s]kip  [q]uit? ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice in ("a", "approve"):
                corrections.append(to_correction(entry, "approve"))
                reviewed += 1
                break
            elif choice in ("d", "deny"):
                corrections.append(to_correction(entry, "deny"))
                reviewed += 1
                break
            elif choice in ("s", "skip"):
                break
            elif choice in ("q", "quit"):
                print(f"\nQuitting. Reviewed {reviewed}/{len(entries)}.")
                if corrections:
                    _write_corrections(corrections, output_path)
                return
            else:
                print("      Invalid choice. Use a/d/s/q.")

    print(f"\nReview complete. {reviewed}/{len(entries)} reviewed.")
    if corrections:
        _write_corrections(corrections, output_path)


def _write_corrections(corrections: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for c in corrections:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(corrections)} correction(s) to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="AlignLayer review CLI")
    p.add_argument("--pending", required=True, help="Path to pending-interrupts.jsonl")
    p.add_argument("--out", default="data/traces/corrections.jsonl", help="Output corrections file")
    args = p.parse_args()

    if not Path(args.pending).exists():
        print(f"File not found: {args.pending}", file=sys.stderr)
        sys.exit(1)

    review(args.pending, args.out)


if __name__ == "__main__":
    main()
