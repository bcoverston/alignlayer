#!/usr/bin/env python3
"""
Ingest boundary-targeted expert-generated commands into scores-cache.jsonl.

Sources:
  1. Agent aa0f2d7 output  → T-1/T0 (160 entries)
  2. Agent a7f54cf output  → T3/T4  (140 entries)
  3. /tmp/boundary_out.json → T2/T3  (160 entries, from gen_boundary.py)
  4. /tmp/dryrun_cmds.json → T-1    (123 entries, no scores)

Skipped:
  - /tmp/mitre_cmds.json (paths/fragments, not usable shell commands)
"""

from __future__ import annotations

import json
import re
import sys
import hashlib
import os
from pathlib import Path

SCORES_CACHE = Path("data/synthetic/scores-cache.jsonl")
TASK_OUTPUT_DIR = Path(os.environ.get(
    "ALIGNLAYER_TASK_OUTPUT_DIR",
    "/private/tmp/claude-501/tasks",
))


# ---------------------------------------------------------------------------
# Extract JSON array from agent output JSONL
# ---------------------------------------------------------------------------

def extract_json_array_from_agent_output(agent_id: str) -> list[dict]:
    """Parse agent output JSONL and find the largest JSON array in assistant messages."""
    output_path = TASK_OUTPUT_DIR / f"{agent_id}.output"
    if not output_path.exists():
        print(f"  ✗ {agent_id}.output not found")
        return []

    best: list = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message", {})
            if msg.get("role") != "assistant":
                continue
            for block in msg.get("content", []):
                if block.get("type") != "text":
                    continue
                text = block.get("text", "")
                if len(text) < 100:
                    continue
                # Extract JSON array by bracket matching (handles code-block wrappers)
                start = text.find("[")
                end = text.rfind("]")
                if start >= 0 and end > start:
                    try:
                        arr = json.loads(text[start : end + 1])
                        if isinstance(arr, list) and len(arr) > len(best):
                            best = arr
                    except Exception:
                        pass
    return best


# ---------------------------------------------------------------------------
# Convert to scores-cache entry
# ---------------------------------------------------------------------------

TIER_RISK_DEFAULTS = {
    -2: 0.20,
    -1: 0.15,
     0: 0.05,
     1: 0.30,
     2: 0.45,
     3: 0.62,
     4: 0.90,
}


def make_entry(cmd: str, tier: int, risk: float, blast: float, source: str) -> dict:
    uid = hashlib.sha256(cmd.encode()).hexdigest()[:12]
    return {
        "id": uid,
        "tool": "bash",
        "args": {"cmd": cmd},
        "text": cmd,
        "risk": round(risk, 4),
        "tier": tier,
        "expected_tier": tier,
        "reasoning": f"Expert-generated boundary example (source={source})",
        "flags": [],
        "scorer": "expert-generated",
        "heuristic_blast": round(blast, 4),
        "heuristic_delta": round(abs(risk - TIER_RISK_DEFAULTS.get(tier, 0.5)), 4),
        "source": source,
    }


def convert_scored(entries: list[dict], source: str) -> list[dict]:
    """Convert entries with {command, risk_score, blast_radius, tier}."""
    out = []
    for e in entries:
        cmd = e.get("command", "").strip()
        if not cmd:
            continue
        tier = int(e.get("tier", 0))
        risk = float(e.get("risk_score", TIER_RISK_DEFAULTS.get(tier, 0.5)))
        blast = float(e.get("blast_radius", risk * 0.9))
        out.append(make_entry(cmd, tier, risk, blast, source))
    return out


def convert_dryrun(entries: list[dict]) -> list[dict]:
    """Convert dryrun_cmds.json: {command, description}, all T-1."""
    out = []
    for e in entries:
        cmd = e.get("command", "").strip()
        if not cmd or len(cmd) < 4:
            continue
        out.append(make_entry(cmd, tier=-1, risk=0.15, blast=0.07, source="dryrun_mining"))
    return out


# ---------------------------------------------------------------------------
# Dedup and append
# ---------------------------------------------------------------------------

def load_existing_texts(path: Path) -> set[str]:
    texts = set()
    if not path.exists():
        return texts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.add(json.loads(line)["text"])
            except Exception:
                pass
    return texts


def append_entries(path: Path, entries: list[dict], existing: set[str]) -> int:
    added = 0
    with open(path, "a") as f:
        for e in entries:
            if e["text"] in existing:
                continue
            existing.add(e["text"])
            f.write(json.dumps(e) + "\n")
            added += 1
    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading existing scores cache...")
    existing = load_existing_texts(SCORES_CACHE)
    print(f"  {len(existing):,} existing entries")

    total_added = 0

    # Source 1: T-1/T0 boundary from agent aa0f2d7
    print("\nSource 1: T-1↔T0 expert (aa0f2d7)...")
    raw = extract_json_array_from_agent_output("aa0f2d7")
    print(f"  Extracted {len(raw)} entries")
    entries = convert_scored(raw, "boundary_t1_t0")
    added = append_entries(SCORES_CACHE, entries, existing)
    print(f"  Added {added} (skipped {len(entries)-added} dupes)")
    total_added += added

    # Source 2: T3/T4 boundary from agent a7f54cf
    print("\nSource 2: T3↔T4 expert (a7f54cf)...")
    raw = extract_json_array_from_agent_output("a7f54cf")
    print(f"  Extracted {len(raw)} entries")
    entries = convert_scored(raw, "boundary_t3_t4")
    added = append_entries(SCORES_CACHE, entries, existing)
    print(f"  Added {added} (skipped {len(entries)-added} dupes)")
    total_added += added

    # Source 3: T2/T3 boundary from gen_boundary.py
    print("\nSource 3: T2↔T3 expert (/tmp/boundary_out.json)...")
    boundary_path = Path("/tmp/boundary_out.json")
    if boundary_path.exists():
        raw = json.load(open(boundary_path))
        print(f"  Loaded {len(raw)} entries")
        entries = convert_scored(raw, "boundary_t2_t3")
        added = append_entries(SCORES_CACHE, entries, existing)
        print(f"  Added {added} (skipped {len(entries)-added} dupes)")
        total_added += added
    else:
        print("  ✗ not found")

    # Source 4: T-1 dryrun patterns
    print("\nSource 4: T-1 dryrun patterns (/tmp/dryrun_cmds.json)...")
    dryrun_path = Path("/tmp/dryrun_cmds.json")
    if dryrun_path.exists():
        raw = json.load(open(dryrun_path))
        print(f"  Loaded {len(raw)} entries")
        entries = convert_dryrun(raw)
        added = append_entries(SCORES_CACHE, entries, existing)
        print(f"  Added {added} (skipped {len(entries)-added} dupes)")
        total_added += added
    else:
        print("  ✗ not found")

    print(f"\nTotal added: {total_added:,}")
    print(f"New cache size: {len(existing):,}")


if __name__ == "__main__":
    main()
