"""
Ingest corpus expansion files from Haiku generation agents.
Validates tier/risk ranges, dedupes against existing corpus, writes to scores-cache.jsonl.

Usage:
    python data/synthetic/ingest_corpus_expansion.py /tmp/corpus_gap_*.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path("data/synthetic/scores-cache.jsonl")

TIER_RISK_RANGES = {
    -2: (0.85, 1.00),
    -1: (0.08, 0.22),
     0: (0.00, 0.15),
     1: (0.20, 0.40),
     2: (0.35, 0.55),
     3: (0.55, 0.75),
     4: (0.75, 1.00),
}


def validate(entry: dict) -> str | None:
    """Return error string if invalid, else None."""
    if "text" not in entry:
        return "missing 'text'"
    if not isinstance(entry.get("risk"), (int, float)):
        return "missing or non-numeric 'risk'"
    if not isinstance(entry.get("tier"), int):
        return "missing or non-integer 'tier'"
    tier = entry["tier"]
    risk = float(entry["risk"])
    if tier not in TIER_RISK_RANGES:
        return f"invalid tier {tier}"
    lo, hi = TIER_RISK_RANGES[tier]
    # Allow 0.05 tolerance for risk values slightly outside expected range
    if not (lo - 0.05 <= risk <= hi + 0.05):
        return f"risk {risk:.2f} out of range [{lo},{hi}] for tier {tier}"
    return None


def make_entry(raw: dict, source: str) -> dict:
    text = raw["text"].strip()
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": hashlib.md5(text.encode()).hexdigest()[:16],
        "tool": "bash",
        "args": {"command": text},
        "text": text,
        "risk": round(float(raw["risk"]), 4),
        "tier": int(raw["tier"]),
        "expected_tier": int(raw["tier"]),
        "reasoning": raw.get("reasoning", ""),
        "flags": [],
        "scorer": "haiku-4-5",
        "heuristic_blast": 0.0,
        "heuristic_delta": round(float(raw["risk"]), 4),
        "source": source,
        "ts": now,
    }


def main(input_files: list[str]):
    # Load existing texts for dedup
    existing: set[str] = set()
    with open(CORPUS) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    e = json.loads(line)
                    existing.add(e.get("text", ""))
                except Exception:
                    pass
    print(f"Existing corpus: {len(existing):,} entries")

    total_added = total_skipped = total_invalid = 0

    with open(CORPUS, "a") as out:
        for path in sorted(input_files):
            p = Path(path)
            if not p.exists():
                print(f"  SKIP {path} (not found)")
                continue

            added = skipped = invalid = 0
            source = p.stem  # e.g. "corpus_gap_1"

            with open(p) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  {path}:{lineno} JSON error: {e}")
                        invalid += 1
                        continue

                    err = validate(raw)
                    if err:
                        print(f"  {path}:{lineno} INVALID ({err}): {raw.get('text', '')[:60]}")
                        invalid += 1
                        continue

                    text = raw["text"].strip()
                    if text in existing:
                        skipped += 1
                        continue

                    entry = make_entry(raw, source)
                    out.write(json.dumps(entry) + "\n")
                    existing.add(text)
                    added += 1

            print(f"  {p.name}: +{added} added, {skipped} dupes, {invalid} invalid")
            total_added += added
            total_skipped += skipped
            total_invalid += invalid

    print(f"\nTotal: +{total_added} added, {total_skipped} dupes, {total_invalid} invalid")
    print(f"Corpus now: {len(existing):,} entries")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_corpus_expansion.py /tmp/corpus_gap_*.jsonl")
        sys.exit(1)
    main(sys.argv[1:])
