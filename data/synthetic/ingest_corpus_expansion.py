"""
Ingest corpus expansion files from Haiku generation agents.
Validates tier/risk ranges, dedupes against existing corpus, writes to scores-cache.jsonl.

Usage:
    python data/synthetic/ingest_corpus_expansion.py /tmp/corpus_gap_*.jsonl

    # Review mode — print entries for human approval before writing:
    python data/synthetic/ingest_corpus_expansion.py --review /tmp/traces_*.jsonl

    # Load a denylist of patterns to exclude (one regex per line):
    python data/synthetic/ingest_corpus_expansion.py --denylist .corpus-denylist /tmp/traces_*.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path("data/synthetic/scores-cache.jsonl")
DEFAULT_DENYLIST = Path(".corpus-denylist")

TIER_RISK_RANGES = {
    -2: (0.85, 1.00),
    -1: (0.08, 0.22),
     0: (0.00, 0.15),
     1: (0.20, 0.40),
     2: (0.35, 0.55),
     3: (0.55, 0.75),
     4: (0.75, 1.00),
}


def load_denylist(path: Path) -> list[re.Pattern]:
    """Load newline-separated regex patterns from a denylist file."""
    if not path.exists():
        return []
    patterns = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                patterns.append(re.compile(line, re.IGNORECASE))
            except re.error as e:
                print(f"  [denylist] invalid pattern {line!r}: {e}", file=sys.stderr)
    return patterns


def is_denied(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


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
        "scorer": raw.get("scorer", "expert-labeled"),
        "heuristic_blast": 0.0,
        "heuristic_delta": round(float(raw["risk"]), 4),
        "source": source,
        "ts": now,
    }


def review_entry(raw: dict) -> bool:
    """Prompt for human approval. Returns True to include, False to skip."""
    text = raw.get("text", "")[:120]
    tier = raw.get("tier")
    risk = raw.get("risk")
    print(f"\n  T{tier:+d}  risk={risk:.2f}  {text}")
    while True:
        resp = input("  Include? [y/n/q] ").strip().lower()
        if resp == "y":
            return True
        if resp == "n":
            return False
        if resp == "q":
            print("Aborted.")
            sys.exit(0)


def main(input_files: list[str], review: bool, denylist_path: Path):
    deny_patterns = load_denylist(denylist_path)
    if deny_patterns:
        print(f"Denylist: {len(deny_patterns)} patterns loaded from {denylist_path}")

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

    total_added = total_skipped = total_invalid = total_denied = 0

    with open(CORPUS, "a") as out:
        for path in sorted(input_files):
            p = Path(path)
            if not p.exists():
                print(f"  SKIP {path} (not found)")
                continue

            added = skipped = invalid = denied = 0
            source = p.stem

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

                    if is_denied(text, deny_patterns):
                        denied += 1
                        continue

                    if text in existing:
                        skipped += 1
                        continue

                    if review and not review_entry(raw):
                        denied += 1
                        continue

                    entry = make_entry(raw, source)
                    out.write(json.dumps(entry) + "\n")
                    existing.add(text)
                    added += 1

            denied_str = f", {denied} filtered" if denied else ""
            print(f"  {p.name}: +{added} added, {skipped} dupes, {invalid} invalid{denied_str}")
            total_added += added
            total_skipped += skipped
            total_invalid += invalid
            total_denied += denied

    print(f"\nTotal: +{total_added} added, {total_skipped} dupes, {total_invalid} invalid", end="")
    if total_denied:
        print(f", {total_denied} filtered", end="")
    print()
    print(f"Corpus now: {len(existing):,} entries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest corpus expansion files")
    ap.add_argument("files", nargs="+", help="Input JSONL files")
    ap.add_argument("--review", action="store_true",
                    help="Interactively approve each entry before writing")
    ap.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST,
                    help=f"Path to denylist file (default: {DEFAULT_DENYLIST})")
    args = ap.parse_args()
    main(args.files, args.review, args.denylist)
