#!/usr/bin/env python3
"""
Ingest tldr-pages commands from /tmp/tldr_cmds.json into scores-cache.jsonl.

Format: [{"command": "...", "description": "..."}, ...]
Source: tldr agent a732d8a — 8,215 commands from common/linux/osx pages.

Tier assignment uses the same heuristic signal as pairs.py heuristic_blast_radius,
mapped to tiers with a simple rule table.  These are real-world documented commands,
so they skew T0–T1 with a long tail up to T3.
"""

from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path

SCORES_CACHE = Path("data/synthetic/scores-cache.jsonl")
TLDR_PATH = Path("/tmp/tldr_cmds.json")

# ---------------------------------------------------------------------------
# Lightweight heuristic (mirrors pairs.py logic, no import coupling)
# ---------------------------------------------------------------------------

_IRREVERSIBILITY = {
    "push", "send", "deploy", "drop", "delete", "rm", "truncate",
    "overwrite", "destroy", "nuke", "reset", "purge", "wipe",
    "revoke", "terminate", "kill",
}
_BOUNDARY = {
    "curl", "wget", "fetch", "http://", "https://", "upload",
    "email", "smtp", "webhook", "s3://", "gs://", "azure", "ssh",
    "scp", "rsync", "ftp", "sftp", "docker push", "npm publish",
    "git push", "aws ", "gcloud ", "kubectl apply", "kubectl delete",
}
_DRY_FLAGS = {"--dry-run", "-n", "--simulate", "--check", "--preview", "--no-act"}
_FORCE_FLAGS = {"-f", "--force", "--hard", "--no-backup", "--overwrite"}
_INTER_FLAGS = {"-i", "--interactive", "--confirm"}
_READ_CMDS = {
    "ls", "cat", "less", "more", "head", "tail", "grep", "find", "ps",
    "top", "htop", "df", "du", "stat", "file", "wc", "sort", "uniq",
    "diff", "git status", "git log", "git diff", "git show", "git branch",
    "kubectl get", "kubectl describe", "kubectl logs", "docker ps",
    "docker images", "docker inspect", "systemctl status", "journalctl",
    "ping", "traceroute", "nslookup", "dig", "host", "whois",
    "man", "help", "which", "whereis", "type", "echo", "printf",
    "date", "uptime", "who", "w", "id", "whoami", "hostname", "uname",
    "env", "printenv", "set",
}


def _flags(cmd: str) -> set[str]:
    tokens = cmd.split()
    result: set[str] = set()
    for t in tokens:
        if t.startswith("--"):
            result.add(t.split("=")[0])
        elif t.startswith("-") and len(t) > 1:
            for c in t[1:]:
                result.add(f"-{c}")
    return result


def assign_tier(cmd: str, desc: str) -> tuple[int, float]:
    """Return (tier, risk_score) for a tldr command."""
    low = cmd.lower()
    desc_low = desc.lower()
    flags = _flags(cmd)

    # T-1: dry-run/preview flags
    if flags & _DRY_FLAGS:
        return -1, 0.15

    # Check read-only prefix
    first_token = low.split()[0] if low.split() else ""
    is_read = any(low.startswith(r) for r in _READ_CMDS)
    is_read = is_read or any(
        kw in desc_low for kw in ("display", "list", "show", "print", "view",
                                   "check", "query", "inspect", "read", "search")
    )

    # T3/T4 signals: external writes
    has_boundary = any(t in low for t in _BOUNDARY)
    has_force = bool(flags & _FORCE_FLAGS)

    if has_boundary:
        if has_force or "delete" in low or "destroy" in low:
            return 4, 0.90
        # git push / docker push / etc without force
        if any(t in low for t in ("push", "deploy", "publish", "apply")):
            return 3, 0.65
        return 3, 0.55

    # T2: local destructive
    has_irr = any(t in low for t in _IRREVERSIBILITY)
    if has_irr:
        if has_force:
            return 2, 0.50
        return 2, 0.40

    # T1: local write (non-destructive)
    _LOCAL_WRITE = {
        "mkdir", "touch", "cp", "mv", "ln", "chmod", "chown",
        "git add", "git commit", "git checkout", "git merge", "git rebase",
        "npm install", "pip install", "apt install", "brew install",
        "make", "cmake", "cargo build", "go build",
        "docker build", "docker run", "docker create",
        "systemctl start", "systemctl enable", "systemctl restart",
        "crontab",
    }
    is_write = any(low.startswith(w) for w in _LOCAL_WRITE) or any(
        kw in desc_low for kw in ("install", "create", "build", "write",
                                   "add", "configure", "set ", "enable")
    )

    if is_write:
        return 1, 0.28

    # T0: read-only default for tldr (docs describe safe usage)
    if is_read or not has_irr:
        return 0, 0.08

    return 0, 0.08


# ---------------------------------------------------------------------------
# Dedup + append
# ---------------------------------------------------------------------------

def load_existing(path: Path) -> set[str]:
    texts: set[str] = set()
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


def main() -> None:
    if not TLDR_PATH.exists():
        print(f"✗ {TLDR_PATH} not found", file=sys.stderr)
        sys.exit(1)

    raw = json.load(open(TLDR_PATH))
    print(f"Loaded {len(raw):,} tldr commands")

    existing = load_existing(SCORES_CACHE)
    print(f"Existing cache: {len(existing):,} entries")

    tier_counts: dict[int, int] = {}
    added = 0
    skipped_dupe = 0
    skipped_short = 0

    with open(SCORES_CACHE, "a") as f:
        for item in raw:
            cmd = item.get("command", "").strip()
            desc = item.get("description", "").strip()
            if len(cmd) < 4:
                skipped_short += 1
                continue
            if cmd in existing:
                skipped_dupe += 1
                continue

            tier, risk = assign_tier(cmd, desc)
            blast = round(risk * 0.85, 4)
            uid = hashlib.sha256(cmd.encode()).hexdigest()[:12]

            entry = {
                "id": uid,
                "tool": "bash",
                "args": {"cmd": cmd},
                "text": cmd,
                "risk": round(risk, 4),
                "tier": tier,
                "expected_tier": tier,
                "reasoning": f"tldr-pages: {desc[:120]}",
                "flags": [],
                "scorer": "tldr-heuristic",
                "heuristic_blast": blast,
                "heuristic_delta": 0.0,
                "source": "tldr_pages",
            }
            f.write(json.dumps(entry) + "\n")
            existing.add(cmd)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            added += 1

    print(f"\nAdded:          {added:,}")
    print(f"Skipped (dupe): {skipped_dupe:,}")
    print(f"Skipped (short):{skipped_short:,}")
    print(f"\nTier distribution of added entries:")
    for t in sorted(tier_counts):
        label = f"T{t}" if t >= 0 else f"T{t}"
        print(f"  {label}: {tier_counts[t]:,}")
    print(f"\nNew cache size: {len(existing):,}")


if __name__ == "__main__":
    main()
