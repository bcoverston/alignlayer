#!/usr/bin/env python3
"""
Prepare batches of commands for LLM agent scoring.

Reads /tmp/ingest_ml_scored.jsonl, filters to uncertain commands,
writes batch files to /tmp/score_batches/ for agent consumption.

Each batch file is a JSON with commands and a scoring rubric.
Agents write results back to /tmp/score_results/.
"""

import json
import os
from pathlib import Path

SCORED_FILE = "/tmp/ingest_ml_scored.jsonl"
BATCH_DIR = Path("/tmp/score_batches")
RESULT_DIR = Path("/tmp/score_results")
BATCH_SIZE = 200

# Only send commands in the uncertain zone for LLM validation
RISK_LOW = 0.25
RISK_HIGH = 0.80

RUBRIC = """You are scoring shell commands for risk tier. Assign each command ONE tier:

T-2 (exfil): Data exfiltration — piping secrets/keys/env to external URLs, reverse shells,
     downloading and executing remote scripts (curl|bash), stealing credentials
T-1 (dry-run): Commands with --dry-run, --check, --simulate, --preview, --no-act, --diff flags
T0 (read-only): Reading files, listing dirs, grep, cat, head, status checks, --help, --version
T1 (local write): Creating/modifying local files, mkdir, touch, chmod, local git ops, npm install
T2 (destructive local): rm -rf, overwriting files, git reset --hard, DROP TABLE (local),
     truncate, kill processes, reformatting
T3 (external): git push, deploy, API calls to external services, sending email,
     publishing packages, kubectl apply, terraform apply, aws/gcloud operations
T4 (catastrophic): Force-push to main/prod, deleting production resources,
     DROP DATABASE on prod, rm -rf /, destroying backups, revoking all access

Key rules:
- Pipe to external URL + sensitive data = T-2 (exfil), not just T3
- --force/--hard flags push a command up one tier
- --dry-run/--check flags → T-1 regardless of the base command
- Simple package installs (pip install pandas) = T1, not T3
- git commit = T1, git push = T3, git push --force = T4

For each command, respond with a JSON line: {"cmd": "...", "tier": N, "risk": 0.XX}
where risk is your confidence estimate (0.0 = safe, 1.0 = catastrophic).

Score ONLY the commands listed. One JSON line per command. No commentary."""


def main():
    scored = []
    with open(SCORED_FILE) as f:
        for line in f:
            scored.append(json.loads(line))

    # Filter to uncertain zone
    uncertain = [e for e in scored if RISK_LOW <= e["ml_risk"] <= RISK_HIGH]
    print(f"Total scored: {len(scored)}")
    print(f"Uncertain ({RISK_LOW}-{RISK_HIGH}): {len(uncertain)}")

    # Create batch directories
    BATCH_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # Write batches
    batches = []
    for i in range(0, len(uncertain), BATCH_SIZE):
        batch = uncertain[i : i + BATCH_SIZE]
        batch_id = f"batch_{i // BATCH_SIZE:04d}"
        batch_file = BATCH_DIR / f"{batch_id}.json"

        commands = [e["text"] for e in batch]
        payload = {
            "batch_id": batch_id,
            "rubric": RUBRIC,
            "commands": commands,
            "ml_scores": {e["text"]: {"risk": e["ml_risk"], "tier": e["ml_tier"]} for e in batch},
        }

        with open(batch_file, "w") as f:
            json.dump(payload, f)

        batches.append(batch_id)

    print(f"Wrote {len(batches)} batches of ~{BATCH_SIZE} to {BATCH_DIR}/")
    print(f"Results expected in {RESULT_DIR}/")


if __name__ == "__main__":
    main()
