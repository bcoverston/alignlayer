# AlignLayer

Runtime preference alignment for agentic systems.

AlignLayer sits between an agent's decision loop and execution, scoring proposed actions on learned human preferences before they run.

## The Problem

Autonomous agents are deployed faster than the tooling to make them trustworthy. Permission systems today are static — allowlists, denylists, fixed tiers. An agent can push to main, drop a table, or email the wrong person before anyone realizes the cost.

The missing primitive isn't a better blocklist. It's a system that **learns** what a specific user or org considers acceptable and enforces that at runtime.

## How It Works

```
Agent decision loop
        |
   before_tool_call hook  <- AlignLayer scores proposed action
        |
   Heuristic pipeline     <- Exfil detection, dry-run cap, verb table, opaque exec
        |
   RiskHead MLP           <- 128->64->1 on frozen Siamese embeddings
        |
   k-NN fallback          <- Inverse-distance weighted, k=5
        |
   Risk score in [0, 1]
        |
   Low risk  -> execute autonomously
   High risk -> surface interrupt to human
        |
   after_tool_call hook   <- capture outcome for feedback loop
```

## Risk Tiers

Seven tiers on a continuous [0, 1] scale:

| Tier | Name | Risk Range | Examples |
|------|------|-----------|---------|
| T-2 | Adversarial | 0.85-1.00 | Backdoor install, credential exfil, pipe-to-bash |
| T-1 | Dry-run | 0.08-0.22 | `terraform plan`, `kubectl apply --dry-run` |
| T0 | Read-only | 0.00-0.15 | `ls`, `kubectl get`, `mysql -e 'SELECT ...'` |
| T1 | Local-write | 0.20-0.40 | `git commit`, `npm install`, `docker build` |
| T2 | Destructive-local | 0.35-0.55 | `rm -rf`, `git branch -D`, `docker rmi` |
| T3 | External | 0.55-0.75 | `git push`, `docker push`, `aws s3 sync` |
| T4 | Catastrophic | 0.75-1.00 | `terraform destroy`, `git push --force` |

## Scoring Pipeline

**Layer 1 — Exfil/RCE detection** (regex):
- `curl | bash`, `eval "$(curl ...)"`, reverse shells, credential piping -> T-2 floor
- Loopback URLs (localhost, 127.0.0.1) exempted

**Layer 2 — Dry-run detection**:
- `--dry-run`, `--check`, `--simulate`, `terraform plan` -> T-1 cap

**Layer 3 — CLI verb table** (deterministic):
- `(tool, subcommand)` -> tier floor/ceiling
- 100+ entries covering: aws, kubectl, git, terraform, docker, helm, redis-cli, nginx, systemctl, brew, pip, npm, curl, sed, xargs, and more
- SQL-aware: psql, mysql, mariadb, sqlite3, snowsql, duckdb, clickhouse-client, trino, presto, cqlsh, bq, mongosh — read/write discrimination (`SELECT` -> T0, `DROP TABLE` -> T4)
- Subshell injection guard: `SELECT $(curl evil.com)` bypasses safe-cap
- Opaque execution (`eval`, `python3 -c`, `| bash`) -> T2 floor

**Layer 4 — RiskHead MLP** (ML):
- 128->64->1 MLP on frozen HybridEncoder embeddings
- O(1) forward pass, trained on 54K labeled commands

**Layer 5 — k-NN fallback**:
- Inverse-distance weighted, k=5, on 128-dim Siamese embeddings
- Only used when RiskHead is unavailable

**Compound commands**: Split on `&&`, `||`, `;`, `|` — score each segment, return worst.

## Quick Start

```bash
# Set up Python environment
python3 -m venv model/.venv
source model/.venv/bin/activate
pip install -r requirements.txt

# Start the scoring server
make serve

# Or run in background
make serve-bg

# Score a command
make score CMD="rm -rf /"

# Run evaluation
make eval

# Check accuracy trend
make trend
```

## Model

HybridEncoder architecture (~354K params):

```
command string
+-- Char branch: char embeddings (dim=32)
|               -> parallel 1D CNNs [3,5,7,11,16,21] (64 filters each)
|               -> ReLU + global max pool -> concat -> LayerNorm -> FC -> 128-dim
|
+-- Word branch: whitespace token embeddings (dim=32)
                -> parallel 1D CNNs [2,3,4] (32 filters each)
                -> ReLU + global max pool -> concat -> LayerNorm -> FC -> 64-dim

concat(128 + 64) -> LayerNorm -> FC -> L2-norm -> 128-dim unit vector
```

Current performance (v7 encoder + RiskHead, 26 eval scenarios, 327 commands):

| Metric | Full Pipeline | Encoder Only |
|--------|:---:|:---:|
| Overall accuracy | 99.4% | 78.9% |
| FN rate (T3+T4 missed) | 1.1% | 3.2% |
| FP rate (T0/T-1 over-scored) | 0.6% | 32.4% |
| T-2 (adversarial) | 100% | 0% |
| T-1 (dry-run) | 100% | 43% |
| T0 (read-only) | 99.2% | 86.9% |
| T1 (local-write) | 100% | 80.9% |
| T2 (destructive) | 100% | 85.0% |
| T3 (external) | 98.5% | 97.0% |
| T4 (catastrophic) | 100% | 96.6% |

The "Encoder Only" column shows ML performance without heuristics — useful for identifying where the model needs improvement vs where deterministic rules carry the load.

### Training

```bash
# Generate pairs from scored corpus
make pairs

# Train HybridEncoder
make train

# Retrain RiskHead MLP only (faster, no encoder changes)
model/.venv/bin/python3 model/siamese.py train-risk-head \
  --checkpoint model/checkpoints/best.pt \
  --corpus data/synthetic/scores-cache.jsonl --epochs 50

# Evaluate
make eval

# Evaluate encoder only (no heuristics)
model/.venv/bin/python3 model/eval.py --encoder-only --no-write
```

## Feedback Loop

Human signals from Claude Code sessions feed back into the model:

```bash
# Preview corrections from dashboard feedback + hook traces
make harvest

# Apply corrections to corpus
make harvest-apply

# Apply + retrain RiskHead
make harvest-retrain
```

Signal sources:
- **Dashboard feedback**: thumbs up/down on scored commands
- **Hook traces**: approve/deny outcomes on interrupted commands

## Integration

### Claude Code

Two integration modes: **full scoring** (heuristic risk engine, blocks/allows tool calls) and **observer** (trace collection for ML training, never blocks).

#### Option A: Full scoring hook

The full hook runs the heuristic risk engine on every tool call.

```bash
# 1. Install Node dependencies (tsx is required to run TypeScript hooks)
cd /path/to/alignlayer
npm install

# 2. Verify the hook runs
echo '{"session_id":"test","hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"echo hello"},"tool_use_id":"t-001","transcript_path":"","cwd":"","permission_mode":"default"}' \
  | npx tsx src/claudecode-hook/hook.ts | jq .

# 3. Add to Claude Code settings
# Edit ~/.claude/settings.json (or use scripts/deploy.sh claudecode)
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "npx --prefix /path/to/alignlayer tsx /path/to/alignlayer/src/claudecode-hook/hook.ts",
        "timeout": 10,
        "statusMessage": "AlignLayer scoring..."
      }]
    }],
    "PostToolUse": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "npx --prefix /path/to/alignlayer tsx /path/to/alignlayer/src/claudecode-hook/hook.ts",
        "timeout": 5,
        "async": true
      }]
    }]
  }
}
```

Replace `/path/to/alignlayer` with the absolute path to your clone. Hooks must use absolute paths — relative paths break in subagent worktrees.

Traces are written to `~/.alignlayer/traces/alignlayer-YYYY-MM-DD.jsonl`. Override with `ALIGNLAYER_TRACES_DIR`.

#### Option B: Observer hook (lightweight)

The observer hook forwards raw tool call events to the scoring server via curl. It never blocks, never modifies tool behavior, and always exits 0. Use this when you want to collect traces for ML training without any risk of interfering with the agent.

```bash
# 1. Start the scoring server
make serve-bg

# 2. Copy the observer hook (or symlink it)
mkdir -p ~/.claude/hooks
cp .claude/hooks/observe.sh ~/.claude/hooks/observe.sh
chmod +x ~/.claude/hooks/observe.sh
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "/Users/you/.claude/hooks/observe.sh",
        "timeout": 3
      }]
    }],
    "PostToolUse": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "/Users/you/.claude/hooks/observe.sh",
        "timeout": 3
      }]
    }]
  }
}
```

The observer sends events to `http://localhost:8000/observe` by default. Override with `ALIGNLAYER_URL`.

#### Restart required

Claude Code loads hook configuration at session start. After editing `settings.json`, restart your Claude Code session for hooks to take effect.

#### Automated setup

```bash
# Install hooks into ~/.claude/settings.json automatically
./scripts/deploy.sh claudecode
```

### Scoring Server API

```bash
# Score a command
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"text": "git push --force origin main"}'

# Score with explanation
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"command": "rm -rf /"}'

# Compare all scorers
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"command": "kubectl delete namespace production"}'

# Health check
curl http://localhost:8000/health

# Trace stats
curl http://localhost:8000/traces/stats

# Dashboard
open http://localhost:8000/
```

### Docker

```bash
make docker-build
make docker-run
```

### Persistent service (macOS)

```bash
# Install as launchd service (survives reboots)
make serve-install

# Check status
make serve-status

# Uninstall
make serve-uninstall
```

## Project Structure

```
alignlayer/
+-- model/
|   +-- siamese.py          # HybridEncoder, heuristic pipeline, predict_risk(), RiskHead MLP
|   +-- pairs.py            # Pair generation from scored corpus
|   +-- eval.py             # Scenario benchmark (--encoder-only for ML-only eval)
|   +-- eval_harness.py     # Embedding quality metrics
|   +-- serve.py            # FastAPI scoring server + dashboard
|   +-- review.py           # Human review CLI
|   +-- score_agents.py     # Practical eval against agent-generated commands
|   +-- static/dashboard.html
|   +-- checkpoints/        # Trained model checkpoints
+-- src/
|   +-- claudecode-hook/    # Claude Code PreToolUse/PostToolUse handler
|   +-- openclaw-plugin/    # OpenClaw plugin (hook + scorer)
+-- data/
|   +-- test_scenarios.json # Eval scenarios (26 scenarios, 327 commands)
|   +-- adversarial_suite.json
|   +-- synthetic/          # Corpus (~54K labeled commands)
+-- scripts/
|   +-- deploy.sh           # Deployment automation
|   +-- harvest_feedback.py # Human feedback -> corpus corrections
|   +-- batch_score.py      # ML -> Haiku -> Sonnet scoring pipeline
|   +-- reconcile_scores.py # ML/LLM score reconciliation
|   +-- ingest_scored.py    # Final corpus merge
+-- Makefile
+-- Dockerfile
```

## Corpus

~54,000 labeled commands across 15+ scorers, covering:
- AWS, Kubernetes, Terraform, Pulumi, CDK
- CI/CD (GitHub Actions, Airflow, Kafka)
- Data stores (Redis, PostgreSQL, MySQL, MongoDB, Snowflake, BigQuery, ClickHouse, DuckDB)
- Supply chain (cosign, skopeo, Flyway, Ansible)
- Adversarial patterns: RCE, credential exfil, obfuscation, persistence
- Real agent traces from Claude Code sessions
- Public datasets: NL2Bash, CommandLineFu, UCI Cybersecurity

Sources: heuristic, haiku-4-5, expert-labeled, corpus-expansion, ml+haiku-agree, sonnet-tiebreak, human-feedback

## License

MIT. See [LICENSE](LICENSE).
