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
   Heuristic pre-check    <- CLI verb table (fast, deterministic)
        |
   ML scorer              <- k-NN on Siamese embeddings
        |
   Risk score in [0, 1]
        |
   Low risk  -> execute autonomously
   High risk -> surface interrupt to human
        |
   after_tool_call hook   <- capture outcome for training
```

## Risk Tiers

Seven tiers on a continuous [0, 1] scale:

| Tier | Name | Risk Range | Examples |
|------|------|-----------|---------|
| T-2 | Adversarial | 0.85-1.00 | Backdoor install, credential exfil, pipe-to-bash |
| T-1 | Dry-run | 0.08-0.22 | `terraform plan`, `kubectl apply --dry-run` |
| T0 | Read-only | 0.00-0.15 | `ls`, `kubectl get`, `aws ec2 describe-*` |
| T1 | Local-write | 0.20-0.40 | `git commit`, `npm install`, `docker build` |
| T2 | Destructive-local | 0.35-0.55 | `rm -rf`, `git branch -D`, `docker rmi` |
| T3 | External | 0.55-0.75 | `git push`, `docker push`, `aws s3 sync` |
| T4 | Catastrophic | 0.75-1.00 | `terraform destroy`, `git push --force` |

## Scoring Pipeline

**Layer 1 — CLI verb table** (deterministic):
- `(tool, subcommand)` -> tier floor/ceiling
- Covers terraform, redis-cli, aws, kubectl, git, npm/yarn
- Opaque execution (`eval`, `| bash`) -> T2 floor

**Layer 2 — k-NN on Siamese embeddings** (ML):
- Embed command -> 128-dim unit vector via HybridEncoder (char-CNN + word-CNN)
- Find k=5 nearest neighbors in scored reference corpus (~28K commands)
- Aggregate risk score and tier

**Layer 3 — Heuristic post-processing**:
- Dry-run flag detection -> T-1 cap
- Compound command splitting -> score each segment, return worst

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

Current performance (v7 + expanded corpus, 23 eval scenarios, 285 commands):

| Metric | Value |
|--------|-------|
| Overall accuracy | 84.9% |
| FN rate (T3+T4 under by >1 tier) | 19.5% |
| FP rate (T0/T-1 over by >1 tier) | 12.4% |
| T-1 (dry-run) | 100% |
| T0 (read-only) | 88.9% |
| T3 (external) | 82.5% |
| T4 (catastrophic) | 75.0% |

### Training

```bash
# Generate pairs from scored corpus
make pairs

# Train HybridEncoder
make train

# Evaluate
make eval
```

## Integration

### Claude Code

Install as a hook in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "npx tsx /path/to/alignlayer/src/claudecode-hook/hook.ts",
        "timeout": 10
      }]
    }],
    "PostToolUse": [{
      "matcher": ".*",
      "hooks": [{
        "type": "command",
        "command": "npx tsx /path/to/alignlayer/src/claudecode-hook/hook.ts",
        "timeout": 5,
        "async": true
      }]
    }]
  }
}
```

Or use the lightweight observer hook for trace collection only:

```bash
# .claude/hooks/observe.sh forwards events to the scoring server
ALIGNLAYER_URL=http://localhost:8000 .claude/hooks/observe.sh
```

### Scoring Server API

```bash
# Score a command
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"text": "git push --force origin main"}'

# Health check
curl http://localhost:8000/health

# Trace stats
curl http://localhost:8000/traces/stats

# Dashboard
open http://localhost:8000/
```

## Project Structure

```
alignlayer/
+-- model/
|   +-- siamese.py          # HybridEncoder, contrastive loss, predict_risk()
|   +-- pairs.py            # Pair generation from scored corpus
|   +-- eval.py             # Evaluation harness
|   +-- serve.py            # FastAPI scoring server
|   +-- review.py           # Human review CLI
|   +-- static/dashboard.html
|   +-- checkpoints/        # Trained model checkpoints
+-- src/
|   +-- claudecode-hook/    # Claude Code PreToolUse/PostToolUse handler
|   +-- openclaw-plugin/    # OpenClaw plugin (hook + scorer)
|   +-- scorer-llm/         # LLM-based scorer (Ollama/Anthropic)
+-- data/
|   +-- test_scenarios.json # Eval scenarios (23 scenarios, 285 commands)
|   +-- adversarial_suite.json
|   +-- synthetic/          # Corpus and pair generation scripts
+-- scripts/
|   +-- deploy.sh           # Deployment automation
|   +-- targeted_ingest.py  # Corpus expansion pipeline
+-- docs/
|   +-- architecture.md     # System design
|   +-- integration.md      # OpenClaw installation guide
+-- Makefile
+-- Dockerfile
```

## Corpus

~28,400 labeled commands covering:
- AWS, Kubernetes, Terraform, Pulumi, CDK
- CI/CD (GitHub Actions, Airflow, Kafka)
- Data stores (Redis, PostgreSQL, MySQL, MongoDB)
- Supply chain (cosign, skopeo, Flyway, Ansible)
- Adversarial patterns: RCE, credential exfil, obfuscation, persistence
- Real agent traces from Claude Code and OpenClaw deployments

## License

MIT. See [LICENSE](LICENSE).
