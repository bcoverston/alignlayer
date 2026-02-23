# AlignLayer

Runtime preference alignment for agentic systems.

## The Problem

Autonomous agents are deployed faster than the tooling to make them trustworthy. Permission systems today are static — allowlists, denylists, fixed tiers. They treat every shell command the same regardless of context. An agent can push to main, drop a table, or email the wrong person before anyone realizes the cost.

The missing primitive isn't a better blocklist. It's a system that **learns** what a specific user or org considers acceptable and enforces that at runtime, dynamically, without human review of every action.

## Core Insight

Risk is two-dimensional:

- **P(wrong action)** — confidence the current plan is still valid
- **Blast radius** — reversibility and downstream impact

These are independent axes. High confidence + irreversible = still dangerous. Low confidence + cheap to undo = probably fine.

Human corrections are the richest training signal. "I'd prefer you didn't push to main" encodes the violated preference and the preferred alternative.

## Architecture

```
Agent decision loop
        ↓
   before_tool_call hook  ← AlignLayer scores proposed action
        ↓
   Heuristic pre-check    ← CLI verb table (fast, deterministic)
        ↓
   ML scorer              ← k-NN on Siamese embeddings
        ↓
   Risk score ∈ [0, 1]    ← combined signal
        ↓
   Low risk  → execute autonomously
   High risk → surface lightweight interrupt to human
        ↓
   after_tool_call hook   ← capture outcome
        ↓
   Human correction?      ← contrastive training signal
```

## Risk Model

Seven tiers, scored on a continuous [0, 1] scale:

| Tier | Name | Risk Range | Examples |
|------|------|-----------|---------|
| T-2 | Adversarial | 0.85–1.00 | Backdoor install, credential exfil, pipe-to-bash |
| T-1 | Dry-run | 0.08–0.22 | `terraform plan`, `kubectl apply --dry-run` |
| T0  | Read-only | 0.00–0.15 | `ls`, `kubectl get`, `aws ec2 describe-*` |
| T1  | Local-write | 0.20–0.40 | `git commit`, `npm install`, `docker build` |
| T2  | Destructive-local | 0.35–0.55 | `rm -rf`, `git branch -D`, `docker rmi` |
| T3  | External | 0.55–0.75 | `git push`, `docker push`, `aws s3 sync`, `curl -X POST` |
| T4  | Catastrophic | 0.75–1.00 | `terraform destroy`, `git push --force`, `aws ec2 terminate-instances` |

## Scoring Pipeline

**Layer 1 — CLI verb table** (deterministic, fires first):
- `(tool, subcommand)` → tier floor or ceiling
- Covers known patterns: terraform, redis-cli, aws, kubectl, git, npm/yarn
- Opaque execution (`python3 -c`, `eval`, `| bash`) → T2 floor

**Layer 2 — k-NN on Siamese embeddings** (ML):
- Embed command → 128-dim unit vector
- Find k=5 nearest neighbors in scored reference corpus (~28K commands)
- Aggregate risk score and tier by majority vote

**Layer 3 — Heuristic post-processing**:
- Dry-run flag detection → T-1 cap
- Compound command splitting (`&&`, `||`, `;`) → score each segment, return worst

## Model

AlignLayer uses a **Siamese network** trained on action pairs via contrastive loss.

### Current architecture: HybridEncoder (v7, in training)

```
command string
├── Char branch: char embeddings (dim=32)
│               → parallel 1D CNNs [3,5,7,11,16,21] (64 filters each)
│               → ReLU + global max pool → concat → LayerNorm → FC → 128-dim
│
└── Word branch: whitespace token embeddings (dim=32)
                → parallel 1D CNNs [2,3,4] (32 filters each)
                → ReLU + global max pool → concat → LayerNorm → FC → 64-dim

concat(128 + 64) → LayerNorm → FC → L2-norm → 128-dim unit vector
```

The word branch gives the model token-level semantics: `describe-instances` and `terminate-instances` are now different vocabulary entries, breaking the char-level ambiguity that caused false positives on AWS read commands.

### Model progression

| Version | Corpus | Overall | FN T3+T4 | FP T0/T-1 | Notes |
|---------|--------|---------|----------|-----------|-------|
| v3 | 22K | 75.0% | 42.7% | — | char-only, 90K params |
| v4 | 22K | 87.5% | 17.3% | — | char-only, 182K params, larger kernels |
| v5 | 26K | 78.8% | 25.3% | 17.5% | corpus expansion wave 1+2; T0/T1 underrepresented |
| v6 | 28K | 80.4% | 24.0% | 17.5% | added 1.4K read-only counterparts; partial fix |
| v7 | 28K | — | — | — | HybridEncoder + heuristics; training now |

v4 + heuristics (current production): **83.3% overall, T0 96%, T4 94.7%, FP 7%**

### Training

```bash
# Char-only (legacy)
python model/siamese.py train --pairs data/synthetic/pairs-v4.jsonl --max-pairs 1000000 --epochs 15

# HybridEncoder
python model/siamese.py train --pairs data/synthetic/pairs-v4.jsonl --max-pairs 1000000 \
  --epochs 15 --hybrid --corpus data/synthetic/scores-cache.jsonl
```

### Corpus

~28,200 labeled commands covering:
- AWS (EC2, S3, RDS, IAM, ECS, EKS, Lambda, CloudWatch, and more)
- Kubernetes (kubectl, Helm, ArgoCD, Flux)
- Infrastructure-as-Code (Terraform, Pulumi, CDK)
- CI/CD (GitHub Actions via `gh`, Airflow, Kafka)
- Data stores (Redis, PostgreSQL, MySQL, MongoDB)
- Supply chain (cosign, skopeo, ngrok, Flyway, Liquibase, Ansible)
- Adversarial patterns (T-2): RCE, credential exfil, obfuscation, persistence
- Real agent traces from live deployments (Claude Code + OpenClaw)

## Integration

AlignLayer deploys as an OpenClaw plugin. See [`docs/integration.md`](docs/integration.md) for the full installation guide.

**Quick start:**
```bash
# Start the ML scoring server
make serve   # or: make docker-run

# Install the plugin into OpenClaw
cp src/openclaw-plugin/* /opt/openclaw/plugins/alignlayer/
```

**Plugin hooks:**
- `before_tool_call` — score proposed action, interrupt if risk ≥ 0.55
- `after_tool_call` — capture outcome for training signal
- `tool_result_persist` — annotate results with risk metadata

The scoring server (`model/serve.py`) exposes `POST /score` backed by the trained model. The plugin falls back to pure heuristics if the server is unreachable.

## Project Structure

```
alignlayer/
├── src/
│   └── openclaw-plugin/
│       ├── HOOK.md              # OpenClaw plugin manifest
│       ├── handler.ts           # before/after_tool_call hooks
│       └── scorer.ts            # Heuristic + ML risk scoring engine
├── model/
│   ├── siamese.py               # Siamese network (HybridEncoder + CommandEncoder)
│   ├── pairs.py                 # Pair generation from scored corpus
│   ├── eval.py                  # Evaluation harness (20 scenarios, 240 commands)
│   ├── serve.py                 # FastAPI scoring server
│   └── checkpoints/
│       ├── best-v4.pt           # Best char-only model (87.5% pre-heuristics)
│       ├── best-v5.pt           # Regression — corpus imbalance
│       ├── best-v6.pt           # Partial fix — read-only rebalancing
│       └── best-v7.pt           # HybridEncoder (in training)
├── data/
│   ├── sprint_plan.md           # Engineering roadmap
│   ├── eval_history.jsonl       # Eval run history
│   └── synthetic/
│       └── ingest_corpus_expansion.py  # Corpus ingestion + sanitization
├── docs/
│   ├── architecture.md          # Detailed system design
│   └── integration.md           # OpenClaw installation guide
└── Dockerfile                   # Scoring server container
```

## Current Status

**Deployed:** OpenClaw plugin (heuristic scorer + trace capture) running in production.

**In training:** v7 HybridEncoder — expected to recover T2 accuracy and push overall above 90%.

**Next:**
- Wire handler.ts → ML scoring server (currently heuristic-only in plugin)
- Phase 1 approval UX: surface pending interrupts for human review
- MCP tool coverage: mine MCP registries for corpus expansion
- Expand EXEC_TOOLS to cover Node/REPL execution surface

## Philosophy

Agents that earn autonomy. Infrastructure that makes it safe to give it.

Not a blocklist. Not a cage. A system that learns what trust looks like — and extends it incrementally, based on demonstrated behavior.
