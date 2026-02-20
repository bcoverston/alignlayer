# AlignLayer

**Runtime preference alignment for agentic systems.**

## One-Liner

AlignLayer sits between an agent's decision loop and execution, scoring proposed actions on learned human preferences before they run.

## The Problem

Autonomous agents are deployed faster than the tooling to make them trustworthy. Permission systems today are static — allowlists, deny lists, fixed tiers. They treat every shell command the same regardless of context. An agent can push to main, drop a table, or email the wrong person before anyone realizes the cost.

The missing primitive isn't a better blocklist. It's a system that **learns** what a specific user or org considers acceptable and enforces that at runtime, dynamically, without human review of every action.

## Core Insight

Risk is two-dimensional:

- **P(wrong action)** — confidence the current plan is still valid
- **Blast radius** — reversibility and downstream impact

These are independent axes. High confidence + irreversible = still dangerous. Low confidence + cheap to undo = probably fine.

Human corrections are the richest training signal. "I'd prefer you didn't push to main" encodes the violated preference AND the preferred alternative.

## Architecture

```
Agent decision loop
        ↓
   before_tool_call hook  ← AlignLayer scores proposed action
        ↓
   Risk score = f(P(wrong), blast_radius)
        ↓
   Low risk  → execute autonomously
   High risk → surface lightweight interrupt to human
        ↓
   after_tool_call hook   ← capture outcome
        ↓
   Human correction?      ← contrastive training signal
```

## Risk Model — Siamese Network Bootstrap

**Cold start problem**: How do you score actions before you have human corrections?

**Solution**: Train a Siamese network on action pairs to learn the **topology of risk** before any human feedback exists.

### Training Pairs

Pairs encode relational risk — the model learns that semantic similarity ≠ risk similarity:

| Pair | Risk Delta | What It Teaches |
|------|-----------|-----------------|
| `(internal, internal)` | Low | Same-scope actions have similar risk |
| `(internal, external)` | High | Crossing boundaries = risk escalation |
| `(commit, push)` | High | Adjacent actions, different blast radius |
| `(commit, commit)` | Low | Same action = baseline |
| `(draft, send)` | High | Reversible → irreversible |
| `(stage, deploy)` | High | Same pattern, different domain |

The model learns **the shape of escalation itself**, not individual tool semantics. `commit/push` has the same risk topology as `draft/send` and `stage/deploy`.

### Loss Function

- **Start with contrastive loss** — binary similar/dissimilar. Matches the pair structure naturally. Gets to MVP fastest.
- **Move to triplet loss** when finer resolution is needed — anchor/positive/negative gives gradients of risk, not just binary.

### Training Data Sources

1. **Own agent traces** — instrument personal OpenClaw instance, label pairs from own corrections. High quality, low volume.
2. **OpenClaw spike** — community/open-source traces. High volume, diverse, noisier.
3. **Synthetic pairs** — generate from tool schemas and API docs. Cheap bootstrap.
4. **Hybrid** — synthetic to bootstrap geometry, real traces to refine boundaries.

## Integration — OpenClaw Plugin

AlignLayer integrates as an OpenClaw plugin using the existing hook system:

### Plugin Hooks (Primary)

- **`before_tool_call`** — Score proposed action. Block or allow based on threshold. This is the core injection point.
- **`after_tool_call`** — Capture outcome for training signal.
- **`tool_result_persist`** — Annotate tool results with risk metadata before transcript persistence.

### Exec Approvals (UX Template)

OpenClaw already has approval infrastructure:
- Lightweight interrupts surfaced to chat channels
- `/approve <id> allow-once | allow-always | deny`
- Per-agent allowlists with glob patterns

AlignLayer replaces static allowlists with a learned model while reusing the existing approval UX.

### Session Transcripts (Training Corpus)

- All tool calls captured as structured JSONL in session transcripts
- Every `before_tool_call` / `after_tool_call` pair is a training sample
- Human corrections via `/approve` are labeled preference data

## Distribution Strategy

Framework-agnostic telemetry plugins. Zero workflow disruption — install a plugin, get observability. Each plugin is simultaneously a data collection point and a distribution channel.

1. **Phase 0 — OpenClaw** — own deployment as primary data source. Heuristic scorer, JSONL traces.
2. **Phase 0.1 — Claude Code** — `PreToolUse` hook gives a richer, simpler integration: synchronous stdin/stdout, full tool context, native block/allow/ask decisions. Observational now; test harness for Phase 3 ML scoring.
3. **Phase 0.2 — OpenClaw blocking** — minimal upstream PR adds `exec_approval_requested` plugin hook, enabling AlignLayer to drive approval resolution via gateway WS. Closes the observe→block gap.
4. **Phase 3+ — LangChain / CrewAI** — framework layer as multi-agent orchestration standardizes.

Each opted-in user generates preference signal that improves the risk model for everyone. Dataset compounds; moat widens with usage.

## Competitive Landscape

- **Invariant Labs** — static rule-based agent security
- **LangSmith** — post-hoc observability only
- **Nobody** scores proposed actions on learned human preferences at runtime

This is a new category.

## Market

- 65% of enterprise leaders cite agentic system complexity as top barrier (KPMG Q4 2025)
- Gartner: 40% of enterprise apps will embed agents by end of 2026
- CrowdStrike published advisory on OpenClaw security gaps

## Project Structure

```
alignlayer/
├── ALIGNLAYER.md                # This file
├── docs/
│   ├── architecture.md          # System design, risk axes, trace schema
│   └── openclaw-blocking-pr.md  # Spec for OpenClaw exec_approval_requested hook PR
├── src/
│   ├── openclaw-plugin/
│   │   ├── HOOK.md              # OpenClaw hook manifest
│   │   ├── handler.ts           # agent-events bus subscriber; scores + traces tool calls
│   │   └── scorer.ts            # Heuristic risk engine (blast_radius × plan_confidence)
│   └── claudecode-hook/
│       └── hook.ts              # Claude Code PreToolUse/PostToolUse handler (self-contained)
├── model/
│   ├── siamese.py               # Siamese network architecture (Phase 2)
│   ├── train.py                 # Training pipeline (Phase 2)
│   └── pairs.py                 # Pair generation from traces (Phase 1)
├── data/
│   ├── synthetic/               # Generated pairs from tool schemas (Phase 1)
│   └── traces/                  # Captured agent traces (gitignored)
└── research/
    └── notes.md                 # Running research notes
```

## Philosophy

Agents that earn autonomy. Infrastructure that makes it safe to give it.

Not a blocklist. Not a cage. A system that learns what trust looks like — and extends it incrementally, based on demonstrated behavior.

Free agency — and how to actually implement it.
