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
   Risk score = f(P(wrong), blast_radius)
        ↓
   Low risk  → execute autonomously
   High risk → surface lightweight interrupt to human
        ↓
   after_tool_call hook   ← capture outcome
        ↓
   Human correction?      ← contrastive training signal
```

## Risk Model

AlignLayer bootstraps with a **Siamese network** trained on action pairs to learn the topology of risk before any human feedback exists.

The model learns the shape of escalation itself, not individual tool semantics. `commit/push` has the same risk topology as `draft/send` and `stage/deploy`.

Once deployed, human corrections via lightweight interrupts become labeled preference data that continuously refine the model.

## Integration

AlignLayer integrates as an OpenClaw plugin using the existing hook system:

- **`before_tool_call`** — score proposed action, block or allow based on threshold
- **`after_tool_call`** — capture outcome for training signal
- **`tool_result_persist`** — annotate tool results with risk metadata

Replaces static allowlists with a learned model while reusing OpenClaw's existing approval UX.

## Project Structure

```
alignlayer/
├── docs/
│   ├── architecture.md          # Detailed system design
│   └── risk-topology.md         # Siamese pair research
├── src/
│   └── openclaw-plugin/
│       ├── HOOK.md              # OpenClaw plugin metadata
│       ├── handler.ts           # before/after_tool_call hooks
│       └── scorer.ts            # Risk scoring engine
├── model/
│   ├── siamese.py               # Siamese network architecture
│   ├── train.py                 # Training pipeline
│   └── pairs.py                 # Pair generation from traces
├── data/
│   ├── synthetic/               # Generated pairs from tool schemas
│   └── traces/                  # Captured agent traces (gitignored)
└── research/
    └── notes.md                 # Running research notes
```

## Philosophy

Agents that earn autonomy. Infrastructure that makes it safe to give it.

Not a blocklist. Not a cage. A system that learns what trust looks like — and extends it incrementally, based on demonstrated behavior.
