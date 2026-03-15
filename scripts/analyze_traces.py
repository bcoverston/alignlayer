"""
Analyze dogfood traces: compare TS hook scorer vs Python ML model.

Reads all trace JSONL files, extracts Bash tool calls, scores them through
the ML model, and reports disagreements between the two scorers.

Usage:
    model/.venv/bin/python3 scripts/analyze_traces.py [--traces-dir ~/.alignlayer/traces]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from siamese import load_model, build_reference_index, predict_risk

RISK_THRESHOLD = 0.55


def load_traces(traces_dir: Path) -> list[dict]:
    """Load all before_tool_call events from trace JSONL files."""
    entries = []
    for f in sorted(traces_dir.glob("alignlayer-*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "before_tool_call":
                entries.append(entry)
    return entries


def extract_bash_commands(entries: list[dict]) -> list[dict]:
    """Extract entries where tool is Bash and args contain a command string."""
    bash = []
    for e in entries:
        tool = e.get("tool", "")
        if tool.lower() != "bash":
            continue
        args = e.get("args", {})
        cmd = args.get("command") or args.get("cmd") or args.get("input") or ""
        if not cmd or not isinstance(cmd, str):
            continue
        bash.append({
            "cmd": cmd.strip(),
            "ts_risk": e.get("risk_score"),
            "ts_decision": e.get("decision"),
            "ts_blast": e.get("blast_radius"),
            "timestamp": e.get("timestamp", ""),
            "session_id": e.get("session_id", ""),
        })
    return bash


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(Path.home() / ".alignlayer" / "traces"))
    ap.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    ap.add_argument("--corpus", default="data/synthetic/scores-cache.jsonl")
    ap.add_argument("--output", default="data/trace_analysis.json")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    print(f"Loading traces from {traces_dir}")
    entries = load_traces(traces_dir)
    print(f"  {len(entries)} before_tool_call events")

    bash_cmds = extract_bash_commands(entries)
    print(f"  {len(bash_cmds)} Bash commands")

    # Dedup by command string (same command scored multiple times)
    seen = set()
    unique = []
    for b in bash_cmds:
        if b["cmd"] not in seen:
            seen.add(b["cmd"])
            unique.append(b)
    print(f"  {len(unique)} unique commands")

    # Tool distribution across all events
    tool_dist = Counter(e.get("tool", "?") for e in entries)
    print(f"\nTool distribution (all events):")
    for tool, count in tool_dist.most_common(15):
        print(f"  {tool:20s}: {count:5d}")

    # TS decision distribution
    ts_decisions = Counter(e.get("decision") for e in entries)
    print(f"\nTS hook decisions (all events):")
    for d, c in ts_decisions.most_common():
        print(f"  {str(d):12s}: {c:5d}")

    # Load ML model
    print(f"\nLoading ML model: {args.checkpoint}")
    model, dev = load_model(args.checkpoint)
    ref_embs, ref_entries = build_reference_index(args.corpus, model, dev)
    print(f"  {len(ref_entries):,} reference entries\n")

    # Score unique Bash commands through ML model
    results = []
    for i, b in enumerate(unique):
        cmd = b["cmd"]
        # Skip very long commands (multi-line scripts)
        if len(cmd) > 500:
            continue
        ml = predict_risk(cmd, model, dev, ref_embs, ref_entries)
        ml_decision = "interrupt" if ml["risk"] >= RISK_THRESHOLD else "allow"
        ts_decision = b["ts_decision"] or "allow"

        results.append({
            "cmd": cmd[:200],
            "ts_risk": b["ts_risk"],
            "ts_decision": ts_decision,
            "ml_risk": round(ml["risk"], 3),
            "ml_tier": ml["tier"],
            "ml_decision": ml_decision,
            "ml_source": ml.get("heuristic") or ml.get("source", "?"),
            "agree": ml_decision == ts_decision,
        })

        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(unique)}...")

    print(f"\n{'='*70}")
    print(f"  TRACE ANALYSIS: {len(results)} unique Bash commands")
    print(f"{'='*70}")

    agree = sum(1 for r in results if r["agree"])
    disagree = [r for r in results if not r["agree"]]
    print(f"\nAgreement: {agree}/{len(results)} ({agree/len(results):.1%})")
    print(f"Disagreements: {len(disagree)}")

    # Categorize disagreements
    ts_allow_ml_interrupt = [r for r in disagree if r["ts_decision"] == "allow" and r["ml_decision"] == "interrupt"]
    ts_interrupt_ml_allow = [r for r in disagree if r["ts_decision"] == "interrupt" and r["ml_decision"] == "allow"]

    if ts_allow_ml_interrupt:
        print(f"\n{'─'*70}")
        print(f"  TS allows, ML would interrupt ({len(ts_allow_ml_interrupt)}):")
        print(f"  (ML is more cautious — potential false negatives in TS)")
        print(f"{'─'*70}")
        for r in sorted(ts_allow_ml_interrupt, key=lambda x: -x["ml_risk"])[:30]:
            print(f"  T{r['ml_tier']:+d} ml={r['ml_risk']:.3f} ts={r['ts_risk']:.2f} [{r['ml_source']:12s}] {r['cmd'][:65]}")

    if ts_interrupt_ml_allow:
        print(f"\n{'─'*70}")
        print(f"  TS interrupts, ML would allow ({len(ts_interrupt_ml_allow)}):")
        print(f"  (TS is more cautious — potential false positives in TS)")
        print(f"{'─'*70}")
        for r in sorted(ts_interrupt_ml_allow, key=lambda x: x["ml_risk"])[:30]:
            print(f"  T{r['ml_tier']:+d} ml={r['ml_risk']:.3f} ts={r['ts_risk']:.2f} [{r['ml_source']:12s}] {r['cmd'][:65]}")

    # ML tier distribution
    tier_dist = Counter(r["ml_tier"] for r in results)
    print(f"\nML tier distribution:")
    for t in sorted(tier_dist):
        c = tier_dist[t]
        pct = c / len(results)
        bar = "█" * int(pct * 40)
        print(f"  T{t:+d}: {c:4d} ({pct:5.1%})  {bar}")

    # ML source distribution
    src_dist = Counter(r["ml_source"] for r in results)
    print(f"\nML scoring source:")
    for s, c in src_dist.most_common():
        print(f"  {s:18s}: {c:4d} ({c/len(results):5.1%})")

    # ML decision distribution
    ml_decisions = Counter(r["ml_decision"] for r in results)
    print(f"\nML decisions:")
    for d, c in ml_decisions.most_common():
        print(f"  {d:12s}: {c:4d} ({c/len(results):5.1%})")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "summary": {
            "total_events": len(entries),
            "bash_commands": len(bash_cmds),
            "unique_commands": len(unique),
            "scored": len(results),
            "agreement": agree,
            "disagreements": len(disagree),
            "ts_allow_ml_interrupt": len(ts_allow_ml_interrupt),
            "ts_interrupt_ml_allow": len(ts_interrupt_ml_allow),
        },
        "disagreements": disagree,
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
