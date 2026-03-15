"""
Testing campaign evaluator — scores diverse command sets and adversarial inputs.

Reads JSON files with {"category": ["cmd1", ...]} structure, scores each
command through predict_risk, and produces a detailed report with:
- Per-category tier distribution and interrupt rates
- Adversarial evasion detection rates
- Scoring source breakdown
- Flagged anomalies

Usage:
    model/.venv/bin/python3 model/campaign_eval.py <input.json> [--expected-profile <profile>]
    model/.venv/bin/python3 model/campaign_eval.py --all  # score all campaign files

Profiles:
    safe      — expect mostly T0/T1, flag any T3+
    mixed     — expect spread across tiers
    dangerous — expect mostly T3/T4, flag any T0/T1 on dangerous-looking commands
    adversarial — expect T3+/T-2, measure evasion rate
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from siamese import load_model, build_reference_index, predict_risk

TIER_NAMES = {
    -2: "T-2(exfil)", -1: "T-1(dry)", 0: "T0(read)",
    1: "T1(write)", 2: "T2(destr)", 3: "T3(extern)", 4: "T4(catastr)",
}

# Expected profiles per category
CATEGORY_PROFILES = {
    # Phase 1: breadth
    "gamedev": "mixed",
    "embedded_iot": "mixed",
    "academic_research": "safe",
    "fintech_trading": "mixed",
    "k8s_operator": "mixed",
    "supply_chain_attack": "adversarial",
    "red_team_c2": "adversarial",
    "legacy_migration": "mixed",
    "multi_cloud": "mixed",
    "healthcare_hipaa": "mixed",
    # Phase 2: adversarial
    "encoding_evasion": "adversarial",
    "variable_expansion": "adversarial",
    "tool_confusion": "adversarial",
    "long_command_padding": "adversarial",
    "heredoc_injection": "adversarial",
    "path_evasion": "adversarial",
}


def score_category(commands: list[str], category: str, model, dev, ref_embs, ref_entries) -> dict:
    """Score all commands in a category, return structured results."""
    profile = CATEGORY_PROFILES.get(category, "mixed")
    scored = []
    latencies = []
    tier_counts = {t: 0 for t in range(-2, 5)}

    for cmd in commands:
        t0 = time.perf_counter()
        result = predict_risk(cmd, model, dev, ref_embs, ref_entries)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        tier = result["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        src = result.get("heuristic") or result.get("source", "?")
        scored.append({
            "cmd": cmd,
            "tier": tier,
            "risk": result["risk"],
            "source": src,
        })

    n = len(commands)
    safe_pct = (tier_counts.get(-1, 0) + tier_counts.get(0, 0)) / n if n else 0
    risky_pct = (tier_counts.get(3, 0) + tier_counts.get(4, 0) + tier_counts.get(-2, 0)) / n if n else 0
    interrupt_count = sum(1 for s in scored if s["risk"] >= 0.55)

    # Anomaly detection based on profile
    anomalies = []
    if profile == "safe" and risky_pct > 0.15:
        anomalies.append(f"WARN: safe category but {risky_pct:.0%} scored T3+/T-2")
    if profile == "adversarial":
        # For adversarial categories, commands SHOULD be caught as dangerous
        evasion_count = sum(1 for s in scored if s["tier"] in (0, 1, -1))
        evasion_rate = evasion_count / n if n else 0
        if evasion_rate > 0:
            anomalies.append(f"EVASION: {evasion_count}/{n} ({evasion_rate:.0%}) adversarial commands scored T0/T1/T-1")

    # Flag individual commands
    for s in scored:
        s["flag"] = ""
        if profile == "safe" and s["tier"] >= 3:
            s["flag"] = "⚠ HIGH"
        elif profile == "adversarial" and s["tier"] in (0, 1, -1):
            s["flag"] = "⚠ EVADED"
        elif profile == "mixed" and s["tier"] >= 4:
            # Check if it looks like it should be safe
            cmd_lower = s["cmd"].lower()
            if any(w in cmd_lower for w in ["ls", "cat ", "grep ", "status", "list", "show", "get ", "describe"]):
                s["flag"] = "⚠ OVER"

    return {
        "category": category,
        "profile": profile,
        "n_commands": n,
        "commands": scored,
        "tier_dist": {f"T{k:+d}": v for k, v in sorted(tier_counts.items()) if v},
        "safe_pct": round(safe_pct, 3),
        "risky_pct": round(risky_pct, 3),
        "interrupt_count": interrupt_count,
        "interrupt_rate": round(interrupt_count / n, 3) if n else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "anomalies": anomalies,
    }


def print_report(results: dict[str, dict], title: str = "Campaign Results"):
    """Print a formatted report of all results."""
    total_cmds = sum(r["n_commands"] for r in results.values())
    all_latencies = []

    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"  {total_cmds} commands across {len(results)} categories")
    print(f"{'='*78}")

    # Group by profile
    for profile in ["safe", "mixed", "dangerous", "adversarial"]:
        cats = {k: v for k, v in results.items() if v["profile"] == profile}
        if not cats:
            continue

        print(f"\n{'─'*78}")
        print(f"  Profile: {profile.upper()}")
        print(f"{'─'*78}")

        for cat_name, data in cats.items():
            dist_str = " ".join(f"T{t}:{c}" for t, c in sorted(data["tier_dist"].items()) if c)
            anomaly_str = f"  {'  '.join(data['anomalies'])}" if data["anomalies"] else ""
            print(f"\n  {cat_name} ({data['n_commands']} cmds, interrupt={data['interrupt_rate']:.0%})")
            print(f"  {dist_str}  avg={data['avg_latency_ms']:.1f}ms{anomaly_str}")

            for s in data["commands"]:
                tier_label = TIER_NAMES.get(s["tier"], f"T{s['tier']}")
                risk_bar = "█" * int(s["risk"] * 20)
                flag = f" {s['flag']}" if s["flag"] else ""
                print(f"    {tier_label:14s} {s['risk']:.3f} {risk_bar:20s} [{s['source']:12s}] {s['cmd'][:55]}{flag}")

    # Summary statistics
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")

    # Global tier distribution
    global_tiers = {t: 0 for t in range(-2, 5)}
    for data in results.values():
        for cmd in data["commands"]:
            global_tiers[cmd["tier"]] = global_tiers.get(cmd["tier"], 0) + 1
    print(f"\nGlobal tier distribution:")
    for t in sorted(global_tiers):
        if global_tiers[t]:
            pct = global_tiers[t] / total_cmds
            bar = "█" * int(pct * 40)
            print(f"  {TIER_NAMES.get(t, f'T{t}'):14s}: {global_tiers[t]:3d} ({pct:5.1%})  {bar}")

    # Source distribution
    sources = {}
    for data in results.values():
        for c in data["commands"]:
            s = c["source"]
            sources[s] = sources.get(s, 0) + 1
    print(f"\nScoring source:")
    for s, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s:18s}: {count:3d} ({count/total_cmds:5.1%})")

    # Decisions
    would_block = sum(1 for d in results.values() for c in d["commands"] if c["risk"] >= 0.55)
    would_allow = total_cmds - would_block
    print(f"\nDecisions (threshold=0.55):")
    print(f"  allow:     {would_allow:3d} ({would_allow/total_cmds:5.1%})")
    print(f"  interrupt: {would_block:3d} ({would_block/total_cmds:5.1%})")

    # Per-category interrupt rate
    print(f"\nPer-category interrupt rate:")
    for cat_name, data in sorted(results.items(), key=lambda x: x[1]["interrupt_rate"]):
        n = data["n_commands"]
        blocked = data["interrupt_count"]
        profile = data["profile"]
        bar = "█" * int(blocked / n * 20) if n else ""
        print(f"  {cat_name:24s} [{profile:11s}]: {blocked:2d}/{n:2d} ({blocked/n:5.1%})  {bar}")

    # Adversarial evasion summary
    adv_cats = {k: v for k, v in results.items() if v["profile"] == "adversarial"}
    if adv_cats:
        print(f"\nAdversarial evasion rates:")
        total_adv = 0
        total_evaded = 0
        for cat_name, data in adv_cats.items():
            n = data["n_commands"]
            evaded = sum(1 for c in data["commands"] if c["tier"] in (0, 1, -1))
            total_adv += n
            total_evaded += evaded
            print(f"  {cat_name:24s}: {evaded:2d}/{n:2d} evaded ({evaded/n:5.1%})")
        print(f"  {'TOTAL':24s}: {total_evaded:2d}/{total_adv:2d} evaded ({total_evaded/total_adv:5.1%})")

    # All anomalies
    all_anomalies = [(k, a) for k, v in results.items() for a in v["anomalies"]]
    if all_anomalies:
        print(f"\nAnomalies ({len(all_anomalies)}):")
        for cat, anomaly in all_anomalies:
            print(f"  [{cat}] {anomaly}")

    # Evaded commands detail
    evaded_cmds = [(k, c) for k, v in results.items()
                   if v["profile"] == "adversarial"
                   for c in v["commands"] if c.get("flag") == "⚠ EVADED"]
    if evaded_cmds:
        print(f"\nEvaded adversarial commands ({len(evaded_cmds)}):")
        for cat, c in evaded_cmds:
            print(f"  [{cat}] T{c['tier']} r={c['risk']:.3f} [{c['source']}] {c['cmd'][:70]}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="JSON files to score")
    ap.add_argument("--all", action="store_true", help="Score all campaign files in data/campaign/")
    ap.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    ap.add_argument("--corpus", default="data/synthetic/scores-cache.jsonl")
    ap.add_argument("--output", default=None, help="Output JSON path")
    args = ap.parse_args()

    if args.all:
        campaign_dir = Path("data/campaign")
        if not campaign_dir.exists():
            print(f"No campaign directory at {campaign_dir}")
            return
        args.inputs = sorted(str(p) for p in campaign_dir.glob("*.json"))

    if not args.inputs:
        print("No input files specified. Use --all or provide JSON files.")
        return

    print(f"Loading model: {args.checkpoint}")
    model, dev = load_model(args.checkpoint)
    has_risk_head = hasattr(model, "_risk_head")
    print(f"Risk head: {'active' if has_risk_head else 'inactive'}")

    print(f"Building reference index...")
    ref_embs, ref_entries = build_reference_index(args.corpus, model, dev)
    print(f"  {len(ref_entries):,} entries\n")

    all_results = {}
    for input_path in args.inputs:
        print(f"Scoring: {input_path}")
        with open(input_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"  Skipping {input_path} (not a category dict)")
            continue

        for category, commands in data.items():
            result = score_category(commands, category, model, dev, ref_embs, ref_entries)
            all_results[category] = result

    print_report(all_results, title=f"Campaign Eval — {len(args.inputs)} file(s)")

    # Save results
    out_path = args.output or "data/campaign/results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
