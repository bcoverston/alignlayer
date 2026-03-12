"""
AlignLayer — Phase 2 eval harness.

Beyond val-loss and pair accuracy, this tests what matters for production:
  1. Tier ranking — are embeddings ordered T0 < T1 < T2 < T3 < T4?
  2. Boundary discrimination — can the model separate adjacent tiers?
  3. k-NN risk prediction — MAE vs ground-truth risk scores
  4. Adversarial separation — T-2 commands distinct from T1 look-alikes
  5. Intra-tier cohesion — same-tier commands cluster tightly

Usage:
  model/.venv/bin/python3 model/eval_harness.py \
    --checkpoint model/checkpoints/best.pt \
    --scores data/synthetic/scores-cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from siamese import CommandEncoder, HybridEncoder, encode, tokenize_word, load_model, MARGIN

SCORES_CACHE = "data/synthetic/scores-cache.jsonl"
CHECKPOINT   = "model/checkpoints/best.pt"
TIER_NAMES   = {-2: "T-2(adv-risky)", -1: "T-1(adv-safe)", 0: "T0(read)", 1: "T1(local-wr)", 2: "T2(destruct)", 3: "T3(external)", 4: "T4(catastroph)"}


def load_corpus(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def embed_all(entries: list[dict], model: CommandEncoder | HybridEncoder, dev: torch.device, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    is_hybrid = isinstance(model, HybridEncoder)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            char_x = torch.stack([encode(e["text"]) for e in batch]).to(dev)
            if is_hybrid:
                word_x = torch.stack([tokenize_word(e["text"], model._vocab) for e in batch]).to(dev)
                all_embs.append(model(char_x, word_x).cpu())
            else:
                all_embs.append(model(char_x).cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# 1. Tier ranking: mean inter-tier distance should increase with tier gap
# ---------------------------------------------------------------------------

def eval_tier_ranking(embs: torch.Tensor, entries: list[dict], n_sample: int = 200) -> dict:
    by_tier: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        t = e.get("tier")
        if t is not None:
            by_tier[t].append(i)

    tiers = sorted(by_tier.keys())
    # Mean distance between random samples of each tier pair
    tier_dists: dict[tuple, float] = {}
    for i, t1 in enumerate(tiers):
        for t2 in tiers[i:]:
            idxs1 = random.sample(by_tier[t1], min(n_sample, len(by_tier[t1])))
            idxs2 = random.sample(by_tier[t2], min(n_sample, len(by_tier[t2])))
            e1 = embs[idxs1]
            e2 = embs[idxs2]
            # pairwise distances (min of lens)
            n = min(len(e1), len(e2))
            d = torch.norm(e1[:n] - e2[:n], dim=1).mean().item()
            tier_dists[(t1, t2)] = round(d, 4)

    return tier_dists


# ---------------------------------------------------------------------------
# 2. Boundary discrimination: AUC for adjacent tier separation
# ---------------------------------------------------------------------------

def eval_boundary_discrimination(embs: torch.Tensor, entries: list[dict], n_pairs: int = 2000) -> dict:
    by_tier: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        t = e.get("tier")
        if t is not None:
            by_tier[t].append(i)

    tiers = sorted(by_tier.keys())
    results = {}

    for i in range(len(tiers) - 1):
        t1, t2 = tiers[i], tiers[i + 1]
        if not by_tier[t1] or not by_tier[t2]:
            continue

        # Sample same-tier pairs (label=similar) and cross-tier pairs (label=dissimilar)
        same_idxs = [(random.choice(by_tier[t1]), random.choice(by_tier[t1])) for _ in range(n_pairs // 2)]
        cross_idxs = [(random.choice(by_tier[t1]), random.choice(by_tier[t2])) for _ in range(n_pairs // 2)]

        dists_same  = [torch.norm(embs[a] - embs[b]).item() for a, b in same_idxs]
        dists_cross = [torch.norm(embs[a] - embs[b]).item() for a, b in cross_idxs]

        # Threshold sweep for best accuracy
        all_dists = dists_same + dists_cross
        all_labels = [0] * len(dists_same) + [1] * len(dists_cross)
        thresholds = sorted(set(all_dists))
        best_acc = 0.0
        for th in thresholds[::max(1, len(thresholds)//50)]:
            preds = [1 if d > th else 0 for d in all_dists]
            acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
            best_acc = max(best_acc, acc)

        mean_same  = sum(dists_same)  / len(dists_same)
        mean_cross = sum(dists_cross) / len(dists_cross)
        separation = (mean_cross - mean_same) / (mean_cross + mean_same + 1e-9)

        results[f"T{t1:+d}↔T{t2:+d}"] = {
            "best_acc": round(best_acc, 4),
            "separation": round(separation, 4),
            "mean_dist_same": round(mean_same, 4),
            "mean_dist_cross": round(mean_cross, 4),
        }

    return results


# ---------------------------------------------------------------------------
# 3. k-NN risk prediction: MAE vs ground-truth risk
# ---------------------------------------------------------------------------

def eval_knn_risk(
    embs: torch.Tensor,
    entries: list[dict],
    k: int = 5,
    n_eval: int = 500,
    haiku_only: bool = True,
) -> dict:
    # Use haiku-scored entries as ground truth (more reliable)
    gold = [i for i, e in enumerate(entries) if not haiku_only or e.get("scorer") == "haiku-4-5"]
    if len(gold) < n_eval * 2:
        gold = list(range(len(entries)))

    eval_idxs  = random.sample(gold, min(n_eval, len(gold) // 2))
    ref_idxs   = [i for i in gold if i not in set(eval_idxs)]

    ref_embs   = embs[ref_idxs]
    ref_risks  = torch.tensor([entries[i]["risk"] for i in ref_idxs])
    ref_tiers  = [entries[i]["tier"] for i in ref_idxs]

    risk_errors = []
    tier_correct = 0

    for idx in eval_idxs:
        emb  = embs[idx].unsqueeze(0)
        dists = torch.norm(ref_embs - emb, dim=1)
        topk  = dists.topk(k, largest=False).indices.tolist()
        pred_risk = ref_risks[topk].mean().item()
        true_risk = entries[idx]["risk"]
        risk_errors.append(abs(pred_risk - true_risk))

        votes = [ref_tiers[j] for j in topk]
        pred_tier = max(set(votes), key=votes.count)
        if pred_tier == entries[idx]["tier"]:
            tier_correct += 1

    return {
        "k": k,
        "n_eval": len(eval_idxs),
        "risk_mae": round(sum(risk_errors) / len(risk_errors), 4),
        "risk_p90": round(sorted(risk_errors)[int(len(risk_errors) * 0.9)], 4),
        "tier_acc": round(tier_correct / len(eval_idxs), 4),
    }


# ---------------------------------------------------------------------------
# 4. Adversarial separation: T-2 vs T1 (both look like build/install ops)
# ---------------------------------------------------------------------------

def eval_adversarial_separation(embs: torch.Tensor, entries: list[dict], n_pairs: int = 500) -> dict:
    t_neg2 = [i for i, e in enumerate(entries) if e.get("tier") == -2]
    t1     = [i for i, e in enumerate(entries) if e.get("tier") == 1]
    t0     = [i for i, e in enumerate(entries) if e.get("tier") == 0]

    if not t_neg2:
        return {"error": "no T-2 entries"}

    n = min(n_pairs, len(t_neg2), len(t1))

    # T-2 vs T1 (should be separated — T-2 has hidden blast)
    idxs_neg2 = random.sample(t_neg2, n)
    idxs_t1   = random.sample(t1, min(n, len(t1)))
    idxs_t0   = random.sample(t0, min(n, len(t0)))

    d_neg2_t1 = [torch.norm(embs[a] - embs[b]).item() for a, b in zip(idxs_neg2, idxs_t1[:n])]
    d_neg2_t0 = [torch.norm(embs[a] - embs[b]).item() for a, b in zip(idxs_neg2, idxs_t0[:n])]
    d_t0_t1   = [torch.norm(embs[a] - embs[b]).item()
                 for a, b in zip(random.sample(t0, min(n, len(t0))), random.sample(t1, min(n, len(t1))))]

    return {
        "mean_dist_T-2_vs_T1": round(sum(d_neg2_t1) / len(d_neg2_t1), 4),
        "mean_dist_T-2_vs_T0": round(sum(d_neg2_t0) / len(d_neg2_t0), 4),
        "mean_dist_T0_vs_T1":  round(sum(d_t0_t1)   / len(d_t0_t1),   4),
        "note": "T-2 should be far from T0/T1; if T-2 clusters near T0, adversarial detection will fail",
    }


# ---------------------------------------------------------------------------
# 5. Intra-tier cohesion: mean intra-tier distance (lower = tighter clusters)
# ---------------------------------------------------------------------------

def eval_cohesion(embs: torch.Tensor, entries: list[dict], n_sample: int = 100) -> dict:
    by_tier: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        t = e.get("tier")
        if t is not None:
            by_tier[t].append(i)

    results = {}
    for tier, idxs in sorted(by_tier.items()):
        if len(idxs) < 4:
            continue
        sample = random.sample(idxs, min(n_sample, len(idxs)))
        e = embs[sample]
        # Mean pairwise distance within tier
        dists = []
        for i in range(len(e)):
            for j in range(i + 1, min(i + 20, len(e))):  # limit pairs
                dists.append(torch.norm(e[i] - e[j]).item())
        results[TIER_NAMES.get(tier, str(tier))] = round(sum(dists) / len(dists), 4)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(checkpoint: str, scores_cache: str):
    print(f"Loading model: {checkpoint}")
    model, dev = load_model(checkpoint)

    print(f"Loading corpus: {scores_cache}")
    entries = load_corpus(scores_cache)
    print(f"  {len(entries):,} entries")

    print("Embedding corpus...", end=" ", flush=True)
    embs = embed_all(entries, model, dev)
    print(f"done. shape={embs.shape}")

    print("\n=== 1. Tier ranking (mean pairwise distance) ===")
    tier_dists = eval_tier_ranking(embs, entries)
    tiers = sorted(set(t for pair in tier_dists for t in pair))
    header = "      " + "".join(f"  T{t:+d} " for t in tiers)
    print(header)
    for t1 in tiers:
        row = f"T{t1:+d}  "
        for t2 in tiers:
            key = (min(t1,t2), max(t1,t2))
            row += f"  {tier_dists.get(key, 0):.3f}"
        print(row)

    print("\n=== 2. Boundary discrimination (adjacent tiers) ===")
    boundary = eval_boundary_discrimination(embs, entries)
    for boundary_name, stats in boundary.items():
        flag = "✓" if stats["best_acc"] >= 0.80 else "✗" if stats["best_acc"] < 0.65 else "~"
        print(f"  {flag} {boundary_name}: acc={stats['best_acc']:.3f}  sep={stats['separation']:.3f}  "
              f"d_same={stats['mean_dist_same']:.3f}  d_cross={stats['mean_dist_cross']:.3f}")

    print("\n=== 3. k-NN risk prediction (haiku-scored ground truth) ===")
    knn = eval_knn_risk(embs, entries)
    print(f"  k={knn['k']}  n={knn['n_eval']}  risk_MAE={knn['risk_mae']}  "
          f"risk_p90={knn['risk_p90']}  tier_acc={knn['tier_acc']:.3f}")

    print("\n=== 4. Adversarial separation (T-2 vs T0/T1) ===")
    adv = eval_adversarial_separation(embs, entries)
    for k, v in adv.items():
        print(f"  {k}: {v}")

    print("\n=== 5. Intra-tier cohesion (lower = tighter) ===")
    cohesion = eval_cohesion(embs, entries)
    for tier_name, d in cohesion.items():
        bar = "█" * int(d * 20)
        print(f"  {tier_name:20s}: {d:.4f}  {bar}")

    print("\n=== Summary ===")
    boundary_accs = [v["best_acc"] for v in boundary.values()]
    weak = [k for k, v in boundary.items() if v["best_acc"] < 0.75]
    print(f"  Min boundary acc:    {min(boundary_accs):.3f}")
    print(f"  Mean boundary acc:   {sum(boundary_accs)/len(boundary_accs):.3f}")
    print(f"  k-NN risk MAE:       {knn['risk_mae']}")
    print(f"  k-NN tier acc:       {knn['tier_acc']:.3f}")
    if weak:
        print(f"  Weak boundaries:     {', '.join(weak)}")
    else:
        print("  All boundaries ≥ 0.75 ✓")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--scores",     default=SCORES_CACHE)
    args = p.parse_args()
    run_eval(args.checkpoint, args.scores)


if __name__ == "__main__":
    main()
