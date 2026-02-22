"""
AlignLayer — Phase 2: Siamese Network for command risk scoring.

Architecture
------------
Character-level encoder shared between both inputs:
  cmd text → char embeddings → multi-scale 1D CNN → global max pool → MLP → L2-norm → 128-d embedding

Training objective: contrastive loss
  similar   (label=0, |risk_A - risk_B| < 0.15): embeddings pulled together
  dissimilar (label=1, |risk_A - risk_B| > 0.35): embeddings pushed apart (margin=1.0)

Inference
---------
  embed(cmd) → 128-d unit vector
  distance(a, b) = ||embed(a) - embed(b)||₂  ∈ [0, 2]
  risk(cmd) ≈ nearest-neighbor lookup in scored reference set

Usage
-----
  # Train
  python model/siamese.py train --pairs data/synthetic/pairs-v0.jsonl

  # Embed a command
  python model/siamese.py embed --checkpoint model/checkpoints/best.pt --cmd "git push origin main"

  # Eval
  python model/siamese.py eval --checkpoint model/checkpoints/best.pt --pairs data/synthetic/pairs-v0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VOCAB = " " + "".join(chr(i) for i in range(33, 127))  # printable ASCII
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}     # 1-indexed; 0 = pad
VOCAB_SIZE = len(VOCAB) + 1                              # +1 for pad
MAX_LEN = 256


def encode(cmd: str) -> torch.Tensor:
    """Encode command string to fixed-length int tensor (pad/truncate to MAX_LEN)."""
    idxs = [CHAR2IDX.get(c, 0) for c in cmd[:MAX_LEN]]
    idxs += [0] * (MAX_LEN - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CommandEncoder(nn.Module):
    """
    Character-level multi-scale CNN encoder.

    char embeddings (dim=32)
    → parallel 1D convs with kernel sizes [3, 5, 7, 11, 16, 21] (each → 64 filters)
    → ReLU + global max pool over time
    → concat (6 × 64 = 384) → LayerNorm → FC(384→128) → L2-norm

    Kernels 16 and 21 extend receptive field to capture long-range flag context
    (e.g. "push --force" 12 chars, "push origin main" 16 chars).
    """

    KERNELS = [3, 5, 7, 11, 16, 21]
    FILTERS = 64
    EMBED_DIM = 32
    OUT_DIM = 128

    def __init__(self, kernels: list[int] | None = None):
        super().__init__()
        ks = kernels if kernels is not None else self.KERNELS
        self.embed = nn.Embedding(VOCAB_SIZE, self.EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.EMBED_DIM, self.FILTERS, k, padding=k // 2)
            for k in ks
        ])
        cnn_out = self.FILTERS * len(ks)
        self.norm = nn.LayerNorm(cnn_out)
        self.fc = nn.Linear(cnn_out, self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, MAX_LEN) → (B, OUT_DIM) unit vectors"""
        e = self.embed(x).transpose(1, 2)          # (B, EMBED_DIM, L)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(e))                    # (B, FILTERS, L)
            pooled.append(c.max(dim=2).values)     # (B, FILTERS)
        h = torch.cat(pooled, dim=1)               # (B, 256)
        h = self.norm(h)
        h = self.fc(h)                             # (B, 128)
        return F.normalize(h, dim=1)               # unit sphere


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

MARGIN = 1.0


def contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard contrastive loss.
    labels: 0 = similar, 1 = dissimilar
    D: Euclidean distance in embedding space (∈ [0, 2] for unit vectors)
    """
    d = torch.norm(emb_a - emb_b, dim=1)
    sim_loss = (1 - labels) * 0.5 * d.pow(2)
    dis_loss = labels * 0.5 * F.relu(MARGIN - d).pow(2)
    return (sim_loss + dis_loss).mean()


def weighted_contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    tiers_a: torch.Tensor,
    tiers_b: torch.Tensor,
) -> torch.Tensor:
    """
    Tier-gap weighted contrastive loss.

    weight = |tier_a - tier_b|² (clamped ≥ 1 so same-tier pairs still train).
    Adversarial tiers (-2, -1) are clamped to 0 before gap computation to avoid
    inflated weights (T4 vs T-2 would otherwise get gap=6, weight=36).

    Similar pairs (label=0) use weight=1 always — tier gap is always 0 for them
    so the clamp(min=1) is the only contributor.
    """
    safe_a = tiers_a.float().clamp(min=0)
    safe_b = tiers_b.float().clamp(min=0)
    gap = (safe_a - safe_b).abs()
    weight = gap.pow(2).clamp(min=1.0)

    d = torch.norm(emb_a - emb_b, dim=1)
    sim_loss = (1 - labels) * 0.5 * d.pow(2)
    dis_loss = labels * 0.5 * F.relu(MARGIN - d).pow(2)
    return (weight * (sim_loss + dis_loss)).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Streams pairs from pairs-v*.jsonl. Loads all into memory (pairs are small)."""

    def __init__(self, path: str, max_pairs: int | None = None):
        self.pairs: list[tuple[str, str, int, int, int]] = []  # (cmd_a, cmd_b, label, tier_a, tier_b)
        with open(path) as f:
            for i, line in enumerate(f):
                if max_pairs and i >= max_pairs:
                    break
                if i > 0 and i % 500_000 == 0:
                    print(f"  {i:,} pairs...", flush=True)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    a = obj["action_a"]["text"]
                    b = obj["action_b"]["text"]
                    label = int(obj["label"])
                    tier_a = int(obj["action_a"].get("tier", 0))
                    tier_b = int(obj["action_b"].get("tier", 0))
                    self.pairs.append((a, b, label, tier_a, tier_b))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a, b, label, tier_a, tier_b = self.pairs[idx]
        return (
            encode(a),
            encode(b),
            torch.tensor(label,  dtype=torch.float32),
            torch.tensor(tier_a, dtype=torch.long),
            torch.tensor(tier_b, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("model/checkpoints")
DEFAULT_PAIRS = "data/synthetic/pairs-v0.jsonl"


def train(
    pairs_path: str = DEFAULT_PAIRS,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_frac: float = 0.05,
    max_pairs: int | None = None,
    device: str | None = None,
):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {dev}")

    print("Loading pairs...", end=" ", flush=True)
    dataset = PairDataset(pairs_path, max_pairs=max_pairs)
    print(f"{len(dataset):,} pairs loaded")

    # Train/val split
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    # num_workers=0 avoids fork-based multiprocessing deadlock on macOS/MPS.
    # pin_memory only helps CUDA; skip it for MPS/CPU.
    nw = 0 if dev.type in ("mps", "cpu") else 4
    pm = dev.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pm)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm)

    model = CommandEncoder().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for xa, xb, labels, ta, tb in train_dl:
            xa, xb, labels = xa.to(dev), xb.to(dev), labels.to(dev)
            ta, tb = ta.to(dev), tb.to(dev)
            emb_a = model(xa)
            emb_b = model(xb)
            loss = weighted_contrastive_loss(emb_a, emb_b, labels, ta, tb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_dl)

        # --- val ---
        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for xa, xb, labels, ta, tb in val_dl:
                xa, xb, labels = xa.to(dev), xb.to(dev), labels.to(dev)
                ta, tb = ta.to(dev), tb.to(dev)
                emb_a = model(xa)
                emb_b = model(xb)
                val_loss += weighted_contrastive_loss(emb_a, emb_b, labels, ta, tb).item()
                d = torch.norm(emb_a - emb_b, dim=1)
                pred = (d > MARGIN / 2).float()
                correct += (pred == labels).sum().item()
                total += len(labels)
        val_loss /= len(val_dl)
        acc = correct / total

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  acc={acc:.3f}  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = CHECKPOINT_DIR / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss, "acc": acc}, ckpt)
            print(f"  ✓ checkpoint saved ({ckpt})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, device: str | None = None) -> tuple[CommandEncoder, torch.device]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    sd = state["model"]
    # Detect architecture from checkpoint: norm.weight shape = FILTERS * n_kernels
    cnn_out = sd["norm.weight"].shape[0]
    n_kernels = cnn_out // CommandEncoder.FILTERS
    all_kernels = [3, 5, 7, 11, 16, 21]
    kernels = all_kernels[:n_kernels]
    model = CommandEncoder(kernels=kernels).to(dev)
    model.load_state_dict(sd)
    model.eval()
    return model, dev


def embed_command(cmd: str, model: CommandEncoder, dev: torch.device) -> torch.Tensor:
    x = encode(cmd).unsqueeze(0).to(dev)
    with torch.no_grad():
        return model(x).squeeze(0).cpu()


def build_reference_index(
    scores_cache: str,
    model: CommandEncoder,
    dev: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Embed all scored commands. Returns (embeddings, metadata_list).
    Used for nearest-neighbor risk lookup at inference time.
    """
    entries = []
    with open(scores_cache) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    all_embs = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        x = torch.stack([encode(e["text"]) for e in batch]).to(dev)
        with torch.no_grad():
            embs = model(x).cpu()
        all_embs.append(embs)

    return torch.cat(all_embs, dim=0), entries


# ---------------------------------------------------------------------------
# Heuristic pre-processing: dry-run detection + compound decomposition
# ---------------------------------------------------------------------------

_DRY_RUN_NEGATIONS: frozenset[str] = frozenset({
    "--no-dry-run", "--dry-run=none", "--dry-run=off",
})

_DRY_RUN_FLAGS: frozenset[str] = frozenset({
    "--dry-run", "--dry-run=client", "--dry-run=server",
    "--check", "--simulate", "--preview", "--no-act", "--what-if", "--noop",
})

# Subcommand-level patterns (flag not sufficient; need verb context)
_DRY_RUN_PATTERN = re.compile(
    r"(?:^|\s)terraform\s+plan(?:\s|$)"
    r"|(?:^|\s)make\s+-[a-zA-Z]*n[a-zA-Z]*(?:\s|$)"   # make -n, -nk, etc.
    r"|(?:^|\s)helm\s+\S+\s+.*?--dry-run",             # helm upgrade ... --dry-run
    re.DOTALL,
)


def is_dry_run(cmd: str) -> bool:
    """Return True if cmd is a dry-run/preview whose risk should be capped at T-1."""
    try:
        tokens = frozenset(shlex.split(cmd))
    except ValueError:
        tokens = frozenset(cmd.split())

    if tokens & _DRY_RUN_NEGATIONS:   # explicit opt-out wins
        return False
    if tokens & _DRY_RUN_FLAGS:
        return True
    if _DRY_RUN_PATTERN.search(cmd):
        return True
    return False


def _split_compound(cmd: str) -> list[str]:
    """
    Split a compound shell command on &&, ||, ;, | respecting quotes and $() depth.
    Falls back to regex split on unmatched-quote errors.
    Returns list of non-empty stripped subcommands.
    """
    parts: list[str] = []
    current: list[str] = []
    sq = dq = False   # single / double quote active
    depth = 0         # $( … ) nesting depth
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if c == "'" and not dq:
            sq = not sq
            current.append(c)
        elif c == '"' and not sq:
            dq = not dq
            current.append(c)
        elif sq or dq:
            current.append(c)
        elif c == '$' and i + 1 < len(cmd) and cmd[i + 1] == '(':
            depth += 1
            current.append(c)
        elif c == '(' and depth > 0:
            depth += 1
            current.append(c)
        elif c == ')' and depth > 0:
            depth -= 1
            current.append(c)
        elif depth > 0:
            current.append(c)
        elif c == '|' and i + 1 < len(cmd) and cmd[i + 1] == '|':
            parts.append("".join(current).strip())
            current = []
            i += 1  # skip second |
        elif c == '&' and i + 1 < len(cmd) and cmd[i + 1] == '&':
            parts.append("".join(current).strip())
            current = []
            i += 1  # skip second &
        elif c == '|':
            parts.append("".join(current).strip())
            current = []
        elif c == ';':
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1

    if current:
        parts.append("".join(current).strip())

    result = [p for p in parts if p]
    return result if result else [cmd]


# ---------------------------------------------------------------------------
# k-NN scoring (inner, no heuristics)
# ---------------------------------------------------------------------------

def _predict_single(
    cmd: str,
    model: CommandEncoder,
    dev: torch.device,
    ref_embs: torch.Tensor,
    ref_entries: list[dict],
    k: int,
) -> dict:
    """Raw k-NN prediction for a single command string. No heuristic overrides."""
    if is_dry_run(cmd):
        return {
            "command": cmd,
            "risk": 0.15,
            "blast_radius": 0.0,
            "tier": -1,
            "heuristic": "dry_run",
            "neighbors": [],
        }
    emb = embed_command(cmd, model, dev)
    dists = torch.norm(ref_embs - emb.unsqueeze(0), dim=1)
    topk = dists.topk(k, largest=False)
    neighbors = [ref_entries[i] for i in topk.indices.tolist()]
    avg_risk  = sum(n["risk"] for n in neighbors) / k
    avg_blast = sum(n.get("heuristic_blast", 0.0) for n in neighbors) / k
    votes = [n["tier"] for n in neighbors]
    tier = max(set(votes), key=votes.count)
    return {
        "command": cmd,
        "risk": round(avg_risk, 4),
        "blast_radius": round(avg_blast, 4),
        "tier": tier,
        "neighbors": [{"command": n["text"], "risk": n["risk"], "tier": n["tier"], "dist": round(d, 4)}
                      for n, d in zip(neighbors, topk.values.tolist())],
    }


def predict_risk(
    cmd: str,
    model: CommandEncoder,
    dev: torch.device,
    ref_embs: torch.Tensor,
    ref_entries: list[dict],
    k: int = 5,
    debug: bool = False,
) -> dict:
    """
    Risk prediction with heuristic pre-processing:
      1. Dry-run detection: if cmd contains --dry-run / terraform plan / etc., cap at T-1.
      2. Compound decomposition: split on &&, ||, ;, |; return max-risk subcommand.
    """
    subcommands = _split_compound(cmd)

    if len(subcommands) == 1:
        result = _predict_single(cmd, model, dev, ref_embs, ref_entries, k)
        result["command"] = cmd
        return result

    # Score each subcommand independently, return worst case
    sub_results = [_predict_single(sub, model, dev, ref_embs, ref_entries, k)
                   for sub in subcommands]
    best = max(sub_results, key=lambda r: r["risk"])
    result = dict(best)
    result["command"] = cmd   # restore original compound command
    if debug:
        result["decomposed"] = sub_results
    return result


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_model(checkpoint: str, pairs_path: str, max_pairs: int = 100_000):
    model, dev = load_model(checkpoint)
    print(f"Loaded checkpoint: {checkpoint}")

    dataset = PairDataset(pairs_path, max_pairs=max_pairs)
    dl = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)

    total_loss = correct = total = 0
    model.eval()
    with torch.no_grad():
        for xa, xb, labels, ta, tb in dl:
            xa, xb, labels = xa.to(dev), xb.to(dev), labels.to(dev)
            ta, tb = ta.to(dev), tb.to(dev)
            emb_a = model(xa)
            emb_b = model(xb)
            total_loss += weighted_contrastive_loss(emb_a, emb_b, labels, ta, tb).item()
            d = torch.norm(emb_a - emb_b, dim=1)
            pred = (d > MARGIN / 2).float()
            correct += (pred == labels).sum().item()
            total += len(labels)

    print(f"Val loss: {total_loss / len(dl):.4f}  Acc: {correct/total:.4f}  ({total:,} pairs)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="AlignLayer Siamese network")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--pairs",      default=DEFAULT_PAIRS)
    tr.add_argument("--epochs",     type=int,   default=20)
    tr.add_argument("--batch-size", type=int,   default=512)
    tr.add_argument("--lr",         type=float, default=1e-3)
    tr.add_argument("--max-pairs",  type=int,   default=None)
    tr.add_argument("--device",     default=None)

    ev = sub.add_parser("eval")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--pairs",      default=DEFAULT_PAIRS)
    ev.add_argument("--max-pairs",  type=int, default=100_000)

    em = sub.add_parser("embed")
    em.add_argument("--checkpoint", required=True)
    em.add_argument("--cmd",        required=True)
    em.add_argument("--scores-cache", default="data/synthetic/scores-cache.jsonl")

    args = p.parse_args()

    if args.cmd == "train":
        train(
            pairs_path=args.pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_pairs=args.max_pairs,
            device=args.device,
        )

    elif args.cmd == "eval":
        eval_model(args.checkpoint, args.pairs, args.max_pairs)

    elif args.cmd == "embed":
        model, dev = load_model(args.checkpoint)
        ref_embs, ref_entries = build_reference_index(args.scores_cache, model, dev)
        result = predict_risk(args.cmd, model, dev, ref_embs, ref_entries)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
