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

    def __init__(
        self,
        path: str,
        max_pairs: int | None = None,
        vocab: dict[str, int] | None = None,
    ):
        self.vocab = vocab  # None → char-only mode; dict → hybrid mode
        self.pairs: list[tuple[str, str, int, int, int]] = []
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
        char_a = encode(a)
        char_b = encode(b)
        t_label  = torch.tensor(label,  dtype=torch.float32)
        t_tier_a = torch.tensor(tier_a, dtype=torch.long)
        t_tier_b = torch.tensor(tier_b, dtype=torch.long)
        if self.vocab is not None:
            return (char_a, tokenize_word(a, self.vocab),
                    char_b, tokenize_word(b, self.vocab),
                    t_label, t_tier_a, t_tier_b)
        return (char_a, char_b, t_label, t_tier_a, t_tier_b)


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
    hybrid: bool = False,
    corpus_path: str | None = None,
):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {dev}")

    # Build word vocab from corpus if hybrid mode requested
    vocab: dict[str, int] | None = None
    if hybrid:
        if not corpus_path:
            raise ValueError("--corpus required for --hybrid training")
        print("Building word vocab from corpus...", end=" ", flush=True)
        commands: list[str] = []
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        commands.append(json.loads(line)["text"])
                    except Exception:
                        pass
        vocab = build_word_vocab(commands)
        print(f"{len(vocab):,} tokens")

    print("Loading pairs...", end=" ", flush=True)
    dataset = PairDataset(pairs_path, max_pairs=max_pairs, vocab=vocab)
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

    model: CommandEncoder | HybridEncoder
    if hybrid and vocab is not None:
        model = HybridEncoder(vocab_size=len(vocab)).to(dev)
        model._vocab = vocab
    else:
        model = CommandEncoder().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    def _forward(batch) -> tuple[torch.Tensor, torch.Tensor]:
        if hybrid and vocab is not None:
            char_a, word_a, char_b, word_b, labels, ta, tb = batch
            char_a, word_a = char_a.to(dev), word_a.to(dev)
            char_b, word_b = char_b.to(dev), word_b.to(dev)
            labels, ta, tb = labels.to(dev), ta.to(dev), tb.to(dev)
            emb_a = model(char_a, word_a)  # type: ignore[arg-type]
            emb_b = model(char_b, word_b)  # type: ignore[arg-type]
        else:
            xa, xb, labels, ta, tb = batch
            xa, xb, labels = xa.to(dev), xb.to(dev), labels.to(dev)
            ta, tb = ta.to(dev), tb.to(dev)
            emb_a = model(xa)  # type: ignore[arg-type]
            emb_b = model(xb)  # type: ignore[arg-type]
        return emb_a, emb_b, labels, ta, tb  # type: ignore[return-value]

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for batch in train_dl:
            emb_a, emb_b, labels, ta, tb = _forward(batch)
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
            for batch in val_dl:
                emb_a, emb_b, labels, ta, tb = _forward(batch)
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
            ckpt_dict: dict = {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss, "acc": acc}
            if hybrid and vocab is not None:
                ckpt_dict["vocab"] = vocab
            torch.save(ckpt_dict, ckpt)
            print(f"  ✓ checkpoint saved ({ckpt})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    return model


# ---------------------------------------------------------------------------
# Risk head training (second stage, frozen encoder)
# ---------------------------------------------------------------------------

def train_risk_head(
    checkpoint: str = "model/checkpoints/best.pt",
    corpus_path: str = "data/synthetic/scores-cache.jsonl",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-3,
    val_frac: float = 0.1,
    device: str | None = None,
):
    """Train RiskHead MLP on frozen encoder embeddings. Saves into the same checkpoint."""
    model, dev = load_model(checkpoint, device)
    model.eval()

    # Load corpus
    entries: list[dict] = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    print(f"Corpus: {len(entries):,} entries")

    # Embed entire corpus (frozen)
    print("Embedding corpus...", end=" ", flush=True)
    is_hybrid = isinstance(model, HybridEncoder)
    all_embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            char_x = torch.stack([encode(e["text"]) for e in batch]).to(dev)
            if is_hybrid:
                word_x = torch.stack([tokenize_word(e["text"], model._vocab) for e in batch]).to(dev)
                all_embs.append(model(char_x, word_x).cpu())
            else:
                all_embs.append(model(char_x).cpu())
    embs = torch.cat(all_embs, dim=0)
    risks = torch.tensor([e["risk"] for e in entries], dtype=torch.float32)
    print(f"done. shape={embs.shape}")

    # Train/val split
    n = len(embs)
    perm = torch.randperm(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    head = RiskHead(in_dim=embs.shape[1]).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        # Shuffle training batches
        shuf = train_idx[torch.randperm(len(train_idx))]
        train_loss = 0.0
        n_batches = 0
        for i in range(0, len(shuf), batch_size):
            idx = shuf[i : i + batch_size]
            e_batch = embs[idx].to(dev)
            r_batch = risks[idx].to(dev)
            pred = head(e_batch)
            loss = F.mse_loss(pred, r_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss /= n_batches

        # Val
        head.eval()
        with torch.no_grad():
            v_emb = embs[val_idx].to(dev)
            v_risk = risks[val_idx].to(dev)
            v_pred = head(v_emb)
            val_loss = F.mse_loss(v_pred, v_risk).item()
            val_mae = (v_pred - v_risk).abs().mean().item()

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            saved = "  ✓ best"

        if epoch % 5 == 0 or epoch == 1 or saved:
            print(f"Epoch {epoch:3d}/{epochs}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}  val_mae={val_mae:.4f}  lr={scheduler.get_last_lr()[0]:.2e}{saved}")

    # Save risk head into the encoder checkpoint
    if best_state is not None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt["risk_head"] = best_state
        torch.save(ckpt, checkpoint)
        print(f"\n✓ Risk head saved into {checkpoint} (val_mse={best_val:.5f})")
    else:
        print("\nNo improvement — risk head not saved.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: str, device: str | None = None
) -> tuple[CommandEncoder | HybridEncoder, torch.device]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    state = torch.load(checkpoint, map_location=dev, weights_only=False)
    sd = state["model"]

    if "vocab" in state:
        # HybridEncoder checkpoint
        vocab = state["vocab"]
        model: CommandEncoder | HybridEncoder = HybridEncoder(vocab_size=len(vocab)).to(dev)
        model.load_state_dict(sd)
        model._vocab = vocab  # type: ignore[union-attr]
    else:
        # Legacy char-only checkpoint — detect kernel count from norm.weight
        cnn_out = sd["norm.weight"].shape[0]
        n_kernels = cnn_out // CommandEncoder.FILTERS
        kernels = [3, 5, 7, 11, 16, 21][:n_kernels]
        model = CommandEncoder(kernels=kernels).to(dev)
        model.load_state_dict(sd)

    model.eval()

    # Load risk head if present in checkpoint
    if "risk_head" in state:
        in_dim = state["risk_head"]["net.0.weight"].shape[1]
        hidden = state["risk_head"]["net.0.weight"].shape[0]
        risk_head = RiskHead(in_dim=in_dim, hidden=hidden).to(dev)
        risk_head.load_state_dict(state["risk_head"])
        risk_head.eval()
        model._risk_head = risk_head  # type: ignore[union-attr]

    return model, dev


def embed_command(
    cmd: str, model: CommandEncoder | HybridEncoder, dev: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        if isinstance(model, HybridEncoder):
            char_ids = encode(cmd).unsqueeze(0).to(dev)
            word_ids = tokenize_word(cmd, model._vocab).unsqueeze(0).to(dev)
            return model(char_ids, word_ids).squeeze(0).cpu()
        x = encode(cmd).unsqueeze(0).to(dev)
        return model(x).squeeze(0).cpu()


def build_reference_index(
    scores_cache: str,
    model: CommandEncoder | HybridEncoder,
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

    is_hybrid = isinstance(model, HybridEncoder)
    all_embs = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        with torch.no_grad():
            if is_hybrid:
                char_x = torch.stack([encode(e["text"]) for e in batch]).to(dev)
                word_x = torch.stack([tokenize_word(e["text"], model._vocab) for e in batch]).to(dev)  # type: ignore[union-attr]
                embs = model(char_x, word_x).cpu()
            else:
                x = torch.stack([encode(e["text"]) for e in batch]).to(dev)
                embs = model(x).cpu()
        all_embs.append(embs)

    return torch.cat(all_embs, dim=0), entries


# ---------------------------------------------------------------------------
# Word-level vocabulary and tokenizer
# ---------------------------------------------------------------------------

WORD_MAX_LEN = 32       # max tokens per command for word branch
_WORD_PAD = 0
_WORD_OOV = 1
_WORD_RESERVED = 2      # first real token index


def build_word_vocab(commands: list[str], max_size: int = 4096) -> dict[str, int]:
    """Build token vocabulary from command strings. 0=pad, 1=OOV, 2+=tokens."""
    from collections import Counter
    counter: Counter[str] = Counter()
    for cmd in commands:
        for tok in cmd.strip().split():
            counter[tok.lower()] += 1
    vocab: dict[str, int] = {"<PAD>": _WORD_PAD, "<OOV>": _WORD_OOV}
    for tok, _ in counter.most_common(max_size):
        vocab[tok] = len(vocab)
    return vocab


def tokenize_word(cmd: str, vocab: dict[str, int]) -> torch.Tensor:
    """Tokenize command into word-level IDs (pad/truncate to WORD_MAX_LEN)."""
    tokens = cmd.strip().split()[:WORD_MAX_LEN]
    ids = [vocab.get(t.lower(), _WORD_OOV) for t in tokens]
    ids += [_WORD_PAD] * (WORD_MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Word-level CNN branch
# ---------------------------------------------------------------------------

class WordBranch(nn.Module):
    """
    Token-level CNN encoder branch.

    word tokens (each → embed dim 32)
    → parallel 1D convs with kernels [2, 3, 4] (each → 32 filters)
    → ReLU + global max pool → concat (3 × 32 = 96) → LayerNorm → FC(96→64)

    Smaller than CharEncoder by design — word sequences are short (≤32 tokens)
    and the branch provides complementary signal, not a full replacement.
    """
    KERNELS   = [2, 3, 4]
    FILTERS   = 32
    EMBED_DIM = 32
    OUT_DIM   = 64

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, self.EMBED_DIM, padding_idx=_WORD_PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.EMBED_DIM, self.FILTERS, k, padding=k // 2)
            for k in self.KERNELS
        ])
        cnn_out = self.FILTERS * len(self.KERNELS)
        self.norm = nn.LayerNorm(cnn_out)
        self.fc   = nn.Linear(cnn_out, self.OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, WORD_MAX_LEN) → (B, OUT_DIM)"""
        e = self.embed(x).transpose(1, 2)      # (B, EMBED_DIM, L)
        pooled = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]
        h = self.norm(torch.cat(pooled, dim=1))
        return self.fc(h)                       # (B, 64) — not normalized here


# ---------------------------------------------------------------------------
# Hybrid encoder (char + word)
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """
    Char-level CNN (128-dim) ‖ Word-level CNN (64-dim) → fused → 128-dim unit vector.

    The char branch captures n-gram morphology and flag syntax; the word branch
    adds token-level semantics so `describe-instances` and `terminate-instances`
    map to different regions of the embedding space.

    Vocab is stored as an instance attribute (not an nn.Parameter) and is
    persisted in the checkpoint dict alongside the state_dict.
    """
    OUT_DIM = 128

    def __init__(self, vocab_size: int):
        super().__init__()
        self.char_branch = CommandEncoder()          # 128-dim, normalized
        self.word_branch = WordBranch(vocab_size)    # 64-dim, unnormalized
        fused = CommandEncoder.OUT_DIM + WordBranch.OUT_DIM   # 192
        self.norm = nn.LayerNorm(fused)
        self.fc   = nn.Linear(fused, self.OUT_DIM)
        # vocab stored as plain dict — persisted in checkpoint, not as nn params
        self._vocab: dict[str, int] = {}

    def forward(self, char_ids: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: (B, MAX_LEN)      — character token IDs
        word_ids: (B, WORD_MAX_LEN) — word token IDs
        → (B, 128) unit vectors
        """
        char_emb = self.char_branch(char_ids)     # (B, 128) normalized
        word_emb = self.word_branch(word_ids)     # (B,  64) unnormalized
        h = self.norm(torch.cat([char_emb, word_emb], dim=1))   # (B, 192)
        return F.normalize(self.fc(h), dim=1)     # (B, 128)


# ---------------------------------------------------------------------------
# Risk head: MLP that maps embeddings directly to risk scores
# ---------------------------------------------------------------------------

class RiskHead(nn.Module):
    """
    Small MLP that maps 128-d embeddings to scalar risk ∈ [0, 1].

    Replaces k-NN at inference time — no more noisy neighbor votes.
    Trained as a second stage on frozen encoder embeddings.
    """
    def __init__(self, in_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (B, 128) → (B,) risk scores in [0, 1]"""
        return self.net(emb).squeeze(-1)


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


# ---------------------------------------------------------------------------
# CLI verb table (heuristic pre-check before k-NN)
# ---------------------------------------------------------------------------
#
# Each entry: (tool_re, verb_re, tier, risk, is_cap)
# is_cap=True  → use as a ceiling (known-safe reads)
# is_cap=False → use as a floor  (known-risky mutations)
#
# Only fires when the full command string starts with the tool name; verb_re is
# matched against the full remainder after the tool (handles multi-word CLIs
# like `aws ec2 describe-instances` and `kubectl rollout status`).

_VERB_TABLE: list[tuple[re.Pattern, re.Pattern, int, float, bool]] = [
    # terraform
    (re.compile(r"^terraform$"),  re.compile(r"^destroy\b"),                          4,  0.92, False),
    (re.compile(r"^terraform$"),  re.compile(r"^apply\b"),                            4,  0.82, False),
    # plan is a dry-run — T-1, not T1
    (re.compile(r"^terraform$"),  re.compile(r"^plan\b"),                            -1,  0.15, True),
    (re.compile(r"^terraform$"),  re.compile(r"^(show|output|validate|fmt|graph|version|workspace|state\s+list|providers)\b"), 0, 0.08, True),

    # redis-cli — verb may appear after -h host -p 6379 etc., so match anywhere
    # in the remainder (not anchored to start).
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(flushall|shutdown)\b", re.I),         4,  0.92, False),
    (re.compile(r"^redis-cli$"),  re.compile(r"\bflushdb\b", re.I),                      2,  0.48, True),
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(del|unlink|expire|persist|rename)\b", re.I), 2, 0.40, False),
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(save|bgsave|bgrewriteaof)\b", re.I),   2,  0.42, True),
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(config\s+set)\b", re.I),               3,  0.62, False),
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(get|keys|scan|info|dbsize|ttl|type|llen|smembers|hgetall|zrange|config\s+get|ping)\b", re.I), 0, 0.05, True),

    # aws — match service+operation in the remainder
    (re.compile(r"^aws$"),        re.compile(r"terminate-instances|stop-instances|delete-|deregister-|revoke-|disable-"),  3, 0.65, False),
    (re.compile(r"^aws$"),        re.compile(r"\bdescribe-|\blist-|\bget-|\bshow-|\bs3\s+ls\b|\blogs\s+(tail|filter-log-events)\b"), 0, 0.05, True),

    # kubectl
    (re.compile(r"^kubectl$"),    re.compile(r"^delete\b"),                            3,  0.65, False),
    (re.compile(r"^kubectl$"),    re.compile(r"^apply\b"),                             3,  0.62, False),
    (re.compile(r"^kubectl$"),    re.compile(r"^(get|describe|logs?|top|explain|diff|version|cluster-info|auth|rollout\s+status|config\s+view)\b"), 0, 0.05, True),

    # git — force push before general push (first match wins)
    (re.compile(r"^git$"),        re.compile(r"^push\b.*--force\b|^push\b.*-f\b"),     4,  0.88, False),
    (re.compile(r"^git$"),        re.compile(r"^push\b"),                             3,  0.62, False),

    # aws write ops not covered by terminate/delete pattern
    (re.compile(r"^aws$"),        re.compile(r"\bsns\s+publish\b|\bsqs\s+send-message\b|\bs3\s+cp\b|\bs3\s+mv\b|\bs3\s+sync\b|\bec2\s+run-instances\b"), 3, 0.62, False),

    # find -delete: file deletion via find is T4 (operates recursively, often at scale)
    (re.compile(r"^find$"),       re.compile(r"\B-delete\b"),                          4,  0.80, False),

    # system power/init commands
    (re.compile(r"^(sudo|reboot|shutdown|poweroff|init|systemctl)$"),
                                  re.compile(r"^(reboot|shutdown|poweroff|halt|stop|disable)"),
                                                                                       4,  0.82, False),

    # npm / yarn publish
    (re.compile(r"^(npm|yarn|pnpm)$"), re.compile(r"^(publish|unpublish|deprecate)\b"), 3, 0.68, False),

    # kubectl create/patch in production contexts
    (re.compile(r"^kubectl$"),    re.compile(r"^(create|patch|replace|set)\b"),          3,  0.62, False),

    # aws s3 rm (especially --recursive)
    (re.compile(r"^aws$"),        re.compile(r"\bs3\s+rm\b"),                            4,  0.82, False),

    # aws autoscaling / iam / organizations destructive ops
    (re.compile(r"^aws$"),        re.compile(r"\biam\s+delete-\b|\bautoscaling\s+suspend-\b|\bautoscaling\s+delete-\b"), 3, 0.65, False),
    (re.compile(r"^aws$"),        re.compile(r"\borganizations\s+leave-organization\b"),   4,  0.88, False),

    # ssh with destructive remote commands
    (re.compile(r"^ssh$"),        re.compile(r"(sudo\s+reboot|sudo\s+shutdown|sudo\s+rm\s|rm\s+-rf|sudo\s+systemctl\s+stop|init\s+0)"), 4, 0.88, False),
    # ssh read-only: cap at T0
    (re.compile(r"^ssh$"),        re.compile(r"(cat\s|tail\s|head\s|less\s|grep\s|ps\s|top\s|uptime|df\s|free\s|uname|hostname|whoami|id\b|env\b|printenv|systemctl\s+status\b|service\s+\S+\s+status\b|redis-cli\s+(ping|info)\b|pg_isready\b|netstat\b|ss\b|iptables\s+-L)"), 0, 0.08, True),

    # docker / podman destructive
    (re.compile(r"^(docker|podman)$"), re.compile(r"^(system\s+prune|volume\s+prune|image\s+prune)\b"), 2, 0.48, False),
    (re.compile(r"^(docker|podman)$"), re.compile(r"^push\b"),                           3,  0.62, False),
    (re.compile(r"^(docker|podman)$"), re.compile(r"^(ps|images|inspect|logs|stats|top|port|diff|history|version)\b"), 0, 0.05, True),

    # terraform state show/list — read-only inspection
    (re.compile(r"^terraform$"),  re.compile(r"^state\s+show\b"),                        0,  0.08, True),

    # SQL/query clients: DDL/DML mutations are risky; reads are safe
    # Covers: psql, mysql, mariadb, sqlite3, snowsql, duckdb, clickhouse-client, trino, presto, cqlsh
    # Write rules first (order matters — first match wins)
    (re.compile(r"^(psql|mysql|mariadb|sqlite3|snowsql|duckdb|clickhouse-client|trino|presto|cqlsh)$"), re.compile(r"(DROP\s+(DATABASE|SCHEMA|TABLE|KEYSPACE)\s)", re.I),  4,  0.82, False),
    (re.compile(r"^(psql|mysql|mariadb|sqlite3|snowsql|duckdb|clickhouse-client|trino|presto|cqlsh)$"), re.compile(r"(CREATE\s+TABLE|ALTER\s+TABLE|GRANT\s|REVOKE\s)", re.I), 3, 0.62, False),
    (re.compile(r"^(psql|mysql|mariadb|sqlite3|snowsql|duckdb|clickhouse-client|trino|presto|cqlsh)$"), re.compile(r"(INSERT\s|UPDATE\s|DELETE\s|TRUNCATE\s|VACUUM\s|REINDEX\s|DROP\s+INDEX)", re.I), 2, 0.45, False),
    # Read-only: SELECT, EXPLAIN, SHOW, DESCRIBE, psql \d, sqlite3 dot-commands
    (re.compile(r"^(psql|mysql|mariadb|sqlite3|snowsql|duckdb|clickhouse-client|trino|presto|cqlsh)$"), re.compile(r"(SELECT\s|EXPLAIN\s|SHOW\s|DESCRIBE\s|\\\\?d[tl+s]|\.tables|\.schema|\.header)", re.I), 0, 0.08, True),

    # BigQuery CLI (bq)
    (re.compile(r"^bq$"),         re.compile(r"^(rm|remove)\b"),                               4,  0.82, False),
    (re.compile(r"^bq$"),         re.compile(r"^(mk|load|insert|update|cp)\b"),                 2,  0.45, False),
    (re.compile(r"^bq$"),         re.compile(r"^(query|ls|show|head|extract|version)\b"),       0,  0.08, True),

    # mongosh / mongo: db.collection.drop() vs db.collection.find()
    (re.compile(r"^(mongosh|mongo)$"), re.compile(r"\.(drop|remove|delete|update|insert|replace|rename)\s*\(", re.I), 3, 0.65, False),
    (re.compile(r"^(mongosh|mongo)$"), re.compile(r"\.(find|count|aggregate|distinct|explain|getIndexes|stats)\s*\(", re.I), 0, 0.08, True),

    # gh (GitHub CLI)
    (re.compile(r"^gh$"),         re.compile(r"^workflow\s+(enable|disable)\b"),           3,  0.62, False),
    (re.compile(r"^gh$"),         re.compile(r"^(run\s+view|pr\s+(view|list|status|checks)|issue\s+(view|list)|repo\s+view)\b"), 0, 0.05, True),

    # git read-only
    (re.compile(r"^git$"),        re.compile(r"^(log|diff|show|status|branch\s+-[avr]|tag\s+-l|remote\s+-v|stash\s+list)\b"), 0, 0.05, True),

    # find without -delete or destructive -exec is read-only
    (re.compile(r"^find$"),       re.compile(r"^(?!.*(-delete|xargs\s+rm|-exec\s+(rm|mv|chmod|chown)))"), 0, 0.08, True),

    # nginx: -T (dump config) is read-only, -s reload/stop is risky
    (re.compile(r"^nginx$"),      re.compile(r"-s\s+(reload|stop|quit)"),                  3,  0.62, False),
    (re.compile(r"^nginx$"),      re.compile(r"^-[tT]\b"),                               0,  0.05, True),
    (re.compile(r"^journalctl$"), re.compile(r"--vacuum"), 2, 0.45, False),
    (re.compile(r"^journalctl$"), re.compile(r""), 0, 0.08, True),
    (re.compile(r"^ssh-keygen$"), re.compile(r"^-l\b"),                                  0,  0.05, True),
    (re.compile(r"^ssh-keyscan$"), re.compile(r""),                                      0,  0.08, True),

    # pip/npm local install (not from URL) — T1 cap
    (re.compile(r"^pip3?$"),      re.compile(r"^install\s+(?!https?://)"),                 1,  0.30, True),
    (re.compile(r"^(npm|yarn|pnpm)$"), re.compile(r"^install\b"),                         1,  0.30, True),

    # Pure filter commands — T0 cap (these only process stdin, never write)
    (re.compile(r"^(grep|egrep|fgrep|rg)$"), re.compile(r""),                             0,  0.05, True),
    (re.compile(r"^(head|tail|wc|sort|uniq|cut|tr|less|more)$"), re.compile(r""),          0,  0.05, True),
    (re.compile(r"^(awk|jq|yq|column|fmt|fold)$"), re.compile(r""),                       0,  0.05, True),
    (re.compile(r"^(uname|whoami|hostname|uptime|id|pwd|true|false)$"), re.compile(r""),   0,  0.03, True),

    # diff is always read-only
    (re.compile(r"^diff$"),       re.compile(r""),                                         0,  0.05, True),

    # cat: only cap if no output redirection (> or >>). Exfil patterns caught by _EXFIL_EXEC_RE.
    # NOTE: can't detect > in verb table since it's in remainder after shlex.
    # Leave cat to k-NN — the corpus handles it.

    # curl read-only: no -d/--data/-X POST/--upload/--form (exfil caught by _EXFIL_EXEC_RE)
    (re.compile(r"^curl$"),       re.compile(r"(-d\s|--data|--upload|-X\s*(POST|PUT|PATCH|DELETE)|--form|-F\s)"), 3, 0.62, False),
    (re.compile(r"^curl$"),       re.compile(r""),                                         0,  0.08, True),

    # sed without -i is a filter (writes to stdout)
    (re.compile(r"^sed$"),        re.compile(r"-i\b"),                                     1,  0.30, False),
    (re.compile(r"^sed$"),        re.compile(r""),                                         0,  0.08, True),

    # tar: create/extract to local paths is T1
    (re.compile(r"^tar$"),        re.compile(r""),                                         1,  0.25, True),

    # cp is local write (T1), not T4
    (re.compile(r"^cp$"),         re.compile(r""),                                         1,  0.28, True),

    # helm: template/lint/show/list are read-only; install/upgrade/delete are risky
    (re.compile(r"^helm$"),       re.compile(r"^(install|upgrade|rollback|delete|uninstall)\b"), 3,  0.62, False),
    (re.compile(r"^helm$"),       re.compile(r"^(template|lint|show|list|get|status|version|search|repo\s+list|env)\b"), 0, 0.05, True),

    # wget: downloads to local file — T1 cap (exfil chains caught by _EXFIL_EXEC_RE)
    (re.compile(r"^wget$"),       re.compile(r""),                                         1,  0.28, True),

    # base64: encode/decode is a pure filter (stdout) — T0 cap
    (re.compile(r"^base64$"),     re.compile(r""),                                         0,  0.05, True),

    # Common inspection/info commands — T0 cap
    (re.compile(r"^(date|env|printenv|type|which|file|stat|du|df|free|lsof|ps|top|htop|vmstat|iostat|netstat|ss|ip|ifconfig|ping|traceroute|dig|nslookup|host)$"),
                                  re.compile(r""),                                         0,  0.05, True),

    # Scanning/security audit tools — T0 cap (read-only analysis)
    (re.compile(r"^openssl$"),    re.compile(r""),                                         0,  0.05, True),
    (re.compile(r"^(nmap|nikto|gobuster|trivy|bandit|safety)$"), re.compile(r""),          0,  0.08, True),

    # systemctl: status/is-active are reads; start/stop/restart/enable/disable are mutations
    (re.compile(r"^systemctl$"),  re.compile(r"^(restart|stop|start|enable|disable|mask|unmask)\b"), 3, 0.62, False),
    (re.compile(r"^systemctl$"),  re.compile(r""),                                         0,  0.05, True),

    # brew: list/search/info are reads; install/upgrade/uninstall are local writes
    (re.compile(r"^brew$"),       re.compile(r"^(install|upgrade|uninstall|remove|tap|untap)\b"), 1, 0.28, True),
    (re.compile(r"^brew$"),       re.compile(r""),                                         0,  0.05, True),

    # tmux/screen: session management is T0, kill is T2, send-keys is risky (proxy exec)
    (re.compile(r"^tmux$"),       re.compile(r"^send-keys?\b"),                                  3,  0.65, False),
    (re.compile(r"^tmux$"),       re.compile(r"^kill-(server|session)\b"),                        2,  0.45, False),
    (re.compile(r"^tmux$"),       re.compile(r"^(ls|list-|display-|show-|has-session|info)\b"),   0,  0.05, True),
    (re.compile(r"^tmux$"),       re.compile(r""),                                               0,  0.08, True),
    (re.compile(r"^screen$"),     re.compile(r"-[Xx]\s+(quit|kill)\b"),                             2,  0.45, False),
    (re.compile(r"^screen$"),     re.compile(r""),                                               0,  0.08, True),

    # xargs: risk depends on the command being invoked
    (re.compile(r"^xargs$"),  re.compile(r"\b(rm|rmdir)\b"),                              4,  0.80, False),
    (re.compile(r"^xargs$"),  re.compile(r"\b(chmod|chown|mv|kill)\b"),                   2,  0.45, False),
    (re.compile(r"^xargs$"),  re.compile(r"\b(grep|cat|ls|echo|file|stat|wc|head|tail)\b"), 0, 0.05, True),

    # nc/ncat/socat: listeners and reverse shells are high-risk
    (re.compile(r"^(nc|ncat|netcat)$"), re.compile(r"-[a-z]*e\s+/bin/(ba)?sh"),           -2, 0.95, False),
    (re.compile(r"^(nc|ncat|netcat)$"), re.compile(r"-[a-z]*l"),                           3, 0.65, False),
    (re.compile(r"^socat$"),  re.compile(r"EXEC:|exec:|SYSTEM:|system:"),                  -2, 0.95, False),
    (re.compile(r"^socat$"),  re.compile(r"TCP-LISTEN|tcp-listen"),                         3, 0.65, False),

    # done/fi/esac — closing keywords only, never risky on their own
    (re.compile(r"^(done|fi|esac)$"), re.compile(r""), 0, 0.03, True),
]

_EXFIL_EXEC_RE = re.compile(
    r'eval\s+"\$\(curl'
    r"|curl\s+.*\|\s*(sudo\s+)?(ba)?sh"
    r"|wget\s+.*\|\s*(sudo\s+)?(ba)?sh"
    r"|curl\s+.*\|\s*python"
    r"|base64\s+-d\s*\|\s*(ba)?sh"
    r"|curl\s+.*-d\s+.*@/etc/(passwd|shadow)"
    r"|nc\s+-[a-z]*e\s+/bin/(ba)?sh"
    r"|/dev/tcp/"
    r"|urllib\.request\.urlopen\("
    r"|requests\.get\(.*\)\.text.*exec\("
    # download-and-execute chains: wget/curl && ... && ./script or bash script
    r"|wget\s+.*&&.*\./\w+.*\.sh"
    r"|curl\s+.*-o\s+\S+.*&&.*\./\w+"
    r"|curl\s+.*-o\s+\S+.*&&.*(?:ba)?sh\s+\S+"
    r"|wget\s+.*-O\s+\S+.*&&.*(?:ba)?sh\s+\S+"
    # npm/pip install from URL (typosquat / supply chain attack)
    r"|npm\s+install\s+https?://"
    r"|pip\s+install\s+https?://"
    # install + immediate code execution (supply chain)
    r"|npm\s+install\s+.*&&.*node\s+-e\s"
    # git clone + build/install/execute in one shot (untrusted repo execution)
    r"|git\s+clone\s+.*&&.*make\s+(install|all)\b"
    r"|git\s+clone\s+.*&&.*\./\w+"
    r"|git\s+clone\s+.*&&.*/install\b"
    # credential exfil via pipe to nc/curl
    r"|cat\s+.*\.(pem|key|id_rsa|id_ed25519|credentials|env)\s*\|"
    r"|cat\s+~/\.ssh/\S+\s*\|"
    # env var credential exfil
    r"|AWS_ACCESS_KEY_ID=\S+\s+AWS_SECRET_ACCESS_KEY=\S+\s+aws\s+s3\s+(sync|cp)\s+/"
    # curl posting secrets/tokens to external URLs
    r"|curl\s+.*Authorization.*\$\w*(SECRET|TOKEN|KEY|PASS)"
    # curl file upload: -F file=@<path> or --form file=@<path>
    r"|curl\s+.*(-F|--form)\s+\S*=@\S+",
    re.IGNORECASE,
)

_LOOPBACK_RE = re.compile(
    r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?:[:/]|$)",
    re.IGNORECASE,
)

_OPAQUE_EXEC_RE = re.compile(
    r"(?:python3?|node|perl|ruby|bash|sh)\s+-[ce]\s+"
    r"|(?<!\w)eval\s+[\$'\"`(]"
    r"|\|\s*(?:bash|sh)\b"
    r"|base64\s+-d\s*\|",
    re.IGNORECASE,
)


_CMD_PREFIXES = {"sudo", "env", "nice", "nohup", "caffeinate", "time", "do", "then", "else"}

def _parse_cli_verb(cmd: str) -> tuple[str, str]:
    """Return (tool, remainder) from a command string, stripping sudo/env/etc."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if not parts:
        return "", ""
    # Strip common command prefixes (sudo, env, nice, etc.) and their flags
    while len(parts) > 1:
        low = parts[0].lower()
        if low in _CMD_PREFIXES:
            parts = parts[1:]
            # Skip flags belonging to the prefix (e.g., nice -n 10, sudo -u user)
            while len(parts) > 1 and parts[0].startswith("-"):
                parts = parts[1:]
                # Skip flag argument if numeric (nice -n 10)
                if len(parts) > 1 and parts[0].isdigit():
                    parts = parts[1:]
        elif "=" in low and not low.startswith("-"):
            # Skip env var assignments (FOO=bar cmd ...)
            parts = parts[1:]
        else:
            break
    tool = parts[0].lower()
    # Strip directory paths: /usr/bin/rm → rm, ./script.sh → script.sh
    if "/" in tool:
        tool = tool.rsplit("/", 1)[-1]
    remainder = " ".join(parts[1:]).lower()
    # Unwrap `python3 -m <module>` → tool=module, remainder=rest
    if tool in ("python3", "python", "python3.10", "python3.11", "python3.12", "python3.13", "python3.14") \
       and len(parts) >= 3 and parts[1] == "-m":
        tool = parts[2].lower()
        remainder = " ".join(parts[3:]).lower()
    return tool, remainder


_SUBSHELL_RE = re.compile(r"\$\(|`")

def _verb_table_lookup(cmd: str) -> dict | None:
    """
    Check cmd against the verb table. Returns result dict or None if no match.
    Floors and caps are both expressed as full results — caller does not need
    to know the distinction.
    """
    tool, remainder = _parse_cli_verb(cmd)
    # First positional arg (skip flags) used for tools where verb = subcommand
    first_pos = next((t for t in remainder.split() if not t.startswith("-")), "")
    for tool_re, verb_re, tier, risk, is_cap in _VERB_TABLE:
        if not tool_re.match(tool):
            continue
        target = remainder if tool in {
            "aws", "kubectl", "git", "redis-cli", "ssh", "docker", "podman",
            "find", "gh", "terraform", "nginx", "ssh-keygen", "ssh-keyscan",
            "psql", "mysql", "mariadb", "sqlite3", "snowsql", "duckdb",
            "clickhouse-client", "trino", "presto", "cqlsh", "bq",
            "mongosh", "mongo", "tmux", "screen",
            "pip", "pip3", "npm", "yarn", "pnpm", "journalctl", "curl", "sed",
            "helm", "systemctl", "brew", "xargs",
            "nc", "ncat", "netcat", "socat",
        } else first_pos
        if verb_re.search(target):
            # SQL/query clients: don't cap as safe if command embeds subshell ($() or ``)
            # — prevents SELECT wrapping injection like: SELECT $(curl evil.com)
            _SQL_TOOLS = {"psql", "mysql", "mariadb", "sqlite3", "snowsql", "duckdb",
                          "clickhouse-client", "trino", "presto", "cqlsh", "mongosh", "mongo"}
            if is_cap and tool in _SQL_TOOLS and _SUBSHELL_RE.search(cmd):
                continue
            return {"tier": tier, "risk": risk, "is_cap": is_cap, "heuristic": "verb_table"}
    return None


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
            current.append('(')
            i += 1  # skip the '(' — already counted in depth
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
    if not result:
        return [cmd]

    # Extract inner commands from $() in for/while loop headers.
    # "for x in $(cmd ...)" → replace the loop header with the inner command.
    expanded = []
    for part in result:
        if re.match(r"^(for|while)\s", part, re.I):
            inners = [m.group(1).strip() for m in re.finditer(r"\$\((.+?)\)", part)]
            if inners:
                expanded.extend(inners)  # replace loop header with inner commands
                continue
        expanded.append(part)
    return expanded


# ---------------------------------------------------------------------------
# k-NN scoring (inner, no heuristics)
# ---------------------------------------------------------------------------

def _risk_to_tier(risk: float) -> int:
    """Deterministic mapping from continuous risk score to tier.

    T-2 is never assigned here — it comes only from the exfil heuristic.
    T-1 is never assigned here — it comes only from dry-run detection.
    """
    if risk < 0.18:
        return 0   # T0: read-only
    if risk < 0.38:
        return 1   # T1: local-write
    if risk < 0.55:
        return 2   # T2: destructive-local
    if risk < 0.74:
        return 3   # T3: external
    return 4       # T4: catastrophic


def predict_encoder_only(
    cmd: str,
    model: CommandEncoder,
    dev: torch.device,
    ref_embs: torch.Tensor,
    ref_entries: list[dict],
    k: int,
) -> dict:
    """Pure encoder prediction — no heuristics at all. RiskHead MLP or k-NN."""
    emb = embed_command(cmd, model, dev)

    risk_head: RiskHead | None = getattr(model, "_risk_head", None)
    if risk_head is not None:
        with torch.no_grad():
            pred_risk = risk_head(emb.unsqueeze(0).to(dev)).item()
        tier = _risk_to_tier(pred_risk)
        return {
            "command": cmd,
            "risk": round(pred_risk, 4),
            "blast_radius": 0.0,
            "tier": tier,
            "source": "risk_head",
            "heuristic": None,
            "neighbors": [],
        }

    dists = torch.norm(ref_embs - emb.unsqueeze(0), dim=1)
    topk = dists.topk(k, largest=False)
    neighbors = [ref_entries[i] for i in topk.indices.tolist()]
    top_dists = topk.values.tolist()
    eps = 1e-6
    weights = [1.0 / max(d, eps) for d in top_dists]
    w_sum = sum(weights)
    avg_risk = sum(w * n["risk"] for w, n in zip(weights, neighbors)) / w_sum
    tier = _risk_to_tier(avg_risk)
    return {
        "command": cmd,
        "risk": round(avg_risk, 4),
        "blast_radius": 0.0,
        "tier": tier,
        "source": "knn",
        "heuristic": None,
        "neighbors": [{"command": n["text"], "risk": n["risk"], "tier": n["tier"], "dist": round(d, 4)}
                      for n, d in zip(neighbors, top_dists)],
    }


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

    # Risk head path: direct MLP prediction (no corpus lookup needed)
    risk_head: RiskHead | None = getattr(model, "_risk_head", None)
    if risk_head is not None:
        with torch.no_grad():
            pred_risk = risk_head(emb.unsqueeze(0).to(dev)).item()
        tier = _risk_to_tier(pred_risk)
        return {
            "command": cmd,
            "risk": round(pred_risk, 4),
            "blast_radius": 0.0,
            "tier": tier,
            "source": "risk_head",
            "neighbors": [],
        }

    # Fallback: k-NN with inverse-distance weighting
    dists = torch.norm(ref_embs - emb.unsqueeze(0), dim=1)
    topk = dists.topk(k, largest=False)
    neighbors = [ref_entries[i] for i in topk.indices.tolist()]
    top_dists = topk.values.tolist()

    eps = 1e-6
    weights = [1.0 / max(d, eps) for d in top_dists]
    w_sum = sum(weights)
    avg_risk  = sum(w * n["risk"] for w, n in zip(weights, neighbors)) / w_sum
    avg_blast = sum(w * n.get("heuristic_blast", 0.0) for w, n in zip(weights, neighbors)) / w_sum
    tier = _risk_to_tier(avg_risk)
    return {
        "command": cmd,
        "risk": round(avg_risk, 4),
        "blast_radius": round(avg_blast, 4),
        "tier": tier,
        "source": "knn",
        "neighbors": [{"command": n["text"], "risk": n["risk"], "tier": n["tier"], "dist": round(d, 4)}
                      for n, d in zip(neighbors, topk.values.tolist())],
    }


_RESTORE_CMDS = re.compile(
    r"^git\s+(checkout|restore)\b"
    r"|^svn\s+revert\b"
    r"|^hg\s+revert\b",
    re.IGNORECASE,
)

def _apply_compensating_cap(
    subcommands: list[str], sub_results: list[dict], result: dict
) -> dict:
    """
    Detect delete-then-restore patterns (e.g., rm -rf X && git checkout X)
    and cap at T2. The intent is an atomic reset, not destruction.
    """
    for i in range(len(subcommands) - 1):
        sub_a, sub_b = subcommands[i], subcommands[i + 1]
        tool_a, _ = _parse_cli_verb(sub_a)
        # Check: destructive command followed by restore of same path
        if tool_a in ("rm", "del") and _RESTORE_CMDS.match(sub_b.strip()):
            result = dict(result)
            result["risk"] = min(result["risk"], 0.50)
            result["tier"] = min(result["tier"], 2)
            result["heuristic"] = "compensating_restore"
            return result
    return result


def compare_scorers(
    cmd: str,
    model: CommandEncoder,
    dev: torch.device,
    ref_embs: torch.Tensor,
    ref_entries: list[dict],
    k: int = 5,
) -> dict:
    """Run every scorer independently on a command and return all results.

    Unlike predict_risk (which short-circuits on the first matching heuristic),
    this runs every path so the caller can see where scorers agree/disagree.
    """
    results: dict[str, dict] = {}

    # 1. Dry-run detection
    dr = is_dry_run(cmd)
    results["dry_run"] = {
        "triggered": dr,
        "risk": 0.12 if dr else None,
        "tier": -1 if dr else None,
    }

    # 2. Exfil regex
    _cmd_stripped = cmd.lower().lstrip()
    _is_commit = (
        _cmd_stripped.startswith("git commit")
        or (_cmd_stripped.startswith("git add") and "commit" in _cmd_stripped)
    )
    exfil_match = bool(_EXFIL_EXEC_RE.search(cmd)) and not _is_commit
    loopback = bool(_LOOPBACK_RE.search(cmd))
    results["exfil_exec"] = {
        "triggered": exfil_match and not loopback,
        "pattern_match": exfil_match,
        "loopback_exempt": loopback,
        "risk": 0.95 if (exfil_match and not loopback) else None,
        "tier": -2 if (exfil_match and not loopback) else None,
    }

    # 3. Verb table
    verb_hit = _verb_table_lookup(cmd)
    results["verb_table"] = {
        "triggered": verb_hit is not None,
        "risk": verb_hit["risk"] if verb_hit else None,
        "tier": verb_hit["tier"] if verb_hit else None,
        "is_cap": verb_hit["is_cap"] if verb_hit else None,
        "tool_verb": None,
    }
    if verb_hit:
        tool, remainder = _parse_cli_verb(cmd)
        results["verb_table"]["tool_verb"] = f"{tool} {remainder[:40]}"

    # 4. Opaque exec
    opaque = bool(_OPAQUE_EXEC_RE.search(cmd))
    results["opaque_exec"] = {
        "triggered": opaque,
        "risk": 0.45 if opaque else None,
        "tier": 2 if opaque else None,
        "note": "floor — ML may score higher" if opaque else None,
    }

    # 5. Risk head MLP
    emb = embed_command(cmd, model, dev)
    risk_head_model: RiskHead | None = getattr(model, "_risk_head", None)
    if risk_head_model is not None:
        with torch.no_grad():
            pred_risk = risk_head_model(emb.unsqueeze(0).to(dev)).item()
        results["risk_head"] = {
            "triggered": True,
            "risk": round(pred_risk, 4),
            "tier": _risk_to_tier(pred_risk),
        }
    else:
        results["risk_head"] = {"triggered": False, "risk": None, "tier": None}

    # 6. k-NN
    dists = torch.norm(ref_embs - emb.unsqueeze(0), dim=1)
    topk = dists.topk(k, largest=False)
    neighbors = []
    for i, d in zip(topk.indices.tolist(), topk.values.tolist()):
        e = ref_entries[i]
        neighbors.append({
            "command": e.get("text", ""),
            "risk": round(e["risk"], 3),
            "tier": e["tier"],
            "dist": round(d, 4),
        })
    eps = 1e-6
    weights = [1.0 / max(d, eps) for d in topk.values.tolist()]
    w_sum = sum(weights)
    knn_risk = sum(w * ref_entries[i]["risk"]
                   for w, i in zip(weights, topk.indices.tolist())) / w_sum
    results["knn"] = {
        "triggered": True,
        "risk": round(knn_risk, 4),
        "tier": _risk_to_tier(knn_risk),
        "neighbors": neighbors,
    }

    # 7. Final prediction (the actual output of predict_risk)
    final = predict_risk(cmd, model, dev, ref_embs, ref_entries, k)
    winner = final.get("heuristic") or final.get("source", "unknown")

    return {
        "command": cmd,
        "final": {
            "risk": round(final["risk"], 4),
            "tier": final["tier"],
            "source": winner,
            "decision": "interrupt" if final["risk"] >= 0.55 else "allow",
        },
        "scorers": results,
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
    Risk prediction with heuristic pre-processing (priority order):
      1. Dry-run detection: --dry-run / terraform plan / etc. → cap at T-1 (0.12).
      2. Exfil/RCE: curl|bash, eval "$(curl ..)", etc. → floor at T-2 (0.95).
      3. Verb table: deterministic (tool, verb) → tier/risk for known CLI patterns.
      4. Opaque execution: python3 -c / eval / pipe-to-sh → T2 floor.
      5. Compound decomposition: split on &&, ||, ;, |; return max-risk subcommand.
    """
    # Check exfil patterns against the full unsplit command first — pipe-based
    # patterns like `curl ... | bash` are destroyed by compound splitting.
    # Loopback URLs (localhost, 127.0.0.1, etc.) are exempted — local dev traffic.
    # Skip exfil check on git commit commands — commit messages are data, not
    # execution, and heredoc messages often contain words like "curl", "exfil",
    # "bash" that trigger false positives.
    _cmd_stripped = cmd.lower().lstrip()
    _is_commit = (
        _cmd_stripped.startswith("git commit")
        or (_cmd_stripped.startswith("git add") and "commit" in _cmd_stripped)
    )
    if not _is_commit and _EXFIL_EXEC_RE.search(cmd) and not _LOOPBACK_RE.search(cmd):
        return {
            "command": cmd, "risk": 0.95, "tier": -2,
            "heuristic": "exfil_exec", "neighbors": [], "blast_radius": 0.0,
        }

    subcommands = _split_compound(cmd)

    # Score each subcommand independently (works for both single and compound)
    sub_results: list[dict] = []
    for sub in subcommands:
        # --- Dry-run cap (highest priority — overrides verb table floors) ---
        if is_dry_run(sub):
            sub_results.append({
                "command": sub, "risk": 0.12, "tier": -1,
                "heuristic": "dry_run_cap", "neighbors": [], "blast_radius": 0.0,
            })
            continue

        # --- Exfil / remote code execution floor (T-2) ---
        if _EXFIL_EXEC_RE.search(sub) and not _LOOPBACK_RE.search(sub):
            ml = _predict_single(sub, model, dev, ref_embs, ref_entries, k)
            ml["risk"]      = max(ml["risk"], 0.95)
            ml["tier"]      = -2
            ml["heuristic"] = "exfil_exec"
            sub_results.append(ml)
            continue

        # --- Verb table pre-check ---
        verb_hit = _verb_table_lookup(sub)
        if verb_hit is not None:
            if verb_hit["is_cap"]:
                # Known-safe: ML may still override upward if compound has worse seg
                sub_results.append({
                    "command": sub, "risk": verb_hit["risk"],
                    "blast_radius": 0.0, "tier": verb_hit["tier"],
                    "heuristic": verb_hit["heuristic"], "neighbors": [],
                })
                continue
            else:
                # Known-risky: floor — don't let ML underestimate
                ml = _predict_single(sub, model, dev, ref_embs, ref_entries, k)
                if ml["risk"] < verb_hit["risk"]:
                    ml["risk"]     = verb_hit["risk"]
                    ml["tier"]     = verb_hit["tier"]
                    ml["heuristic"] = verb_hit["heuristic"]
                sub_results.append(ml)
                continue

        # --- Opaque execution floor (T2) ---
        if _OPAQUE_EXEC_RE.search(sub):
            ml = _predict_single(sub, model, dev, ref_embs, ref_entries, k)
            if ml["tier"] < 2:
                ml["risk"]     = max(ml["risk"], 0.45)
                ml["tier"]     = max(ml["tier"], 2)
                ml["heuristic"] = "opaque_exec"
            sub_results.append(ml)
            continue

        sub_results.append(_predict_single(sub, model, dev, ref_embs, ref_entries, k))

    best = max(sub_results, key=lambda r: r["risk"])
    result = dict(best)
    result["command"] = cmd

    # Compensating-action detection: if a destructive command is followed by a
    # restoring command on the same path, cap at T2 (the intent is atomic reset).
    if len(subcommands) >= 2 and result["tier"] >= 3:
        result = _apply_compensating_cap(subcommands, sub_results, result)

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
    tr.add_argument("--hybrid",     action="store_true",
                    help="Train HybridEncoder (char + word branches)")
    tr.add_argument("--corpus",     default="data/synthetic/scores-cache.jsonl",
                    help="Scored corpus for building word vocab (--hybrid only)")

    ev = sub.add_parser("eval")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--pairs",      default=DEFAULT_PAIRS)
    ev.add_argument("--max-pairs",  type=int, default=100_000)

    em = sub.add_parser("embed")
    em.add_argument("--checkpoint", required=True)
    em.add_argument("--cmd", dest="embed_cmd", required=True)
    em.add_argument("--scores-cache", default="data/synthetic/scores-cache.jsonl")

    rh = sub.add_parser("train-risk-head")
    rh.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    rh.add_argument("--corpus",     default="data/synthetic/scores-cache.jsonl")
    rh.add_argument("--epochs",     type=int,   default=50)
    rh.add_argument("--batch-size", type=int,   default=256)
    rh.add_argument("--lr",         type=float, default=3e-3)
    rh.add_argument("--device",     default=None)

    args = p.parse_args()

    if args.cmd == "train":
        train(
            pairs_path=args.pairs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_pairs=args.max_pairs,
            device=args.device,
            hybrid=args.hybrid,
            corpus_path=args.corpus,
        )

    elif args.cmd == "eval":
        eval_model(args.checkpoint, args.pairs, args.max_pairs)

    elif args.cmd == "embed":
        model, dev = load_model(args.checkpoint)
        ref_embs, ref_entries = build_reference_index(args.scores_cache, model, dev)
        result = predict_risk(args.embed_cmd, model, dev, ref_embs, ref_entries)
        print(json.dumps(result, indent=2))

    elif args.cmd == "train-risk-head":
        train_risk_head(
            checkpoint=args.checkpoint,
            corpus_path=args.corpus,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()
