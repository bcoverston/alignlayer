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
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(flushdb|flushall)\b", re.I),         2,  0.48, False),
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(del|unlink|expire|persist|rename|save|bgsave|bgrewriteaof|config\s+set|shutdown)\b", re.I), 2, 0.40, False),
    (re.compile(r"^redis-cli$"),  re.compile(r"\b(get|keys|scan|info|dbsize|ttl|type|llen|smembers|hgetall|zrange|config\s+get|ping)\b", re.I), 0, 0.05, True),

    # aws — match service+operation in the remainder
    (re.compile(r"^aws$"),        re.compile(r"terminate-instances|stop-instances|delete-|deregister-|revoke-|disable-"),  3, 0.65, False),
    (re.compile(r"^aws$"),        re.compile(r"\bdescribe-|\blist-|\bget-|\bshow-"),  0,  0.05, True),

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

    # aws autoscaling / iam destructive ops
    (re.compile(r"^aws$"),        re.compile(r"\biam\s+delete-\b|\bautoscaling\s+suspend-\b|\bautoscaling\s+delete-\b"), 3, 0.65, False),

    # ssh with destructive remote commands
    (re.compile(r"^ssh$"),        re.compile(r"(sudo\s+reboot|sudo\s+shutdown|sudo\s+rm\s|rm\s+-rf|sudo\s+systemctl\s+stop|init\s+0)"), 4, 0.88, False),
    # ssh read-only: cap at T0
    (re.compile(r"^ssh$"),        re.compile(r"(cat\s|tail\s|head\s|less\s|grep\s|ps\s|top\s|uptime|df\s|free\s|uname|hostname|whoami|id\b|env\b|printenv)"), 0, 0.08, True),

    # docker / podman destructive
    (re.compile(r"^(docker|podman)$"), re.compile(r"^(system\s+prune|volume\s+prune|image\s+prune)\b"), 2, 0.48, False),
    (re.compile(r"^(docker|podman)$"), re.compile(r"^push\b"),                           3,  0.62, False),
    (re.compile(r"^(docker|podman)$"), re.compile(r"^(ps|images|inspect|logs|stats|top|port|diff|history|version)\b"), 0, 0.05, True),

    # terraform state show/list — read-only inspection
    (re.compile(r"^terraform$"),  re.compile(r"^state\s+show\b"),                        0,  0.08, True),

    # psql read-only: metacommands (\dt, \dl, \l, \d+) and SELECT
    (re.compile(r"^psql$"),       re.compile(r"-c\s+.*\\\\?d[tl+]|SELECT\s", re.I),      0,  0.08, True),

    # gh (GitHub CLI) — read-only commands
    (re.compile(r"^gh$"),         re.compile(r"^(run\s+view|pr\s+(view|list|status|checks)|issue\s+(view|list)|repo\s+view)\b"), 0, 0.05, True),

    # git read-only
    (re.compile(r"^git$"),        re.compile(r"^(log|diff|show|status|branch\s+-[avr]|tag\s+-l|remote\s+-v|stash\s+list)\b"), 0, 0.05, True),

    # find without -delete or destructive -exec is read-only
    (re.compile(r"^find$"),       re.compile(r"^(?!.*(-delete|xargs\s+rm|-exec\s+(rm|mv|chmod|chown)))"), 0, 0.08, True),

    # nginx -T (dump config), journalctl (read logs), ssh-keygen -l (fingerprint)
    (re.compile(r"^nginx$"),      re.compile(r"^-[tT]\b"),                               0,  0.05, True),
    (re.compile(r"^journalctl$"), re.compile(r"--vacuum"), 2, 0.45, False),
    (re.compile(r"^journalctl$"), re.compile(r""), 0, 0.08, True),
    (re.compile(r"^ssh-keygen$"), re.compile(r"^-l\b"),                                  0,  0.05, True),
    (re.compile(r"^ssh-keyscan$"), re.compile(r""),                                      0,  0.08, True),

    # pip/npm local install (not from URL) — T1 cap
    (re.compile(r"^pip3?$"),      re.compile(r"^install\s+(?!https?://)"),                 1,  0.30, True),
    (re.compile(r"^(npm|yarn|pnpm)$"), re.compile(r"^install\b"),                         1,  0.30, True),
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
    # git clone + build/install in one shot (untrusted repo execution)
    r"|git\s+clone\s+.*&&.*make\s+(install|all)\b"
    # credential exfil via pipe to nc/curl
    r"|cat\s+.*\.(pem|key|id_rsa|id_ed25519|credentials|env)\s*\|"
    r"|cat\s+~/\.ssh/\S+\s*\|"
    # env var credential exfil
    r"|AWS_ACCESS_KEY_ID=\S+\s+AWS_SECRET_ACCESS_KEY=\S+\s+aws\s+s3\s+(sync|cp)\s+/"
    # curl posting secrets/tokens to external URLs
    r"|curl\s+.*Authorization.*\$\w*(SECRET|TOKEN|KEY|PASS)",
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


def _parse_cli_verb(cmd: str) -> tuple[str, str]:
    """Return (tool, remainder) from a command string."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if not parts:
        return "", ""
    tool = parts[0].lower()
    remainder = " ".join(parts[1:]).lower()
    return tool, remainder


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
            "find", "gh", "psql", "terraform", "nginx", "ssh-keygen", "ssh-keyscan",
            "pip", "pip3", "npm", "yarn", "pnpm",
        } else first_pos
        if verb_re.search(target):
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
    top_dists = topk.values.tolist()

    # Inverse-distance weighting: closer neighbors get more influence.
    # Clamp minimum distance to avoid division by zero for exact matches.
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
    if _EXFIL_EXEC_RE.search(cmd) and not _LOOPBACK_RE.search(cmd):
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


if __name__ == "__main__":
    main()
