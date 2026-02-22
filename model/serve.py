"""
AlignLayer scoring server.

Loads the model + k-NN index once at startup, then serves scoring requests.

Usage:
    pip install fastapi uvicorn
    uvicorn model.serve:app --host 0.0.0.0 --port 8000

    # or directly:
    python model/serve.py

Environment:
    ALIGNLAYER_CHECKPOINT   path to .pt file (default: model/checkpoints/best.pt)
    ALIGNLAYER_CORPUS       path to scores-cache.jsonl (default: data/synthetic/scores-cache.jsonl)
    ALIGNLAYER_K            k-NN neighbors (default: 5)
    ALIGNLAYER_THRESHOLD    interrupt threshold 0-1 (default: 0.55)
    PORT                    listen port (default: 8000)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Allow `python model/serve.py` from repo root
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from pydantic import BaseModel

from siamese import build_reference_index, load_model, predict_risk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get("ALIGNLAYER_CHECKPOINT", "model/checkpoints/best.pt")
CORPUS     = os.environ.get("ALIGNLAYER_CORPUS",     "data/synthetic/scores-cache.jsonl")
K          = int(os.environ.get("ALIGNLAYER_K",         "5"))
THRESHOLD  = float(os.environ.get("ALIGNLAYER_THRESHOLD", "0.55"))

# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------

app = FastAPI(title="AlignLayer scorer", version="0.1.0")

_state: dict = {}


@app.on_event("startup")
def _load() -> None:
    t0 = time.time()
    print(f"Loading model: {CHECKPOINT}", flush=True)
    model, dev = load_model(CHECKPOINT)

    print(f"Building k-NN index from {CORPUS}...", flush=True)
    ref_embs, ref_entries = build_reference_index(CORPUS, model, dev)

    _state["model"]       = model
    _state["dev"]         = dev
    _state["ref_embs"]    = ref_embs
    _state["ref_entries"] = ref_entries
    print(f"Ready in {time.time()-t0:.1f}s — {len(ref_entries):,} corpus entries, k={K}", flush=True)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    command: str


class ScoreResponse(BaseModel):
    command:  str
    risk:     float
    tier:     int
    decision: str   # "allow" | "interrupt"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus_size": len(_state.get("ref_entries", []))}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    result = predict_risk(
        req.command,
        _state["model"],
        _state["dev"],
        _state["ref_embs"],
        _state["ref_entries"],
        k=K,
    )
    decision = "interrupt" if result["risk"] >= THRESHOLD else "allow"
    return ScoreResponse(
        command=req.command,
        risk=round(result["risk"], 4),
        tier=result["tier"],
        decision=decision,
    )


# ---------------------------------------------------------------------------
# Direct invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
