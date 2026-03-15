"""
AlignLayer scoring server.

Loads the model + k-NN index once at startup, then serves scoring requests.
Also accepts Claude Code hook payloads for observational data collection.

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
    ALIGNLAYER_TRACES_DIR   trace output directory (default: data/traces)
    PORT                    listen port (default: 8000)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python model/serve.py` from repo root
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from siamese import build_reference_index, embed_command, load_model, predict_risk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT = os.environ.get("ALIGNLAYER_CHECKPOINT", "model/checkpoints/best.pt")
CORPUS     = os.environ.get("ALIGNLAYER_CORPUS",     "data/synthetic/scores-cache.jsonl")
K          = int(os.environ.get("ALIGNLAYER_K",         "5"))

# Mutable runtime config
_config: dict[str, Any] = {
    "threshold": float(os.environ.get("ALIGNLAYER_THRESHOLD", "0.55")),
}
FEEDBACK_FILE = Path("data/traces/feedback.jsonl")
TRACES_DIR = Path(os.environ.get("ALIGNLAYER_TRACES_DIR", "data/traces"))
# Hook traces (written by the TS hook, separate from server traces)
HOOK_TRACES_DIR = Path.home() / ".alignlayer" / "traces"

logger = logging.getLogger("alignlayer.serve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------

app = FastAPI(title="AlignLayer scorer", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {}


@app.on_event("startup")
def _load() -> None:
    t0 = time.time()
    logger.info("Loading model: %s", CHECKPOINT)
    model, dev = load_model(CHECKPOINT)

    logger.info("Building k-NN index from %s...", CORPUS)
    ref_embs, ref_entries = build_reference_index(CORPUS, model, dev)

    _state["model"]       = model
    _state["dev"]         = dev
    _state["ref_embs"]    = ref_embs
    _state["ref_entries"] = ref_entries

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Ready in %.1fs — %d corpus entries, k=%d, traces → %s",
                time.time() - t0, len(ref_entries), K, TRACES_DIR)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    if request.url.path != "/observe":  # observe is noisy, log separately
        logger.info("%s %s %.0fms %d", request.method, request.url.path, elapsed, response.status_code)
    return response


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


class BatchScoreRequest(BaseModel):
    commands: list[str]


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_one(command: str) -> ScoreResponse:
    result = predict_risk(
        command,
        _state["model"],
        _state["dev"],
        _state["ref_embs"],
        _state["ref_entries"],
        k=K,
    )
    decision = "interrupt" if result["risk"] >= _config["threshold"] else "allow"
    return ScoreResponse(
        command=command,
        risk=round(result["risk"], 4),
        tier=result["tier"],
        decision=decision,
    )


def _score_one_full(command: str) -> dict:
    """Score with full neighbor detail for trace enrichment."""
    return predict_risk(
        command,
        _state["model"],
        _state["dev"],
        _state["ref_embs"],
        _state["ref_entries"],
        k=K,
    )


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------

def _append_trace(entry: dict[str, Any]) -> None:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    trace_file = TRACES_DIR / f"claude-code-{date}.jsonl"
    with open(trace_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Endpoints — scoring
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus_size": len(_state.get("ref_entries", []))}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    return _score_one(req.command)


@app.post("/batch", response_model=BatchScoreResponse)
def batch(req: BatchScoreRequest) -> BatchScoreResponse:
    return BatchScoreResponse(results=[_score_one(cmd) for cmd in req.commands])


# ---------------------------------------------------------------------------
# Endpoints — explain
# ---------------------------------------------------------------------------

@app.post("/explain")
def explain(req: ScoreRequest) -> dict:
    """Detailed scoring breakdown: which heuristic, why, nearest neighbors."""
    result = predict_risk(
        req.command,
        _state["model"], _state["dev"],
        _state["ref_embs"], _state["ref_entries"],
        k=K,
    )
    tier = result["tier"]
    risk = result["risk"]
    source = result.get("heuristic") or result.get("source", "unknown")
    decision = "interrupt" if risk >= _config["threshold"] else "allow"

    # Always include nearest neighbors for context (predict_risk may skip them
    # when heuristics or RiskHead fire early)
    neighbors = result.get("neighbors", [])
    if not neighbors:
        emb = embed_command(req.command, _state["model"], _state["dev"])
        dists = torch.norm(_state["ref_embs"] - emb.unsqueeze(0), dim=1)
        topk = dists.topk(K, largest=False)
        neighbors = []
        for i, d in zip(topk.indices.tolist(), topk.values.tolist()):
            e = _state["ref_entries"][i]
            neighbors.append({"command": e.get("text", e.get("command", "")),
                              "risk": e["risk"], "tier": e["tier"], "dist": d})

    neighbors = [
        {"command": n["command"], "risk": round(n["risk"], 3),
         "tier": n["tier"], "dist": round(n.get("dist", 0), 4)}
        for n in neighbors[:5]
    ]

    # Build human-readable explanation
    explanations = []
    if source == "exfil_exec":
        explanations.append("Matched exfiltration/RCE pattern (pipe-to-shell, credential exfil, file upload)")
    elif source == "dry_run_cap":
        explanations.append("Detected dry-run/preview flag — risk capped at T-1")
    elif source == "verb_table":
        explanations.append(f"Matched verb table rule for this tool+subcommand combination")
    elif source == "opaque_exec":
        explanations.append("Opaque code execution (eval, python -c, pipe-to-sh) — floor at T2")
    elif source == "risk_head":
        explanations.append("Scored by RiskHead MLP on command embedding")
    elif source == "knn":
        explanations.append("Scored by k-NN lookup against reference corpus")

    if risk >= _config["threshold"]:
        explanations.append(f"Risk {risk:.3f} >= threshold {_config['threshold']:.2f} → would interrupt")
    else:
        explanations.append(f"Risk {risk:.3f} < threshold {_config['threshold']:.2f} → allowed")

    return {
        "command": req.command,
        "risk": round(risk, 4),
        "tier": tier,
        "decision": decision,
        "source": source,
        "explanation": explanations,
        "neighbors": neighbors,
        "threshold": _config["threshold"],
    }


# ---------------------------------------------------------------------------
# Endpoints — config (runtime-adjustable)
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    threshold: float | None = None


@app.get("/config")
def get_config() -> dict:
    return dict(_config)


@app.put("/config")
def update_config(update: ConfigUpdate) -> dict:
    if update.threshold is not None:
        if not (0.0 <= update.threshold <= 1.0):
            return {"error": "threshold must be between 0 and 1"}
        _config["threshold"] = update.threshold
        logger.info("Threshold updated to %.2f", update.threshold)
    return dict(_config)


# ---------------------------------------------------------------------------
# Endpoints — feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    command: str
    risk: float
    tier: int
    source: str | None = None
    decision: str          # "allow" | "interrupt"
    judgment: str          # "correct" | "incorrect"
    note: str | None = None


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    """Record human judgment on a scoring decision."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": req.command,
        "risk": req.risk,
        "tier": req.tier,
        "source": req.source,
        "decision": req.decision,
        "judgment": req.judgment,
        "note": req.note,
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("feedback: %s on %s (risk=%.2f tier=%d) %s",
                req.judgment, req.decision, req.risk, req.tier, req.command[:60])
    return {"status": "ok"}


@app.get("/feedback")
def get_feedback() -> dict:
    """Retrieve all feedback entries."""
    entries = []
    if FEEDBACK_FILE.exists():
        for line in FEEDBACK_FILE.read_text().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    correct = sum(1 for e in entries if e.get("judgment") == "correct")
    incorrect = sum(1 for e in entries if e.get("judgment") == "incorrect")

    # Group incorrect by decision type for analysis
    false_positives = [e for e in entries if e["judgment"] == "incorrect" and e["decision"] == "interrupt"]
    false_negatives = [e for e in entries if e["judgment"] == "incorrect" and e["decision"] == "allow"]

    return {
        "total": len(entries),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(correct / len(entries), 4) if entries else None,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Endpoints — Claude Code hook observer
# ---------------------------------------------------------------------------

EXEC_TOOLS = {"Bash", "bash", "shell", "run", "computer"}


def _extract_command(payload: dict[str, Any]) -> str | None:
    """Pull the shell command out of a hook payload, if present."""
    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return None
    return tool_input.get("command") or tool_input.get("cmd") or tool_input.get("input") or None


@app.post("/observe")
async def observe(request: Request) -> dict:
    """
    Receive Claude Code hook payloads (PreToolUse / PostToolUse).

    For exec-class tools (Bash, shell, etc.), scores the command and writes
    an enriched trace entry. For non-exec tools, logs the raw event.

    Always returns 200 — this endpoint is observational only.
    """
    try:
        payload = await request.json()
    except Exception:
        return {"status": "bad_request"}

    event = payload.get("hook_event_name", "unknown")
    tool = payload.get("tool_name", "unknown")
    session_id = payload.get("session_id", "")
    ts = datetime.now(timezone.utc).isoformat()

    trace: dict[str, Any] = {
        "ts": ts,
        "event": event,
        "session_id": session_id,
        "tool": tool,
        "tool_input": payload.get("tool_input"),
    }

    # Score exec-class tools
    cmd = _extract_command(payload) if tool in EXEC_TOOLS else None
    if cmd:
        try:
            result = _score_one_full(cmd)
            trace["ml_risk"] = result.get("risk")
            trace["ml_tier"] = result.get("tier")
            trace["ml_heuristic"] = result.get("heuristic")
            trace["ml_neighbors"] = [
                {"command": n["command"], "risk": n["risk"], "tier": n["tier"], "dist": n["dist"]}
                for n in result.get("neighbors", [])[:3]
            ]
            trace["ml_decision"] = "interrupt" if result.get("risk", 0) >= _config["threshold"] else "allow"

            tier = result.get("tier", 0)
            risk = result.get("risk", 0)
            marker = " ⚠" if risk >= _config["threshold"] else ""
            logger.info("observe %s T%+d risk=%.2f %s%s",
                        event, tier, risk, cmd[:80], marker)
        except Exception as exc:
            logger.warning("observe scoring error: %s", exc)
            trace["ml_error"] = str(exc)
    else:
        # Non-exec tool or no command — log tool name + args summary
        tool_input = payload.get("tool_input", {})
        summary = ""
        if isinstance(tool_input, dict):
            # Grab a useful identifier without logging full content
            summary = tool_input.get("file_path") or tool_input.get("pattern") or tool_input.get("query") or ""
        logger.info("observe %s %s %s", event, tool, str(summary)[:80])

    # Include tool result for PostToolUse (truncated)
    if event == "PostToolUse":
        response = payload.get("tool_response")
        if isinstance(response, str) and len(response) > 500:
            trace["tool_response_preview"] = response[:500]
        elif response is not None:
            trace["tool_response_preview"] = response

    _append_trace(trace)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Endpoints — trace stats
# ---------------------------------------------------------------------------

@app.get("/traces/stats")
def trace_stats() -> dict:
    """Summary stats across all trace files."""
    total = 0
    by_tool: dict[str, int] = {}
    by_tier: dict[int, int] = {}
    interrupts = 0

    for f in sorted(TRACES_DIR.glob("claude-code-*.jsonl")):
        for line in open(f):
            try:
                entry = json.loads(line)
                total += 1
                tool = entry.get("tool", "unknown")
                by_tool[tool] = by_tool.get(tool, 0) + 1
                tier = entry.get("ml_tier")
                if tier is not None:
                    by_tier[tier] = by_tier.get(tier, 0) + 1
                if entry.get("ml_decision") == "interrupt":
                    interrupts += 1
            except Exception:
                pass

    return {
        "total_events": total,
        "by_tool": dict(sorted(by_tool.items(), key=lambda x: -x[1])),
        "by_tier": {f"T{k:+d}": v for k, v in sorted(by_tier.items())},
        "would_interrupt": interrupts,
        "trace_files": [f.name for f in sorted(TRACES_DIR.glob("claude-code-*.jsonl"))],
    }


# ---------------------------------------------------------------------------
# Endpoints — trace data APIs (renderer-agnostic)
# ---------------------------------------------------------------------------

@app.get("/traces/tail")
def trace_tail(n: int = 50) -> dict:
    """Last N trace entries, newest first."""
    entries: list[dict] = []
    for f in sorted(TRACES_DIR.glob("claude-code-*.jsonl"), reverse=True):
        for line in reversed(open(f).readlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
            if len(entries) >= n:
                break
        if len(entries) >= n:
            break
    return {"entries": entries, "count": len(entries)}


@app.get("/traces/distributions")
def trace_distributions() -> dict:
    """Risk, tier, and tool distributions across all traces."""
    risks: list[float] = []
    tiers: list[int] = []
    tools: dict[str, int] = {}
    heuristics: dict[str, int] = {}
    decisions: dict[str, int] = {"allow": 0, "interrupt": 0}
    hourly: dict[str, int] = {}

    for f in sorted(TRACES_DIR.glob("claude-code-*.jsonl")):
        for line in open(f):
            try:
                e = json.loads(line)
                tool = e.get("tool", "unknown")
                tools[tool] = tools.get(tool, 0) + 1

                risk = e.get("ml_risk")
                if risk is not None:
                    risks.append(risk)
                tier = e.get("ml_tier")
                if tier is not None:
                    tiers.append(tier)
                h = e.get("ml_heuristic")
                if h:
                    heuristics[h] = heuristics.get(h, 0) + 1
                d = e.get("ml_decision")
                if d in decisions:
                    decisions[d] += 1

                ts = e.get("ts", "")
                if len(ts) >= 13:
                    hour_key = ts[:13]  # YYYY-MM-DDTHH
                    hourly[hour_key] = hourly.get(hour_key, 0) + 1
            except Exception:
                pass

    # Compute risk histogram (10 buckets)
    risk_hist = [0] * 10
    for r in risks:
        bucket = min(int(r * 10), 9)
        risk_hist[bucket] += 1

    # Tier counts
    tier_counts = {}
    for t in sorted(set(tiers)):
        tier_counts[f"T{t:+d}"] = tiers.count(t)

    return {
        "total_scored": len(risks),
        "risk_histogram": {f"{i/10:.1f}-{(i+1)/10:.1f}": risk_hist[i] for i in range(10)},
        "risk_mean": round(sum(risks) / len(risks), 4) if risks else 0,
        "tier_counts": tier_counts,
        "tool_counts": dict(sorted(tools.items(), key=lambda x: -x[1])),
        "heuristic_counts": dict(sorted(heuristics.items(), key=lambda x: -x[1])),
        "decisions": decisions,
        "hourly_volume": dict(sorted(hourly.items())),
    }


# ---------------------------------------------------------------------------
# Endpoints — threshold simulator
# ---------------------------------------------------------------------------

@app.get("/traces/threshold-sim")
def threshold_sim() -> dict:
    """Replay all scored commands at multiple thresholds."""
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]
    risks: list[float] = []

    for f in sorted(TRACES_DIR.glob("claude-code-*.jsonl")):
        for line in open(f):
            try:
                e = json.loads(line)
                risk = e.get("ml_risk")
                if risk is not None:
                    risks.append(risk)
            except Exception:
                pass

    if not risks:
        return {"total": 0, "thresholds": {}}

    result = {}
    for t in thresholds:
        interrupts = sum(1 for r in risks if r >= t)
        allows = len(risks) - interrupts
        result[str(t)] = {
            "allow": allows,
            "interrupt": interrupts,
            "interrupt_rate": round(interrupts / len(risks), 4),
        }

    return {"total": len(risks), "thresholds": result}


# ---------------------------------------------------------------------------
# Endpoints — human outcomes (from hook traces)
# ---------------------------------------------------------------------------

def _load_hook_traces() -> list[dict]:
    """Load before_tool_call events from the hook trace directory."""
    entries = []
    if not HOOK_TRACES_DIR.exists():
        return entries
    for f in sorted(HOOK_TRACES_DIR.glob("alignlayer-*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


@app.get("/traces/outcomes")
def human_outcomes() -> dict:
    """Aggregate human approve/deny decisions from hook traces."""
    entries = _load_hook_traces()

    total_asks = 0
    approved = 0
    # Build set of asked tool_use_ids from before_tool_call with decision=interrupt
    asked_ids: set[str] = set()
    approved_cmds: list[dict] = []
    denied_cmds: list[dict] = []

    # First pass: identify asked tool calls
    for e in entries:
        if e.get("event") == "before_tool_call" and e.get("decision") == "interrupt":
            asked_ids.add(e.get("turn_id", ""))
            total_asks += 1

    # Second pass: check which ones got PostToolUse (= approved)
    approved_ids: set[str] = set()
    for e in entries:
        if e.get("event") == "after_tool_call" and e.get("turn_id", "") in asked_ids:
            approved_ids.add(e["turn_id"])
            if e.get("human_outcome") == "approved":
                approved += 1

    # Collect details for approved/denied
    for e in entries:
        if e.get("event") != "before_tool_call" or e.get("turn_id", "") not in asked_ids:
            continue
        cmd = ""
        args = e.get("args", {})
        if isinstance(args, dict):
            cmd = args.get("command", args.get("cmd", ""))
        info = {
            "cmd": str(cmd)[:200],
            "risk": e.get("risk_score"),
            "tool": e.get("tool", ""),
            "timestamp": e.get("timestamp", ""),
        }
        if e["turn_id"] in approved_ids:
            approved_cmds.append(info)
        else:
            denied_cmds.append(info)

    denied = total_asks - len(approved_ids)

    return {
        "total_hook_events": len(entries),
        "total_asks": total_asks,
        "approved": len(approved_ids),
        "denied": denied,
        "approval_rate": round(len(approved_ids) / total_asks, 4) if total_asks > 0 else None,
        "recent_approved": approved_cmds[-10:],
        "recent_denied": denied_cmds[-10:],
    }


# ---------------------------------------------------------------------------
# Endpoints — dashboard
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (Path(__file__).parent / "static" / "dashboard.html").read_text()


# ---------------------------------------------------------------------------
# Direct invocation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
