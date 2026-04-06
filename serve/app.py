"""
serve/app.py — ChefMate Substitution API
=========================================
GET  /health
POST /substitute   { "recipe_ingredients": ["flour","egg"], "missing": "butter" }
                → { "substitutions": [{"ingredient": "margarine", "score": 0.92}, ...] }
"""
import json, os, time
from pathlib import Path
from typing import List

import torch
import mlflow.pytorch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse

app = FastAPI(title="ChefMate Substitution API", version="1.0")

# ── Prometheus metrics ──────────────────────────────────────
REQUEST_COUNT   = Counter("substitute_requests_total", "Total requests")
REQUEST_LATENCY = Histogram("substitute_latency_seconds", "Request latency")
HIT_COUNTER     = Counter("substitute_hits_total", "Times top-1 was confirmed correct")

# ── Load model + vocab on startup ──────────────────────────
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI    = os.getenv("MODEL_URI", "models:/chefmate-substitution/Production")
VOCAB_PATH   = os.getenv("VOCAB_PATH", "data/processed/vocab.json")
TOP_K        = int(os.getenv("TOP_K", "10"))

model, vocab, id2ing = None, None, None

@app.on_event("startup")
def load_model():
    global model, vocab, id2ing
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.pytorch.load_model(MODEL_URI)
    model.eval()
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    id2ing = {v: k for k, v in vocab.items()}
    print(f"Model loaded from {MODEL_URI}. Vocab size: {len(vocab)}")


# ── Schemas ─────────────────────────────────────────────────
class SubRequest(BaseModel):
    recipe_ingredients: List[str]
    missing: str

class SubResult(BaseModel):
    ingredient: str
    score: float

class SubResponse(BaseModel):
    substitutions: List[SubResult]
    latency_ms: float


# ── Endpoints ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/substitute", response_model=SubResponse)
def substitute(req: SubRequest):
    REQUEST_COUNT.inc()
    if model is None:
        raise HTTPException(503, "Model not loaded")

    missing_lower = req.missing.strip().lower()
    if missing_lower not in vocab:
        raise HTTPException(422, f"Unknown ingredient: {req.missing}")

    vocab_size = len(vocab)
    ctx = torch.zeros(1, vocab_size)
    for ing in req.recipe_ingredients:
        if ing.strip().lower() in vocab:
            ctx[0, vocab[ing.strip().lower()]] = 1.0

    src = torch.tensor([vocab[missing_lower]], dtype=torch.long)

    t0 = time.time()
    with torch.no_grad():
        logits = model(ctx, src)[0]           # (vocab_size,)
        scores, indices = logits.topk(TOP_K)
        scores = torch.softmax(scores, dim=0)
    latency_ms = (time.time() - t0) * 1000

    REQUEST_LATENCY.observe(latency_ms / 1000)

    results = [
        SubResult(ingredient=id2ing[idx.item()], score=round(sc.item(), 4))
        for idx, sc in zip(indices, scores)
        if idx.item() in id2ing and id2ing[idx.item()] != missing_lower
    ]
    return SubResponse(substitutions=results, latency_ms=round(latency_ms, 2))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
