"""FastAPI application: serves the static frontend and a small JSON API.
The embedding model and FAISS index are loaded once at startup."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import audit, config, llm, pipeline, retrieval
from .schemas import AskRequest, AskResponse, Settings

WEB_DIR = config.ROOT_DIR / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the model and load the prebuilt index before serving traffic.
    retrieval.get_model()
    retrieval.load_index()
    yield


app = FastAPI(title="GroundCheck", version="1.0.0", lifespan=lifespan)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    # Rate limiting is keyed by client IP. Behind a proxy (Hugging Face, Render),
    # the real client is in X-Forwarded-For; fall back to the socket address.
    fwd = request.headers.get("x-forwarded-for", "")
    client_id = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "global")
    return pipeline.run(body.query, body.settings, client_id=client_id)


@app.get("/api/settings")
def settings_defaults() -> JSONResponse:
    """Default tuning values plus their allowed ranges, so the frontend can
    build the tuning panel and offer a one-click reset to defaults."""
    return JSONResponse({
        "defaults": Settings().model_dump(),
        "llm_available": llm.LLM_AVAILABLE,
        "bounds": {
            "retrieval_min_score": {"min": 0.0, "max": 1.0, "step": 0.01},
            "grounding_min": {"min": 0.0, "max": 1.0, "step": 0.01},
            "top_k": {"min": 1, "max": 12, "step": 1},
        },
    })


@app.get("/api/examples")
def examples() -> JSONResponse:
    with open(config.EXAMPLES_PATH, "r", encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


@app.get("/api/eval-summary")
def eval_summary() -> JSONResponse:
    path = config.EVAL_SUMMARY_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return JSONResponse(json.load(fh))
    return JSONResponse({
        "total": 0,
        "passed": 0,
        "note": "Evaluation has not been run yet. Run scripts/run_eval.py.",
    })


@app.get("/api/audit")
def list_audit() -> JSONResponse:
    """Recent audit records (lightweight), most recent first. Persisted to disk,
    so this survives a restart."""
    return JSONResponse({
        "count": audit.store.count(),
        "persisted": audit.store.persist,
        "recent": audit.store.recent(20),
    })


@app.get("/api/audit/{audit_id}")
def get_audit(audit_id: str) -> JSONResponse:
    record = audit.store.get(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return JSONResponse(record)


@app.get("/api/corpus")
def corpus_map() -> JSONResponse:
    """A 2D PCA projection of every document embedding, plus aggregate counts,
    for the corpus map. Computed once and cached."""
    return JSONResponse({
        "stats": retrieval.corpus_stats(),
        "points": retrieval.corpus_projection(),
    })


@app.get("/api/health")
def health() -> JSONResponse:
    corpus = retrieval.load_corpus()
    return JSONResponse({
        "status": "ok",
        "llm": llm.LLM_AVAILABLE,
        "corpus": len(corpus),
    })


# Static assets (logo, fonts, css, js). Mounted last so API routes win.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
