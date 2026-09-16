"""FastAPI application: serves the static frontend and a small JSON API.
The embedding model and FAISS index are loaded once at startup."""

from __future__ import annotations

import ipaddress
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import audit, config, llm, local_ai, pipeline, retrieval
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
        "llm_available": llm.llm_available(),
        "bounds": {
            "retrieval_min_score": {"min": 0.0, "max": 1.0, "step": 0.01},
            "grounding_min": {"min": 0.0, "max": 1.0, "step": 0.01},
            "top_k": {"min": 1, "max": 12, "step": 1},
            "temperature": {"min": 0.0, "max": 1.5, "step": 0.1},
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
    provider = llm.active_provider()
    return JSONResponse({
        "status": "ok",
        "llm": provider is not None,
        "provider": provider,
        "corpus": len(corpus),
    })


# --- Local AI ---------------------------------------------------------------

class LocalModelRequest(BaseModel):
    model: str | None = None


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")


def _require_local_ai_admin(request: Request) -> None:
    """Downloading and switching models changes the server for everyone, so by
    default only requests from this machine may do it."""
    policy = config.LOCAL_AI_ADMIN
    if policy == "all":
        return
    if policy == "local":
        host = _client_ip(request)
        try:
            if host == "testclient" or ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
    raise HTTPException(status_code=403, detail="Managing local models is only allowed from this machine.")


@app.get("/api/local-ai")
def local_ai_status(request: Request) -> JSONResponse:
    body = local_ai.status()
    try:
        _require_local_ai_admin(request)
        body["can_manage"] = True
    except HTTPException:
        body["can_manage"] = False
    return JSONResponse(body)


@app.post("/api/local-ai/pull")
def local_ai_pull(body: LocalModelRequest, request: Request) -> StreamingResponse:
    """Download a catalogue model, streaming progress as newline-delimited JSON."""
    _require_local_ai_admin(request)
    if not body.model or body.model not in {m.name for m in local_ai.CATALOGUE}:
        raise HTTPException(status_code=400, detail="Choose a model from the catalogue.")

    def events():
        try:
            for event in local_ai.pull(body.model):
                yield json.dumps(event) + "\n"
        except local_ai.OllamaError as exc:
            yield json.dumps({"status": "error", "error": str(exc)}) + "\n"
        finally:
            local_ai.forget_availability()

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/local-ai/select")
async def local_ai_select(body: LocalModelRequest, request: Request) -> JSONResponse:
    """Make a downloaded model the one that drafts answers, and load it into
    memory. A null model switches back to the cloud model or extractive mode."""
    _require_local_ai_admin(request)
    previous = local_ai.selected_model()
    if body.model is None:
        local_ai.select_model(None)
        local_ai.forget_availability()
        if previous:
            await run_in_threadpool(local_ai.unload, previous)
        return JSONResponse({"selected": None, "provider": llm.active_provider()})

    if body.model not in {m.name for m in local_ai.CATALOGUE}:
        raise HTTPException(status_code=400, detail="Choose a model from the catalogue.")
    if not await run_in_threadpool(local_ai.is_installed, body.model):
        raise HTTPException(status_code=409, detail=f"Download {body.model} before selecting it.")
    try:
        await run_in_threadpool(local_ai.load, body.model)
    except local_ai.OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    local_ai.select_model(body.model)
    local_ai.forget_availability()
    if previous and previous != body.model:
        await run_in_threadpool(local_ai.unload, previous)
    return JSONResponse({"selected": body.model, "provider": llm.active_provider()})


# Static assets (logo, fonts, css, js). Mounted last so API routes win.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
