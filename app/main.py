"""FastAPI application: serves the static frontend and a small JSON API.
The embedding model and search index are loaded once at startup."""

from __future__ import annotations

import ipaddress
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import audit, auth, config, db, governance, llm, local_ai, pipeline, retrieval
from .schemas import AskRequest, AskResponse, Settings

WEB_DIR = config.ROOT_DIR / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the model and load the prebuilt index before serving traffic.
    retrieval.get_model()
    retrieval.load_index()
    yield


app = FastAPI(title="GroundCheck", version="1.0.0", lifespan=lifespan)


# --- Accounts ---------------------------------------------------------------
#
# With AUTH_REQUIRED=false (the default, for the demo) every endpoint is open,
# as before. With AUTH_REQUIRED=true, requests need a signed-in session and a
# role: clinicians ask questions, reviewers also read the audit trail, and
# admins also manage users and local AI models.

SESSION_COOKIE = "gc_session"
STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


def _session_token(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="strict", path="/",
        secure=config.SESSION_COOKIE_SECURE, max_age=int(config.SESSION_HOURS * 3600),
    )


@app.middleware("http")
async def reject_cross_site_writes(request: Request, call_next):
    """Refuse state-changing API requests sent from another site. The session
    cookie is SameSite=Strict as well; this also covers older browsers."""
    if request.method in STATE_CHANGING and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != f"{request.url.scheme}://{request.url.netloc}":
            forwarded_host = request.headers.get("x-forwarded-host")
            if not (forwarded_host and origin.split("://", 1)[-1].rstrip("/") == forwarded_host):
                return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
    return await call_next(request)


# Reachable without signing in, even when accounts are required.
PUBLIC_API = {"/api/health", "/api/auth/me", "/api/auth/login", "/api/auth/mfa",
              "/api/auth/logout", "/api/auth/first-admin"}


@app.middleware("http")
async def require_sign_in(request: Request, call_next):
    """With accounts required, every API endpoint needs a signed-in session
    unless it is listed as public. New endpoints are protected by default."""
    path = request.url.path
    if config.AUTH_REQUIRED and path.startswith("/api/") and path not in PUBLIC_API:
        token = _session_token(request)
        principal = await run_in_threadpool(auth.session_principal, token) if token and db.ready() else None
        if principal is None:
            return JSONResponse({"detail": "Sign in to continue."}, status_code=401)
    return await call_next(request)


def current_user(request: Request) -> auth.Principal | None:
    """The signed-in user, or None. Raises 401 when accounts are required and
    nobody is signed in."""
    principal = auth.session_principal(_session_token(request)) if db.ready() else None
    if principal is None and config.AUTH_REQUIRED:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return principal


def require_role(role: str):
    def dependency(user: auth.Principal | None = Depends(current_user)) -> auth.Principal | None:
        if config.AUTH_REQUIRED and (user is None or not user.can(role)):
            raise HTTPException(status_code=403, detail="Your role doesn't allow this.")
        return user
    return dependency


class SignInRequest(BaseModel):
    email: str
    password: str


class CodeRequest(BaseModel):
    code: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class PasswordRequest(BaseModel):
    password: str


class NewUserRequest(BaseModel):
    email: str
    password: str
    role: str = "clinician"
    name: str = ""


class UserUpdateRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    name: str | None = None
    password: str | None = None


def _auth_error(exc: auth.AuthError, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


def _require_database() -> None:
    if not db.ready():
        raise HTTPException(status_code=503, detail="Accounts need a database, and it isn't available.")


@app.get("/api/auth/me")
def auth_me(request: Request) -> JSONResponse:
    token = _session_token(request)
    pending = None
    principal = None
    if db.ready() and token:
        principal = auth.session_principal(token)
        if principal is None:
            pending = auth.session_principal(token, allow_mfa_pending=True)
    return JSONResponse({
        "auth_required": config.AUTH_REQUIRED,
        "user": principal.__dict__ if principal else None,
        "mfa_pending": pending is not None,
        "needs_first_admin": db.ready() and auth.count_users() == 0,
    })


@app.post("/api/auth/login")
def auth_login(body: SignInRequest, request: Request, response: Response) -> dict:
    _require_database()
    try:
        result = auth.sign_in(body.email, body.password, ip=_client_ip(request),
                              user_agent=request.headers.get("user-agent", ""))
    except auth.AuthError as exc:
        raise _auth_error(exc, 401) from exc
    _set_session_cookie(response, result.token)
    return {"mfa_required": result.mfa_required,
            "user": None if result.mfa_required else result.principal.__dict__}


@app.post("/api/auth/mfa")
def auth_mfa(body: CodeRequest, request: Request) -> dict:
    _require_database()
    token = _session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in again.")
    try:
        principal = auth.verify_mfa(token, body.code)
    except auth.AuthError as exc:
        raise _auth_error(exc, 401) from exc
    return {"user": principal.__dict__}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict:
    if db.ready():
        auth.sign_out(_session_token(request))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"signed_out": True}


@app.post("/api/auth/first-admin")
def auth_first_admin(body: NewUserRequest, request: Request, response: Response) -> dict:
    """Create the first admin account. Only works while there are no users, and
    only from this machine, so a new install can't be claimed remotely."""
    _require_database()
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="Create the first admin from the machine running GroundCheck.")
    if auth.count_users() > 0:
        raise HTTPException(status_code=409, detail="An admin already exists. Sign in instead.")
    try:
        auth.create_user(body.email, body.password, role="admin", name=body.name)
        result = auth.sign_in(body.email, body.password, ip=_client_ip(request),
                              user_agent=request.headers.get("user-agent", ""))
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc
    _set_session_cookie(response, result.token)
    return {"user": result.principal.__dict__}


def _signed_in(user: auth.Principal | None = Depends(current_user)) -> auth.Principal:
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


@app.post("/api/auth/password")
def auth_change_password(body: PasswordChangeRequest, request: Request, response: Response,
                         user: auth.Principal = Depends(_signed_in)) -> dict:
    try:
        auth.set_password(user.id, body.new_password, current_password=body.current_password)
        result = auth.sign_in(user.email, body.new_password, ip=_client_ip(request),
                              user_agent=request.headers.get("user-agent", ""))
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc
    _set_session_cookie(response, result.token)
    return {"changed": True, "mfa_required": result.mfa_required}


@app.post("/api/auth/mfa/setup")
def auth_mfa_setup(user: auth.Principal = Depends(_signed_in)) -> dict:
    import segno

    try:
        setup = auth.begin_mfa_setup(user.id)
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc
    qr = segno.make(setup["otpauth_uri"], error="m").svg_data_uri(scale=5, border=2)
    return {**setup, "qr_svg_data_uri": qr}


@app.post("/api/auth/mfa/confirm")
def auth_mfa_confirm(body: CodeRequest, user: auth.Principal = Depends(_signed_in)) -> dict:
    try:
        auth.confirm_mfa_setup(user.id, body.code)
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc
    return {"mfa_enabled": True}


@app.post("/api/auth/mfa/disable")
def auth_mfa_disable(body: PasswordRequest, user: auth.Principal = Depends(_signed_in)) -> dict:
    try:
        auth.disable_mfa(user.id, body.password)
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc
    return {"mfa_enabled": False}


def _admin(user: auth.Principal | None = Depends(current_user)) -> auth.Principal:
    if user is None or not user.can("admin"):
        raise HTTPException(status_code=403, detail="Only admins can manage users.")
    return user


@app.get("/api/users")
def users_list(_: auth.Principal = Depends(_admin)) -> dict:
    return {"users": auth.list_users()}


@app.post("/api/users")
def users_create(body: NewUserRequest, _: auth.Principal = Depends(_admin)) -> dict:
    try:
        return {"user": auth.create_user(body.email, body.password, role=body.role, name=body.name).__dict__}
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc


@app.patch("/api/users/{user_id}")
def users_update(user_id: int, body: UserUpdateRequest, admin: auth.Principal = Depends(_admin)) -> dict:
    try:
        if body.password is not None:
            auth.set_password(user_id, body.password)
        return {"user": auth.update_user(user_id, role=body.role, is_active=body.is_active,
                                         name=body.name, acting_user_id=admin.id)}
    except auth.AuthError as exc:
        raise _auth_error(exc) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request,
        user: auth.Principal | None = Depends(require_role("clinician"))) -> AskResponse:
    # Rate limiting is keyed by client IP. Behind a proxy (Hugging Face, Render),
    # the real client is in X-Forwarded-For; fall back to the socket address.
    fwd = request.headers.get("x-forwarded-for", "")
    client_id = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "global")
    return pipeline.run(body.query, body.settings, client_id=client_id,
                        user_id=user.id if user else None)


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
def list_audit(user: auth.Principal | None = Depends(current_user)) -> JSONResponse:
    """Recent audit records (lightweight), most recent first. Persisted to disk,
    so this survives a restart."""
    return JSONResponse({
        "count": audit.store.count(),
        "persisted": audit.store.backend() is not None,
        "backend": audit.store.backend(),
        # Clinicians see their own questions; reviewers and admins see all.
        "recent": audit.store.recent(20, user_id=_audit_scope(user)),
    })


def _audit_scope(user: auth.Principal | None) -> int | None:
    if config.AUTH_REQUIRED and user is not None and not user.can("reviewer"):
        return user.id
    return None


@app.get("/api/audit/{audit_id}")
def get_audit(audit_id: str, user: auth.Principal | None = Depends(current_user)) -> JSONResponse:
    record = audit.store.get(audit_id)
    scope = _audit_scope(user)
    if record is None or (scope is not None and record.get("user_id") != scope):
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
    corpus = retrieval.all_metadata()
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


def _is_local_request(request: Request) -> bool:
    host = _client_ip(request)
    try:
        return host == "testclient" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_manager(request: Request, role: str = "admin") -> auth.Principal | None:
    """Guard for actions that change the service for everyone. With accounts
    required, the user needs the role. In the open demo, ADMIN_ACCESS decides
    where requests may come from."""
    if config.AUTH_REQUIRED:
        principal = auth.session_principal(_session_token(request)) if db.ready() else None
        if principal is None or not principal.can(role):
            raise HTTPException(status_code=403, detail="Your role doesn't allow this.")
        return principal
    if config.ADMIN_ACCESS == "all" or (config.ADMIN_ACCESS == "local" and _is_local_request(request)):
        return None
    raise HTTPException(status_code=403, detail="This is only allowed from the machine running GroundCheck.")


def _require_local_ai_admin(request: Request) -> None:
    require_manager(request, "admin")


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


# --- Organisation documents -------------------------------------------------
#
# Upload needs a reviewer; approving, rejecting, retiring, rebuilding the index
# and running a document's evaluation need an admin. An approval by the person
# who uploaded the document is refused when accounts are required.

class ReviewRequest(BaseModel):
    note: str = ""


def _parse_date(value: str | None, name: str):
    from datetime import datetime, timezone

    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a date like 2026-10-01.") from exc


def _knowledge_error(exc: Exception) -> HTTPException:
    from . import documents, knowledge

    if isinstance(exc, (documents.DocumentError, knowledge.KnowledgeError)):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@app.post("/api/sources")
async def sources_upload(request: Request) -> dict:
    from . import documents, knowledge

    user = require_manager(request, "reviewer")
    _require_database()
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="Choose a file to upload.")
    data = await upload.read(documents.MAX_FILE_BYTES + 1)
    try:
        return {"source": await run_in_threadpool(
            knowledge.import_document, upload.filename or "", data,
            title=str(form.get("title") or "") or None,
            owner=str(form.get("owner") or ""),
            effective_from=_parse_date(str(form.get("effective_from") or ""), "Effective date"),
            expires_on=_parse_date(str(form.get("expires_on") or ""), "Expiry date"),
            uploaded_by=user.id if user else None,
        )}
    except Exception as exc:  # noqa: BLE001 - mapped to a 400 or re-raised
        raise _knowledge_error(exc) from exc


@app.get("/api/sources")
def sources_list(request: Request) -> dict:
    from . import knowledge

    require_manager(request, "reviewer")
    _require_database()
    return {"sources": knowledge.list_sources(), "index": knowledge.index_status(),
            "include_demo_corpus": config.INCLUDE_DEMO_CORPUS}


@app.get("/api/sources/{source_id}")
def sources_get(source_id: int, request: Request) -> dict:
    from . import knowledge

    require_manager(request, "reviewer")
    _require_database()
    try:
        return {"source": knowledge.get_source(source_id)}
    except Exception as exc:  # noqa: BLE001
        raise _knowledge_error(exc) from exc


@app.post("/api/sources/{source_id}/evaluate")
async def sources_evaluate(source_id: int, request: Request) -> dict:
    from . import knowledge

    require_manager(request, "admin")
    _require_database()
    try:
        return {"evaluation": await run_in_threadpool(knowledge.evaluate_source, source_id)}
    except Exception as exc:  # noqa: BLE001
        raise _knowledge_error(exc) from exc


@app.post("/api/sources/{source_id}/{decision}")
def sources_review(source_id: int, decision: str, body: ReviewRequest, request: Request) -> dict:
    from . import knowledge

    if decision not in ("approve", "reject", "retire"):
        raise HTTPException(status_code=404, detail="Not found")
    user = require_manager(request, "admin")
    _require_database()
    try:
        source = knowledge.review(source_id, decision, user.id if user else None, body.note,
                                  allow_self_approval=not config.AUTH_REQUIRED)
    except Exception as exc:  # noqa: BLE001
        raise _knowledge_error(exc) from exc
    if decision in ("approve", "retire"):
        knowledge.rebuild_index_in_background()
    return {"source": source, "index": knowledge.index_status()}


@app.post("/api/index/rebuild")
def index_rebuild(request: Request) -> dict:
    from . import knowledge

    require_manager(request, "admin")
    knowledge.rebuild_index_in_background()
    return {"index": knowledge.index_status()}


@app.get("/api/index")
def index_get(request: Request) -> dict:
    from . import knowledge

    require_manager(request, "reviewer")
    return {"index": knowledge.index_status(), "passages": len(retrieval.all_metadata())}


# --- Review and governance ------------------------------------------------------
#
# Any signed-in user can flag an answer. Reviewers work the queue and read
# reports; admins also edit the hazard log.

class FlagRequest(BaseModel):
    note: str


class AssignRequest(BaseModel):
    user_id: int | None = None


class CommentRequest(BaseModel):
    note: str


class ResolveRequest(BaseModel):
    outcome: str
    note: str = ""
    expected_decision: str | None = None


class HazardRequest(BaseModel):
    title: str
    cause: str = ""
    effect: str = ""
    severity: int
    likelihood: int
    controls: str = ""
    residual_severity: int
    residual_likelihood: int
    status: str = "open"
    owner: str = ""
    related_case_id: int | None = None


def _governance_error(exc: Exception) -> HTTPException:
    if isinstance(exc, governance.GovernanceError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@app.post("/api/audit/{audit_id}/flag")
def flag_answer(audit_id: str, body: FlagRequest, request: Request,
                user: auth.Principal | None = Depends(current_user)) -> dict:
    _require_database()
    if not config.AUTH_REQUIRED:
        require_manager(request, "clinician")
    try:
        return {"case": governance.flag_answer(audit_id, body.note, user.id if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.get("/api/reviews")
def reviews_list(request: Request, status: str = "open", mine: bool = False, overdue: bool = False) -> dict:
    user = require_manager(request, "reviewer")
    _require_database()
    if status not in ("open", "resolved", "dismissed", "all"):
        raise HTTPException(status_code=400, detail="Status must be open, resolved, dismissed or all.")
    reviewers = [{"id": u["id"], "name": u["name"] or u["email"]} for u in auth.list_users()
                 if u["is_active"] and u["role"] in ("reviewer", "admin")]
    return {**governance.list_cases(status, assigned_to=user.id if (mine and user) else None,
                                    overdue_only=overdue),
            "reviewers": reviewers, "me": user.id if user else None}


@app.get("/api/reviews/{case_id}")
def reviews_get(case_id: int, request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    try:
        return {"case": governance.get_case(case_id)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.post("/api/reviews/{case_id}/assign")
def reviews_assign(case_id: int, body: AssignRequest, request: Request) -> dict:
    user = require_manager(request, "reviewer")
    _require_database()
    try:
        return {"case": governance.assign(case_id, body.user_id, user.id if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.post("/api/reviews/{case_id}/comment")
def reviews_comment(case_id: int, body: CommentRequest, request: Request) -> dict:
    user = require_manager(request, "reviewer")
    _require_database()
    try:
        return {"case": governance.comment(case_id, body.note, user.id if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.post("/api/reviews/{case_id}/resolve")
def reviews_resolve(case_id: int, body: ResolveRequest, request: Request) -> dict:
    user = require_manager(request, "reviewer")
    _require_database()
    try:
        return {"case": governance.resolve(case_id, body.outcome, body.note, user.id if user else None,
                                           body.expected_decision)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.post("/api/reviews/{case_id}/reopen")
def reviews_reopen(case_id: int, body: CommentRequest, request: Request) -> dict:
    user = require_manager(request, "reviewer")
    _require_database()
    try:
        return {"case": governance.reopen(case_id, body.note, user.id if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.get("/api/review-tests")
def review_tests_list(request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    return {"cases": governance.list_eval_cases()}


@app.post("/api/review-tests/run")
async def review_tests_run(request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    return await run_in_threadpool(governance.run_eval_cases)


@app.get("/api/hazards")
def hazards_list(request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    return {"hazards": governance.list_hazards()}


@app.post("/api/hazards")
def hazards_create(body: HazardRequest, request: Request) -> dict:
    user = require_manager(request, "admin")
    _require_database()
    try:
        return {"hazard": governance.save_hazard(body.model_dump(), user.id if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.put("/api/hazards/{hazard_id}")
def hazards_update(hazard_id: int, body: HazardRequest, request: Request) -> dict:
    user = require_manager(request, "admin")
    _require_database()
    try:
        return {"hazard": governance.save_hazard(body.model_dump(), user.id if user else None, hazard_id)}
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.get("/api/governance/report")
def governance_report(request: Request, days: int = 30) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    try:
        return governance.report(days)
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.get("/api/governance/safety-case")
def governance_safety_case(request: Request, days: int = 30) -> Response:
    require_manager(request, "reviewer")
    _require_database()
    try:
        text = governance.safety_case_markdown(days)
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc
    return Response(text, media_type="text/markdown; charset=utf-8", headers={
        "Content-Disposition": 'attachment; filename="groundcheck-safety-case.md"'})


# Static assets (logo, fonts, css, js). Mounted last so API routes win.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
