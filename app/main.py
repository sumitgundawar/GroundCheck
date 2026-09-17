"""FastAPI application: serves the static frontend and a small JSON API.
The embedding model and search index are loaded once at startup."""

from __future__ import annotations

import asyncio
import hmac
import html
import time
import ipaddress
import logging
import json
from urllib.parse import quote
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select

from . import (
    audit, auth, config, db, encryption, governance, integrity, llm, local_ai, monitoring, pipeline, retention, retrieval,
    sso,
)
from .schemas import AskRequest, AskResponse, Settings

WEB_DIR = config.ROOT_DIR / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Refuse to start with a malformed encryption or signing key, rather than
    # run without the protection that was asked for.
    encryption.keyring()
    integrity.signing_summary()
    # Warm the model and load the prebuilt index before serving traffic. On a
    # new server whose index lives on an empty data volume, build it first.
    retrieval.get_model()
    checks = None
    if db.ready():
        from .imaging import store as imaging_store

        imaging_store.recover_interrupted()
        if config.ALERT_INTERVAL_SECONDS > 0:
            checks = asyncio.create_task(_alert_loop())
    try:
        retrieval.load_index()
    except FileNotFoundError:
        from . import knowledge

        logging.getLogger("groundcheck").warning("No search index found in %s; building it now.", config.INDEX_DIR)
        knowledge.rebuild_index()
    try:
        yield
    finally:
        if checks is not None:
            checks.cancel()


async def _alert_loop() -> None:
    """Evaluate alert rules in the background, starting a minute after startup."""
    await asyncio.sleep(min(60, config.ALERT_INTERVAL_SECONDS))
    while True:
        try:
            await run_in_threadpool(monitoring.evaluate)
        except Exception:  # noqa: BLE001 - keep checking
            logging.getLogger("groundcheck").exception("Alert evaluation failed")
        await asyncio.sleep(config.ALERT_INTERVAL_SECONDS)


app = FastAPI(title="GroundCheck", version=config.VERSION, lifespan=lifespan,
              docs_url="/docs" if config.API_DOCS else None, redoc_url="/redoc" if config.API_DOCS else None,
              openapi_url="/openapi.json" if config.API_DOCS else None)


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
              "/api/auth/logout", "/api/auth/first-admin", "/api/auth/sso/start", "/api/auth/sso/callback",
              "/api/ehr/launch", "/api/ehr/callback"}


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
        "sso": {"enabled": sso.enabled(), "provider": config.OIDC_PROVIDER_NAME},
        "password_sign_in": config.PASSWORD_SIGN_IN or not sso.enabled(),
    })


SSO_STATE_COOKIE = "gc_sso_state"


def _base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{proto}://{host}"


def _continue_page(path: str, message: str = "Signing you in") -> HTMLResponse:
    """A page that moves on to a path on this site. A page, not a redirect: a
    redirect chain that began at the identity provider counts as cross-site,
    and the browser wouldn't send the new SameSite=Strict session cookie."""
    target = html.escape(path, quote=True)
    return HTMLResponse(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}">'
        f"<title>{html.escape(message)}</title><p>{html.escape(message)}… <a href=\"{target}\">Continue</a></p></html>",
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.get("/api/auth/sso/start")
def auth_sso_start(request: Request, next: str = "/") -> Response:  # noqa: A002 - query parameter name
    _require_database()
    try:
        url, state = sso.start(_base_url(request), next)
    except sso.SsoError as exc:
        return _continue_page(f"/?sso_error={quote(str(exc))}", "Sign-in failed")
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(SSO_STATE_COOKIE, state, httponly=True, samesite="lax", secure=config.SESSION_COOKIE_SECURE,
                        max_age=sso.LOGIN_MINUTES * 60, path="/api/auth/sso")
    return response


@app.get("/api/auth/sso/callback")
def auth_sso_callback(request: Request, code: str = "", state: str = "", error: str = "",
                      error_description: str = "") -> Response:
    _require_database()
    if error:
        message = f"{config.OIDC_PROVIDER_NAME} didn't sign you in: {error_description or error}"[:300]
        response = _continue_page(f"/?sso_error={quote(message)}", "Sign-in failed")
    else:
        try:
            done = sso.finish(code, state, request.cookies.get(SSO_STATE_COOKIE), _base_url(request),
                              ip=_client_ip(request), user_agent=request.headers.get("user-agent", ""))
        except sso.SsoError as exc:
            response = _continue_page(f"/?sso_error={quote(str(exc))}", "Sign-in failed")
        else:
            response = _continue_page(done.next_path)
            _set_session_cookie(response, done.token)
    response.delete_cookie(SSO_STATE_COOKIE, path="/api/auth/sso")
    return response


@app.post("/api/auth/login")
def auth_login(body: SignInRequest, request: Request, response: Response) -> dict:
    _require_database()
    if not config.PASSWORD_SIGN_IN and sso.enabled():
        raise HTTPException(status_code=403, detail=f"Sign in with {config.OIDC_PROVIDER_NAME} instead.")
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


@app.delete("/api/users/{user_id}")
def users_delete(user_id: int, admin: auth.Principal = Depends(_admin)) -> dict:
    try:
        return {"user": auth.delete_user(user_id, acting_user_id=admin.id)}
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
    return pipeline.run(body.query, body.settings, client_id=client_id, patient=body.patient,
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


@app.get("/api/usage")
def usage_dashboard(request: Request, days: int = 30) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    try:
        return governance.usage(days)
    except Exception as exc:  # noqa: BLE001
        raise _governance_error(exc) from exc


@app.get("/api/formulary")
def formulary_list(q: str = "") -> dict:
    """The medicines patient-aware checks know, with the kinds of rules each has."""
    from . import formulary

    f = formulary.index().formulary
    needle = formulary.normalise(q)
    rows = []
    for m in f.medicines:
        names = " ".join(formulary.normalise(n) for n in [m.name, *m.aliases, *m.classes])
        if needle and needle not in names:
            continue
        rows.append({
            "name": m.name, "aliases": m.aliases, "classes": m.classes, "high_alert": m.high_alert,
            "adult_dose": m.adult_dose.model_dump() if m.adult_dose else None,
            "rules": {
                "allergies": len(m.allergy_groups), "conditions": len(m.contraindicated_conditions),
                "interactions": len(m.interactions), "kidney": len(m.renal), "liver": len(m.hepatic),
                "children": m.paediatric_dose is not None, "weight": m.weight_dose is not None,
                "labs": len(m.labs), "pregnancy": m.pregnancy, "breastfeeding": m.breastfeeding,
            },
        })
    return {"name": f.name, "version": f.version, "synthetic": f.synthetic, "total": len(f.medicines),
            "medicines": rows[:500], "names": sorted({n for m in f.medicines for n in [m.name, *m.classes]})}


@app.get("/api/formulary/{name}")
def formulary_get(name: str) -> dict:
    from . import formulary

    medicine = formulary.index().find(name)
    if medicine is None:
        raise HTTPException(status_code=404, detail="That medicine isn't in the formulary.")
    return {"medicine": medicine.model_dump()}


@app.get("/api/embeddings")
def embeddings_summary() -> dict:
    from . import knowledge

    return {**retrieval.embedding_summary(), "index": knowledge.index_status()}


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


# --- EHR integration --------------------------------------------------------------
#
# SMART on FHIR launch and patient loading (/api/ehr/...), and a CDS Hooks
# service (/cds-services) that EHRs call during medication ordering.

EHR_COOKIE = "gc_ehr"
EHR_STATE_COOKIE = "gc_ehr_state"


class EhrLoadRequest(BaseModel):
    server: str
    patient_id: str


@app.get("/api/ehr/config")
def ehr_config(request: Request) -> dict:
    from . import smart

    base = _base_url(request)
    return {"smart_enabled": smart.enabled(), "smart_client_id": config.SMART_CLIENT_ID,
            "smart_allowed_issuers": config.SMART_ALLOWED_ISSUERS, "smart_launch_url": f"{base}/api/ehr/launch",
            "smart_redirect_url": smart.redirect_url(base), "smart_scopes": config.SMART_SCOPES,
            "open_servers": config.FHIR_OPEN_SERVERS, "cds_discovery_url": f"{base}/cds-services",
            "cds_unsigned": config.CDS_HOOKS_ALLOW_UNSIGNED, "cds_trusted": sorted(config.CDS_HOOKS_TRUSTED),
            "context_minutes": config.EHR_CONTEXT_MINUTES}


@app.get("/api/ehr/launch")
def ehr_launch(request: Request, iss: str = "", launch: str = "") -> Response:
    from . import smart

    _require_database()
    try:
        url, state = smart.start(iss, launch or None, _base_url(request))
    except smart.SmartError as exc:
        return _continue_page(f"/?ehr_error={quote(str(exc))}", "Launch failed")
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(EHR_STATE_COOKIE, state, httponly=True, samesite="lax", secure=config.SESSION_COOKIE_SECURE,
                        max_age=smart.LAUNCH_MINUTES * 60, path="/api/ehr")
    return response


@app.get("/api/ehr/callback")
def ehr_callback(request: Request, code: str = "", state: str = "", error: str = "",
                 error_description: str = "") -> Response:
    from . import smart

    _require_database()
    if error:
        response = _continue_page(f"/?ehr_error={quote(('The EHR ended the launch: ' + (error_description or error))[:300])}",
                                  "Launch failed")
    else:
        try:
            token = smart.finish(code, state, request.cookies.get(EHR_STATE_COOKIE), _base_url(request))
        except smart.SmartError as exc:
            response = _continue_page(f"/?ehr_error={quote(str(exc))}", "Launch failed")
        else:
            response = _continue_page("/?ehr=1#/ask", "Opening the patient")
            response.set_cookie(EHR_COOKIE, token, httponly=True, samesite="strict", path="/",
                                secure=config.SESSION_COOKIE_SECURE, max_age=config.EHR_CONTEXT_MINUTES * 60)
    response.delete_cookie(EHR_STATE_COOKIE, path="/api/ehr")
    return response


@app.get("/api/ehr/context")
def ehr_context(request: Request, user: auth.Principal | None = Depends(require_role("clinician"))) -> dict:
    from . import smart

    _require_database()
    return {"context": smart.get(request.cookies.get(EHR_COOKIE))}


@app.delete("/api/ehr/context")
def ehr_clear(request: Request, response: Response,
              user: auth.Principal | None = Depends(require_role("clinician"))) -> dict:
    from . import smart

    _require_database()
    smart.clear(request.cookies.get(EHR_COOKIE))
    response.delete_cookie(EHR_COOKIE, path="/")
    return {"cleared": True}


@app.post("/api/ehr/load")
async def ehr_load(body: EhrLoadRequest, request: Request, response: Response,
                   user: auth.Principal | None = Depends(require_role("clinician"))) -> dict:
    from . import smart

    _require_database()
    if not config.AUTH_REQUIRED:
        require_manager(request, "clinician")
    try:
        token, context = await run_in_threadpool(smart.load_open, body.server, body.patient_id)
    except smart.SmartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.set_cookie(EHR_COOKIE, token, httponly=True, samesite="strict", path="/",
                        secure=config.SESSION_COOKIE_SECURE, max_age=config.EHR_CONTEXT_MINUTES * 60)
    return {"context": context}


class EhrNoteRequest(BaseModel):
    audit_id: str
    comment: str = ""


@app.post("/api/ehr/notes")
async def ehr_note(body: EhrNoteRequest, request: Request,
                   user: auth.Principal | None = Depends(require_role("clinician"))) -> dict:
    from . import smart

    _require_database()
    record = audit.store.get(body.audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail="That answer isn't in the audit trail.")
    if user is not None and record.get("user_id") not in (None, user.id):
        raise HTTPException(status_code=403, detail="You can only save answers you asked.")
    with db.session() as s:
        row = s.scalar(select(db.AuditRecord).where(db.AuditRecord.audit_id == body.audit_id))
        created = row.created_at.isoformat() if row else None
    reviewer = (user.name or user.email) if user else "a clinician"
    try:
        result = await run_in_threadpool(smart.write_note, request.cookies.get(EHR_COOKIE),
                                         {**record, "created_at": created}, reviewer, body.comment[:1000])
    except smart.SmartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logging.getLogger("groundcheck.ehr").info("Saved audit record %s to the EHR as %s", body.audit_id, result["reference"])
    return {"reference": result["reference"]}


CDS_CORS = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type", "Access-Control-Max-Age": "600"}


@app.options("/cds-services")
@app.options("/cds-services/{service_id}")
@app.options("/cds-services/{service_id}/feedback")
def cds_preflight() -> Response:
    return Response(status_code=204, headers=CDS_CORS)


@app.get("/cds-services")
def cds_discovery() -> JSONResponse:
    from . import cds_hooks

    return JSONResponse(cds_hooks.discovery(), headers=CDS_CORS)


@app.post("/cds-services/{service_id}")
async def cds_service(service_id: str, request: Request) -> JSONResponse:
    from . import cds_hooks

    if service_id not in {s["id"] for s in cds_hooks.discovery()["services"]}:
        return JSONResponse({"detail": "No such service."}, status_code=404, headers=CDS_CORS)
    try:
        await run_in_threadpool(cds_hooks.verify, request.headers.get("authorization"),
                                f"{_base_url(request)}/cds-services/{service_id}")
        body = await request.json()
        if not isinstance(body, dict):
            raise cds_hooks.CdsError("The request must be a JSON object.")
        result = await run_in_threadpool(cds_hooks.cards_for, body)
    except cds_hooks.CdsError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status, headers=CDS_CORS)
    except ValueError:
        return JSONResponse({"detail": "The request isn't valid JSON."}, status_code=400, headers=CDS_CORS)
    logging.getLogger("groundcheck.cds").info("CDS %s: %d cards for hook instance %s", service_id,
                                              len(result["cards"]), cds_hooks.safe_hook_instance(body))
    return JSONResponse(result, headers=CDS_CORS)


@app.post("/cds-services/{service_id}/feedback")
async def cds_feedback(service_id: str, request: Request) -> JSONResponse:
    from . import cds_hooks

    try:
        await run_in_threadpool(cds_hooks.verify, request.headers.get("authorization"),
                                f"{_base_url(request)}/cds-services/{service_id}/feedback")
        body = await request.json()
    except cds_hooks.CdsError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status, headers=CDS_CORS)
    except ValueError:
        return JSONResponse({"detail": "The request isn't valid JSON."}, status_code=400, headers=CDS_CORS)
    for item in (body.get("feedback") or [])[:50] if isinstance(body, dict) else []:
        logging.getLogger("groundcheck.cds").info(
            "CDS feedback %s: card %s %s, reason %s", service_id, str(item.get("card", ""))[:64],
            str(item.get("outcome", ""))[:20], str((item.get("overrideReason") or {}).get("reason", {}).get("code", ""))[:40])
    return JSONResponse({}, headers=CDS_CORS)


# --- Data protection -----------------------------------------------------------

@app.get("/api/data-protection")
def data_protection(request: Request) -> dict:
    """Encryption, audit signing and retention status. Key ids only, never keys."""
    require_manager(request, "admin")
    _require_database()
    return {"encryption": encryption.keyring().summary(), "audit_signing": integrity.signing_summary(),
            "audit_chain": integrity.head(), "retention": retention.plan()}


@app.post("/api/audit/verify")
async def audit_verify(request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    return await run_in_threadpool(integrity.verify)


@app.post("/api/retention/run")
def retention_run(request: Request) -> dict:
    user = require_manager(request, "admin")
    _require_database()
    return {"run": retention.apply(user.id if user else None), "retention": retention.plan()}


# --- Training studio and model library --------------------------------------
#
# Training reads folders on this machine and uses its GPUs, so it's for
# admins. Reviewers can see the library, and anyone signed in can use a model
# on an image.

class TrainingRequest(BaseModel):
    name: str
    dataset: str
    architecture: str = "small-cnn"
    pretrained: bool = False
    device: str
    epochs: int = 20
    image_size: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    notes: str = ""


def _training_error(exc: Exception) -> HTTPException:
    from .training import datasets, library, runs

    if isinstance(exc, (datasets.DatasetError, runs.TrainingError, library.LibraryError)):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@app.get("/api/training/setup")
async def training_setup(request: Request) -> dict:
    from .training import architectures, datasets, hardware, runs

    require_manager(request, "admin")
    return {
        "hardware": await run_in_threadpool(hardware.detect),
        **architectures.options(),
        "roots": [str(r) for r in datasets.roots()],
        "target_accuracy": config.MODEL_TARGET_ACCURACY,
        "active_run": runs.active(),
    }


@app.get("/api/training/browse")
def training_browse(request: Request, path: str = "") -> dict:
    from .training import datasets

    require_manager(request, "admin")
    try:
        return datasets.browse(path or None)
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.get("/api/training/dataset")
async def training_dataset(request: Request, path: str) -> dict:
    from .training import datasets

    require_manager(request, "admin")
    try:
        folder, _, items, summary = await run_in_threadpool(datasets.read, path)
        checks = await run_in_threadpool(datasets.inspect, folder, items)
        checks.pop("_leaked_paths", None)
        return {**summary, "checks": checks, "warnings": summary["warnings"] + checks["warnings"]}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.get("/api/training/image")
def training_image(request: Request, dataset: str, file: str) -> Response:
    """A small preview of one image in a dataset."""
    import io

    from .training import datasets, preprocess

    require_manager(request, "admin")
    try:
        path = datasets.resolve_file(datasets.resolve(dataset), file)
        image = preprocess.open_image(path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc
    image.thumbnail((160, 160))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return Response(buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=300"})


@app.post("/api/training/runs")
async def training_start(body: TrainingRequest, request: Request) -> dict:
    from .training import runs

    user = require_manager(request, "admin")
    try:
        return {"run": await run_in_threadpool(runs.start, body.model_dump(exclude_none=True),
                                               (user.name or user.email) if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.get("/api/training/runs")
def training_runs(request: Request) -> dict:
    from .training import runs

    require_manager(request, "admin")
    return {"runs": runs.list_runs()}


@app.get("/api/training/runs/{run_id}")
def training_run(run_id: str, request: Request) -> dict:
    from .training import runs

    require_manager(request, "admin")
    try:
        return {"run": runs.get(run_id)}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.post("/api/training/runs/{run_id}/cancel")
def training_cancel(run_id: str, request: Request) -> dict:
    from .training import runs

    require_manager(request, "admin")
    try:
        return {"run": runs.cancel(run_id)}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.post("/api/training/runs/{run_id}/resume")
def training_resume(run_id: str, request: Request) -> dict:
    from .training import runs

    require_manager(request, "admin")
    try:
        return {"run": runs.resume(run_id)}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.get("/api/models")
def models_list(request: Request) -> dict:
    from .training import library

    require_manager(request, "reviewer")
    return {"models": library.list_models(), "target_accuracy": config.MODEL_TARGET_ACCURACY}


@app.get("/api/models/{model_id}")
def models_get(model_id: str, request: Request) -> dict:
    from .training import library

    require_manager(request, "reviewer")
    try:
        return {"model": library.get_model(model_id)}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.delete("/api/models/{model_id}")
def models_delete(model_id: str, request: Request) -> dict:
    from .training import library

    require_manager(request, "admin")
    try:
        library.delete_model(model_id)
        return {"deleted": model_id}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.get("/api/models/{model_id}/download")
def models_download(model_id: str, request: Request) -> Response:
    from .training import library

    require_manager(request, "admin")
    try:
        data = library.export_zip(model_id)
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{model_id}.zip"'})


@app.post("/api/models/{model_id}/predict")
async def models_predict(model_id: str, request: Request,
                         user: auth.Principal | None = Depends(require_role("clinician"))) -> dict:
    from .training import library

    if not config.AUTH_REQUIRED:
        require_manager(request, "clinician")
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="Choose an image.")
    data = await upload.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Images can be up to 20 MB.")
    try:
        return await run_in_threadpool(library.predict, model_id, data)
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


class ModelImagingRequest(BaseModel):
    modality: str
    window: str | None = None
    orientation: str = "identity"
    patch_mm: float | None = None
    note: str = ""


@app.put("/api/models/{model_id}/imaging")
def models_set_imaging(model_id: str, body: ModelImagingRequest, request: Request) -> dict:
    from .training import library

    require_manager(request, "admin")
    try:
        return {"imaging": library.set_imaging(model_id, body.model_dump())}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


@app.delete("/api/models/{model_id}/imaging")
def models_remove_imaging(model_id: str, request: Request) -> dict:
    from .training import library

    require_manager(request, "admin")
    try:
        return {"imaging": library.set_imaging(model_id, None)}
    except Exception as exc:  # noqa: BLE001
        raise _training_error(exc) from exc


# --- CT and MRI imaging -------------------------------------------------------

def _imaging_error(exc: Exception) -> HTTPException:
    from .imaging import dicomweb, store

    if isinstance(exc, LookupError):
        return HTTPException(status_code=404, detail="No such series or report.")
    if isinstance(exc, store.ImagingError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, dicomweb.PacsError):
        return HTTPException(status_code=502, detail=str(exc))
    raise exc


def _imaging_user(request: Request, role: str = "clinician") -> auth.Principal | None:
    user = require_manager(request, role)
    if not db.ready():
        raise HTTPException(status_code=503, detail="Imaging needs a database, and it isn't available.")
    return user


@app.get("/api/imaging")
def imaging_home(request: Request) -> dict:
    from .imaging import dicom, dicomweb, store

    _imaging_user(request)
    return {"series": store.list_series(), "models": store.imaging_models(), "pacs": dicomweb.enabled(),
            "windows": {k: list(v) for k, v in dicom.WINDOWS.items()},
            "max_upload_mb": config.IMAGING_MAX_UPLOAD_MB}


@app.post("/api/imaging/upload")
async def imaging_upload(request: Request) -> dict:
    from .imaging import store

    user = _imaging_user(request)
    limit = config.IMAGING_MAX_UPLOAD_MB * 1024**2
    if int(request.headers.get("content-length") or 0) > limit + 1024**2:
        raise HTTPException(status_code=413, detail=f"Uploads can be up to {config.IMAGING_MAX_UPLOAD_MB} MB.")
    form = await request.form(max_files=3000, max_part_size=200 * 1024**2)
    uploads, total = [], 0
    for item in form.getlist("files"):
        if not hasattr(item, "read"):
            continue
        data = await item.read()
        total += len(data)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"Uploads can be up to {config.IMAGING_MAX_UPLOAD_MB} MB.")
        uploads.append((item.filename or "upload", data))
    if not uploads:
        raise HTTPException(status_code=400, detail="Choose DICOM files, a folder of them, or a zip.")
    try:
        return await run_in_threadpool(store.import_series, uploads, "upload", str(form.get("label") or ""),
                                       user.id if user else None)
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


@app.get("/api/imaging/series/{series_id}")
def imaging_series(series_id: int, request: Request) -> dict:
    from .imaging import store

    _imaging_user(request)
    try:
        return {"series": store.get_series(series_id)}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


@app.delete("/api/imaging/series/{series_id}")
def imaging_delete(series_id: int, request: Request) -> dict:
    from .imaging import store

    _imaging_user(request, "reviewer")
    try:
        store.delete_series(series_id)
        return {"deleted": series_id}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


@app.get("/api/imaging/series/{series_id}/slices/{index}.png")
async def imaging_slice(series_id: int, index: int, request: Request, window: str | None = None) -> Response:
    from .imaging import store

    _imaging_user(request)
    try:
        png = await run_in_threadpool(store.slice_png, series_id, index, window)
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


class ImagingAnalyseRequest(BaseModel):
    model_id: str


@app.post("/api/imaging/series/{series_id}/analyses")
def imaging_analyse(series_id: int, body: ImagingAnalyseRequest, request: Request) -> dict:
    from .imaging import store

    user = _imaging_user(request)
    try:
        return {"analysis": store.start_analysis(series_id, body.model_id, user.id if user else None)}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


class ImagingReportRequest(BaseModel):
    findings: str = ""
    impression: str = ""
    agreement: str = "not_used"
    analysis_id: int | None = None
    sign: bool = False
    report_id: int | None = None


@app.post("/api/imaging/series/{series_id}/reports")
def imaging_report(series_id: int, body: ImagingReportRequest, request: Request) -> dict:
    from .imaging import store

    user = _imaging_user(request)
    try:
        return {"report": store.save_report(
            series_id, findings=body.findings, impression=body.impression, agreement=body.agreement,
            analysis_id=body.analysis_id, sign=body.sign, report_id=body.report_id,
            author_id=user.id if user else None, author_name=(user.name or user.email) if user else "Local user")}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


@app.get("/api/imaging/reports/{report_id}/sr.dcm")
async def imaging_report_download(report_id: int, request: Request) -> Response:
    from .imaging import store

    _imaging_user(request)
    try:
        data, info = await run_in_threadpool(store.report_sr, report_id, False)
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc
    return Response(data, media_type="application/dicom",
                    headers={"Content-Disposition": f'attachment; filename="report-{report_id}.dcm"'})


@app.post("/api/imaging/reports/{report_id}/send")
async def imaging_report_send(report_id: int, request: Request) -> dict:
    from .imaging import dicomweb, store

    _imaging_user(request)
    try:
        data, info = await run_in_threadpool(store.report_sr, report_id, True)
        await run_in_threadpool(dicomweb.store, [data], info["study"])
        await run_in_threadpool(store.mark_sent, report_id)
        return {"sent": True, "sop_instance_uid": info["sop_instance_uid"]}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


@app.get("/api/imaging/pacs/studies")
async def imaging_pacs_studies(request: Request, patient_id: str = "", patient_name: str = "", accession: str = "",
                               study_date: str = "", modality: str = "") -> dict:
    from .imaging import dicomweb

    _imaging_user(request)
    try:
        return {"studies": await run_in_threadpool(dicomweb.search_studies, patient_id, patient_name, accession,
                                                   study_date, modality)}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


@app.get("/api/imaging/pacs/studies/{study_uid}/series")
async def imaging_pacs_series(study_uid: str, request: Request) -> dict:
    from .imaging import dicomweb

    _imaging_user(request)
    try:
        return {"series": await run_in_threadpool(dicomweb.search_series, study_uid)}
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


class PacsRetrieveRequest(BaseModel):
    study_uid: str
    series_uid: str
    label: str = ""


@app.post("/api/imaging/pacs/retrieve")
async def imaging_pacs_retrieve(body: PacsRetrieveRequest, request: Request) -> dict:
    from .imaging import dicomweb, store

    user = _imaging_user(request)
    try:
        files = await run_in_threadpool(dicomweb.retrieve_series, body.study_uid, body.series_uid)
        return await run_in_threadpool(store.import_series, files, "pacs", body.label, user.id if user else None)
    except Exception as exc:  # noqa: BLE001
        raise _imaging_error(exc) from exc


# --- Monitoring -----------------------------------------------------------------

@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Request counts and latency for /metrics, labelled by route template so
    IDs in paths don't create a metric per record."""
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        template = getattr(route, "path", None) or ("static" if request.method == "GET" else "unmatched")
        monitoring.observe_request(request.method, template, status, time.perf_counter() - started)


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    if config.METRICS_TOKEN:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {config.METRICS_TOKEN}".encode()):
            return Response("Unauthorized\n", status_code=401, media_type="text/plain")
    elif not _is_local_request(request):
        return Response("Set METRICS_TOKEN to scrape metrics from another machine.\n", status_code=403,
                        media_type="text/plain")
    return Response(monitoring.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/healthz/live", include_in_schema=False)
def health_live() -> dict:
    """The process is running. For a container orchestrator's liveness probe."""
    return {"status": "ok"}


@app.get("/healthz/ready", include_in_schema=False)
def health_ready() -> JSONResponse:
    """Ready for traffic: the database answers and the search index is loaded."""
    checks = {"database": db.ready(), "index": retrieval.is_loaded()}
    ok = all(checks.values())
    return JSONResponse({"status": "ok" if ok else "unavailable", "checks": checks}, status_code=200 if ok else 503)


@app.get("/api/monitoring")
def monitoring_overview(request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    return monitoring.overview()


@app.post("/api/monitoring/evaluate")
async def monitoring_evaluate(request: Request) -> dict:
    require_manager(request, "reviewer")
    _require_database()
    changes = await run_in_threadpool(monitoring.evaluate)
    return {"changes": changes, **(await run_in_threadpool(monitoring.overview))}


@app.post("/api/alerts/{alert_id}/acknowledge")
def alerts_acknowledge(alert_id: int, request: Request) -> dict:
    user = require_manager(request, "reviewer")
    _require_database()
    try:
        return {"alert": monitoring.acknowledge(alert_id, user.id if user else None,
                                                (user.name or user.email) if user else "Local user")}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="No such alert.") from exc


# The dashboard loads nothing from other sites, so the policy allows only this
# origin. FastAPI's interactive docs (/docs, /redoc) load from a CDN, so they
# get a looser policy and can be switched off with API_DOCS=false.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
DOCS_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    headers = response.headers
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("Referrer-Policy", "same-origin")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
    headers.setdefault("Content-Security-Policy",
                       DOCS_POLICY if path in ("/docs", "/redoc", "/docs/oauth2-redirect") else CONTENT_SECURITY_POLICY)
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if forwarded_proto == "https":
        headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if path.startswith("/api/"):
        headers.setdefault("Cache-Control", "no-store")
    return response


@app.middleware("http")
async def revalidate_static_files(request: Request, call_next):
    """Ask browsers to check the dashboard's files on every load. Unchanged
    files still come back as a quick 304, and an update is never hidden
    behind a stale cached copy."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


# Static assets (logo, fonts, css, js). Mounted last so API routes win.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
