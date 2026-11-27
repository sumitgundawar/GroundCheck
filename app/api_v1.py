"""The versioned API for machine clients: /api/v1.

Requests carry an API key (`Authorization: Bearer gck_...` or `X-API-Key`),
never in the URL. A gateway in front of the routes does, in this order: a key
in the query string is refused; the key is checked; the per-minute limit is
taken; the scope the route needs is checked; the body is size-limited. Only
then is the body parsed and validated, so a bad request from a real key is
counted, and a request with no valid key never reaches the parser.

Every response carries X-Request-Id. Every response to an authenticated
request also carries X-RateLimit-Limit, -Remaining and -Reset (seconds). Every
error is {"error": {"code", "message"}}, with "details" for validation errors.
Error messages never echo what the caller sent.

Metering counts each authenticated request once, as the decision it got, or as
`error` (refused before it was evaluated) or `throttled` (429). A request that
fails before a key is identified cannot be attributed and is not counted.
Quota is taken only once a request is valid and is given back if the server
fails, so it is spent only on requests that were evaluated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from urllib.parse import unquote_plus
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader, HTTPBearer
from pydantic import BaseModel
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import apikeys, audit, config, db, metering, monitoring, pipeline, retrieval, sites, verify
from .schemas import AskResponse, TooLong, V1AskRequest, V1VerifyRequest, VerifyResponse

log = logging.getLogger("groundcheck.api")

REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
KEY_PARAMS = frozenset({"api_key", "apikey", "api-key", "key", "token", "access_token", "x-api-key", "authorization"})
BILLABLE = ("ask", "verify")

# (method, path) -> (endpoint name, scope). Audit is matched by prefix.
ROUTES = {("POST", "/api/v1/ask"): ("ask", "ask"), ("POST", "/api/v1/verify"): ("verify", "verify"),
          ("GET", "/api/v1/usage"): ("usage", "usage")}
AUDIT_PREFIX = "/api/v1/audit/"
CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
         409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 422: "validation_error",
         429: "rate_limited", 500: "internal_error", 503: "service_unavailable"}


def in_envelope(path: str) -> bool:
    """Paths whose errors use the {"error": {...}} shape: the v1 API and key
    administration. The dashboard's own routes keep {"detail": ...}."""
    return path in ("/api/v1", "/api/keys") or path.startswith(("/api/v1/", "/api/keys/"))


def envelope(status: int, code: str, message: str, headers: dict[str, str] | None = None,
             details: list[dict] | None = None) -> JSONResponse:
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse({"error": error}, status_code=status, headers=headers)


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# --- A request in flight --------------------------------------------------------

class Call:
    """One authenticated request: whose it is, what it will be metered as."""

    def __init__(self, key: apikeys.KeyPrincipal, endpoint: str):
        self.key, self.endpoint = key, endpoint
        self.site_id = key.site_id
        self.metered = False
        self.reserved = False

    def reserve(self) -> None:
        """Take one request from the monthly quota (billable endpoints only)."""
        if self.endpoint in BILLABLE:
            apikeys.reserve_quota(self.key)
            self.reserved = True

    def ok(self, decision: str) -> None:
        if not self.metered:
            self.metered = True
            metering.record(self.endpoint, decision, self.site_id, self.key.id)

    def failed(self, status: int) -> None:
        if self.metered:
            return
        self.metered = True
        if self.reserved and status >= 500:
            apikeys.refund_quota(self.key)
        metering.record(self.endpoint, "throttled" if status == 429 else "error", self.site_id, self.key.id)


def current_call(request: Request) -> Call:
    return request.state.call


def resolve_site(call: Call, site_key: str | None) -> int | None:
    """The site a request speaks for: the key's own, or, for a key that is not
    tied to a site, the one it names."""
    if site_key is None:
        return call.site_id
    with db.session() as s:
        site = sites.by_key(s, site_key)
        if site is None:
            if call.key.site_id is not None:      # a site's key cannot tell a site that does not exist from one that does
                raise apikeys.ApiKeyError(403, "site_forbidden", "This API key belongs to a different site.")
            raise apikeys.ApiKeyError(422, "unknown_site", "There is no site with that key.")
        if call.key.site_id is not None and site.id != call.key.site_id:
            raise apikeys.ApiKeyError(403, "site_forbidden", "This API key belongs to a different site.")
        call.site_id = site.id
        return site.id


# --- The gateway -------------------------------------------------------------------

def _header(scope, name: bytes) -> str | None:
    for k, v in scope["headers"]:
        if k == name:
            return v.decode("latin-1")
    return None


def _gate(scope) -> tuple[Call | None, dict[str, str]]:
    """Authenticate and rate-limit one request. Raises ApiKeyError."""
    method, path = scope["method"], scope["path"]
    query = scope.get("query_string", b"").decode("latin-1")
    names = {unquote_plus(part.split("=", 1)[0]).lower() for part in query.split("&") if part}
    if names & KEY_PARAMS:
        raise apikeys.ApiKeyError(400, "api_key_in_query", "Send the API key in the Authorization header. "
                                  "A key in the URL ends up in logs, so it is refused.")
    target = ROUTES.get((method, path))
    if target is None and method == "GET" and path.startswith(AUDIT_PREFIX):
        target = ("audit", "audit:read")
    if target is None:
        return None, {}
    if not db.ready():
        raise apikeys.ApiKeyError(503, "service_unavailable", "The service is not ready.")
    key = apikeys.authenticate(apikeys.token_from_headers(_header(scope, b"authorization"),
                                                          _header(scope, b"x-api-key")))
    call = Call(key, target[0])
    scope["state"]["call"] = call
    remaining, wait = apikeys.check_rate(key)
    scope["state"]["rate"] = {"X-RateLimit-Limit": str(key.rpm_limit), "X-RateLimit-Remaining": str(remaining),
                              "X-RateLimit-Reset": str(wait)}
    if not key.has(target[1]):
        raise apikeys.ApiKeyError(403, "insufficient_scope", f"This API key does not have the '{target[1]}' scope.")
    return call, scope["state"]["rate"]


class Gateway:
    """Pure ASGI middleware for /api/v1 and /api/keys."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not in_envelope(scope["path"]):
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        inbound = _header(scope, b"x-request-id")
        state["request_id"] = inbound if inbound and REQUEST_ID.match(inbound) else "req_" + secrets.token_hex(12)
        started = False

        async def wrapped(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                extra = {"X-Request-Id": state["request_id"], **(state.get("rate") or {})}
                for k, v in extra.items():
                    if k.lower().encode() not in present:
                        headers.append((k.lower().encode(), v.encode()))
                message = {**message, "headers": headers}
                call = state.get("call")
                if call is not None and not call.metered:
                    if status >= 400:
                        await run_in_threadpool(call.failed, status)
                    else:
                        await run_in_threadpool(call.ok, "answer")   # unreachable: routes meter themselves
            await send(message)

        async def reply(response: Response) -> None:
            await response(scope, receive, wrapped)

        try:
            if scope["path"].startswith("/api/v1"):
                try:
                    await run_in_threadpool(_gate, scope)
                except apikeys.ApiKeyError as exc:
                    # A request stopped after its key was found is still counted, by the hook above.
                    return await reply(envelope(exc.status, exc.code, exc.message,
                                                {**(state.get("rate") or {}), **exc.headers}))
            if scope["method"] in ("POST", "PUT", "PATCH"):
                receive, refusal = await _limited(scope, receive)
                if refusal:
                    return await reply(envelope(*refusal))
            await self.app(scope, receive, wrapped)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - nothing leaves as a bare 500
            log.exception("Unhandled error in %s %s", scope["method"], scope["path"])
            if started:
                raise
            await reply(envelope(500, "internal_error", "Something went wrong on the server."))


async def _limited(scope, receive):
    """The request body, size-checked. Returns (receive that replays it,
    None) or (receive, (status, code, message)) when it is refused."""
    limit = config.API_MAX_BODY_BYTES
    declared = _header(scope, b"content-length")
    if declared is not None:
        if not declared.isdigit():
            return receive, (400, "bad_request", "The Content-Length header is not valid.")
        if int(declared) > limit:
            return receive, (413, "payload_too_large", f"The request is larger than {limit} bytes.")
    chunks, size = [], 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return receive, (400, "bad_request", "The request was interrupted.")
        size += len(message.get("body", b""))
        if size > limit:
            return receive, (413, "payload_too_large", f"The request is larger than {limit} bytes.")
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            break
    body, sent = b"".join(chunks), False

    async def replay():
        nonlocal sent
        if sent:
            return await receive()
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return replay, None


# --- Errors -------------------------------------------------------------------------

def _field(loc) -> str:
    parts = [str(p) for p in loc if p not in ("body", "query", "path")]
    return ".".join(parts)[:100]


def _clean(message: str) -> str:
    return re.sub(r"^Value error, ", "", message)[:200]


def install(app: FastAPI) -> None:
    """Register the error shapes and the gateway on the application."""

    @app.exception_handler(apikeys.ApiKeyError)
    async def _key_error(request: Request, exc: apikeys.ApiKeyError):
        return envelope(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if not in_envelope(request.url.path):
            return await http_exception_handler(request, exc)
        message = exc.detail if isinstance(exc.detail, str) and exc.status_code < 500 else "Request failed."
        return envelope(exc.status_code, CODES.get(exc.status_code, "error"), message, dict(exc.headers or {}))

    @app.exception_handler(OverflowError)
    async def _out_of_range(request: Request, exc: OverflowError):
        """A number too large for the database's integer: no such record, or a value nobody could mean."""
        if request.path_params:
            if in_envelope(request.url.path):
                return envelope(404, "not_found", "Not found.")
            return JSONResponse({"detail": "Not found."}, status_code=404)
        if in_envelope(request.url.path):
            return envelope(422, "validation_error", "A number in the request is out of range.")
        return JSONResponse({"detail": "A number in the request is out of range."}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        if not in_envelope(request.url.path):
            # The default reply echoes the offending input, and a NaN in it cannot be written as JSON.
            return JSONResponse({"detail": [{"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "invalid")),
                                             "type": str(e.get("type", "value_error"))} for e in exc.errors()]},
                                status_code=422)
        errors = exc.errors()
        if any(isinstance((e.get("ctx") or {}).get("error"), TooLong) for e in errors):
            text = next(str(e["ctx"]["error"]) for e in errors if isinstance((e.get("ctx") or {}).get("error"), TooLong))
            return envelope(413, "payload_too_large", text)
        details = [{"field": _field(e.get("loc", ())), "message": _clean(str(e.get("msg", "invalid")))}
                   for e in errors[:10]]
        return envelope(422, "validation_error", "The request is not valid.", details=details)

    app.add_middleware(Gateway)


# --- Routes ------------------------------------------------------------------------

bearer = HTTPBearer(auto_error=False, description="An API key: `Authorization: Bearer gck_...`.")
header_key = APIKeyHeader(name="X-API-Key", auto_error=False, description="An API key, as an alternative to Bearer.")
router = APIRouter(prefix="/api/v1", tags=["v1"], dependencies=[Depends(bearer), Depends(header_key)])
FAILURES = {code: {"model": ErrorEnvelope, "description": text} for code, text in {
    401: "Missing, invalid, revoked or expired API key.", 403: "The key lacks the scope, or belongs to another site.",
    413: "The request or a text in it is too large.", 422: "The request is not valid.",
    429: "Rate limit or monthly quota reached. See Retry-After."}.items()}


@router.post("/ask", response_model=AskResponse, responses=FAILURES, summary="Ask a question GroundCheck answers")
def ask(body: V1AskRequest, call: Call = Depends(current_call)) -> AskResponse:
    site_id = resolve_site(call, body.site)
    call.reserve()
    with retrieval.site_scope(site_id):
        response = pipeline.run(body.question, client_id=f"key:{call.key.id}", patient=body.patient,
                                service=pipeline.ServiceCaller(call.key.id, site_id))
    call.ok(response.decision)
    return response


@router.post("/verify", response_model=VerifyResponse, responses={**FAILURES, 404: FAILURES[401]},
             summary="Check an answer that another model wrote")
def verify_answer(body: V1VerifyRequest, call: Call = Depends(current_call)) -> VerifyResponse:
    site_id = resolve_site(call, body.site)
    with retrieval.site_scope(site_id):
        for source_id in body.sources or []:
            if retrieval.record_for(source_id) is None:
                raise apikeys.ApiKeyError(422, "unknown_source", f"There is no source '{source_id[:64]}'.")
        call.reserve()
        response, extras = verify.verify(body.question, body.answer, body.sources, body.patient)
    audit_id = audit.store.new_id()
    response.audit_id = audit_id
    if body.patient is not None and not body.patient.is_empty():
        extras["patient"] = body.patient.model_dump(exclude_defaults=True)
    if response.decision != "pass" and response.reason_code != "input_blocked":
        response.review = pipeline._route(
            review=True, check_only=False, outcome="refused", audit_id=audit_id, query=extras["redacted_query"],
            reason=response.explanation.summary, reason_code=None, site_id=site_id)
    audit.store.save(audit_id, response, extras, site_id=site_id, api_key_id=call.key.id)
    call.ok(response.decision)
    return response


@router.get("/usage", responses={**FAILURES, 200: {"description": "JSON, or CSV with format=csv.", "content": {
    "application/json": {}, "text/csv": {}}}}, summary="Request counts by key, day, site, endpoint or decision")
def usage(call: Call = Depends(current_call),
          group_by: Literal["key", "day", "site", "endpoint", "decision"] = "key",
          start: str | None = Query(default=None, alias="from", max_length=10),
          end: str | None = Query(default=None, alias="to", max_length=10),
          format: Literal["json", "csv"] = "json") -> Response:  # noqa: A002 - query parameter name
    try:
        report = metering.usage(group_by, start, end, call.key.site_id)
    except metering.UsageError as exc:
        raise apikeys.ApiKeyError(422, "validation_error", str(exc)) from exc
    call.ok("pass")
    if format == "csv":
        return Response(metering.to_csv(report), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="usage.csv"'})
    return JSONResponse(report)


@router.get("/audit/{audit_id}", responses={**FAILURES, 404: FAILURES[401]}, summary="Read one audit record")
def audit_record(audit_id: str, call: Call = Depends(current_call)) -> dict:
    record = audit.store.get(audit_id) if len(audit_id) <= 64 else None
    if (record is None or "response" not in record
            or (call.key.site_id is not None and record.get("site_id") != call.key.site_id)):
        raise apikeys.ApiKeyError(404, "not_found", "No such audit record.")
    created = None
    with db.session() as s:
        row = s.scalar(select(db.AuditRecord).where(db.AuditRecord.audit_id == audit_id))
        if row is not None:
            created = row.created_at.isoformat()
    call.ok("pass")
    return {"audit_id": audit_id, "endpoint": record.get("endpoint", "ask"), "created_at": created,
            "site_id": record.get("site_id"), "api_key_id": record.get("api_key_id"),
            "decision": record["response"].get("decision"), "redacted_query": record.get("redacted_query", ""),
            "patient": record.get("patient"), "response": record["response"]}


@router.get("/health", summary="Is the service up (no key needed)")
def health() -> dict:
    ready = db.ready() and retrieval.is_loaded()
    return {"status": "ok" if ready and monitoring.recording_as_configured() else "degraded",
            "version": config.VERSION, "index": {"documents": retrieval.corpus_size() if retrieval.is_loaded() else 0}}


def openapi_document() -> dict:
    """The OpenAPI document for /api/v1 alone."""
    sub = FastAPI(title="GroundCheck API", version="1.0.0",
                  description="Ground, check and audit answers from any model. Errors are "
                              "{\"error\": {\"code\", \"message\"}}.")
    sub.include_router(router)
    return json.loads(json.dumps(sub.openapi(), sort_keys=True))
