"""SMART on FHIR: launching GroundCheck from an EHR, and loading its patient.

EHR launch: the EHR opens /api/ehr/launch?iss=<its FHIR base>&launch=<token>.
Standalone launch: /api/ehr/launch?iss=<FHIR base>. Either way GroundCheck
discovers the authorisation server from the issuer's
.well-known/smart-configuration, sends the browser to it with PKCE, and on
return exchanges the code for an access token and the patient in context.

The patient is read once (app/fhir.py) and kept, encrypted, for
EHR_CONTEXT_MINUTES for this browser. The access token isn't stored.

Only issuers in SMART_ALLOWED_ISSUERS can launch GroundCheck, so a crafted
launch link can't make the server fetch from anywhere else. For development,
servers in FHIR_OPEN_SERVERS can be read without SMART."""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import timedelta
from urllib.parse import urlencode

import httpx
from sqlalchemy import delete, select

from . import config, db, fhir
from .db import EhrContext, EhrLaunch, utcnow

LAUNCH_MINUTES = 10
# Tests replace this with a mock transport; None uses the network.
transport: httpx.BaseTransport | None = None


class SmartError(Exception):
    """A launch or EHR load failed. The message is safe to show."""


def enabled() -> bool:
    return bool(config.SMART_CLIENT_ID and config.SMART_ALLOWED_ISSUERS)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redirect_url(base_url: str) -> str:
    return config.SMART_REDIRECT_URL or f"{base_url.rstrip('/')}/api/ehr/callback"


def discover(issuer: str, transport_override: httpx.BaseTransport | None = None) -> dict:
    try:
        with httpx.Client(timeout=10.0, transport=transport_override or transport, follow_redirects=False) as client:
            response = client.get(f"{issuer}/.well-known/smart-configuration", headers={"Accept": "application/json"})
            if response.status_code == 200:
                document = response.json()
            else:
                metadata = client.get(f"{issuer}/metadata", headers={"Accept": "application/fhir+json"}).json()
                security = (metadata.get("rest") or [{}])[0].get("security", {})
                oauth = next((e for e in security.get("extension", [])
                              if e.get("url", "").endswith("StructureDefinition/oauth-uris")), {})
                uris = {e.get("url"): e.get("valueUri") for e in oauth.get("extension", [])}
                document = {"authorization_endpoint": uris.get("authorize"), "token_endpoint": uris.get("token")}
    except (httpx.HTTPError, ValueError) as exc:
        raise SmartError("Couldn't reach the EHR's sign-in service.") from exc
    for key in ("authorization_endpoint", "token_endpoint"):
        url = str(document.get(key) or "")
        if not (url.startswith("https://") or url.startswith(("http://localhost", "http://127.0.0.1"))):
            raise SmartError("The EHR's SMART configuration is missing or not secure.")
    return document


def start(issuer: str, launch: str | None, base_url: str) -> tuple[str, str]:
    if not enabled():
        raise SmartError("SMART on FHIR isn't set up on this server.")
    try:
        issuer = fhir.validate_base_url(issuer, config.SMART_ALLOWED_ISSUERS)
    except fhir.FhirError as exc:
        raise SmartError("This EHR isn't allowed to launch GroundCheck.") from exc
    document = discover(issuer)
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    now = utcnow()
    with db.session() as s:
        s.execute(delete(EhrLaunch).where(EhrLaunch.expires_at < now))
        s.add(EhrLaunch(state_hash=_hash(state), issuer=issuer, code_verifier=verifier,
                        token_endpoint=document["token_endpoint"], created_at=now,
                        expires_at=now + timedelta(minutes=LAUNCH_MINUTES)))
    params = {"response_type": "code", "client_id": config.SMART_CLIENT_ID, "redirect_uri": redirect_url(base_url),
              "scope": config.SMART_SCOPES if launch else config.SMART_SCOPES.replace("launch ", "", 1),
              "state": state, "aud": issuer, "code_challenge": challenge, "code_challenge_method": "S256"}
    if launch:
        params["launch"] = launch
    endpoint = document["authorization_endpoint"]
    return f"{endpoint}{'&' if '?' in endpoint else '?'}{urlencode(params)}", state


def finish(code: str, state: str, cookie_state: str | None, base_url: str) -> str:
    if not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
        raise SmartError("The launch didn't start in this browser, or took too long. Launch GroundCheck again.")
    now = utcnow()
    with db.session() as s:
        pending = s.scalar(select(EhrLaunch).where(EhrLaunch.state_hash == _hash(state)))
        if pending is None or pending.expires_at < now:
            raise SmartError("The launch didn't start in this browser, or took too long. Launch GroundCheck again.")
        issuer, verifier, token_endpoint = pending.issuer, pending.code_verifier, pending.token_endpoint
        s.delete(pending)
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_url(base_url),
            "code_verifier": verifier, "client_id": config.SMART_CLIENT_ID}
    auth = (config.SMART_CLIENT_ID, config.SMART_CLIENT_SECRET) if config.SMART_CLIENT_SECRET else None
    try:
        with httpx.Client(timeout=15.0, transport=transport, follow_redirects=False) as client:
            response = client.post(token_endpoint, data=form, auth=auth, headers={"Accept": "application/json"})
        tokens = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SmartError("Couldn't complete the launch with the EHR.") from exc
    if response.status_code != 200 or not tokens.get("access_token"):
        raise SmartError("The EHR didn't accept the launch. Launch GroundCheck again.")
    patient_id = tokens.get("patient")
    if not patient_id:
        raise SmartError("The EHR didn't say which patient to open. Launch GroundCheck from a patient's record.")
    try:
        with fhir.FhirClient(issuer, tokens["access_token"], transport=transport) as client:
            loaded = fhir.load_patient(client, patient_id, {"system": "SMART on FHIR", "server": issuer})
    except fhir.FhirError as exc:
        raise SmartError(str(exc)) from exc
    return save(loaded)


def load_open(server: str, patient_id: str) -> tuple[str, dict]:
    try:
        base = fhir.validate_base_url(server, config.FHIR_OPEN_SERVERS)
        with fhir.FhirClient(base, transport=transport) as client:
            loaded = fhir.load_patient(client, patient_id.strip(), {"system": "FHIR", "server": base})
    except fhir.FhirError as exc:
        raise SmartError(str(exc)) from exc
    token = save(loaded)
    return token, get(token)


def save(loaded: fhir.EhrPatient) -> str:
    token = secrets.token_urlsafe(32)
    now = utcnow()
    with db.session() as s:
        s.execute(delete(EhrContext).where(EhrContext.expires_at < now))
        s.add(EhrContext(token_hash=_hash(token), created_at=now,
                         expires_at=now + timedelta(minutes=config.EHR_CONTEXT_MINUTES),
                         data={"patient": loaded.context.model_dump(exclude_defaults=True), "source": loaded.source,
                               "warnings": loaded.warnings, "loaded_at": now.isoformat()}))
    return token


def get(token: str | None) -> dict | None:
    if not token:
        return None
    with db.session() as s:
        row = s.scalar(select(EhrContext).where(EhrContext.token_hash == _hash(token)))
        if row is None:
            return None
        if row.expires_at < utcnow():
            s.delete(row)
            return None
        return {**row.data, "expires_at": row.expires_at.isoformat()}


def clear(token: str | None) -> None:
    if token:
        with db.session() as s:
            s.execute(delete(EhrContext).where(EhrContext.token_hash == _hash(token)))
