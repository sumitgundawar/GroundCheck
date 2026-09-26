"""SMART on FHIR: launching GroundCheckHealth from an EHR, and loading its patient.

EHR launch: the EHR opens /api/ehr/launch?iss=<its FHIR base>&launch=<token>.
Standalone launch: /api/ehr/launch?iss=<FHIR base>. Either way GroundCheckHealth
discovers the authorisation server from the issuer's
.well-known/smart-configuration, sends the browser to it with PKCE, and on
return exchanges the code for an access token and the patient in context.

The patient is read once (app/fhir.py) and kept, encrypted, for
EHR_CONTEXT_MINUTES for this browser. The access token isn't stored.

Only issuers in SMART_ALLOWED_ISSUERS can launch GroundCheckHealth, so a crafted
launch link can't make the server fetch from anywhere else. For development,
servers in FHIR_OPEN_SERVERS can be read without SMART."""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timedelta
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
        raise SmartError("This EHR isn't allowed to launch GroundCheckHealth.") from exc
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
    scopes = config.SMART_SCOPES + (" patient/DocumentReference.write" if config.SMART_WRITE_NOTES else "")
    params = {"response_type": "code", "client_id": config.SMART_CLIENT_ID, "redirect_uri": redirect_url(base_url),
              "scope": scopes if launch else scopes.replace("launch ", "", 1),
              "state": state, "aud": issuer, "code_challenge": challenge, "code_challenge_method": "S256"}
    if launch:
        params["launch"] = launch
    endpoint = document["authorization_endpoint"]
    return f"{endpoint}{'&' if '?' in endpoint else '?'}{urlencode(params)}", state


def finish(code: str, state: str, cookie_state: str | None, base_url: str) -> str:
    if not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
        raise SmartError("The launch didn't start in this browser, or took too long. Launch GroundCheckHealth again.")
    now = utcnow()
    with db.session() as s:
        pending = s.scalar(select(EhrLaunch).where(EhrLaunch.state_hash == _hash(state)))
        if pending is None or pending.expires_at < now:
            raise SmartError("The launch didn't start in this browser, or took too long. Launch GroundCheckHealth again.")
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
        raise SmartError("The EHR didn't accept the launch. Launch GroundCheckHealth again.")
    patient_id = tokens.get("patient")
    if not patient_id:
        raise SmartError("The EHR didn't say which patient to open. Launch GroundCheckHealth from a patient's record.")
    try:
        with fhir.FhirClient(issuer, tokens["access_token"], transport=transport) as client:
            loaded = fhir.load_patient(client, patient_id, {"system": "SMART on FHIR", "server": issuer})
    except fhir.FhirError as exc:
        raise SmartError(str(exc)) from exc
    granted = str(tokens.get("scope", ""))
    can_write = config.SMART_WRITE_NOTES and any(s in granted for s in (
        "DocumentReference.write", "DocumentReference.c", "DocumentReference.*", "patient/*.write", "patient/*.*"))
    access = None
    if can_write:
        expires_in = tokens.get("expires_in") if isinstance(tokens.get("expires_in"), int) else 3600
        access = {"token": tokens["access_token"], "expires_at": (utcnow() + timedelta(seconds=expires_in)).isoformat()}
    return save(loaded, access)


def load_open(server: str, patient_id: str) -> tuple[str, dict]:
    try:
        base = fhir.validate_base_url(server, config.FHIR_OPEN_SERVERS)
        with fhir.FhirClient(base, transport=transport) as client:
            loaded = fhir.load_patient(client, patient_id.strip(), {"system": "FHIR", "server": base})
    except fhir.FhirError as exc:
        raise SmartError(str(exc)) from exc
    token = save(loaded)
    return token, get(token)


def save(loaded: fhir.EhrPatient, access: dict | None = None) -> str:
    token = secrets.token_urlsafe(32)
    now = utcnow()
    data = {"patient": loaded.context.model_dump(exclude_defaults=True), "source": loaded.source,
            "warnings": loaded.warnings, "loaded_at": now.isoformat()}
    if access:
        data["access"] = access  # encrypted at rest with the rest of the context
    with db.session() as s:
        s.execute(delete(EhrContext).where(EhrContext.expires_at < now))
        s.add(EhrContext(token_hash=_hash(token), created_at=now,
                         expires_at=now + timedelta(minutes=config.EHR_CONTEXT_MINUTES), data=data))
    return token


def _row(token: str | None):
    if not token:
        return None
    with db.session() as s:
        row = s.scalar(select(EhrContext).where(EhrContext.token_hash == _hash(token)))
        if row is None or row.expires_at < utcnow():
            return None
        return dict(row.data)


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
        data = {k: v for k, v in row.data.items() if k != "access"}
        access = row.data.get("access")
        data["can_write_notes"] = bool(access and datetime.fromisoformat(access["expires_at"]) > utcnow())
        return {**data, "expires_at": row.expires_at.isoformat()}


def clear(token: str | None) -> None:
    if token:
        with db.session() as s:
            s.execute(delete(EhrContext).where(EhrContext.token_hash == _hash(token)))


NOTE_TYPE = {"system": "http://loinc.org", "code": "11506-3", "display": "Progress note"}


def note_text(record: dict, reviewer: str, comment: str) -> str:
    response = record.get("response", {})
    answered = response.get("decision") == "answer"
    lines = [f"Question: {record.get('redacted_query', '')}", f"Decision: {'Answered' if answered else 'Refused'}", ""]
    if answered:
        lines += ["Answer:", response.get("answer_text", ""), ""]
        cited = {sid for c in response.get("claims", []) for sid in c.get("source_ids", [])}
        sources = [f"- {s['id']}: {s.get('title', '')}" for s in response.get("sources", []) if s.get("id") in cited]
        if sources:
            lines += ["Sources:", *sources, ""]
    else:
        lines += [f"Reason: {response.get('refused_reason', '')}", ""]
    findings = response.get("patient_findings", [])
    if findings:
        lines += ["Checked for this patient:",
                  *[f"- [{f['severity']}] {f['medicine']}: {f['message']}" for f in findings], ""]
    if comment.strip():
        lines += [f"Clinician comment: {comment.strip()}", ""]
    lines += [f"Drafted by GroundCheckHealth from approved sources and reviewed by {reviewer} before saving. "
              f"GroundCheckHealth is not a medical device. Audit record {record.get('audit_id', '')}."]
    return "\n".join(lines)


def write_note(context_token: str | None, record: dict, reviewer: str, comment: str = "") -> dict:
    """Save a reviewed answer to the patient's record as a preliminary note."""
    data = _row(context_token)
    if not data or data.get("source", {}).get("system") != "SMART on FHIR" or not data.get("access"):
        raise SmartError("Saving to the record needs a SMART launch from the EHR with permission to write notes.")
    access = data["access"]
    if datetime.fromisoformat(access["expires_at"]) <= utcnow():
        raise SmartError("The EHR's permission has expired. Launch GroundCheckHealth from the record again.")
    if not record.get("patient"):
        raise SmartError("Only an answer asked for this patient can be saved to their record.")
    loaded_at = datetime.fromisoformat(data["loaded_at"])
    created = record.get("created_at")
    if created and datetime.fromisoformat(created) < loaded_at:
        raise SmartError("That answer was asked before this patient was opened.")
    now = utcnow()
    text = note_text(record, reviewer, comment)
    document = {
        "resourceType": "DocumentReference", "status": "current", "docStatus": "preliminary",
        "type": {"coding": [NOTE_TYPE], "text": "GroundCheckHealth answer"},
        "category": [{"coding": [{"system": "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category",
                                  "code": "clinical-note", "display": "Clinical Note"}]}],
        "subject": {"reference": data["source"]["patient"]},
        "date": now.isoformat(),
        "author": [{"display": reviewer}],
        "description": f"GroundCheckHealth: {str(record.get('redacted_query', ''))[:120]}",
        "content": [{"attachment": {"contentType": "text/plain; charset=utf-8", "title": "GroundCheckHealth answer",
                                    "creation": now.isoformat(),
                                    "data": base64.b64encode(text.encode("utf-8")).decode("ascii")}}],
        "context": {"related": [{"identifier": {"system": "https://groundcheckhealth.com/audit",
                                                "value": record.get("audit_id", "")}}]},
    }
    server = data["source"]["server"]
    try:
        with httpx.Client(timeout=15.0, transport=transport, follow_redirects=False) as client:
            response = client.post(f"{server}/DocumentReference", json=document,
                                   headers={"Authorization": f"Bearer {access['token']}",
                                            "Content-Type": "application/fhir+json", "Accept": "application/fhir+json"})
    except httpx.HTTPError as exc:
        raise SmartError("Couldn't reach the EHR to save the note.") from exc
    if response.status_code in (401, 403):
        raise SmartError("The EHR refused to save the note. Check GroundCheckHealth's write permission with the EHR team.")
    if response.status_code not in (200, 201):
        raise SmartError(f"The EHR couldn't save the note ({response.status_code}).")
    location = response.headers.get("location", "")
    try:
        created_id = response.json().get("id") if response.content else None
    except ValueError:
        created_id = None
    if not created_id and "/DocumentReference/" in location:
        created_id = location.split("/DocumentReference/", 1)[1].split("/")[0]
    return {"id": created_id, "reference": f"DocumentReference/{created_id}" if created_id else None, "text": text}
