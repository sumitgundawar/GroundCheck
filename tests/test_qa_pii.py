"""No raw identifier survives anywhere it should not: every place a question, answer, comment, file name or payload can end
up (responses, the database, the audit file, logs, exports, telemetry, traces) is scanned for the canaries sent in."""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import apikeys, auth, config, db, guards_input, pipeline, retrieval, telemetry  # noqa: E402
from app.main import app  # noqa: E402

PASSWORD = "correct horse battery staple"
JSON = {"content-type": "application/json"}

# What a canary looks like in the text, and what must never come back out. Short fragments are the identifying part.
CANARY_TEXT = ("patient John Smith DOB 14/03/1971 NHS 943 476 5919 MRN 8877665 email zq.canary@example.org "
               "phone 07700 900123 SSN 078-05-1120 lives at 42 Wallaby Way")
RAW = ["Smith", "14/03/1971", "943 476 5919", "9434765919", "8877665", "zq.canary", "07700 900123", "07700900123",
       "078-05-1120", "Wallaby"]
ASK = f"{CANARY_TEXT}. What is the standard dose of Caloradine?"
UNKNOWN = f"{CANARY_TEXT}. What is the standard dose of Quelvarin?"     # refused, so it also goes to the review queue
ANSWER = f"For {CANARY_TEXT}: Caloradine is 15 mg once daily."


def leaked(*haystacks) -> list[str]:
    found = []
    for hay in haystacks:
        text = hay if isinstance(hay, str) else json.dumps(hay, default=str)
        found += [raw for raw in RAW if raw.lower() in text.lower()]
    return sorted(set(found))


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def world(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    pipeline.clear_answer_cache()
    guards_input.key_limiter._by_client.clear()
    auth.create_user("admin@example.org", PASSWORD, role="admin", name="Admin")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/auth/login", json={"email": "admin@example.org", "password": PASSWORD}).status_code == 200
    made = apikeys.create("pii", ["ask", "verify", "usage", "audit:read"])
    return client, {"Authorization": "Bearer " + made["key"]}


def stored() -> str:
    """Everything the database and the audit file hold, as text."""
    out = []
    path = config.DATABASE_URL.removeprefix("sqlite:///")
    if config.DATABASE_URL.startswith("sqlite") and os.path.exists(path):
        con = sqlite3.connect(path)
        out.append("\n".join(con.iterdump()))
        con.close()
    log = Path(config.AUDIT_LOG_PATH)
    if log.exists():
        out.append(log.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def surfaces(client) -> dict[str, str]:
    """Every read-only view of what was recorded."""
    paths = ["/api/audit?limit=200", "/api/reviews?status=all", "/api/governance/report", "/api/usage", "/api/monitoring",
             "/api/incidents", "/api/incidents/export.csv", "/api/governance/pilot-report", "/api/metrics/adoption",
             "/api/telemetry", "/metrics", "/api/governance/surveillance", "/api/hazards", "/api/review-tests"]
    out = {}
    for path in paths:
        r = client.get(path)
        out[path] = r.text if r.status_code < 500 else ""
        assert r.status_code < 500, (path, r.status_code)
    pack = client.get("/api/governance/evidence-pack")
    if pack.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(pack.content)) as z:
            out["evidence-pack"] = "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())
    return out


# --- The question path -----------------------------------------------------------------------------------------------

def test_a_dashboard_question_leaks_nothing_anywhere(world, caplog):
    client, _ = world
    caplog.set_level(logging.DEBUG)
    answered = client.post("/api/ask", json={"query": ASK})
    refused = client.post("/api/ask", json={"query": UNKNOWN})
    assert answered.status_code == refused.status_code == 200
    assert leaked(answered.text, refused.text) == []
    audit_id = answered.json()["audit_id"]
    detail = client.get(f"/api/audit/{audit_id}")
    assert leaked(detail.text) == []
    views = surfaces(client)
    for name, text in views.items():
        assert leaked(text) == [], name
    assert leaked(stored()) == []
    assert leaked(caplog.text) == []


def test_the_audit_record_keeps_the_redacted_question_and_says_what_was_removed(world):
    client, _ = world
    r = client.post("/api/ask", json={"query": ASK}).json()
    assert r["redaction"]["kinds"] and "[NAME]" in r["redaction"]["redacted_query"]
    assert "Caloradine" in r["redaction"]["redacted_query"]


# --- The API ------------------------------------------------------------------------------------------------------------

def test_the_api_leaks_nothing_in_questions_answers_audit_usage_or_logs(world, caplog):
    client, headers = world
    caplog.set_level(logging.DEBUG)
    asked = client.post("/api/v1/ask", json={"question": ASK}, headers=headers)
    refused = client.post("/api/v1/ask", json={"question": UNKNOWN}, headers=headers)
    checked = client.post("/api/v1/verify", json={"question": ASK, "answer": ANSWER}, headers=headers)
    wrong = client.post("/api/v1/verify", json={"question": ASK, "answer": ANSWER.replace("15 mg", "150 mg")}, headers=headers)
    assert [x.status_code for x in (asked, refused, checked, wrong)] == [200] * 4
    for response in (asked, refused, checked, wrong):
        assert leaked(response.text) == []
    for response in (asked, refused, checked, wrong):
        record = client.get(f"/api/v1/audit/{response.json()['audit_id']}", headers=headers)
        assert record.status_code == 200 and leaked(record.text) == []
    for query in ("group_by=key", "group_by=day&format=csv", "group_by=endpoint", "group_by=decision&format=csv"):
        assert leaked(client.get(f"/api/v1/usage?{query}", headers=headers).text) == []
    assert leaked(stored()) == []
    assert leaked(caplog.text) == []
    for view in surfaces(client).values():
        assert leaked(view) == []


def test_the_patient_context_is_stored_without_identifiers(world):
    client, headers = world
    patient = {"age_years": 70, "egfr": 45, "weight_kg": 70, "medicines": ["warfarin"], "conditions": ["John Smith's gout"],
               "allergies": ["penicillin"]}
    r = client.post("/api/v1/ask", json={"question": "What is the standard dose of Caloradine?", "patient": patient}, headers=headers)
    assert r.status_code == 200
    record = client.get(f"/api/v1/audit/{r.json()['audit_id']}", headers=headers)
    assert leaked(record.text, stored()) == []


@pytest.mark.parametrize("field", ["question", "answer"])
def test_identifiers_split_across_formatting_are_still_removed(world, field):
    client, headers = world
    body = {"question": "What is the standard dose of Caloradine?", "answer": "Caloradine is 15 mg once daily."}
    body[field] = "NHS number 943-476-5919 / MRN: 8877665 / DOB 14.03.1971 / zq.canary@example.org. " + body[field]
    r = client.post("/api/v1/verify", json=body, headers=headers)
    assert r.status_code == 200 and leaked(r.text, stored()) == []


# --- Feedback, incidents, review notes, file names -----------------------------------------------------------------------

def test_a_feedback_comment_is_cleaned_before_it_is_stored(world):
    client, _ = world
    audit_id = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?"}).json()["audit_id"]
    r = client.post(f"/api/audit/{audit_id}/feedback", json={"rating": "down", "reason": "wrong", "comment": CANARY_TEXT})
    assert r.status_code == 200 and leaked(r.text, stored()) == []
    assert leaked(client.get(f"/api/audit/{audit_id}/feedback").text) == []
    for view in surfaces(client).values():
        assert leaked(view) == []


def test_an_incident_is_cleaned_before_it_is_stored_or_exported(world):
    client, _ = world
    r = client.post("/api/incidents", json={"title": "Wrong dose for John Smith", "category": "answer", "harm": "none",
                                            "description": CANARY_TEXT})
    assert r.status_code == 200, r.text
    incident_id = r.json()["incident"]["id"]
    client.post(f"/api/incidents/{incident_id}/comments", json={"note": CANARY_TEXT})
    client.patch(f"/api/incidents/{incident_id}", json={"status": "investigating", "note": CANARY_TEXT})
    assert leaked(r.text, client.get(f"/api/incidents/{incident_id}").text, client.get("/api/incidents/export.csv").text,
                  client.get("/api/incidents").text, stored()) == []


def test_review_notes_are_cleaned_before_they_are_stored(world):
    client, _ = world
    refused = client.post("/api/ask", json={"query": UNKNOWN}).json()
    case_id = refused["review"]["case_id"]
    assert case_id
    client.post(f"/api/reviews/{case_id}/comment", json={"note": CANARY_TEXT})
    client.post(f"/api/reviews/{case_id}/resolve", json={"resolution": "answered", "note": CANARY_TEXT})
    assert leaked(client.get(f"/api/reviews/{case_id}").text, client.get("/api/reviews?status=all").text, stored()) == []


def test_a_source_file_name_with_a_name_in_it_is_not_kept(world, monkeypatch):
    from app import knowledge

    monkeypatch.setattr(knowledge, "rebuild_index_in_background", lambda *a, **k: None)
    client, _ = world
    data = b"# Protocol\n\n## Dose\n\nGive Zoxalin 5 mg once daily.\n"
    r = client.post("/api/sources", files={"file": ("Dr Zhou for patient John Smith 14-03-1971.md", data, "text/markdown")},
                    data={"title": "Zoxalin protocol"})
    assert r.status_code == 200, r.text
    assert leaked(r.text, client.get("/api/sources").text, stored()) == []


# --- CDS Hooks, EHR and FHIR -------------------------------------------------------------------------------------------

def test_a_cds_request_with_a_patient_resource_leaks_nothing(world, monkeypatch, caplog):
    client, _ = world
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(config, "CDS_HOOKS_ALLOW_UNSIGNED", True)
    services = client.get("/cds-services").json()["services"]
    assert services
    patient = {"resourceType": "Patient", "id": "p1", "name": [{"family": "Smith", "given": ["John"]}], "birthDate": "1971-03-14",
               "identifier": [{"system": "nhs", "value": "9434765919"}, {"system": "mrn", "value": "8877665"}],
               "telecom": [{"system": "email", "value": "zq.canary@example.org"}, {"system": "phone", "value": "07700 900123"}],
               "address": [{"line": ["42 Wallaby Way"]}]}
    for service in services:
        body = {"hook": service.get("hook", "patient-view"), "hookInstance": "d1b2c3d4-0000-4000-8000-000000000001",
                "context": {"patientId": "p1", "userId": "Practitioner/1",
                            "draftOrders": {"resourceType": "Bundle", "entry": [{"resource": {
                                "resourceType": "MedicationRequest", "status": "draft", "intent": "order",
                                "medicationCodeableConcept": {"text": f"Caloradine 150 mg once daily for {CANARY_TEXT}"}}}]}},
                "prefetch": {"patient": patient}}
        r = client.post(f"/cds-services/{service['id']}", json=body)
        assert r.status_code < 500
        assert leaked(r.text) == []
        fb = {"feedback": [{"card": "c1", "outcome": "overridden", "outcomeTimestamp": "2026-11-24T10:00:00Z",
                            "overrideReason": {"reason": {"code": "John Smith 8877665"}, "userComment": CANARY_TEXT}}]}
        assert client.post(f"/cds-services/{service['id']}/feedback", json=fb).status_code < 500
    assert leaked(caplog.text) == []
    assert leaked(stored()) == []
    for view in surfaces(client).values():
        assert leaked(view) == []


# --- Telemetry and traces -----------------------------------------------------------------------------------------------

def test_the_telemetry_payload_holds_nothing_from_questions(world):
    client, _ = world
    client.post("/api/ask", json={"query": ASK})
    assert leaked(telemetry.payload("install-id")) == []


def test_traces_carry_no_question_text_or_identifiers(world, monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app import tracing

    exporter = InMemorySpanExporter()
    tracing.use_exporter(exporter)
    try:
        client, headers = world
        client.post("/api/ask", json={"query": ASK})
        client.post("/api/v1/verify", json={"question": ASK, "answer": ANSWER}, headers=headers)
        spans = exporter.get_finished_spans()
        assert spans
        dumped = json.dumps([{"name": s.name, "attributes": dict(s.attributes or {}),
                              "events": [(e.name, dict(e.attributes or {})) for e in s.events]} for s in spans], default=str)
        assert leaked(dumped) == []
    finally:
        tracing.reset()


# --- Identifiers written in unusual ways ----------------------------------------------------------

UNSEEN = [
    ("QA-6 a name written in lower case", "Caloradine dose for john smith", "smith"),
    ("QA-6 an initial and a surname", "Caloradine for J. Smith", "smith"),
    ("QA-7 an e-mail written with [at] and [dot]", "email: zq.canary [at] example [dot] org", "canary"),
    ("QA-7 an e-mail with spaces around the at sign", "zq.canary @ example.org", "canary"),
    ("QA-8 an NHS number spelled in words", "NHS number nine four three four seven six five nine one nine", "nine four three"),
]


@pytest.mark.parametrize("why,text,fragment", UNSEEN, ids=[u[0].split()[0] + "-" + str(i) for i, u in enumerate(UNSEEN)])
def test_identifiers_the_deidentifier_misses(why, text, fragment):
    assert fragment not in guards_input.deidentify(text).text.lower()


def test_an_accented_name_is_removed():
    assert "quillfeather" not in guards_input.deidentify("patient María José Quillfeather needs Caloradine").text.lower()
