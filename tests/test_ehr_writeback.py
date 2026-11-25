"""Writing a Provenance (and optionally an AuditEvent) next to a saved note:
FHIR R4 structure, off by default, and a failed write never costs the clinician
their saved note, but is recorded in the audit trail."""

from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, fhir, fhir_profiles as fp, integrity, smart  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-044")

SERVER = "https://ehr.example.org/fhir"
SCOPES = "launch patient/DocumentReference.write patient/Provenance.write patient/AuditEvent.write"


class Ehr:
    """A FHIR server that accepts creates and keeps what it was sent."""

    def __init__(self, fail: dict[str, object] | None = None):
        self.created: dict[str, list[dict]] = {}
        self.fail = fail or {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        kind = request.url.path.rsplit("/", 1)[-1]
        failure = self.fail.get(kind)
        if isinstance(failure, Exception):
            raise failure
        if isinstance(failure, httpx.Response):
            return failure
        body = json.loads(request.content)
        self.created.setdefault(kind, []).append(body)
        return httpx.Response(201, json={"resourceType": kind, "id": f"{kind[:3].lower()}-{len(self.created[kind])}"})


def setup(monkeypatch, ehr: Ehr, provenance=True, audit_event=False, scope=SCOPES, notes=True):
    monkeypatch.setattr(smart, "transport", httpx.MockTransport(ehr.handler))
    monkeypatch.setattr(config, "SMART_WRITE_NOTES", notes)
    for name, on in (("SMART_WRITE_PROVENANCE", provenance), ("SMART_WRITE_AUDITEVENT", audit_event)):
        if on is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "true" if on else "false")
    loaded = fhir.EhrPatient(context=fhir.PatientContext(), source={"system": "SMART on FHIR", "server": SERVER,
                                                                      "patient": "Patient/p1"})
    access = {"token": "tok", "expires_at": (db.utcnow() + timedelta(hours=1)).isoformat()}
    if scope is not None:
        access["scope"] = scope
    return smart.save(loaded, access)


RECORD = {"patient": {"age_years": 50}, "audit_id": "a1b2c3d4e5f60718", "redacted_query": "What is the dose?",
          "response": {"decision": "answer", "answer_text": "10 mg.", "claims": [], "sources": []}}


def failures():
    with db.session() as s:
        rows = s.scalars(select(db.AuditRecord).where(db.AuditRecord.decision == "event")).all()
        return [r.record for r in rows if r.record.get("event") == "ehr_writeback_failed"]


# --- Off by default -------------------------------------------------------------------------------

def test_provenance_is_off_by_default(database, monkeypatch):
    monkeypatch.delenv("SMART_WRITE_PROVENANCE", raising=False)
    monkeypatch.delenv("SMART_WRITE_AUDITEVENT", raising=False)
    assert smart.provenance_enabled() is False and smart.audit_event_enabled() is False
    ehr = Ehr()
    token = setup(monkeypatch, ehr, provenance=None, audit_event=None)
    result = smart.write_note(token, RECORD, "Dr Who")
    assert result["reference"] == "DocumentReference/doc-1"
    assert list(ehr.created) == ["DocumentReference"]
    assert "provenance" not in result and "audit_event" not in result


@pytest.mark.parametrize("value", ["false", "0", "no", "", "off", "FALSE"])
def test_only_explicit_true_values_enable_it(monkeypatch, value):
    monkeypatch.setenv("SMART_WRITE_PROVENANCE", value)
    assert smart.provenance_enabled() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", " True "])
def test_true_values_enable_it(monkeypatch, value):
    monkeypatch.setenv("SMART_WRITE_PROVENANCE", value)
    assert smart.provenance_enabled() is True


def test_audit_event_needs_its_own_flag_and_provenance_on(database, monkeypatch):
    ehr = Ehr()
    token = setup(monkeypatch, ehr, provenance=False, audit_event=True)
    assert smart.audit_event_enabled() is False
    smart.write_note(token, RECORD, "Dr Who")
    assert list(ehr.created) == ["DocumentReference"]
    ehr = Ehr()
    token = setup(monkeypatch, ehr, provenance=True, audit_event=False)
    smart.write_note(token, RECORD, "Dr Who")
    assert list(ehr.created) == ["DocumentReference", "Provenance"]


# --- What is written ---------------------------------------------------------------------------------

def test_provenance_links_the_note_to_groundcheck_the_reviewer_and_the_audit_record(database, monkeypatch):
    ehr = Ehr()
    token = setup(monkeypatch, ehr)
    result = smart.write_note(token, RECORD, "Dr Jane Who")
    assert result["provenance"] == {"status": "written", "reference": "Provenance/pro-1"}
    [prov] = ehr.created["Provenance"]
    assert fp.validate_r4(prov) == []
    assert prov["target"] == [{"reference": "DocumentReference/doc-1"}]
    assert prov["entity"][0]["what"]["identifier"] == {"system": fhir.AUDIT_SYSTEM, "value": RECORD["audit_id"]}
    who = [a["who"]["display"] for a in prov["agent"]]
    assert who == ["GroundCheckHealth", "Dr Jane Who"]
    assert [a["type"]["coding"][0]["code"] for a in prov["agent"]] == ["assembler", "verifier"]
    assert prov["activity"]["coding"][0]["code"] == "CREATE"
    recorded = datetime.fromisoformat(prov["recorded"].replace("Z", "+00:00"))
    assert abs((datetime.now(timezone.utc) - recorded).total_seconds()) < 60


def test_the_note_itself_is_valid_r4(database, monkeypatch):
    ehr = Ehr()
    smart.write_note(setup(monkeypatch, ehr), RECORD, "Dr Who", "a comment")
    [doc] = ehr.created["DocumentReference"]
    assert fp.validate_r4(doc) == []
    assert fp.valid_instant(doc["date"]) and fp.valid_instant(doc["content"][0]["attachment"]["creation"])


def test_audit_event_is_written_when_enabled_and_is_valid_r4(database, monkeypatch):
    ehr = Ehr()
    token = setup(monkeypatch, ehr, audit_event=True)
    result = smart.write_note(token, RECORD, "Dr Who")
    assert result["audit_event"] == {"status": "written", "reference": "AuditEvent/aud-1"}
    [event] = ehr.created["AuditEvent"]
    assert fp.validate_r4(event) == []
    assert event["action"] == "C" and event["outcome"] == "0"
    assert [e["what"]["reference"] for e in event["entity"]] == ["DocumentReference/doc-1", "Patient/p1"]
    assert event["entity"][0]["detail"][0]["valueString"] == RECORD["audit_id"]
    assert [a["requestor"] for a in event["agent"]] == [True, False]


def test_no_patient_details_or_note_text_go_into_provenance_or_audit_event(database, monkeypatch):
    ehr = Ehr()
    record = {**RECORD, "redacted_query": "Zzyzx Surname 943 476 5919",
              "response": {"decision": "answer", "answer_text": "Secret answer text", "claims": [], "sources": []}}
    smart.write_note(setup(monkeypatch, ehr, audit_event=True), record, "Dr Who")
    blob = json.dumps([ehr.created["Provenance"], ehr.created["AuditEvent"]])
    for text in ("Zzyzx", "943 476", "Secret answer", "What is"):
        assert text not in blob


def test_scopes_requested_at_launch_follow_the_flags(database, monkeypatch):
    def fake_discover(issuer, override=None):
        return {"authorization_endpoint": "https://ehr.example.org/auth/authorize", "token_endpoint": "https://ehr.example.org/auth/token"}

    monkeypatch.setattr(smart, "discover", fake_discover)
    monkeypatch.setattr(config, "SMART_CLIENT_ID", "gc")
    monkeypatch.setattr(config, "SMART_ALLOWED_ISSUERS", [SERVER])

    def scope(**flags):
        for name, on in flags.items():
            monkeypatch.setenv(name, "true" if on else "false")
        url, _ = smart.start(SERVER, "launch-token", "https://gc.example.org")
        return dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))["scope"]

    monkeypatch.setattr(config, "SMART_WRITE_NOTES", True)
    plain = scope(SMART_WRITE_PROVENANCE=False, SMART_WRITE_AUDITEVENT=False)
    assert "Provenance" not in plain and "AuditEvent" not in plain and "DocumentReference.write" in plain
    assert "Provenance.write" in scope(SMART_WRITE_PROVENANCE=True, SMART_WRITE_AUDITEVENT=False)
    assert "AuditEvent" not in scope(SMART_WRITE_PROVENANCE=True, SMART_WRITE_AUDITEVENT=False)
    assert "AuditEvent.write" in scope(SMART_WRITE_PROVENANCE=True, SMART_WRITE_AUDITEVENT=True)
    monkeypatch.setattr(config, "SMART_WRITE_NOTES", False)
    assert "Provenance" not in scope(SMART_WRITE_PROVENANCE=True, SMART_WRITE_AUDITEVENT=True)


# --- Failures never break the note, and are audited ------------------------------------------------------

FAILURES = {
    "server error": httpx.Response(500, text="boom"),
    "forbidden": httpx.Response(403),
    "unauthorised": httpx.Response(401),
    "unprocessable": httpx.Response(422, json={"resourceType": "OperationOutcome"}),
    "timeout": httpx.ReadTimeout("slow"),
    "connection refused": httpx.ConnectError("no route"),
    "protocol error": httpx.RemoteProtocolError("bad"),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_a_failed_provenance_write_keeps_the_note_and_is_audited(database, monkeypatch, name):
    ehr = Ehr(fail={"Provenance": FAILURES[name]})
    token = setup(monkeypatch, ehr)
    before = len(failures())
    result = smart.write_note(token, RECORD, "Dr Who")
    assert result["reference"] == "DocumentReference/doc-1" and result["text"]
    assert result["provenance"]["status"] == "failed"
    after = failures()
    assert len(after) == before + 1
    last = after[-1]
    assert last["detail"]["resource"] == "Provenance" and last["detail"]["document"] == "DocumentReference/doc-1"
    assert last["detail"]["audit_id_ref"] == RECORD["audit_id"]
    assert integrity.verify()["ok"]


def test_audit_event_failure_does_not_stop_a_successful_provenance(database, monkeypatch):
    ehr = Ehr(fail={"AuditEvent": httpx.Response(500)})
    result = smart.write_note(setup(monkeypatch, ehr, audit_event=True), RECORD, "Dr Who")
    assert result["provenance"]["status"] == "written" and result["audit_event"]["status"] == "failed"
    assert [f["detail"]["resource"] for f in failures()][-1] == "AuditEvent"


def test_both_failing_is_two_audit_records(database, monkeypatch):
    ehr = Ehr(fail={"AuditEvent": httpx.ConnectError("x"), "Provenance": httpx.ConnectError("x")})
    before = len(failures())
    result = smart.write_note(setup(monkeypatch, ehr, audit_event=True), RECORD, "Dr Who")
    assert result["provenance"]["status"] == result["audit_event"]["status"] == "failed"
    assert len(failures()) == before + 2


def test_a_server_that_returns_no_id_still_counts_as_written(database, monkeypatch):
    class Quiet(Ehr):
        def handler(self, request):
            response = super().handler(request)
            return httpx.Response(201, content=b"") if request.url.path.endswith("Provenance") else response

    ehr = Quiet()
    result = smart.write_note(setup(monkeypatch, ehr), RECORD, "Dr Who")
    assert result["provenance"] == {"status": "written", "reference": None}


def test_a_note_the_ehr_did_not_identify_cannot_be_linked(database, monkeypatch):
    class NoId(Ehr):
        def handler(self, request):
            if request.url.path.endswith("DocumentReference"):
                self.created.setdefault("DocumentReference", []).append(json.loads(request.content))
                return httpx.Response(200, content=b"")
            return super().handler(request)

    ehr = NoId()
    result = smart.write_note(setup(monkeypatch, ehr), RECORD, "Dr Who")
    assert result["reference"] is None
    assert result["provenance"]["status"] == "skipped" and "Provenance" not in ehr.created


def test_missing_write_permission_for_provenance_is_reported_and_audited_without_posting(database, monkeypatch):
    ehr = Ehr()
    token = setup(monkeypatch, ehr, scope="launch patient/DocumentReference.write")
    before = len(failures())
    result = smart.write_note(token, RECORD, "Dr Who")
    assert result["provenance"]["status"] == "not_permitted"
    assert "Provenance" not in ehr.created and len(failures()) == before + 1


@pytest.mark.parametrize("scope,ok", [
    ("patient/Provenance.write", True), ("patient/*.write", True), ("patient/*.*", True), ("system/Provenance.*", True),
    ("patient/Provenance.c", True), ("patient/Provenance.cruds", True), ("patient/*.cu", True), ("user/Provenance.write", True),
    ("patient/Provenance.read", False), ("patient/Provenance.rs", False), ("patient/Observation.write", False),
    ("patient/AuditEvent.write", False), ("Provenance.write", False), ("patient/Provenance.", False), ("", False),
    ("patient/ProvenanceX.write", False), ("patient/Provenance.writeX", False),
])
def test_scope_matching_for_provenance(scope, ok):
    assert smart.scope_allows_write(scope, "Provenance") is ok


def test_a_scope_list_not_recorded_by_an_older_launch_still_tries(database, monkeypatch):
    ehr = Ehr()
    token = setup(monkeypatch, ehr, scope=None)
    assert smart.write_note(token, RECORD, "Dr Who")["provenance"]["status"] == "written"


def test_if_the_audit_trail_itself_is_unavailable_the_note_is_still_returned(database, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("audit down")

    monkeypatch.setattr(integrity, "record_event", broken)
    ehr = Ehr(fail={"Provenance": httpx.Response(500)})
    result = smart.write_note(setup(monkeypatch, ehr), RECORD, "Dr Who")
    assert result["reference"] and result["provenance"]["status"] == "failed"


def test_an_invalid_resource_is_never_sent(database, monkeypatch):
    monkeypatch.setattr(fhir, "build_provenance", lambda *a, **k: {"resourceType": "Provenance", "target": []})
    ehr = Ehr()
    result = smart.write_note(setup(monkeypatch, ehr), RECORD, "Dr Who")
    assert result["provenance"]["status"] == "failed" and "invalid FHIR" in result["provenance"]["reason"]
    assert "Provenance" not in ehr.created and failures()


def test_unexpected_errors_while_building_are_contained(database, monkeypatch):
    def boom(*a, **k):
        raise ValueError("x")

    monkeypatch.setattr(fhir, "build_provenance", boom)
    result = smart.write_note(setup(monkeypatch, Ehr()), RECORD, "Dr Who")
    assert result["reference"] and result["provenance"] == {"status": "failed", "reason": "ValueError"}


def test_notes_still_need_the_existing_permissions_with_provenance_on(database, monkeypatch):
    monkeypatch.setattr(config, "SMART_WRITE_NOTES", True)
    monkeypatch.setenv("SMART_WRITE_PROVENANCE", "true")
    with pytest.raises(smart.SmartError, match="SMART launch"):
        smart.write_note(None, RECORD, "Dr Who")


# --- Builders and the R4 structure check ----------------------------------------------------------------

def test_builders_make_valid_resources_for_awkward_inputs():
    for reviewer in ("", "A" * 5000, "Dr Ünïcödé 医生", "O'Brien <b>x</b>"):
        for when in (None, datetime(2026, 11, 24, 10, 0, 0), datetime(2026, 11, 24, 10, 0, 0, 123456, tzinfo=timezone(timedelta(hours=-5)))):
            prov = fhir.build_provenance("DocumentReference/1", "abc", reviewer, when)
            event = fhir.build_audit_event("DocumentReference/1", "Patient/p", "abc", reviewer, "0", when)
            assert fp.validate_r4(prov) == [] and fp.validate_r4(event) == []
            assert len(prov["agent"][1]["who"]["display"]) <= 200
    assert fhir.instant(datetime(2026, 11, 24, 10, 0, 0)) == "2026-11-24T10:00:00Z"
    assert fhir.instant(datetime(2026, 11, 24, 5, 0, 0, tzinfo=timezone(timedelta(hours=-5)))) == "2026-11-24T10:00:00Z"


def test_an_unknown_outcome_becomes_a_serious_failure_not_success():
    assert fhir.build_audit_event("D/1", "Patient/p", "a", "r", outcome="success")["outcome"] == "8"
    assert fhir.build_audit_event("D/1", "Patient/p", "a", "r", outcome="4")["outcome"] == "4"


def good_provenance():
    return fhir.build_provenance("DocumentReference/1", "abc", "Dr Who", datetime(2026, 11, 24, tzinfo=timezone.utc))


def good_event():
    return fhir.build_audit_event("DocumentReference/1", "Patient/p", "abc", "Dr Who", "0", datetime(2026, 11, 24, tzinfo=timezone.utc))


def drop(*path):
    def f(r):
        node = r
        for p in path[:-1]:
            node = node[p]
        del node[path[-1]]
    return f


def put(value, *path):
    def f(r):
        node = r
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = value
    return f


PROVENANCE_BAD = {
    "no target": (drop("target"), "target: is required"),
    "empty target": (put([], "target"), "target: is required (1..*) but empty"),
    "target not array": (put({"reference": "A/1"}, "target"), "must be an array"),
    "target without reference": (put([{"display": "x"}], "target"), "must point at the resource"),
    "empty reference": (put([{}], "target"), "needs a reference"),
    "no recorded": (drop("recorded"), "recorded: is required"),
    "recorded date only": (put("2026-11-24", "recorded"), "valid instant"),
    "recorded no zone": (put("2026-11-24T10:00:00", "recorded"), "valid instant"),
    "recorded number": (put(1700000000, "recorded"), "valid instant"),
    "no agent": (drop("agent"), "agent: is required"),
    "empty agent": (put([], "agent"), "agent: is required (1..*) but empty"),
    "agent without who": (drop("agent", 0, "who"), "agent[0].who: is required"),
    "agent who empty": (put({}, "agent", 0, "who"), "needs a reference, identifier or display"),
    "agent type string": (put("author", "agent", 0, "type"), "CodeableConcept"),
    "agent role not array": (put({"text": "x"}, "agent", 0, "role"), "must be an array"),
    "entity without what": (drop("entity", 0, "what"), "entity[0].what: is required"),
    "entity bad role": (put("whatever", "entity", 0, "role"), "isn't one of"),
    "entity without role": (drop("entity", 0, "role"), "entity[0].role: is required"),
    "unknown element": (put("x", "madeUp"), "madeUp"),
    "activity array": (put([{"text": "x"}], "activity"), "may occur only once"),
    "bad id": (put("has space", "id"), "valid id"),
    "contained not array": (put({}, "contained"), "must be an array"),
}


@pytest.mark.parametrize("name", list(PROVENANCE_BAD))
def test_malformed_provenance_is_rejected(name):
    mutate, expected = PROVENANCE_BAD[name]
    resource = good_provenance()
    mutate(resource)
    errors = fp.validate_r4(resource)
    assert any(expected in e for e in errors), errors


AUDIT_BAD = {
    "no type": (drop("type"), "type: is required"),
    "type string": (put("rest", "type"), "Coding"),
    "type without code": (put({"system": "x"}, "type"), "needs a code or display"),
    "subtype not array": (put({"code": "create"}, "subtype"), "must be an array"),
    "bad action": (put("X", "action"), "isn't one of"),
    "lowercase action": (put("c", "action"), "isn't one of"),
    "no recorded": (drop("recorded"), "recorded: is required"),
    "recorded no zone": (put("2026-11-24T10:00:00", "recorded"), "valid instant"),
    "outcome int": (put(0, "outcome"), "non-empty code"),
    "bad outcome": (put("2", "outcome"), "isn't one of"),
    "no agent": (drop("agent"), "agent: is required"),
    "requestor missing": (drop("agent", 0, "requestor"), "requestor: is required"),
    "requestor string": (put("true", "agent", 0, "requestor"), "true or false"),
    "agent identifies nobody": (lambda r: (r["agent"][0].pop("who"),), "needs who, altId or name"),
    "bad network type": (put({"address": "1.2.3.4", "type": "9"}, "agent", 0, "network"), "isn't one of"),
    "no source": (drop("source"), "source: is required"),
    "source without observer": (drop("source", "observer"), "source.observer: is required"),
    "source array": (put([{"observer": {"display": "x"}}], "source"), "may occur only once"),
    "entity detail without type": (drop("entity", 0, "detail", 0, "type"), "type: is required"),
    "entity what empty": (put({}, "entity", 0, "what"), "needs a reference"),
    "entity not array": (put({"what": {"display": "x"}}, "entity"), "must be an array"),
    "unknown element": (put(1, "unknownThing"), "unknownThing"),
}


@pytest.mark.parametrize("name", list(AUDIT_BAD))
def test_malformed_audit_events_are_rejected(name):
    mutate, expected = AUDIT_BAD[name]
    resource = good_event()
    mutate(resource)
    errors = fp.validate_r4(resource)
    assert any(expected in e for e in errors), errors


DOC_BAD = {
    "no status": (drop("status"), "status: is required"),
    "bad status": (put("final", "status"), "isn't one of"),
    "bad docStatus": (put("draft", "docStatus"), "isn't one of"),
    "no content": (drop("content"), "content: is required"),
    "empty content": (put([], "content"), "required (1..*) but empty"),
    "attachment missing": (put([{}], "content"), "attachment: is required"),
    "bad mime": (put([{"attachment": {"contentType": "text"}}], "content"), "MIME"),
    "bad base64": (put([{"attachment": {"contentType": "text/plain", "data": "***"}}], "content"), "base64"),
    "bad date": (put("2026-11-24", "date"), "valid instant"),
    "author not array": (put({"display": "x"}, "author"), "must be an array"),
}


@pytest.mark.parametrize("name", list(DOC_BAD))
def test_malformed_document_references_are_rejected(name):
    document = {"resourceType": "DocumentReference", "status": "current", "docStatus": "preliminary",
                "subject": {"reference": "Patient/p1"}, "date": "2026-11-24T10:00:00Z", "author": [{"display": "x"}],
                "content": [{"attachment": {"contentType": "text/plain", "data": "aGk="}}]}
    assert fp.validate_r4(document) == []
    mutate, expected = DOC_BAD[name]
    mutate(document)
    assert any(expected in e for e in fp.validate_r4(document)), fp.validate_r4(document)


def test_validate_r4_handles_junk_and_hostile_input():
    for junk in (None, [], "x", 5, {}, {"resourceType": "Observation"}, {"resourceType": ["Provenance"]}):
        assert fp.validate_r4(junk)
    deep = good_provenance()
    node = deep
    for _ in range(100):
        node["madeUp"] = {}
        node = node["madeUp"]
    assert "deep" in fp.validate_r4(deep)[0]
    huge = good_provenance()
    huge["agent"][0]["who"]["display"] = "x" * 2_000_000
    assert "1 MB" in fp.validate_r4(huge)[0]


def test_good_resources_pass_untouched():
    assert fp.validate_r4(good_provenance()) == [] and fp.validate_r4(good_event()) == []
    before = copy.deepcopy(good_event())
    fp.validate_r4(before)
    assert before == good_event()


def test_the_notes_route_reports_how_provenance_and_the_audit_event_went(database, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    audit_id = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?"}).json()["audit_id"]
    outcomes = [
        {"reference": "DocumentReference/doc-1"},
        {"reference": "DocumentReference/doc-1", "provenance": {"status": "written", "reference": "Provenance/pro-1"}},
        {"reference": "DocumentReference/doc-1", "provenance": {"status": "written", "reference": "Provenance/pro-1"},
         "audit_event": {"status": "failed", "detail": "the EHR answered 403"}},
    ]
    for result in outcomes:
        monkeypatch.setattr(smart, "write_note", lambda *a, _r=result, **k: dict(_r, text="note"))
        saved = client.post("/api/ehr/notes", json={"audit_id": audit_id})
        assert saved.status_code == 200 and saved.json() == result
