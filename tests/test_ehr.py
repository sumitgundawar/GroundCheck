"""EHR integration: mapping FHIR to a patient, the FHIR client's safety
rules, the SMART on FHIR launch against a fake EHR, and the CDS Hooks service
with signed requests. Live tests against public sandboxes run when
RUN_LIVE_EHR=1."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import cds_hooks, config, fhir, patient_checks, smart  # noqa: E402

EHR = "https://ehr.example.org/fhir"
RECENT = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()


def obs(code, value=None, unit=None, when=RECENT, concept=None, status="final", oid=None):
    resource = {"resourceType": "Observation", "id": oid or code, "status": status,
                "code": {"coding": [{"system": fhir.LOINC, "code": code}]}, "effectiveDateTime": when}
    if concept:
        resource["valueCodeableConcept"] = {"coding": [{"system": fhir.SNOMED, "code": concept}]}
    else:
        resource["valueQuantity"] = {"value": value, "unit": unit}
    return resource


PATIENT = {"resourceType": "Patient", "id": "p1", "birthDate": "1958-03-14", "gender": "female",
           "name": [{"family": "Doe", "given": ["Jane"]}], "identifier": [{"value": "943 476 5919"}],
           "telecom": [{"value": "07700 900123"}]}
OBSERVATIONS = [
    obs("62238-1", 38, "mL/min/1.73m2", oid="egfr-new"),
    obs("62238-1", 75, "mL/min/1.73m2", when="2020-01-01T00:00:00Z", oid="egfr-old"),
    obs("2160-0", 1.4, "mg/dL"),
    obs("29463-7", 154, "[lb_av]"),
    obs("2823-3", 5.8, "mmol/L"),
    obs("777-3", 250000, "/uL"),
    obs("6301-6", 1.1, None, when="2019-05-01T00:00:00Z"),
    obs("1742-6", 30, "furlongs"),
    obs("2951-2", 140, "mmol/L", status="entered-in-error"),
]
ALLERGIES = [
    {"resourceType": "AllergyIntolerance", "code": {"text": "Quorl"}, "clinicalStatus": {"coding": [{"code": "active"}]}},
    {"resourceType": "AllergyIntolerance", "code": {"coding": [{"display": "Penicillin"}]},
     "clinicalStatus": {"coding": [{"code": "inactive"}]}},
    {"resourceType": "AllergyIntolerance", "code": {"text": "Latex"}, "verificationStatus": {"coding": [{"code": "refuted"}]}},
]
MEDICATIONS = [
    {"resourceType": "MedicationRequest", "status": "active", "intent": "order",
     "medicationCodeableConcept": {"text": "Tessorin 10 mg tablet"}},
    {"resourceType": "MedicationRequest", "status": "stopped", "intent": "order", "medicationCodeableConcept": {"text": "Old"}},
    {"resourceType": "MedicationRequest", "status": "active", "intent": "order", "contained": [
        {"resourceType": "Medication", "id": "m1", "code": {"text": "Mendel solution 5 mL"}}],
     "medicationReference": {"reference": "#m1"}},
]
CONDITIONS = [
    {"resourceType": "Condition", "code": {"text": "Chronic kidney disease, stage 3"}, "clinicalStatus": {"coding": [{"code": "active"}]}},
    {"resourceType": "Condition", "code": {"text": "Resolved thing"}, "clinicalStatus": {"coding": [{"code": "resolved"}]}},
]


# --- Mapping ------------------------------------------------------------------------------

def test_fhir_resources_map_to_a_patient_without_identifiers():
    loaded = fhir.map_patient(PATIENT, OBSERVATIONS, ALLERGIES, MEDICATIONS, CONDITIONS, {"system": "test"},
                              today=date(2026, 9, 17))
    c = loaded.context
    assert (c.age_years, c.sex) == (68, "female")
    assert c.egfr == 38                                   # the newest result wins
    assert c.creatinine_umol_l == pytest.approx(123.8, abs=0.1)
    assert c.weight_kg == pytest.approx(69.9, abs=0.1)
    assert c.labs == {"potassium": 5.8, "platelets": 250.0, "inr": 1.1}
    assert c.allergies == ["Quorl"]
    assert c.medicines == ["Tessorin 10 mg tablet", "Mendel solution 5 mL"]
    assert c.conditions == ["Chronic kidney disease, stage 3"]
    dumped = json.dumps(c.model_dump()) + json.dumps(loaded.source)
    for identifier in ("Doe", "Jane", "943 476 5919", "07700"):
        assert identifier not in dumped
    assert any("inr is from 2019" in w for w in loaded.warnings)
    assert any("alt result used units" in w for w in loaded.warnings)


def test_implausible_results_are_left_out_not_fatal():
    loaded = fhir.map_patient({"resourceType": "Patient"}, [obs("2160-0", 30000, "umol/L"), obs("29463-7", 70, "kg")],
                              [], [], [], {})
    assert loaded.context.creatinine_umol_l is None and loaded.context.weight_kg == 70
    assert any("isn't a plausible value" in w for w in loaded.warnings)


def test_age_pregnancy_and_gender_details():
    baby = fhir.map_patient({"resourceType": "Patient", "birthDate": "2026-03-17", "gender": "male"},
                            [obs("82810-3", concept=fhir.PREGNANT)], [], [], [], {}, today=date(2026, 9, 17))
    assert baby.context.age_years == pytest.approx(0.5, abs=0.01) and baby.context.pregnant is None
    pregnant = fhir.map_patient({"resourceType": "Patient", "gender": "female"}, [obs("82810-3", concept=fhir.PREGNANT)],
                                [], [], [], {})
    assert pregnant.context.pregnant is True


def test_the_client_follows_pages_on_the_same_server_only():
    def handler(request):
        if "page=2" in str(request.url):
            return httpx.Response(200, json={"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Condition", "id": "b"}}]})
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Condition", "id": "a"}}],
                                         "link": [{"relation": "next", "url": f"{EHR}?page=2"}]})
    with fhir.FhirClient(EHR, transport=httpx.MockTransport(handler)) as client:
        assert [r["id"] for r in client.search("Condition", {})] == ["a", "b"]

    def evil(request):
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [],
                                         "link": [{"relation": "next", "url": "https://evil.example/steal"}]})
    with fhir.FhirClient(EHR, transport=httpx.MockTransport(evil)) as client:
        with pytest.raises(fhir.FhirError, match="another site"):
            client.search("Condition", {})


def test_server_urls_are_allow_listed_and_secure():
    with pytest.raises(fhir.FhirError, match="HTTPS"):
        fhir.validate_base_url("http://ehr.example.org/fhir", ["http://ehr.example.org/fhir"])
    with pytest.raises(fhir.FhirError, match="allowed list"):
        fhir.validate_base_url("https://other.example/fhir", [EHR])
    with pytest.raises(fhir.FhirError, match="allowed list"):
        fhir.validate_base_url("https://ehr.example.org/fhirevil", [EHR])
    assert fhir.validate_base_url(EHR + "/", [EHR]) == EHR


# --- A fake EHR: FHIR server and SMART authorisation ---------------------------------------

class FakeEhr:
    def __init__(self):
        self.codes: dict[str, dict] = {}
        self.tokens: set[str] = set()
        self.patient_in_token = True
        self.notes: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        path = request.url.path
        if path.endswith("/.well-known/smart-configuration"):
            return httpx.Response(200, json={"authorization_endpoint": "https://ehr.example.org/auth/authorize",
                                             "token_endpoint": "https://ehr.example.org/auth/token",
                                             "capabilities": ["launch-ehr", "client-public"]})
        if path == "/auth/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            grant = self.codes.pop(form.get("code", ""), None)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(form.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
            if grant is None or challenge != grant["challenge"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            token = secrets.token_urlsafe(16)
            self.tokens.add(token)
            body = {"access_token": token, "token_type": "Bearer", "scope": grant["scope"], "expires_in": 3600}
            if self.patient_in_token:
                body["patient"] = "p1"
            return httpx.Response(200, json=body)
        if path == "/fhir/DocumentReference" and request.method == "POST":
            if request.headers.get("authorization", "").removeprefix("Bearer ") not in self.tokens:
                return httpx.Response(401)
            self.notes.append(json.loads(request.content))
            return httpx.Response(201, headers={"Location": f"{EHR}/DocumentReference/note-{len(self.notes)}/_history/1"})
        if path.startswith("/fhir/"):
            if request.headers.get("authorization", "").removeprefix("Bearer ") not in self.tokens and "open" not in url:
                return httpx.Response(401)
            resource = path.removeprefix("/fhir/")
            bundle = lambda items: {"resourceType": "Bundle", "entry": [{"resource": r} for r in items]}
            if resource == "Patient/p1":
                return httpx.Response(200, json=PATIENT)
            data = {"Observation": OBSERVATIONS, "AllergyIntolerance": ALLERGIES, "MedicationRequest": MEDICATIONS,
                    "MedicationStatement": [], "Condition": CONDITIONS}.get(resource)
            return httpx.Response(200, json=bundle(data)) if data is not None else httpx.Response(404)
        return httpx.Response(404)

    def authorize(self, url: str) -> tuple[str, str]:
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        code = secrets.token_urlsafe(12)
        self.codes[code] = {"challenge": q["code_challenge"], "scope": q["scope"]}
        return code, q["state"]


@pytest.fixture()
def ehr(database, monkeypatch):
    fake = FakeEhr()
    monkeypatch.setattr(smart, "transport", httpx.MockTransport(fake.handler))
    monkeypatch.setattr(config, "SMART_CLIENT_ID", "groundcheck")
    monkeypatch.setattr(config, "SMART_ALLOWED_ISSUERS", [EHR])
    monkeypatch.setattr(config, "FHIR_OPEN_SERVERS", [])
    from app.main import app
    with TestClient(app) as client:
        yield fake, client


def test_an_ehr_launch_loads_the_patient_for_this_browser(ehr):
    fake, client = ehr
    start = client.get(f"/api/ehr/launch?iss={EHR}&launch=abc123", follow_redirects=False)
    assert start.status_code == 302
    params = {k: v[0] for k, v in parse_qs(urlparse(start.headers["location"]).query).items()}
    assert params["aud"] == EHR and params["launch"] == "abc123" and params["code_challenge_method"] == "S256"
    assert "launch" in params["scope"].split()
    code, state = fake.authorize(start.headers["location"])
    done = client.get(f"/api/ehr/callback?code={code}&state={state}", follow_redirects=False)
    assert done.status_code == 200 and "ehr_error" not in done.text and "gc_ehr=" in done.headers["set-cookie"]

    context = client.get("/api/ehr/context").json()["context"]
    assert context["patient"]["egfr"] == 38 and context["patient"]["allergies"] == ["Quorl"]
    assert context["source"] == {**context["source"], "system": "SMART on FHIR", "server": EHR, "patient": "Patient/p1"}
    assert "Doe" not in json.dumps(context)

    # Another browser can't see it, and it can be cleared.
    assert TestClient(client.app).get("/api/ehr/context").json()["context"] is None
    client.delete("/api/ehr/context")
    assert client.get("/api/ehr/context").json()["context"] is None


def test_launches_are_refused_when_unsafe(ehr, monkeypatch):
    fake, client = ehr
    evil = client.get("/api/ehr/launch?iss=https://evil.example/fhir", follow_redirects=False)
    assert "isn%E2%80%99t%20allowed" in evil.text or "allowed" in unquote(evil.text)

    start = client.get(f"/api/ehr/launch?iss={EHR}", follow_redirects=False)
    code, state = fake.authorize(start.headers["location"])
    other = TestClient(client.app)
    assert "didn't start in this browser" in unquote(other.get(f"/api/ehr/callback?code={code}&state={state}",
                                                              follow_redirects=False).text)
    wrong_verifier = client.get(f"/api/ehr/callback?code=forged&state={state}", follow_redirects=False)
    assert "didn't accept the launch" in unquote(wrong_verifier.text) or "didn't start" in unquote(wrong_verifier.text)

    fake.patient_in_token = False
    start = client.get(f"/api/ehr/launch?iss={EHR}", follow_redirects=False)
    code, state = fake.authorize(start.headers["location"])
    no_patient = client.get(f"/api/ehr/callback?code={code}&state={state}", follow_redirects=False)
    assert "which patient" in unquote(no_patient.text)


def test_open_servers_load_directly_and_others_are_refused(ehr, monkeypatch):
    fake, client = ehr
    open_base = "https://ehr.example.org/fhir/open"
    refused = client.post("/api/ehr/load", json={"server": open_base, "patient_id": "p1"})
    assert refused.status_code == 400 and "allowed list" in refused.json()["detail"]
    assert client.post("/api/ehr/load", json={"server": EHR, "patient_id": "../Patient/p1"}).status_code == 400


def test_a_reviewed_answer_is_saved_to_the_record(ehr, monkeypatch):
    fake, client = ehr
    monkeypatch.setattr(config, "SMART_WRITE_NOTES", True)
    early = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?",
                                          "patient": {"age_years": 60, "weight_kg": 70, "egfr": 90}}).json()
    start = client.get(f"/api/ehr/launch?iss={EHR}&launch=l1", follow_redirects=False)
    assert "DocumentReference.write" in parse_qs(urlparse(start.headers["location"]).query)["scope"][0]
    code, state = fake.authorize(start.headers["location"])
    client.get(f"/api/ehr/callback?code={code}&state={state}", follow_redirects=False)
    context = client.get("/api/ehr/context").json()["context"]
    assert context["can_write_notes"] is True and "access" not in context and "token" not in json.dumps(context)

    too_early = client.post("/api/ehr/notes", json={"audit_id": early["audit_id"]})
    assert too_early.status_code == 400 and "before this patient" in too_early.json()["detail"]
    no_patient = client.post("/api/ask", json={"query": "How is Veltris syndrome treated?"}).json()
    assert "asked for this patient" in client.post("/api/ehr/notes", json={"audit_id": no_patient["audit_id"]}).json()["detail"]

    asked = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?",
                                          "patient": context["patient"]}).json()
    saved = client.post("/api/ehr/notes", json={"audit_id": asked["audit_id"], "comment": "Agree, check eGFR weekly."})
    assert saved.status_code == 200 and saved.json()["reference"] == "DocumentReference/note-1"
    [note] = fake.notes
    assert note["docStatus"] == "preliminary" and note["subject"] == {"reference": "Patient/p1"}
    text = base64.b64decode(note["content"][0]["attachment"]["data"]).decode()
    assert "Question: What is the standard dose of Caloradine?" in text and "Agree, check eGFR weekly." in text
    assert f"Audit record {asked['audit_id']}" in text and "reviewed by a clinician" in text
    assert client.post("/api/ehr/notes", json={"audit_id": "missing"}).status_code == 404

    client.delete("/api/ehr/context")
    assert "SMART launch" in client.post("/api/ehr/notes", json={"audit_id": asked["audit_id"]}).json()["detail"]


# --- CDS Hooks ---------------------------------------------------------------------------------

def order_request(dose=15, allergies=(), egfr=90, medicine="Caloradine", selections=None, hook="order-sign"):
    observations = [obs("62238-1", egfr, "mL/min/1.73m2")] if egfr is not None else []
    return {
        "hook": hook, "hookInstance": "d1577c69-dfbe-44ad-ba6d-3e05e953b2ea",
        "context": {"userId": "Practitioner/1", "patientId": "p1", "selections": selections or [], "draftOrders": {
            "resourceType": "Bundle", "entry": [{"resource": {
                "resourceType": "MedicationRequest", "id": "mr1", "status": "draft", "intent": "order",
                "medicationCodeableConcept": {"text": medicine},
                "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": dose, "unit": "mg"}}],
                                       "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}}]}}]}},
        "prefetch": {
            "patient": {"resourceType": "Patient", "id": "p1", "birthDate": "1970-01-01", "gender": "male"},
            "observations": {"resourceType": "Bundle", "entry": [{"resource": o} for o in observations]},
            "allergies": {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "AllergyIntolerance",
                                                                            "code": {"text": a}}} for a in allergies]},
            "medications": {"resourceType": "Bundle", "entry": []},
            "conditions": {"resourceType": "Bundle", "entry": []},
        },
    }


@pytest.fixture()
def cds(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())), "kid": "ehr-1", "alg": "RS384"}
    monkeypatch.setattr(cds_hooks, "transport", httpx.MockTransport(lambda r: httpx.Response(200, json={"keys": [jwk]})))
    monkeypatch.setattr(config, "CDS_HOOKS_TRUSTED", {"https://ehr.example.org": "https://ehr.example.org/jwks"})
    monkeypatch.setattr(config, "CDS_HOOKS_ALLOW_UNSIGNED", False)
    from app.main import app

    def token(audience, **overrides):
        now = int(time.time())
        claims = {"iss": "https://ehr.example.org", "aud": audience, "exp": now + 300, "iat": now,
                  "jti": secrets.token_hex(8), **overrides}
        return jwt.encode(claims, key, algorithm="RS384", headers={"kid": "ehr-1", "jku": "https://ehr.example.org/jwks"})

    with TestClient(app) as client:
        yield client, token


SERVICE = "/cds-services/groundcheck-medication-safety-order-sign"
AUD = "http://testserver" + SERVICE


def test_discovery_lists_both_hooks_with_prefetch(cds):
    client, _ = cds
    response = client.get("/cds-services")
    services = response.json()["services"]
    assert {s["hook"] for s in services} == {"order-select", "order-sign"}
    assert services[0]["prefetch"]["patient"] == "Patient/{{context.patientId}}"
    assert response.headers["access-control-allow-origin"] == "*"
    assert client.options(SERVICE).status_code == 204


def test_calls_must_be_signed_by_a_trusted_ehr(cds, monkeypatch):
    client, token = cds
    assert client.post(SERVICE, json=order_request()).status_code == 401
    assert client.post(SERVICE, json=order_request(), headers={"Authorization": f"Bearer {token('https://wrong.example')}"}).status_code == 401
    assert client.post(SERVICE, json=order_request(), headers={"Authorization": f"Bearer {token(AUD, iss='https://evil.example')}"}).status_code == 401
    expired = token(AUD, exp=int(time.time()) - 120, iat=int(time.time()) - 400)
    assert client.post(SERVICE, json=order_request(), headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    ok = client.post(SERVICE, json=order_request(), headers={"Authorization": f"Bearer {token(AUD)}"})
    assert ok.status_code == 200 and ok.json()["cards"] == []
    monkeypatch.setattr(config, "CDS_HOOKS_ALLOW_UNSIGNED", True)
    assert client.post(SERVICE, json=order_request()).status_code == 200


def test_cards_for_unsafe_orders(cds, monkeypatch):
    client, token = cds
    headers = lambda: {"Authorization": f"Bearer {token(AUD)}"}
    renal = client.post(SERVICE, json=order_request(egfr=40), headers=headers()).json()["cards"]
    assert renal[0]["indicator"] == "critical" and "7.5 mg once daily" in renal[0]["summary"]
    assert "order's 15 mg" in renal[0]["summary"] and "CALO-RENAL-2" in renal[0]["source"]["label"]
    assert len(renal[0]["summary"]) <= 140 and renal[0]["overrideReasons"]

    allergy = client.post(SERVICE, json=order_request(allergies=["Quorl sensitivity"]), headers=headers()).json()["cards"]
    assert allergy[0]["indicator"] == "critical" and allergy[0]["source"]["topic"]["code"] == "allergy"

    missing = client.post(SERVICE, json=order_request(egfr=None), headers=headers()).json()["cards"]
    assert missing[0]["source"]["topic"]["code"] == "missing_data"

    # A medicine the formulary has never heard of must say so. Returning no
    # cards would read, in an ordering screen, as "checked, no concerns".
    unknown = client.post(SERVICE, json=order_request(medicine="Paracetamol"), headers=headers()).json()
    assert len(unknown["cards"]) == 1
    card = unknown["cards"][0]
    assert card["source"]["topic"]["code"] == "not_in_formulary"
    assert card["indicator"] == "warning"
    assert "Paracetamol" in card["summary"] and "not in the formulary" in card["summary"]
    assert "not a statement that the order is safe" in card["detail"]

    select = "/cds-services/groundcheck-medication-safety-order-select"
    body = order_request(egfr=40, hook="order-select", selections=["MedicationRequest/other"])
    assert client.post(select, json=body, headers={"Authorization": f"Bearer {token('http://testserver' + select)}"}).json() == {"cards": []}

    no_patient = order_request()
    del no_patient["prefetch"]["patient"]
    assert client.post(SERVICE, json=no_patient, headers=headers()).status_code == 412
    assert client.post("/cds-services/nope", json={}, headers=headers()).status_code == 404
    feedback = client.post(SERVICE + "/feedback", json={"feedback": [{"card": "x", "outcome": "overridden"}]},
                           headers={"Authorization": f"Bearer {token(AUD + '/feedback')}"})
    assert feedback.status_code == 200


# --- Live sandboxes ------------------------------------------------------------------------------

live = pytest.mark.skipif(os.environ.get("RUN_LIVE_EHR") != "1", reason="set RUN_LIVE_EHR=1 to use public sandboxes")
HAPI = "https://hapi.fhir.org/baseR4"
LAUNCHER = "https://launch.smarthealthit.org/v/r4"


@live
def test_live_hapi_patient_loads():
    with fhir.FhirClient(HAPI) as client:
        observations = client.search("Observation", {"code": f"{fhir.LOINC}|2160-0", "_count": 20}, max_pages=1)
        ids = [o["subject"]["reference"].split("/")[1] for o in observations
               if o.get("subject", {}).get("reference", "").startswith("Patient/")]
        loaded = [fhir.load_patient(client, pid, {"server": HAPI}) for pid in ids[:3]]
    assert any(p.context.creatinine_umol_l for p in loaded)


def _launcher_patient() -> str:
    with httpx.Client(timeout=30) as client:
        bundle = client.get(f"{LAUNCHER}/fhir/Observation", params={"code": f"{fhir.LOINC}|2160-0", "_count": 5},
                            headers={"Accept": "application/fhir+json"}).json()
    return bundle["entry"][0]["resource"]["subject"]["reference"].split("/")[1]


@live
def test_live_smart_launch_against_the_smart_health_it_sandbox(database, monkeypatch):
    patient = _launcher_patient()
    # The launcher's simulated patient-standalone launch, with login and
    # authorisation screens skipped (its launch parameters, base64url-encoded).
    params = [3, patient, "", "AUTO", 1, 1, 0, "", "", "", "", "", "", "", 0, 1, ""]
    encoded = base64.urlsafe_b64encode(json.dumps(params).encode()).decode().rstrip("=")
    issuer = f"{LAUNCHER}/sim/{encoded}/fhir"
    monkeypatch.setattr(config, "SMART_CLIENT_ID", "groundcheck-test")
    monkeypatch.setattr(config, "SMART_ALLOWED_ISSUERS", [f"{LAUNCHER}/sim"])
    monkeypatch.setattr(config, "SMART_WRITE_NOTES", True)
    monkeypatch.setattr(smart, "transport", None)
    from app.main import app
    with TestClient(app) as client:
        start = client.get(f"/api/ehr/launch?iss={issuer}", follow_redirects=False)
        assert start.status_code == 302, start.text
        with httpx.Client(timeout=30, follow_redirects=False) as browser:
            url = start.headers["location"]
            for _ in range(8):
                hop = browser.get(url)
                url = hop.headers.get("location", "")
                if url.startswith("http://testserver/api/ehr/callback"):
                    break
                assert hop.status_code in (301, 302, 303, 307), hop.text[:300]
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        done = client.get(f"/api/ehr/callback?code={q['code']}&state={q['state']}", follow_redirects=False)
        assert "ehr_error" not in done.text, unquote(done.text)
        context = client.get("/api/ehr/context").json()["context"]
        assert context["source"]["patient"] == f"Patient/{patient}"
        assert context["patient"].get("age_years") is not None and context["can_write_notes"]
        asked = client.post("/api/ask", json={"query": "How is Veltris syndrome treated?",
                                              "patient": context["patient"]}).json()
        saved = client.post("/api/ehr/notes", json={"audit_id": asked["audit_id"], "comment": "Automated sandbox test."})
    assert saved.status_code == 200, saved.text
    assert saved.json()["reference"].startswith("DocumentReference/")


def test_lab_units_are_converted_not_taken_at_face_value():
    """A creatinine of 1.2 mg/dL is 106 umol/L, and the kidney calculation
    works in umol/L. Reading the number without its unit would put this
    patient two dose tiers away from where they belong."""
    def obs(value, unit):
        return {"valueQuantity": {"value": value, "unit": unit}}

    assert fhir._quantity("creatinine", obs(1.2, "mg/dL")) == 106.1
    assert fhir._quantity("creatinine", obs(106.1, "umol/L")) == 106.1
    assert fhir._quantity("creatinine", obs(106.1, "µmol/L")) == 106.1
    assert fhir._quantity("creatinine", obs(0.106, "mmol/L")) == 106.0
    # A unit nothing can be made of is left out rather than guessed at.
    assert fhir._quantity("creatinine", obs(1.2, "mg/L")) is None

    assert fhir._quantity("weight", obs(70, "kg")) == 70.0
    assert fhir._quantity("weight", obs(154, "[lb_av]")) == 69.9
    assert fhir._quantity("weight", obs(70000, "g")) == 70.0
    assert fhir._quantity("weight", obs(70, "stone")) is None


def test_the_ehr_is_told_the_formulary_is_invented():
    """An EHR decides whether to install a service from its description. One
    that advertises allergy and interaction checking, without saying its
    formulary knows no real medicine, is how this ends up in an ordering
    screen."""
    service = cds_hooks.discovery()["services"][0]
    assert "demonstration" in service["title"].lower()
    assert service["description"].startswith("DEMONSTRATION ONLY")
    assert "synthetic" in service["usageRequirements"].lower()
    assert "not checked" in service["description"]

    import app.formulary as formulary_module

    real = formulary_module.load().model_copy(update={"synthetic": False})
    original = formulary_module.load
    formulary_module.load = lambda: real
    try:
        live = cds_hooks.discovery()["services"][0]
    finally:
        formulary_module.load = original
    assert "DEMONSTRATION ONLY" not in live["description"]
    assert "demonstration" not in live["title"].lower()


def test_a_very_large_dose_is_not_read_as_a_small_one():
    """The order's dose was formatted with "%g", which switches to exponential
    notation at a million: 1000000 became "1e+06", and the dose parser then
    read that as 6. A million-milligram order produced no card at all."""
    assert cds_hooks._plain(1000000.0) == "1000000"
    assert cds_hooks._plain(123456.7) == "123456.7"       # and no rounding to 6 figures
    assert cds_hooks._plain(15.0) == "15" and cds_hooks._plain(2.5) == "2.5"

    text, amount, unit = cds_hooks._order_dose(
        {"dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": 1000000, "unit": "mg"}}],
                                "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}}]})
    assert "1e+06" not in text and text.startswith("1000000 mg")
    assert patient_checks._VALUE.findall(text) == [("1000000", "mg")]
