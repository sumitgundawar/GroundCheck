"""CDS Hooks 2.0 conformance (https://cds-hooks.hl7.org/2.0/).

Each test names the specification behaviour it covers in its docstring.
Covered: discovery (GET /cds-services, services with hook, id, description,
prefetch), service request shape (hook, hookInstance UUID, context, fhirServer,
fhirAuthorization, prefetch), response shape (cards, uuid, summary <= 140,
indicator, source.label, overrideReasons, links), missing prefetch (FHIR
fallback, or a clear card or error), JWT authentication (iss, aud, exp, iat,
jti, signature, algorithm), the feedback endpoint, and error status codes.
No network."""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import cds_hooks, config, fhir  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-045")

ISS = "https://ehr.example.org"
SERVICE_ID = "groundcheck-medication-safety-order-sign"
SELECT_ID = "groundcheck-medication-safety-order-select"
SERVICE, SELECT = f"/cds-services/{SERVICE_ID}", f"/cds-services/{SELECT_ID}"
AUD = "http://testserver" + SERVICE
HOOK_INSTANCE = "d1577c69-dfbe-44ad-ba6d-3e05e953b2ea"
CARD_KEYS = {"uuid", "summary", "detail", "indicator", "source", "suggestions", "selectionBehavior", "overrideReasons", "links"}


def obs(code, value, unit, when=None, oid=None):
    from datetime import datetime, timedelta, timezone

    when = when or (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    return {"resourceType": "Observation", "id": oid or code, "status": "final",
            "code": {"coding": [{"system": fhir.LOINC, "code": code}]}, "effectiveDateTime": when,
            "valueQuantity": {"value": value, "unit": unit}}


def draft(medicine="Caloradine", dose=15, oid="mr1", **extra):
    return {"resourceType": "MedicationRequest", "id": oid, "status": "draft", "intent": "order",
            "medicationCodeableConcept": {"text": medicine},
            "dosageInstruction": [{"doseAndRate": [{"doseQuantity": {"value": dose, "unit": "mg"}}],
                                   "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}}}], **extra}


def bundle(*resources):
    return {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": r} for r in resources]}


def request(orders=None, egfr=90, allergies=(), hook="order-sign", patient_id="p1", **over):
    orders = [draft()] if orders is None else orders
    body = {
        "hook": hook, "hookInstance": HOOK_INSTANCE,
        "context": {"userId": "Practitioner/1", "patientId": patient_id, "draftOrders": bundle(*orders)},
        "prefetch": {
            "patient": {"resourceType": "Patient", "id": patient_id, "birthDate": "1970-01-01", "gender": "male"},
            "observations": bundle(*([obs("62238-1", egfr, "mL/min/1.73m2")] if egfr is not None else [])),
            "allergies": bundle(*[{"resourceType": "AllergyIntolerance", "code": {"text": a}} for a in allergies]),
            "medications": bundle(), "conditions": bundle(),
        },
    }
    body.update(over)
    return body


def check_card(card: dict) -> None:
    """Card: uuid (string), summary (required, <= 140), indicator in info|warning|critical, source.label (required),
    optional detail/links/overrideReasons, and no unknown properties."""
    assert set(card) <= CARD_KEYS, set(card) - CARD_KEYS
    assert isinstance(card["uuid"], str) and uuid.UUID(card["uuid"])
    assert isinstance(card["summary"], str) and 0 < len(card["summary"]) <= 140
    assert card["indicator"] in ("info", "warning", "critical")
    assert isinstance(card["source"], dict) and isinstance(card["source"]["label"], str) and card["source"]["label"]
    assert isinstance(card.get("detail", ""), str)
    for reason in card.get("overrideReasons", []):
        assert isinstance(reason.get("display"), str) and reason["display"]
    for link in card.get("links", []):
        assert link["url"].startswith("https://") and link["label"] and link["type"] in ("absolute", "smart")
    if "suggestions" in card:
        assert card.get("selectionBehavior") in ("at-most-one", "any")
    topic = card["source"].get("topic")
    if topic is not None:
        assert isinstance(topic.get("code"), str) and topic["code"]


class Ehr:
    """The EHR's JWT signing keys and a FHIR server for fallback reads."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.jwk = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key())), "kid": "ehr-1", "alg": "RS384"}
        self.jwks_response: object = {"keys": [self.jwk]}
        self.fhir_reads: list[str] = []
        self.fhir_fail: object = None
        self.fhir_data: dict = {}

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url == f"{ISS}/jwks":
            if isinstance(self.jwks_response, Exception):
                raise self.jwks_response
            if isinstance(self.jwks_response, httpx.Response):
                return self.jwks_response
            return httpx.Response(200, json=self.jwks_response)
        self.fhir_reads.append(req.url.path)
        if isinstance(self.fhir_fail, Exception):
            raise self.fhir_fail
        if isinstance(self.fhir_fail, httpx.Response):
            return self.fhir_fail
        if req.headers.get("authorization") != "Bearer fhir-token":
            return httpx.Response(401)
        name = req.url.path.removeprefix("/fhir/")
        if name.startswith("Patient/"):
            return httpx.Response(200, json=self.fhir_data.get("Patient", {"resourceType": "Patient", "id": name.split("/")[1]}))
        return httpx.Response(200, json=bundle(*self.fhir_data.get(name, [])))

    def token(self, audience=AUD, key=None, alg="RS384", kid="ehr-1", drop=(), **claims):
        now = int(time.time())
        body = {"iss": ISS, "aud": audience, "exp": now + 300, "iat": now, "jti": secrets.token_hex(8), **claims}
        for name in drop:
            body.pop(name, None)
        return jwt.encode(body, key or self.key, algorithm=alg, headers={"kid": kid})


@pytest.fixture()
def ehr(monkeypatch):
    server = Ehr()
    monkeypatch.setattr(cds_hooks, "transport", httpx.MockTransport(server.handler))
    monkeypatch.setattr(config, "CDS_HOOKS_TRUSTED", {ISS: f"{ISS}/jwks"})
    monkeypatch.setattr(config, "CDS_HOOKS_ALLOW_UNSIGNED", False)
    monkeypatch.setattr(config, "SMART_ALLOWED_ISSUERS", [f"{ISS}/fhir"])
    monkeypatch.setattr(config, "FHIR_OPEN_SERVERS", [])
    from app.main import app

    with TestClient(app) as client:
        server.client = client
        yield server


def call(ehr, body, path=SERVICE, token=None, raw=None, headers=None):
    token = token if token is not None else ehr.token("http://testserver" + path)
    hdrs = {"Authorization": f"Bearer {token}", **(headers or {})}
    if raw is not None:
        return ehr.client.post(path, content=raw, headers={**hdrs, "Content-Type": "application/json"})
    return ehr.client.post(path, json=body, headers=hdrs)


# --- Discovery ----------------------------------------------------------------------------------

def test_discovery_returns_a_services_array_without_authentication(ehr):
    """Discovery: GET {base}/cds-services returns 200 JSON {"services": [...]}; no JWT is needed."""
    response = ehr.client.get("/cds-services")
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/json")
    services = response.json()["services"]
    assert isinstance(services, list) and len(services) == 2


def test_each_service_has_the_required_discovery_fields(ehr):
    """Discovery: each service has hook (required), description (required), id (required, URL-safe), title (recommended),
    prefetch (optional, a map of key to FHIR query template)."""
    for service in ehr.client.get("/cds-services").json()["services"]:
        assert service["hook"] in ("order-select", "order-sign")
        assert isinstance(service["description"], str) and service["description"].strip()
        assert isinstance(service["title"], str) and service["title"].strip()
        assert service["id"].replace("-", "").isalnum() and service["id"] == f"groundcheck-medication-safety-{service['hook']}"
        assert isinstance(service["prefetch"], dict) and service["prefetch"]


def test_service_ids_are_unique_and_resolve_to_a_route(ehr):
    """Discovery: service ids are unique, and POST {base}/cds-services/{id} is the service endpoint."""
    ids = [s["id"] for s in ehr.client.get("/cds-services").json()["services"]]
    assert len(set(ids)) == len(ids)
    for service_id in ids:
        path = f"/cds-services/{service_id}"
        assert ehr.client.post(path, json={}).status_code != 404


def test_prefetch_templates_use_only_context_tokens_and_cover_every_resource_the_service_reads(ehr):
    """Prefetch: templates are FHIR read/search queries that use {{context.<field>}} tokens only; the service's needs are listed."""
    import re

    for service in ehr.client.get("/cds-services").json()["services"]:
        assert set(service["prefetch"]) >= {"patient", "observations", "allergies", "medications", "conditions"}
        for template in service["prefetch"].values():
            assert not template.startswith("/") and not re.match(r"[a-z]+://", template)
            for token in re.findall(r"\{\{([^}]*)\}\}", template):
                assert token == "context.patientId"
        assert service["prefetch"]["patient"] == "Patient/{{context.patientId}}"


def test_discovery_discloses_that_a_synthetic_formulary_is_not_for_clinical_use(ehr):
    """Discovery: usageRequirements / description tell the EHR what the service can safely be used for."""
    for service in ehr.client.get("/cds-services").json()["services"]:
        assert "synthetic" in service["description"].lower() and "DEMONSTRATION" in service["description"]
        assert "Not for clinical use" in service["usageRequirements"]


def test_discovery_preflight_and_cors_for_browser_based_clients(ehr):
    """Transport: CDS Clients in a browser need CORS; OPTIONS preflight is answered."""
    assert ehr.client.get("/cds-services").headers["access-control-allow-origin"] == "*"
    for path in ("/cds-services", SERVICE, SERVICE + "/feedback"):
        response = ehr.client.options(path)
        assert response.status_code == 204 and "Authorization" in response.headers["access-control-allow-headers"]


# --- Responses -----------------------------------------------------------------------------------

def test_a_call_with_no_concerns_returns_200_and_an_empty_cards_array(ehr):
    """Response: HTTP 200, application/json, {"cards": []} when there is nothing to say."""
    response = call(ehr, request())
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/json")
    assert response.json() == {"cards": []}


@pytest.mark.parametrize("path,hook", [(SERVICE, "order-sign"), (SELECT, "order-select")])
def test_each_hook_returns_valid_cards_for_an_unsafe_order(ehr, path, hook):
    """Response: every card has uuid, summary (<= 140 chars), indicator and source.label; indicator is one of
    info, warning, critical; cards are unique."""
    response = call(ehr, request(egfr=40, allergies=["Quorl"], hook=hook), path=path)
    assert response.status_code == 200
    cards = response.json()["cards"]
    assert len(cards) >= 2
    for card in cards:
        check_card(card)
    assert len({c["uuid"] for c in cards}) == len(cards)
    assert {c["indicator"] for c in cards} <= {"critical", "warning", "info"}
    assert cards[0]["indicator"] == "critical"


def test_cards_are_ordered_most_serious_first(ehr):
    """Response: cards are returned in the order they should be shown; critical before warning before info."""
    order = {"critical": 0, "warning": 1, "info": 2}
    body = request(orders=[draft(), draft("Paracetamol", oid="mr2")], egfr=40, allergies=["Quorl"])
    ranks = [order[c["indicator"]] for c in call(ehr, body).json()["cards"]]
    assert ranks == sorted(ranks) and len(ranks) >= 3


def test_override_reasons_are_offered_only_for_blocking_cards_and_are_well_formed(ehr):
    """Card: overrideReasons is an array of Coding-like reasons with a display; absent or empty is allowed."""
    cards = call(ehr, request(egfr=40)).json()["cards"]
    assert cards[0]["overrideReasons"] and all(r["display"] and r["code"] for r in cards[0]["overrideReasons"])
    unknown = call(ehr, request(orders=[draft("Paracetamol")])).json()["cards"]
    assert unknown[0]["overrideReasons"] == []


def test_card_summary_never_exceeds_140_characters(ehr):
    """Card: summary is at most 140 characters, even for very long medicine names and messages."""
    name = "Caloradine " + "extended release " * 60
    for medicine in (name, "Paracetamol " * 100):
        for card in call(ehr, request(orders=[draft(medicine)], egfr=40)).json()["cards"]:
            check_card(card)


def test_every_card_has_unique_uuids_across_calls(ehr):
    """Card: uuid identifies a card for feedback; it is not reused across calls."""
    seen = set()
    for _ in range(5):
        for card in call(ehr, request(egfr=40)).json()["cards"]:
            assert card["uuid"] not in seen
            seen.add(card["uuid"])


# --- Orders ----------------------------------------------------------------------------------------

def test_an_unknown_medicine_always_yields_a_not_checked_card(ehr):
    """Safety: silence reads as 'checked, no concerns'. A medicine outside the formulary always gets a warning card
    saying it was not checked, however the name is expressed."""
    contained = draft("", oid="c1")
    contained.pop("medicationCodeableConcept")
    contained["contained"] = [{"resourceType": "Medication", "id": "m", "code": {"text": "Zorbinex 5 mg"}}]
    contained["medicationReference"] = {"reference": "#m"}
    by_display = draft(oid="c2")
    by_display.pop("medicationCodeableConcept")
    by_display["medicationReference"] = {"reference": "Medication/9", "display": "Vexorin"}
    no_name = draft(oid="c3")
    no_name["medicationCodeableConcept"] = {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "123"}]}
    for order in (draft("Paracetamol"), draft("パラセタモール"), draft("  "), draft("x" * 5000), contained, by_display, no_name,
                  draft("Zzz'; DROP TABLE medicines;--")):
        cards = call(ehr, request(orders=[order])).json()["cards"]
        assert len(cards) == 1, order
        card = cards[0]
        check_card(card)
        assert card["indicator"] == "warning" and card["source"]["topic"]["code"] == "not_in_formulary"
        assert "Not checked" in card["summary"] and "not a statement that the order is safe" in card["detail"]


def test_each_unknown_medicine_in_a_set_of_orders_gets_its_own_card(ehr):
    """Safety: no order is skipped silently."""
    orders = [draft("Paracetamol", oid="a"), draft("Ibuprofen", oid="b"), draft("Caloradine", oid="c"), draft("Amoxicillin", oid="d")]
    cards = call(ehr, request(orders=orders)).json()["cards"]
    assert [c["source"]["topic"]["code"] for c in cards].count("not_in_formulary") == 3


def test_orders_that_are_not_medications_are_ignored(ehr):
    """Hooks: draftOrders may hold any order type; this service only reads MedicationRequest."""
    service_request = {"resourceType": "ServiceRequest", "id": "s1", "status": "draft", "intent": "order"}
    assert call(ehr, request(orders=[service_request, "junk", 5, None])).json() == {"cards": []}
    assert call(ehr, request(orders=[service_request, draft("Paracetamol")])).json()["cards"][0]["source"]["topic"]["code"] == "not_in_formulary"


def test_no_draft_orders_means_no_cards(ehr):
    """order-sign: draftOrders may be empty; nothing to check."""
    body = request(orders=[])
    assert call(ehr, body).json() == {"cards": []}
    del body["context"]["draftOrders"]
    assert call(ehr, body).json() == {"cards": []}


def test_order_select_only_checks_the_selected_orders(ehr):
    """order-select: context.selections lists the draft orders (Type/id) the user just selected."""
    orders = [draft("Paracetamol", oid="other"), draft("Caloradine", oid="mine", dose=15)]
    body = request(orders=orders, egfr=40, hook="order-select")
    body["context"]["selections"] = ["MedicationRequest/mine"]
    cards = call(ehr, body, path=SELECT).json()["cards"]
    assert cards and all(c["source"]["topic"]["code"] != "not_in_formulary" for c in cards)
    body["context"]["selections"] = ["MedicationRequest/other"]
    assert [c["source"]["topic"]["code"] for c in call(ehr, body, path=SELECT).json()["cards"]] == ["not_in_formulary"]


def test_order_sign_ignores_selections(ehr):
    """order-sign: all draft orders are checked; selections is not part of this hook."""
    body = request(orders=[draft("Paracetamol", oid="x")])
    body["context"]["selections"] = ["MedicationRequest/y"]
    assert len(call(ehr, body).json()["cards"]) == 1


# --- Prefetch --------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["observations", "allergies", "medications", "conditions"])
def test_a_missing_prefetch_key_is_reported_not_treated_as_empty(ehr, key):
    """Prefetch: a key the CDS Client could not fulfil is omitted. Treating it as empty would read as 'no allergies';
    the response says which data was not available."""
    body = request(egfr=90)
    del body["prefetch"][key]
    cards = call(ehr, body).json()["cards"]
    [partial] = [c for c in cards if c["source"]["topic"]["code"] == "patient_data_missing"]
    check_card(partial)
    label = {"observations": "lab results", "allergies": "allergies", "medications": "current medicines", "conditions": "conditions"}[key]
    assert label in partial["summary"] and "is a statement that the order is safe" in partial["detail"]
    assert (len(cards) == 2) == (key == "observations")    # without labs the kidney-dependent dose can't be checked either


@pytest.mark.parametrize("value", [None, "x", 5, [], {"resourceType": "OperationOutcome", "issue": []}, {"resourceType": "Encounter"}, {}])
def test_unusable_prefetch_values_count_as_missing(ehr, value):
    """Prefetch: null or an OperationOutcome in place of a resource means the CDS Client could not supply it."""
    body = request()
    body["prefetch"]["allergies"] = value
    cards = call(ehr, body).json()["cards"]
    assert [c["source"]["topic"]["code"] for c in cards] == ["patient_data_missing"]


def test_missing_data_that_would_have_blocked_the_order_is_still_surfaced(ehr):
    """Prefetch: with no lab results the kidney-dependent dose check says it can't be done."""
    cards = call(ehr, request(egfr=None)).json()["cards"]
    assert cards[0]["source"]["topic"]["code"] == "missing_data"


def test_the_patient_prefetch_is_required(ehr):
    """Prefetch: without a patient and no way to read one, the response is 412 Precondition Failed with a clear message."""
    body = request()
    del body["prefetch"]["patient"]
    response = call(ehr, body)
    assert response.status_code == 412 and "patient prefetch is required" in response.json()["detail"]
    body = request()
    del body["prefetch"]
    assert call(ehr, body).status_code == 412


def test_missing_prefetch_is_read_from_the_fhir_server_with_the_clients_token(ehr):
    """Prefetch fallback: a CDS Service MAY fetch missing data from fhirServer using fhirAuthorization.access_token."""
    ehr.fhir_data = {"Patient": {"resourceType": "Patient", "id": "p1", "birthDate": "1970-01-01", "gender": "male"},
                     "Observation": [obs("62238-1", 40, "mL/min/1.73m2")], "AllergyIntolerance": [], "MedicationRequest": [],
                     "Condition": []}
    body = request()
    body["prefetch"] = {}
    body["fhirServer"] = f"{ISS}/fhir"
    body["fhirAuthorization"] = {"access_token": "fhir-token", "token_type": "Bearer", "expires_in": 300,
                                 "scope": "patient/Patient.read patient/Observation.read", "subject": "cds-service"}
    response = call(ehr, body)
    assert response.status_code == 200
    cards = response.json()["cards"]
    assert cards[0]["indicator"] == "critical" and "7.5 mg" in cards[0]["summary"]
    assert ehr.fhir_reads and all(p.startswith("/fhir/") for p in ehr.fhir_reads)


def test_the_fallback_reads_only_what_was_missing(ehr):
    """Prefetch fallback: keys the EHR did supply are not fetched again."""
    ehr.fhir_data = {"AllergyIntolerance": []}
    body = request()
    del body["prefetch"]["allergies"]
    body["fhirServer"], body["fhirAuthorization"] = f"{ISS}/fhir", {"access_token": "fhir-token"}
    assert call(ehr, body).json() == {"cards": []}
    assert ehr.fhir_reads == ["/fhir/AllergyIntolerance"]


@pytest.mark.parametrize("server", ["http://169.254.169.254/latest", "https://evil.example/fhir", "http://ehr.example.org/fhir",
                                    f"{ISS}/fhirevil", "file:///etc/passwd", "ftp://ehr.example.org/fhir", "", 7, None,
                                    "https://ehr.example.org@evil.example/fhir", "https://ehr.example.org.evil.example/fhir"])
def test_the_fallback_never_contacts_a_server_the_deployment_does_not_allow(ehr, server):
    """Security: fhirServer comes from the request, so it is only used when it is on this deployment's allowed list."""
    body = request()
    body["prefetch"] = {}
    body["fhirServer"], body["fhirAuthorization"] = server, {"access_token": "fhir-token"}
    response = call(ehr, body)
    assert response.status_code == 412
    assert ehr.fhir_reads == []


@pytest.mark.parametrize("authorization", [None, {}, {"access_token": ""}, {"access_token": 5}, "token", [], {"token": "x"}])
def test_the_fallback_needs_an_access_token(ehr, authorization):
    """Prefetch fallback: without fhirAuthorization.access_token no FHIR read is attempted."""
    body = request()
    body["prefetch"] = {}
    body["fhirServer"], body["fhirAuthorization"] = f"{ISS}/fhir", authorization
    assert call(ehr, body).status_code == 412 and ehr.fhir_reads == []


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused"), httpx.Response(401), httpx.Response(403),
                                     httpx.Response(404), httpx.Response(500, text="boom"), httpx.Response(200, text="<html>"),
                                     httpx.Response(200, json=["not", "a", "bundle"])])
def test_a_failing_fhir_fallback_gives_a_clear_error_not_a_crash_or_silence(ehr, failure):
    """Prefetch fallback: when the FHIR server can't be read the response is 412 with a reason, never 500 or an
    empty 'all clear'; timeouts return promptly."""
    ehr.fhir_fail = failure
    body = request()
    body["prefetch"] = {}
    body["fhirServer"], body["fhirAuthorization"] = f"{ISS}/fhir", {"access_token": "fhir-token"}
    start = time.perf_counter()
    response = call(ehr, body)
    assert response.status_code == 412 and response.json()["detail"]
    assert time.perf_counter() - start < 10


def test_a_failing_fallback_for_other_data_becomes_a_partly_checked_card(ehr):
    """Prefetch fallback: if only optional data can't be read, the check still runs and says what is missing."""
    ehr.fhir_fail = httpx.Response(500)
    body = request()
    del body["prefetch"]["allergies"]
    body["fhirServer"], body["fhirAuthorization"] = f"{ISS}/fhir", {"access_token": "fhir-token"}
    cards = call(ehr, body).json()["cards"]
    assert [c["source"]["topic"]["code"] for c in cards] == ["patient_data_missing"]
    assert "(" in cards[0]["detail"]


def test_the_fallback_has_a_time_budget(ehr, monkeypatch):
    """Transport: an ordering screen waits seconds, not minutes; reading stops when the budget is spent."""
    monkeypatch.setattr(cds_hooks, "FETCH_BUDGET_SECONDS", -1.0)
    ehr.fhir_data = {"Patient": {"resourceType": "Patient", "id": "p1"}}
    body = request()
    body["prefetch"] = {}
    body["fhirServer"], body["fhirAuthorization"] = f"{ISS}/fhir", {"access_token": "fhir-token"}
    response = call(ehr, body)
    assert response.status_code == 412 and "too long" in response.json()["detail"] and ehr.fhir_reads == []


def test_a_prefetched_patient_other_than_the_one_in_context_is_refused(ehr):
    """Safety: the prefetched Patient must be context.patientId; anything else could be the wrong patient's record."""
    body = request()
    body["prefetch"]["patient"]["id"] = "someone-else"
    response = call(ehr, body)
    assert response.status_code == 412 and "isn't the patient in context" in response.json()["detail"]


def test_resources_of_another_patient_in_the_prefetch_are_ignored_and_reported(ehr):
    """Safety: a lab result or allergy that references a different patient is never used."""
    body = request(egfr=None)
    wrong = obs("62238-1", 30, "mL/min/1.73m2")
    wrong["subject"] = {"reference": "Patient/other"}
    body["prefetch"]["observations"] = bundle(wrong)
    body["prefetch"]["allergies"] = bundle({"resourceType": "AllergyIntolerance", "code": {"text": "Quorl"},
                                            "patient": {"reference": "Patient/other"}})
    cards = call(ehr, body).json()["cards"]
    codes = [c["source"]["topic"]["code"] for c in cards]
    assert "allergy" not in codes and "missing_data" in codes
    assert any("different patient" in c["detail"] for c in cards)


def test_unreadable_patient_data_produces_an_info_card_so_a_missing_check_is_visible(ehr):
    """Safety: invalid or unreadable parts of the record (for example an allergy with no name) are disclosed."""
    body = request()
    body["prefetch"]["allergies"] = bundle({"resourceType": "AllergyIntolerance", "code": {"coding": [{"system": "x"}]}})
    cards = call(ehr, body).json()["cards"]
    assert [c["source"]["topic"]["code"] for c in cards] == ["patient_data_unreadable"]
    assert cards[0]["indicator"] == "info" and "can't be named" in cards[0]["detail"]
    check_card(cards[0])


def test_ordinary_profile_gaps_do_not_add_noise(ehr):
    """Usability: a record that merely lacks optional profile elements gets no extra card."""
    body = request()
    body["prefetch"]["conditions"] = bundle({"resourceType": "Condition", "code": {"text": "Asthma"}})
    body["prefetch"]["allergies"] = bundle({"resourceType": "AllergyIntolerance", "code": {"text": "Latex"},
                                            "clinicalStatus": {"coding": [{"code": "active"}]}})
    assert call(ehr, body).json() == {"cards": []}


# --- Request shape -----------------------------------------------------------------------------------

def test_invalid_json_is_a_400(ehr):
    """Errors: a body that is not JSON is 400 Bad Request."""
    for raw in (b"{not json", b"", b"\xff\xfe", b"[1,2", b'{"hook": '):
        assert call(ehr, None, raw=raw).status_code == 400


def test_a_body_that_is_not_an_object_is_a_400(ehr):
    """Errors: the request is a JSON object."""
    for raw in (b"[]", b'"x"', b"5", b"null", b"true"):
        assert call(ehr, None, raw=raw).status_code == 400


@pytest.mark.parametrize("hook", ["patient-view", "encounter-start", "", None, 5, "ORDER-SIGN", ["order-sign"]])
def test_an_unsupported_or_malformed_hook_is_a_400(ehr, hook):
    """Request: hook (required) names a hook this service handles."""
    assert call(ehr, request(hook=hook)).status_code == 400
    body = request()
    body.pop("hook")
    assert call(ehr, body).status_code == 400


def test_a_hook_that_does_not_match_the_service_id_is_refused(ehr):
    """Request: a service registered for order-select is not called for order-sign; the hook must match the service."""
    for service_id, hook in ((SERVICE_ID, "order-select"), (SELECT_ID, "order-sign")):
        with pytest.raises(cds_hooks.CdsError) as err:
            cds_hooks.cards_for(request(hook=hook), service_id)
        assert err.value.status == 400 and "hook" in str(err.value)
    assert cds_hooks.cards_for(request(hook="order-sign"), SERVICE_ID) == {"cards": []}
    assert cds_hooks.cards_for(request(hook="order-select"), SELECT_ID) == {"cards": []}


@pytest.mark.parametrize("instance", [None, "", "abc", "d1577c69-dfbe-44ad-ba6d", "d1577c69dfbe44adba6d3e05e953b2ea" + "0", 5, ["x"],
                                      "d1577c69-dfbe-44ad-ba6d-3e05e953b2eg", "../../etc/passwd", "<script>"])
def test_hook_instance_must_be_a_uuid(ehr, instance):
    """Request: hookInstance (required) is a UUID that uniquely identifies this call."""
    body = request()
    body["hookInstance"] = instance
    assert call(ehr, body).status_code == 400


@pytest.mark.parametrize("context", [None, "x", [], 5])
def test_context_is_required_and_an_object(ehr, context):
    """Request: context (required) is an object."""
    body = request()
    body["context"] = context
    assert call(ehr, body).status_code == 400
    del body["context"]
    assert call(ehr, body).status_code == 400


@pytest.mark.parametrize("patient_id", [None, "", 5, "../Patient/1", "a b", "x" * 65, "p1; drop", ["p1"], {"id": "p1"}])
def test_patient_id_is_required_and_safe(ehr, patient_id):
    """Context: patientId is required for order-select and order-sign."""
    body = request()
    body["context"]["patientId"] = patient_id
    assert call(ehr, body).status_code == 400


@pytest.mark.parametrize("user_id", [None, "", 5])
def test_user_id_is_required(ehr, user_id):
    """Context: userId is required."""
    body = request()
    body["context"]["userId"] = user_id
    assert call(ehr, body).status_code == 400


@pytest.mark.parametrize("value", ["x", [], 5, True])
def test_draft_orders_must_be_a_bundle_object(ehr, value):
    """Context: draftOrders is a FHIR Bundle."""
    body = request()
    body["context"]["draftOrders"] = value
    assert call(ehr, body).status_code == 400


def test_selections_must_be_a_list_for_order_select(ehr):
    """Context: order-select selections is an array of Type/id strings."""
    body = request(hook="order-select")
    body["context"]["selections"] = "MedicationRequest/1"
    assert call(ehr, body, path=SELECT).status_code == 400


def test_prefetch_must_be_an_object(ehr):
    """Request: prefetch is an object keyed by prefetch name."""
    for value in ("x", [], 5):
        body = request()
        body["prefetch"] = value
        assert call(ehr, body).status_code == 400


def test_unknown_request_fields_are_ignored(ehr):
    """Robustness: CDS Services ignore request properties they don't know."""
    body = request(extraField={"x": 1})
    body["context"]["encounterId"] = "e1"
    assert call(ehr, body).status_code == 200


def test_hostile_payloads_do_not_cause_server_errors(ehr):
    """Robustness: deep nesting, huge strings and many orders are handled, not 500s."""
    deep = {}
    node = deep
    for _ in range(300):
        node["a"] = {}
        node = node["a"]
    body = request()
    body["prefetch"]["patient"]["extension"] = [{"url": "http://x", "extension": [deep]}]
    assert call(ehr, body).status_code == 200
    body = request(orders=[draft("Paracetamol", oid=f"o{i}") for i in range(300)])
    start = time.perf_counter()
    response = call(ehr, body)
    assert response.status_code == 200 and len(response.json()["cards"]) == 300
    assert time.perf_counter() - start < 30
    body = request()
    body["prefetch"]["patient"]["name"] = [{"family": "A" * 2_000_000}]
    assert call(ehr, body).status_code == 200


DEEP = b'{"a":' * 100_000 + b"1" + b"}" * 100_000


def test_read_json_turns_every_bad_body_into_a_400():
    """Robustness: bodies that aren't JSON, aren't objects, or nest beyond the parser's limit are CdsError(400)."""
    for raw in (DEEP, b"", b"{", b"[]", b"null", b"\xff"):
        with pytest.raises(cds_hooks.CdsError) as err:
            cds_hooks.read_json(raw)
        assert err.value.status == 400
    assert cds_hooks.read_json(b'{"hook": "order-sign"}') == {"hook": "order-sign"}


@pytest.mark.xfail(strict=True, reason="app/main.py's route parses with request.json() and catches only ValueError; "
                                       "it should call cds_hooks.read_json (engineer X)")
def test_a_json_body_nested_beyond_the_parsers_limit_is_a_400_not_a_500(ehr):
    """Robustness: a body too deeply nested for the JSON parser is rejected as bad input."""
    assert call(ehr, None, raw=DEEP).status_code == 400


# --- JWT authentication ---------------------------------------------------------------------------------

def post(ehr, token, path=SERVICE, body=None):
    return ehr.client.post(path, json=body or request(), headers={"Authorization": f"Bearer {token}"} if token is not None else {})


def test_a_call_without_a_token_is_401_when_unsigned_calls_are_not_allowed(ehr):
    """Auth: the CDS Client sends Authorization: Bearer <JWT>; without it the call is 401."""
    assert post(ehr, None).status_code == 401


def test_a_valid_token_is_accepted(ehr):
    """Auth: a JWT with iss, aud, exp, iat, jti, signed by a key in the issuer's JWKS, is accepted."""
    assert post(ehr, ehr.token()).status_code == 200


def test_expired_tokens_are_rejected(ehr):
    """Auth: exp in the past (beyond clock-skew leeway) is rejected."""
    now = int(time.time())
    assert post(ehr, ehr.token(exp=now - 120, iat=now - 400)).status_code == 401
    assert post(ehr, ehr.token(exp=now - 10, iat=now - 400)).status_code == 200  # within the 30 s leeway


def test_tokens_for_another_audience_are_rejected(ehr):
    """Auth: aud must be the URL of the service being called."""
    for audience in ("https://wrong.example/cds-services/x", "http://testserver/cds-services", "http://testserver" + SELECT,
                     AUD + "/", "", ["https://other.example"]):
        assert post(ehr, ehr.token(audience=audience)).status_code == 401, audience
    assert post(ehr, ehr.token(audience=[AUD, "https://other.example"])).status_code == 200


def test_the_same_token_is_not_accepted_twice(ehr):
    """Auth: jti is unique per call; a repeated jti is a replay and is rejected."""
    token = ehr.token()
    assert post(ehr, token).status_code == 200
    assert post(ehr, token).status_code == 401
    assert post(ehr, ehr.token(jti=jwt.decode(token, options={"verify_signature": False})["jti"])).status_code == 401


def test_replay_tracking_is_per_issuer_and_bounded(ehr, monkeypatch):
    """Auth: jti uniqueness is per issuer; the replay cache can't grow without bound."""
    assert cds_hooks._first_use("a", "j", time.time() + 100) and cds_hooks._first_use("b", "j", time.time() + 100)
    assert not cds_hooks._first_use("a", "j", time.time() + 100)
    monkeypatch.setattr(cds_hooks, "_MAX_SEEN", 10)
    for i in range(50):
        cds_hooks._first_use("c", f"j{i}", time.time() + 100)
    assert len(cds_hooks._seen_jti) <= 12


def test_a_token_whose_jti_has_expired_from_the_cache_can_not_be_replayed_if_it_is_expired_itself(ehr):
    """Auth: a replayed token past its exp is rejected by exp, whatever the cache holds."""
    now = int(time.time())
    token = ehr.token(exp=now - 200, iat=now - 500)
    assert post(ehr, token).status_code == 401 and post(ehr, token).status_code == 401


def test_unknown_issuers_are_rejected_before_any_key_is_fetched(ehr):
    """Auth: iss must be a trusted EHR; unknown issuers never cause a JWKS fetch."""
    seen = []
    original = cds_hooks._jwks
    cds_hooks._jwks = lambda url: (seen.append(url), original(url))[1]
    try:
        assert post(ehr, ehr.token(iss="https://evil.example")).status_code == 401
        assert post(ehr, ehr.token(iss="")).status_code == 401
    finally:
        cds_hooks._jwks = original
    assert seen == []


def test_tokens_signed_with_another_key_are_rejected(ehr):
    """Auth: the signature must verify against the EHR's published key."""
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert post(ehr, ehr.token(key=other)).status_code == 401


def test_tokens_with_unknown_or_missing_kid_are_rejected(ehr):
    """Auth: the kid header selects the JWKS key; an unknown kid is rejected."""
    assert post(ehr, ehr.token(kid="not-a-key")).status_code == 401
    assert post(ehr, jwt.encode({"iss": ISS, "aud": AUD, "exp": int(time.time()) + 99, "iat": int(time.time()), "jti": "x"},
                                ehr.key, algorithm="RS384")).status_code == 401


@pytest.mark.parametrize("claim", ["exp", "iat", "jti", "iss", "aud"])
def test_required_claims_must_be_present(ehr, claim):
    """Auth: iss, aud, exp, iat and jti are all required claims."""
    assert post(ehr, ehr.token(drop=(claim,))).status_code == 401


def test_tokens_issued_in_the_future_or_not_yet_valid_are_rejected(ehr):
    """Auth: iat and nbf in the future (beyond leeway) are rejected."""
    now = int(time.time())
    assert post(ehr, ehr.token(iat=now + 3600, exp=now + 4000)).status_code == 401
    assert post(ehr, ehr.token(nbf=now + 3600)).status_code == 401


def test_the_none_algorithm_and_symmetric_algorithms_are_rejected(ehr):
    """Auth: only RS256/RS384/ES256/ES384 are accepted; alg none and HS256 (key confusion) are not."""
    now = int(time.time())
    claims = {"iss": ISS, "aud": AUD, "exp": now + 300, "iat": now, "jti": "j1"}
    assert post(ehr, jwt.encode(claims, key=None, algorithm="none", headers={"kid": "ehr-1"})).status_code == 401
    secret = json.dumps(ehr.jwk)
    hs = jwt.encode({**claims, "jti": "j2"}, secret * 2, algorithm="HS256", headers={"kid": "ehr-1"})
    assert post(ehr, hs).status_code == 401


def test_an_ec_token_against_an_rsa_key_is_rejected(ehr):
    """Auth: the token's algorithm must fit the key type published for its kid."""
    key = ec.generate_private_key(ec.SECP384R1())
    assert post(ehr, ehr.token(key=key, alg="ES384")).status_code == 401


@pytest.mark.parametrize("header", ["", "Bearer", "Bearer ", "Basic abc", "Token abc", "bearer", "abc", "Bearer a.b", "Bearer a.b.c",
                                    "Bearer " + "x" * 10000])
def test_malformed_authorization_headers_are_rejected(ehr, header):
    """Auth: a value that isn't 'Bearer <JWT>' is 401."""
    response = ehr.client.post(SERVICE, json=request(), headers={"Authorization": header})
    assert response.status_code == 401


def test_the_bearer_scheme_is_case_insensitive(ehr):
    """Auth: HTTP auth schemes are case-insensitive (RFC 9110)."""
    response = ehr.client.post(SERVICE, json=request(), headers={"Authorization": f"bearer {ehr.token()}"})
    assert response.status_code == 200


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("down"), httpx.Response(500, text="oops"),
                                     httpx.Response(200, json=[]), httpx.Response(200, json={"keys": "x"}),
                                     httpx.Response(200, json={"keys": [None, "x", 5]}), httpx.Response(200, json={}),
                                     httpx.Response(200, text="<html>"), httpx.Response(200, json={"keys": [{"kid": "ehr-1"}]}),
                                     httpx.Response(200, json={"keys": [{"kid": "ehr-1", "kty": "RSA", "n": "x", "e": "y"}]})])
def test_an_unreachable_or_broken_jwks_means_401_not_500(ehr, failure):
    """Auth: if the EHR's keys can't be loaded or used the token can't be verified, so 401, never an error or an accept."""
    ehr.jwks_response = failure
    start = time.perf_counter()
    assert post(ehr, ehr.token()).status_code == 401
    assert time.perf_counter() - start < 10


def test_unsigned_calls_are_accepted_only_when_explicitly_allowed(ehr, monkeypatch):
    """Auth: CDS_HOOKS_ALLOW_UNSIGNED is a sandbox-only switch, off by default."""
    assert post(ehr, None).status_code == 401
    monkeypatch.setattr(config, "CDS_HOOKS_ALLOW_UNSIGNED", True)
    assert post(ehr, None).status_code == 200
    assert post(ehr, "garbage").status_code == 401   # a bad token is still bad even when unsigned calls are allowed


def test_authentication_comes_before_any_processing_or_fhir_reads(ehr):
    """Security: an unauthenticated request can't make the service read from a FHIR server."""
    body = request()
    body["prefetch"] = {}
    body["fhirServer"], body["fhirAuthorization"] = f"{ISS}/fhir", {"access_token": "fhir-token"}
    assert ehr.client.post(SERVICE, json=body).status_code == 401 and ehr.fhir_reads == []


# --- Feedback ---------------------------------------------------------------------------------------------

FEEDBACK = {"feedback": [{"card": str(uuid.uuid4()), "outcome": "overridden", "outcomeTimestamp": "2026-11-24T10:00:00Z",
                          "overrideReason": {"reason": {"code": "clinically-appropriate"}, "userComment": "ok"}},
                         {"card": str(uuid.uuid4()), "outcome": "accepted", "acceptedSuggestions": [{"id": "s1"}],
                          "outcomeTimestamp": "2026-11-24T10:00:01Z"}]}


def test_feedback_is_accepted_with_a_signed_request(ehr):
    """Feedback: POST {base}/cds-services/{id}/feedback with outcome accepted|overridden is answered 200."""
    path = SERVICE + "/feedback"
    assert call(ehr, FEEDBACK, path=path).status_code == 200
    assert call(ehr, {"feedback": []}, path=path).status_code == 200


def test_feedback_needs_authentication_for_its_own_audience(ehr):
    """Feedback: authenticated like any service call, with aud the feedback URL."""
    path = SERVICE + "/feedback"
    assert ehr.client.post(path, json=FEEDBACK).status_code == 401
    assert call(ehr, FEEDBACK, path=path, token=ehr.token(AUD)).status_code == 401


def test_feedback_with_bad_json_is_400(ehr):
    """Feedback: invalid JSON is 400."""
    assert call(ehr, None, path=SERVICE + "/feedback", raw=b"{bad").status_code == 400


ODD_FEEDBACK = [{}, [], "x", {"feedback": "x"}, {"feedback": [None, 5, "x", {"card": None}, {"card": 5, "outcome": []}]},
                {"feedback": [{"overrideReason": "text"}, {"overrideReason": {"reason": "text"}}]}]


def test_feedback_items_survive_any_shape():
    """Feedback: unexpected shapes are skipped, values are clipped, and nothing raises."""
    for body in ODD_FEEDBACK:
        assert isinstance(cds_hooks.feedback_items(body), list)
    [item] = cds_hooks.feedback_items({"feedback": [{"card": "c" * 500, "outcome": "overridden",
                                                     "overrideReason": {"reason": {"code": "r" * 500}}}]})
    assert len(item["card"]) == 64 and len(item["reason"]) == 40 and item["outcome"] == "overridden"
    assert len(cds_hooks.feedback_items({"feedback": [{}] * 500})) == 50


@pytest.mark.xfail(strict=True, reason="app/main.py's feedback route calls item.get on entries that may not be objects; "
                                       "it should use cds_hooks.feedback_items (engineer X)")
def test_odd_feedback_shapes_are_not_a_500_over_http(ehr):
    """Feedback: unexpected shapes never produce a server error."""
    for body in ODD_FEEDBACK:
        assert call(ehr, body, path=SERVICE + "/feedback").status_code in (200, 400), body


def test_a_huge_feedback_batch_is_handled(ehr):
    """Feedback: large batches are bounded and answered."""
    body = {"feedback": [{"card": str(uuid.uuid4()), "outcome": "accepted"} for _ in range(5000)]}
    assert call(ehr, body, path=SERVICE + "/feedback").status_code == 200


def test_unknown_services_are_404(ehr):
    """Errors: an unknown service id is 404."""
    assert call(ehr, request(), path="/cds-services/nope").status_code == 404
