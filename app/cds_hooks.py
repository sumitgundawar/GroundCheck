"""CDS Hooks: medication safety cards inside the EHR's ordering screen.

The EHR calls this service when a clinician selects or signs a medication
order (the order-select and order-sign hooks). GroundCheck builds the patient
from the prefetched FHIR resources (app/fhir.py), checks each draft
MedicationRequest against the formulary (app/patient_checks.py), and returns a
card for every finding: critical for unsafe, warning for checks, info for
notes, each naming its rule and source.

Calls must carry a JWT signed by a trusted EHR (CDS_HOOKS_TRUSTED maps each
issuer to its JWKS URL), audience this service's URL, not expired. Only for
testing with a sandbox, CDS_HOOKS_ALLOW_UNSIGNED accepts unsigned calls.

Specification: https://cds-hooks.hl7.org"""

from __future__ import annotations

import re
import time
import uuid

import httpx

from . import config, fhir, formulary, patient_checks
from .schemas import Claim, PatientContext

SERVICE_ID = "groundcheck-medication-safety"
# Tests replace this with a mock transport; None uses the network.
transport: httpx.BaseTransport | None = None
_codes = ",".join(f"{fhir.LOINC}|{c}" for codes in fhir.OBSERVATION_CODES.values() for c in codes)
PREFETCH = {
    "patient": "Patient/{{context.patientId}}",
    "observations": f"Observation?patient={{{{context.patientId}}}}&code={_codes}&_sort=-date&_count=100",
    "allergies": "AllergyIntolerance?patient={{context.patientId}}",
    "medications": "MedicationRequest?patient={{context.patientId}}&status=active",
    "conditions": "Condition?patient={{context.patientId}}",
}
INDICATOR = {"block": "critical", "warn": "warning", "info": "info"}
_FREQUENCY = {(1, 1, "d"): "once daily", (2, 1, "d"): "twice daily", (1, 12, "h"): "every 12 hours",
              (1, 24, "h"): "once daily", (3, 1, "d"): "three times daily"}


class CdsError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def discovery() -> dict:
    description = ("Checks a medication order against the patient's allergies, other medicines, age, weight, kidney "
                   "and liver function, pregnancy and lab results, using GroundCheck's formulary.")
    return {"services": [
        {"hook": hook, "id": f"{SERVICE_ID}-{hook}", "title": "GroundCheck medication safety",
         "description": description, "prefetch": PREFETCH, "usageRequirements": "Patient-context medication ordering."}
        for hook in ("order-select", "order-sign")
    ]}


def _jwks(url: str) -> dict:
    try:
        with httpx.Client(timeout=10.0, transport=transport) as client:
            return client.get(url).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CdsError("Couldn't load the EHR's signing keys.", 401) from exc


def verify(authorization: str | None, audience: str) -> str:
    """Returns the calling EHR's issuer, or raises CdsError."""
    import jwt

    if not authorization:
        if config.CDS_HOOKS_ALLOW_UNSIGNED:
            return "unsigned"
        raise CdsError("A signed JWT from a trusted EHR is required.", 401)
    token = authorization.removeprefix("Bearer ").strip()
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise CdsError("The authorization token is malformed.", 401) from exc
    issuer = unverified.get("iss", "")
    jwks_url = config.CDS_HOOKS_TRUSTED.get(issuer)
    if not jwks_url:
        raise CdsError("This EHR isn't trusted to call GroundCheck.", 401)
    if header.get("alg") not in ("RS384", "ES384", "RS256", "ES256"):
        raise CdsError("The token's algorithm isn't allowed.", 401)
    keys = _jwks(jwks_url).get("keys", [])
    jwk = next((k for k in keys if k.get("kid") == header.get("kid")), None)
    if jwk is None:
        raise CdsError("The token was signed with an unknown key.", 401)
    try:
        jwt.decode(token, jwt.PyJWK(jwk, algorithm=header["alg"]).key, algorithms=[header["alg"]], audience=audience,
                   issuer=issuer, leeway=30, options={"require": ["exp", "iat", "iss", "aud", "jti"]})
    except jwt.PyJWTError as exc:
        raise CdsError("The token couldn't be verified.", 401) from exc
    return issuer


def _bundle_resources(value, resource_type: str) -> list[dict]:
    if not value:
        return []
    if value.get("resourceType") == resource_type:
        return [value]
    return [e["resource"] for e in value.get("entry", []) if e.get("resource", {}).get("resourceType") == resource_type]


def _order_dose(order: dict) -> tuple[str, float | None, str]:
    """(sentence, amount, unit) describing an order's dose."""
    dosage = (order.get("dosageInstruction") or [{}])[0]
    quantity = next((d.get("doseQuantity") for d in dosage.get("doseAndRate", []) if d.get("doseQuantity")), None)
    repeat = (dosage.get("timing") or {}).get("repeat", {})
    frequency = _FREQUENCY.get((repeat.get("frequency", 1), repeat.get("period", 1), repeat.get("periodUnit", "d")), "")
    if not quantity or not isinstance(quantity.get("value"), (int, float)):
        return dosage.get("text", ""), None, ""
    unit = (quantity.get("unit") or quantity.get("code") or "").replace("ml", "mL")
    amount = float(quantity["value"])
    return f"{amount:g} {unit} {frequency}".strip(), amount, unit


def cards_for(request: dict) -> dict:
    context = request.get("context") or {}
    prefetch = request.get("prefetch") or {}
    hook = request.get("hook", "")
    if hook not in ("order-select", "order-sign"):
        raise CdsError("This service handles order-select and order-sign.")
    patient = prefetch.get("patient")
    if not patient:
        raise CdsError("The patient prefetch is required.", 412)
    loaded = fhir.map_patient(
        patient,
        _bundle_resources(prefetch.get("observations"), "Observation"),
        _bundle_resources(prefetch.get("allergies"), "AllergyIntolerance"),
        _bundle_resources(prefetch.get("medications"), "MedicationRequest"),
        _bundle_resources(prefetch.get("conditions"), "Condition"),
        {"system": "CDS Hooks"},
    )
    orders = _bundle_resources(context.get("draftOrders"), "MedicationRequest")
    selected = set(context.get("selections") or [])
    if hook == "order-select" and selected:
        orders = [o for o in orders if f"MedicationRequest/{o.get('id')}" in selected]

    idx = formulary.index()
    cards = []
    for order in orders:
        name = fhir.concept_text(order.get("medicationCodeableConcept"))
        medicine = idx.find(name) or next(iter(idx.mentioned(name)), None)
        if medicine is None:
            # Never say nothing. In an ordering screen an empty response reads
            # as "checked, no concerns", so a medicine this formulary has never
            # heard of has to say so out loud — it is the one case where
            # silence is read as clearance.
            cards.append({
                "uuid": str(uuid.uuid4()),
                "summary": f"Not checked: {name or 'this medicine'} is not in the formulary",
                "indicator": "warning",
                "detail": (f"**{name or 'This medicine'}** was not checked. It is not in the formulary "
                           f"GroundCheck is running ({formulary.load().name}), so no allergy, interaction, "
                           f"kidney, liver, pregnancy or dose rule has been applied to this order.\n\n"
                           f"This is not a statement that the order is safe."),
                "source": {"label": "GroundCheck formulary",
                           "topic": {"code": "not_in_formulary",
                                     "system": "https://groundcheckhealth.com/cds/finding"}},
                "overrideReasons": [],
            })
            continue
        sentence, _, _ = _order_dose(order)
        claims = [Claim(text=f"{medicine.name} {sentence}.", source_ids=["ORDER"], grounded=True, grounding_score=1.0)]
        # The order itself: "the dose of X" makes the checks treat it as giving the medicine.
        others = PatientContext(**{**loaded.context.model_dump(),
                                   "medicines": [m for m in loaded.context.medicines
                                                 if medicine not in idx.mentioned(m)]})
        review = patient_checks.review(f"dose of {medicine.name}", claims, others, subject="order")
        for finding in review.findings:
            if finding.code == "already_taking":
                continue
            summary = finding.message if len(finding.message) <= 140 else finding.message[:137].rstrip() + "…"
            detail = f"**{medicine.name}** ({sentence or 'dose not stated'}): {finding.message}"
            if loaded.warnings:
                detail += "\n\n" + "\n".join(f"- {w}" for w in loaded.warnings)
            cards.append({
                "uuid": str(uuid.uuid4()),
                "summary": summary,
                "indicator": INDICATOR[finding.severity],
                "detail": detail,
                "source": {"label": "GroundCheck formulary" + (f" ({finding.rule_id})" if finding.rule_id else ""),
                           "topic": {"code": finding.code, "system": "https://groundcheckhealth.com/cds/finding"}},
                "overrideReasons": [
                    {"code": "clinically-appropriate", "system": "https://groundcheckhealth.com/cds/override",
                     "display": "Clinically appropriate for this patient"},
                    {"code": "data-out-of-date", "system": "https://groundcheckhealth.com/cds/override",
                     "display": "The patient data is out of date"},
                ] if finding.severity == "block" else [],
            })
    order_rank = {"critical": 0, "warning": 1, "info": 2}
    cards.sort(key=lambda c: order_rank[c["indicator"]])
    return {"cards": cards}


def safe_hook_instance(request: dict) -> str:
    value = str(request.get("hookInstance", ""))
    return value if re.fullmatch(r"[A-Za-z0-9-]{1,64}", value) else ""


def now() -> float:
    return time.time()
