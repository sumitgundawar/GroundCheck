"""CDS Hooks: medication safety cards inside the EHR's ordering screen.

The EHR calls this service when a clinician selects or signs a medication
order (the order-select and order-sign hooks). GroundCheckHealth builds the patient
from the prefetched FHIR resources (app/fhir.py), checks each draft
MedicationRequest against the formulary (app/patient_checks.py), and returns a
card for every finding: critical for unsafe, warning for checks, info for
notes, each naming its rule and source.

Calls must carry a JWT signed by a trusted EHR (CDS_HOOKS_TRUSTED maps each
issuer to its JWKS URL), audience this service's URL, not expired. Only for
testing with a sandbox, CDS_HOOKS_ALLOW_UNSIGNED accepts unsigned calls.

Specification: https://cds-hooks.hl7.org"""

from __future__ import annotations

from decimal import Decimal

import math
import re
import threading
import time
import uuid

import httpx

from . import config, fhir, formulary, messages, patient_checks
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


HOOKS = ("order-select", "order-sign")
FETCH_SECONDS = 4.0          # one FHIR read made on the EHR's behalf
FETCH_BUDGET_SECONDS = 8.0   # all of them: an ordering screen won't wait longer
_PREFETCH_TYPES = {"observations": "Observation", "allergies": "AllergyIntolerance",
                   "medications": "MedicationRequest", "conditions": "Condition"}
_PREFETCH_LABELS = {"observations": "lab results", "allergies": "allergies", "medications": "current medicines",
                    "conditions": "conditions"}


class CdsError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def discovery() -> dict:
    """What this service tells an EHR it does.

    An EHR decides whether to install a service from this description, so it
    has to say which formulary the checks come from. Advertising allergy and
    interaction checking without saying that the demonstration formulary is
    invented is how a service that knows no real medicine gets installed in a
    real ordering screen.
    """
    loaded = formulary.load()
    description = ("Checks a medication order against the patient's allergies, other medicines, age, weight, kidney "
                   f"and liver function, pregnancy and lab results, using the {loaded.name} formulary. "
                   "A medicine that formulary does not contain is reported as not checked.")
    requirements = "Patient-context medication ordering."
    if loaded.synthetic:
        description = ("DEMONSTRATION ONLY — the formulary behind this service is synthetic: every medicine, "
                       "dose and rule in it is invented, and no real medicine is known to it. It must not be "
                       "installed where it can influence prescribing. " + description)
        requirements = "Demonstration only. Not for clinical use: the formulary is synthetic."
    return {"services": [
        {"hook": hook, "id": f"{SERVICE_ID}-{hook}", "title": "GroundCheckHealth medication safety"
         + (" (demonstration, synthetic formulary)" if loaded.synthetic else ""),
         "description": description, "prefetch": PREFETCH, "usageRequirements": requirements}
        for hook in ("order-select", "order-sign")
    ]}


def _jwks(url: str) -> dict:
    try:
        with httpx.Client(timeout=5.0, transport=transport) as client:
            body = client.get(url).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise CdsError("Couldn't load the EHR's signing keys.", 401) from exc
    if not isinstance(body, dict) or not isinstance(body.get("keys"), list):
        raise CdsError("The EHR's signing keys aren't a JWK set.", 401)
    return body


# A token is accepted once. EHRs sign a new one (new jti) for every call, so a
# repeat is a replay. Kept in memory per process, until the token would have
# expired anyway.
_seen_jti: dict[tuple[str, str], float] = {}
_seen_lock = threading.Lock()
_MAX_SEEN = 100_000


def _first_use(issuer: str, jti: str, expires: float) -> bool:
    now = time.time()
    with _seen_lock:
        if len(_seen_jti) > _MAX_SEEN // 2:
            for key in [k for k, exp in _seen_jti.items() if exp < now]:
                del _seen_jti[key]
            if len(_seen_jti) >= _MAX_SEEN:
                _seen_jti.clear()
        if _seen_jti.get((issuer, jti), 0) >= now:
            return False
        _seen_jti[(issuer, jti)] = expires + 60
        return True


def verify(authorization: str | None, audience: str) -> str:
    """Returns the calling EHR's issuer, or raises CdsError."""
    import jwt

    if not authorization:
        if config.CDS_HOOKS_ALLOW_UNSIGNED:
            return "unsigned"
        raise CdsError("A signed JWT from a trusted EHR is required.", 401)
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise CdsError("The authorization token is malformed.", 401)
    token = token.strip()
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise CdsError("The authorization token is malformed.", 401) from exc
    issuer = unverified.get("iss", "")
    jwks_url = config.CDS_HOOKS_TRUSTED.get(issuer) if isinstance(issuer, str) else None
    if not jwks_url:
        raise CdsError("This EHR isn't trusted to call GroundCheckHealth.", 401)
    if header.get("alg") not in ("RS384", "ES384", "RS256", "ES256"):
        raise CdsError("The token's algorithm isn't allowed.", 401)
    keys = _jwks(jwks_url)["keys"]
    jwk = next((k for k in keys if isinstance(k, dict) and k.get("kid") == header.get("kid")), None)
    if jwk is None:
        raise CdsError("The token was signed with an unknown key.", 401)
    try:
        claims = jwt.decode(token, jwt.PyJWK(jwk, algorithm=header["alg"]).key, algorithms=[header["alg"]],
                            audience=audience, issuer=issuer, leeway=30,
                            options={"require": ["exp", "iat", "iss", "aud", "jti"]})
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise CdsError("The token couldn't be verified.", 401) from exc
    jti = claims.get("jti")
    if not isinstance(jti, str) or not jti or len(jti) > 256:
        raise CdsError("The token's jti is missing or invalid.", 401)
    if not _first_use(issuer, jti, float(claims["exp"])):
        raise CdsError("This token was already used.", 401)
    return issuer


def _bundle_resources(value, resource_type: str) -> list[dict]:
    if not isinstance(value, dict):
        return []
    if value.get("resourceType") == resource_type:
        return [value]
    entries = value.get("entry")
    if not isinstance(entries, list):
        return []
    return [e["resource"] for e in entries if isinstance(e, dict) and isinstance(e.get("resource"), dict)
            and e["resource"].get("resourceType") == resource_type]


def _usable(prefetch: dict, key: str, resource_type: str) -> bool:
    """Whether a prefetch value is the resource or bundle it should be. An
    OperationOutcome, null or the wrong type means the EHR couldn't supply it."""
    value = prefetch.get(key)
    return isinstance(value, dict) and value.get("resourceType") in (resource_type, "Bundle")


def _tail(reference) -> str:
    return reference.rsplit("/", 1)[-1] if isinstance(reference, str) else ""


def _of_other_patient(resource: dict, patient_id: str) -> bool:
    ref = (resource.get("subject") or resource.get("patient")) if isinstance(resource.get("subject") or resource.get("patient"), dict) else None
    ref = ref.get("reference") if ref else None
    return bool(isinstance(ref, str) and ref.startswith("Patient/") and _tail(ref) != patient_id)


def _fetch_missing(request: dict, patient_id: str, need: list[str]) -> tuple[dict, str | None]:
    """Read what the EHR didn't prefetch from its FHIR server with the token it
    gave. Only servers this deployment allows are contacted. Returns
    ({prefetch key: resource or bundle}, reason it stopped early)."""
    server, auth = request.get("fhirServer"), request.get("fhirAuthorization")
    token = auth.get("access_token") if isinstance(auth, dict) else None
    if not (isinstance(server, str) and isinstance(token, str) and token):
        return {}, "no FHIR server access was provided"
    try:
        base = fhir.validate_base_url(server, list(config.SMART_ALLOWED_ISSUERS) + list(config.FHIR_OPEN_SERVERS))
    except fhir.FhirError:
        return {}, "the FHIR server isn't one this deployment may read"
    got: dict = {}
    deadline = time.monotonic() + FETCH_BUDGET_SECONDS
    codes = _codes
    queries = {"patient": ("Patient", None), "observations": ("Observation", {"patient": patient_id, "code": codes, "_sort": "-date"}),
               "allergies": ("AllergyIntolerance", {"patient": patient_id}),
               "medications": ("MedicationRequest", {"patient": patient_id, "status": "active"}),
               "conditions": ("Condition", {"patient": patient_id})}
    with fhir.FhirClient(base, token, transport=transport, timeout=FETCH_SECONDS) as client:
        for key in need:
            if time.monotonic() > deadline:
                return got, "reading the FHIR server took too long"
            kind, params = queries[key]
            try:
                if params is None:
                    got[key] = client.get(f"Patient/{patient_id}")
                else:
                    got[key] = {"resourceType": "Bundle",
                                "entry": [{"resource": r} for r in client.search(kind, params, max_pages=2)]}
            except fhir.FhirError as exc:
                return got, str(exc)
    return got, None


def _plain(amount: float) -> str:
    """A number as a clinician would write it. "%g" turns 1000000 into "1e+06",
    which the dose parser then reads as 6, and rounds anything past six
    significant figures; neither is acceptable for a dose."""
    # Decimal, not a fixed number of places: "%.4f" rounded 0.000001 to "0"
    # and 0.00005 to "0.0001", which is a dose the order did not say.
    text = format(Decimal(str(amount)).normalize(), "f")
    return text if text and text != "-0" else "0"


def _frequency(repeat: dict) -> str:
    """How often, in words the dose checks read. A frequency this cannot express is left out, never turned into once daily
    by a table miss: four times a day must not look like once."""
    count, period, unit = repeat.get("frequency", 1), repeat.get("period", 1), repeat.get("periodUnit", "d")
    numbers = [x for x in (count, period) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0]
    if len(numbers) != 2 or not isinstance(unit, str):
        return ""
    known = _FREQUENCY.get((count, period, unit))
    if known:
        return known
    per_day = {"d": count / period, "h": count * 24 / period, "min": count * 1440 / period}.get(unit)
    if per_day is None:
        return ""
    if per_day <= 1:
        hours = 24 / per_day if per_day else 0
        return f"every {_plain(hours)} hours" if hours and float(hours).is_integer() and per_day < 1 else ("once daily" if per_day == 1 else "")
    if float(per_day).is_integer() and per_day <= 24:
        return f"{int(per_day)} times daily"
    return ""


def _dosages(order: dict) -> list[dict]:
    raw = order.get("dosageInstruction")
    found = [d for d in raw if isinstance(d, dict)] if isinstance(raw, list) else []
    return found or [{}]


def _dose_of(dosage: dict) -> tuple[str, float | None, str]:
    rates = dosage.get("doseAndRate") if isinstance(dosage.get("doseAndRate"), list) else []
    quantity = next((d["doseQuantity"] for d in rates if isinstance(d, dict) and isinstance(d.get("doseQuantity"), dict)
                     and d["doseQuantity"]), None)
    timing = dosage.get("timing") if isinstance(dosage.get("timing"), dict) else {}
    repeat = timing.get("repeat") if isinstance(timing.get("repeat"), dict) else {}
    frequency = _frequency(repeat)
    text = dosage.get("text") if isinstance(dosage.get("text"), str) else ""
    value = quantity.get("value") if quantity else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return text, None, ""
    raw_unit = quantity.get("unit") or quantity.get("code") or ""
    unit = (raw_unit if isinstance(raw_unit, str) else "").replace("ml", "mL")
    amount = float(value)
    return f"{_plain(amount)} {unit} {frequency}".strip(), amount, unit


def _order_dose(order: dict) -> tuple[str, float | None, str]:
    """(sentence, amount, unit) describing an order's first dosage instruction."""
    return _dose_of(_dosages(order)[0])


def _order_sentences(order: dict) -> list[str]:
    """Every dosage instruction is checked, not only the first: a second line is still a dose the patient is given."""
    seen = []
    for dosage in _dosages(order):
        sentence = _dose_of(dosage)[0]
        if sentence not in seen:
            seen.append(sentence)
    return seen


def _order_name(order: dict) -> str:
    name = fhir.concept_text(order.get("medicationCodeableConcept"))
    if name:
        return name
    ref = order.get("medicationReference") if isinstance(order.get("medicationReference"), dict) else {}
    target = ref.get("reference", "")
    if isinstance(target, str) and target.startswith("#"):
        contained = next((c for c in order.get("contained", []) if isinstance(c, dict) and c.get("id") == target[1:]), None)
        name = fhir.concept_text((contained or {}).get("code"))
    return name or (ref.get("display") if isinstance(ref.get("display"), str) else "")


def _clip(text: str, n: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _say(loc: str, key: str, english: str, **values: object) -> str:
    """The wording for a card: the English written in the code, or the locale file's under `key`."""
    return english if loc == "en" else (messages.message(key, loc, **values) or english)


def _notice(summary: str, detail: str, code: str, indicator: str = "warning", loc: str = "en") -> dict:
    return {"uuid": str(uuid.uuid4()), "summary": summary[:140], "indicator": indicator, "detail": detail,
            "source": {"label": _say(loc, "msg.cds.source", "GroundCheckHealth formulary"),
                       "topic": {"code": code, "system": "https://groundcheckhealth.com/cds/finding"}},
            "overrideReasons": []}


def validate_request(request, service_id: str | None = None) -> str:
    """Checks a request against the CDS Hooks 2.0 shape and returns its hook."""
    if not isinstance(request, dict):
        raise CdsError("The request must be a JSON object.")
    hook = request.get("hook")
    if hook not in HOOKS:
        raise CdsError("This service handles order-select and order-sign.")
    if service_id is not None and service_id != f"{SERVICE_ID}-{hook}":
        raise CdsError(f"This service handles the {service_id.removeprefix(SERVICE_ID + '-')} hook, not {hook}.")
    instance = request.get("hookInstance")
    try:
        if not isinstance(instance, str) or str(uuid.UUID(instance)) != instance.lower():
            raise ValueError
    except ValueError:
        raise CdsError("hookInstance must be a UUID.") from None
    context = request.get("context")
    if not isinstance(context, dict):
        raise CdsError("The request needs a context object.")
    patient_id = context.get("patientId")
    if not isinstance(patient_id, str) or not re.fullmatch(r"[A-Za-z0-9\-.]{1,64}", patient_id):
        raise CdsError("context.patientId is required.")
    if not isinstance(context.get("userId"), str) or not context["userId"]:
        raise CdsError("context.userId is required.")
    if request.get("prefetch") is not None and not isinstance(request["prefetch"], dict):
        raise CdsError("prefetch must be an object.")
    if request.get("context", {}).get("draftOrders") is not None and not isinstance(context["draftOrders"], dict):
        raise CdsError("context.draftOrders must be a FHIR Bundle.")
    if hook == "order-select" and not isinstance(context.get("selections", []), list):
        raise CdsError("context.selections must be an array.")
    return hook


def cards_for(request: dict, service_id: str | None = None, accept_language: str | None = None) -> dict:
    hook = validate_request(request, service_id)
    context = request["context"]
    # The language of the cards: the hook's own, else the caller's Accept-Language, else English.
    loc = messages.pick_locale(context.get("language"), context.get("locale"), request.get("language"),
                               request.get("locale"), accept_language)
    patient_id = context["patientId"]
    prefetch = dict(request.get("prefetch") or {})
    missing = [k for k, t in (("patient", "Patient"), *_PREFETCH_TYPES.items()) if not _usable(prefetch, k, t)]
    fetched, stopped = ({}, None)
    if missing:
        fetched, stopped = _fetch_missing(request, patient_id, missing)
        prefetch.update({k: v for k, v in fetched.items() if _usable(fetched, k, "Patient" if k == "patient" else _PREFETCH_TYPES[k])})
    patient = prefetch.get("patient") if _usable(prefetch, "patient", "Patient") else None
    if not patient:
        raise CdsError("The patient prefetch is required, and the patient couldn't be read from the EHR"
                       + (f" ({stopped})." if stopped else "."), 412)
    if patient.get("id") and patient["id"] != patient_id:
        raise CdsError("The prefetched patient isn't the patient in context.", 412)
    gaps = [k for k in _PREFETCH_TYPES if not _usable(prefetch, k, _PREFETCH_TYPES[k])]
    wrong_patient = 0

    def mine(key: str) -> list[dict]:
        nonlocal wrong_patient
        found = _bundle_resources(prefetch.get(key), _PREFETCH_TYPES[key])
        kept = [r for r in found if not _of_other_patient(r, patient_id)]
        wrong_patient += len(found) - len(kept)
        return kept

    loaded = fhir.map_patient(patient, mine("observations"), mine("allergies"), mine("medications"), mine("conditions"),
                              {"system": "CDS Hooks"})
    if wrong_patient:
        loaded.warnings.append(f"{wrong_patient} prefetched resources belonged to a different patient and were ignored.")
    orders = _bundle_resources(context.get("draftOrders"), "MedicationRequest")
    selections = context.get("selections")
    selected = {x for x in selections if isinstance(x, str)} if isinstance(selections, list) else set()
    if hook == "order-select" and selected:
        orders = [o for o in orders if f"MedicationRequest/{o.get('id')}" in selected]

    idx = formulary.index()
    cards = []
    for order in orders:
        name = _order_name(order)
        medicine = idx.find(name) or next(iter(idx.mentioned(name)), None)
        if medicine is None:
            # Never say nothing. In an ordering screen an empty response reads
            # as "checked, no concerns", so a medicine this formulary has never
            # heard of has to say so out loud — it is the one case where
            # silence is read as clearance.
            cards.append({
                "uuid": str(uuid.uuid4()),
                "summary": (f"Not checked: {_clip(name) or 'this medicine'} is not in the formulary" if loc == "en" else
                            messages.message("msg.cds.not_checked.summary", loc, name=_clip(name) or messages.message(
                                "msg.cds.not_checked.fallback", loc))[:140]),
                "indicator": "warning",
                "detail": (f"**{name or 'This medicine'}** was not checked. It is not in the formulary "
                           f"GroundCheckHealth is running ({formulary.load().name}), so no allergy, interaction, "
                           f"kidney, liver, pregnancy or dose rule has been applied to this order.\n\n"
                           f"This is not a statement that the order is safe." if loc == "en" else
                           messages.message("msg.cds.not_checked.detail", loc, formulary=formulary.load().name,
                                            name=name or messages.message("msg.cds.not_checked.fallback_cap", loc))),
                "source": {"label": _say(loc, "msg.cds.source", "GroundCheckHealth formulary"),
                           "topic": {"code": "not_in_formulary",
                                     "system": "https://groundcheckhealth.com/cds/finding"}},
                "overrideReasons": [],
            })
            continue
        sentences = _order_sentences(order)
        sentence = "; ".join(x for x in sentences if x)
        claims = [Claim(text=f"{medicine.name} {x}.", source_ids=["ORDER"], grounded=True, grounding_score=1.0)
                  for x in sentences]
        if not any(patient_checks._VALUE.search(x) for x in sentences):
            given = any(_dosages(order)[0].values())
            kind = "given" if given else "none"
            cards.append(_notice(
                _say(loc, f"msg.cds.dose.summary_{kind}", f"Dose not checked: {_clip(medicine.name, 40)} has no readable dose"
                     if given else f"Dose not checked: {_clip(medicine.name, 40)} has no dose", name=_clip(medicine.name, 40)),
                _say(loc, f"msg.cds.dose.detail_{kind}", f"The order for **{medicine.name}** "
                     + ("gives its dose in a form GroundCheckHealth cannot read (for example a range or free text without an amount), "
                        if given else "does not state a dose, ")
                     + "so no dose limit was applied. Other checks (allergy, interaction, kidney, liver, pregnancy) still ran. "
                     "This is not a statement that the dose is safe.", name=medicine.name),
                "dose_not_checked", "warning" if given else "info", loc))
        # The order itself: "the dose of X" makes the checks treat it as giving the medicine.
        others = PatientContext(**{**loaded.context.model_dump(),
                                   "medicines": [m for m in loaded.context.medicines
                                                 if medicine not in idx.mentioned(m)]})
        review = patient_checks.review(f"dose of {medicine.name}", claims, others, subject="order")
        for finding in review.findings:
            if finding.code == "already_taking":
                continue
            said = messages.finding_text(finding, loc) if loc != "en" else None
            card_loc = loc if said else "en"
            message = said or finding.message
            summary = message if len(message) <= 140 else message[:137].rstrip() + "…"
            shown = sentence
            if card_loc != "en":
                shown = "; ".join(messages.phrase(x, card_loc) or x for x in sentences if x)
            detail = f"**{medicine.name}** ({shown or _say(card_loc, 'msg.cds.dose_not_stated', 'dose not stated')}): {message}"
            if loaded.warnings:
                detail += "\n\n" + (_say(card_loc, "msg.cds.warnings", "") + "\n" if card_loc != "en" else "") + "\n".join(f"- {w}" for w in loaded.warnings)
            cards.append({
                "uuid": str(uuid.uuid4()),
                "summary": summary,
                "indicator": INDICATOR[finding.severity],
                "detail": detail,
                "source": {"label": _say(card_loc, "msg.cds.source", "GroundCheckHealth formulary") + (f" ({finding.rule_id})" if finding.rule_id else ""),
                           "topic": {"code": finding.code, "system": "https://groundcheckhealth.com/cds/finding"}},
                "overrideReasons": [
                    {"code": "clinically-appropriate", "system": "https://groundcheckhealth.com/cds/override",
                     "display": _say(card_loc, "msg.cds.override.clinically", "Clinically appropriate for this patient")},
                    {"code": "data-out-of-date", "system": "https://groundcheckhealth.com/cds/override",
                     "display": _say(card_loc, "msg.cds.override.outdated", "The patient data is out of date")},
                ] if finding.severity == "block" else [],
            })
    if orders and gaps:
        what = ", ".join(_PREFETCH_LABELS[k] for k in gaps)
        said = ", ".join(_say(loc, f"msg.cds.gap.{k}", _PREFETCH_LABELS[k]) for k in gaps)
        more = _say(loc, "msg.cds.partly.stopped", f", and they couldn't be read from its FHIR server ({stopped})",
                    stopped=stopped) if stopped else ""
        cards.append(_notice(
            _say(loc, "msg.cds.partly.summary", f"Partly checked: {what} not available", what=said),
            _say(loc, "msg.cds.partly.detail", f"The EHR did not supply the patient's {what}{more}. Checks that need them "
                 "were not applied, so no card here is a statement that the order is safe.", what=said, stopped=more),
            "patient_data_missing", "warning", loc))
    if orders and (loaded.serious or wrong_patient):
        lines = "\n".join(f"- {w}" for w in loaded.serious + loaded.warnings[-1:] * bool(wrong_patient))
        cards.append(_notice(
            _say(loc, "msg.cds.unreadable.summary", "Some patient data could not be read"),
            _say(loc, "msg.cds.unreadable.detail", "Parts of the patient's record were invalid or unreadable, so a check "
                 "may be missing:\n\n" + lines, lines=lines),
            "patient_data_unreadable", "info", loc))
    order_rank = {"critical": 0, "warning": 1, "info": 2}
    cards.sort(key=lambda c: order_rank[c["indicator"]])
    return {"cards": cards}


def safe_hook_instance(request: dict) -> str:
    value = str(request.get("hookInstance", "")) if isinstance(request, dict) else ""
    return value if re.fullmatch(r"[A-Za-z0-9-]{1,64}", value) else ""


def now() -> float:
    return time.time()


def read_json(raw: bytes) -> dict:
    """A request body as a JSON object, or a CdsError(400). Handles bodies too
    deeply nested for the JSON parser, which raise RecursionError, not ValueError."""
    import json

    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        raise CdsError("The request isn't valid JSON.") from None
    if not isinstance(body, dict):
        raise CdsError("The request must be a JSON object.")
    return body


def _token(value, limit: int) -> str:
    """An identifier or code from the EHR, logged only if it looks like one: free text here could hold a patient's name."""
    text = str(value if value is not None else "")[:limit]
    return text if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", text) else ("" if not text else "other")


def feedback_items(body, limit: int = 50) -> list[dict]:
    """The usable entries of a feedback body: (card, outcome, override reason code), clipped for logging."""
    entries = body.get("feedback") if isinstance(body, dict) else None
    out = []
    for item in (entries if isinstance(entries, list) else [])[:limit]:
        if not isinstance(item, dict):
            continue
        reason = item.get("overrideReason") if isinstance(item.get("overrideReason"), dict) else {}
        code = reason.get("reason") if isinstance(reason.get("reason"), dict) else {}
        out.append({"card": _token(item.get("card"), 64), "outcome": _token(item.get("outcome"), 20),
                    "reason": _token(code.get("code"), 40)})
    return out
