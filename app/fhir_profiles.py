"""FHIR profile checks and R4 structure checks. Pure: no network, never raises.

Two jobs:

1. Profile validation (a US Core / UK Core subset) of the resources patient
   checks are built from: Patient, Observation, AllergyIntolerance,
   MedicationRequest and Condition. `validate_resource` reports every issue;
   `check_warnings` keeps the ones the checks rely on and folds them into
   a few short, PHI-free sentences, so a record missing an element the checks
   need is never silently treated as "nothing to check".

2. Structural validation against FHIR R4 of the resources GroundCheckHealth
   writes back: DocumentReference, Provenance and AuditEvent
   (`validate_r4`). A small table of element types, cardinalities and code
   bindings taken from the R4 definitions, not a full validator.

Hostile input (deep nesting, huge strings, oversized resources) is reported as
an error, not followed."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_DEPTH = 40
MAX_STRING = 1_000_000
MAX_NODES = 200_000
MAX_RESOURCES = 1000
UCUM = "http://unitsofmeasure.org"

_YEAR = r"(?:[0-9](?:[0-9](?:[0-9][1-9]|[1-9]0)|[1-9]00)|[1-9]000)"
_DATE = rf"{_YEAR}(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12][0-9]|3[01]))?)?"
_TIME = r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]{1,9})?"
_ZONE = r"(?:Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))"
DATE_RE = re.compile(rf"^{_DATE}$")
DATETIME_RE = re.compile(rf"^{_YEAR}(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12][0-9]|3[01])(?:T{_TIME}{_ZONE})?)?)?$")
INSTANT_RE = re.compile(rf"^{_YEAR}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T{_TIME}{_ZONE}$")
_TIME_NO_ZONE = re.compile(rf"^{_YEAR}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T{_TIME}$")
ID_RE = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")
CODE_RE = re.compile(r"^[^\s]+(?: [^\s]+)*$")
_DAYS = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

# UCUM codes that carry the quantities the checks use, in their canonical case.
UCUM_CODES = (
    "mL/min/{1.73_m2}", "mL/min/1.73m2", "mL/min", "umol/L", "mmol/L", "mg/dL", "g/dL", "g/L", "mg/L", "ug/L", "ng/mL",
    "kg", "g", "mg", "ug", "[lb_av]", "[oz_av]", "[in_i]", "cm", "m", "mm[Hg]", "Cel", "[degF]", "ms", "s", "min", "h",
    "d", "wk", "mo", "a", "U/L", "[IU]/L", "[iU]/L", "mEq/L", "meq/L", "10*3/uL", "10*9/L", "10*3/mm3", "/uL", "/mm3",
    "{cells}/uL", "%", "{INR}", "1", "L", "mL", "kg/m2", "/min", "{beats}/min", "mmol/mol", "mL/min/{1.73_m2}",
)
_UCUM_CANON = {c.lower(): c for c in UCUM_CODES}
_UCUM_SYNTAX = re.compile(r"^(?:[A-Za-z%\[\]_'*^0-9{}.\-+]+)(?:[/.][A-Za-z%\[\]_'*^0-9{}.\-+]+)*$|^/[A-Za-z%\[\]_'*^0-9{}.\-+]+$")


@dataclass(frozen=True)
class Issue:
    severity: str          # "error" (invalid or required element missing) or "warning" (must-support missing)
    resource: str          # resource type
    path: str
    message: str
    relied_on: bool = False   # GroundCheckHealth's checks use this element

    def __str__(self) -> str:
        return f"{self.resource}.{self.path}: {self.message}"


# --- Hostile input ----------------------------------------------------------------------------

def shape_problem(value) -> str | None:
    """Why a value is too deep, large or odd to process, or None. Iterative, so
    deep nesting can't exhaust the stack."""
    stack = [(value, 1)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > MAX_DEPTH:
            return f"nested more than {MAX_DEPTH} levels deep"
        if nodes > MAX_NODES:
            return f"more than {MAX_NODES:,} elements"
        if isinstance(item, str):
            if len(item) > MAX_STRING:
                return "contains a string over 1 MB"
        elif isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, float) and not math.isfinite(item):
            return "contains a number that isn't finite"
    return None


# --- Primitive types --------------------------------------------------------------------------

def valid_date(value) -> bool:
    if not isinstance(value, str) or not DATE_RE.match(value):
        return False
    if len(value) == 10:
        year, month, day = int(value[:4]), int(value[5:7]), int(value[8:10])
        return day <= (29 if month == 2 and (year % 4 == 0 and (year % 100 or year % 400 == 0)) else _DAYS[month] - (1 if month == 2 else 0))
    return True


def valid_datetime(value) -> bool:
    if not isinstance(value, str) or not DATETIME_RE.match(value):
        return False
    return valid_date(value[:10]) if len(value) >= 10 else True


def valid_instant(value) -> bool:
    return isinstance(value, str) and bool(INSTANT_RE.match(value)) and valid_date(value[:10])


def datetime_problem(value) -> str | None:
    """Why a date/dateTime is wrong, in words, or None."""
    if isinstance(value, str) and _TIME_NO_ZONE.match(value):
        return "has a time but no time zone, which FHIR requires"
    return None if valid_datetime(value) else "isn't a valid FHIR date or dateTime"


def ucum_canonical(code: str) -> str | None:
    """The canonical (case-correct) UCUM code for a known unit, or None."""
    return _UCUM_CANON.get(code.strip().lower()) if isinstance(code, str) else None


def ucum_syntax_ok(code) -> bool:
    return isinstance(code, str) and 0 < len(code) <= 64 and bool(_UCUM_SYNTAX.match(code))


# --- Profile validation -----------------------------------------------------------------------

def profile_of(resource: dict) -> str:
    """"uk-core" when the resource claims a UK Core profile, otherwise "us-core"."""
    profiles = (resource.get("meta") or {}).get("profile") if isinstance(resource.get("meta"), dict) else None
    if isinstance(profiles, list) and any(isinstance(p, str) and ("fhir.org.uk" in p or "UKCore" in p) for p in profiles):
        return "uk-core"
    return "us-core"


def _is_concept(value) -> bool:
    return isinstance(value, dict)


def _codings(concept) -> list[dict]:
    coding = concept.get("coding") if isinstance(concept, dict) else None
    return [c for c in coding if isinstance(c, dict)] if isinstance(coding, list) else []


def _has_code(concept) -> bool:
    return any((isinstance(c.get("code"), str) and c["code"]) or (isinstance(c.get("display"), str) and c["display"])
               for c in _codings(concept)) or \
        bool(isinstance(concept, dict) and isinstance(concept.get("text"), str) and concept["text"].strip())


class _Collector:
    def __init__(self, resource_type: str):
        self.type, self.issues = resource_type, []

    def add(self, severity: str, path: str, message: str, relied: bool = False) -> None:
        self.issues.append(Issue(severity, self.type, path, message, relied))

    def single(self, resource: dict, name: str, kind: str, required: bool, relied: bool = False, ms: bool = False):
        """An element that may occur once. Returns the value when usable."""
        if name not in resource or resource[name] is None:
            if required:
                self.add("error", name, "is required but missing", relied)
            elif ms:
                self.add("warning", name, "is a must-support element and is missing", relied)
            return None
        value = resource[name]
        if isinstance(value, list):
            self.add("error", name, "may occur only once but is an array", relied)
            return None
        if not self.is_kind(value, kind):
            self.add("error", name, f"isn't a valid {kind}", relied)
            return None
        return value

    def many(self, resource: dict, name: str, required: bool, relied: bool = False, ms: bool = False):
        if name not in resource or resource[name] is None:
            if required:
                self.add("error", name, "is required (1..*) but missing", relied)
            elif ms:
                self.add("warning", name, "is a must-support element and is missing", relied)
            return []
        value = resource[name]
        if not isinstance(value, list):
            self.add("error", name, "can occur many times and must be an array", relied)
            return []
        if required and not value:
            self.add("error", name, "is required (1..*) but empty", relied)
        return value

    def is_kind(self, value, kind: str) -> bool:
        if kind == "string":
            return isinstance(value, str) and bool(value.strip())
        if kind == "code":
            return isinstance(value, str) and bool(CODE_RE.match(value))
        if kind == "date":
            return valid_date(value)
        if kind == "dateTime":
            return valid_datetime(value)
        if kind == "instant":
            return valid_instant(value)
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "decimal":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if kind in ("CodeableConcept", "Reference", "Quantity", "Coding", "Period", "Identifier", "Attachment"):
            return isinstance(value, dict)
        return True


def _check_common(c: _Collector, resource: dict) -> None:
    for ext in c.many(resource, "extension", False):
        if not isinstance(ext, dict) or not isinstance(ext.get("url"), str) or not ext["url"]:
            c.add("error", "extension", "every extension needs a url")
    # An unknown modifierExtension can change what the resource means, so the
    # resource can't safely be read as if it weren't there.
    mods = c.many(resource, "modifierExtension", False)
    if mods:
        c.add("warning", "modifierExtension", "has a modifier extension GroundCheckHealth doesn't understand, which may "
              "change the meaning of this resource", True)
    meta = resource.get("meta")
    if meta is not None and not isinstance(meta, dict):
        c.add("error", "meta", "isn't an object")
    if "id" in resource and not (isinstance(resource["id"], str) and ID_RE.match(resource["id"])):
        c.add("error", "id", "isn't a valid FHIR id (letters, digits, - and ., up to 64)")


def _reference_ok(value) -> bool:
    if not isinstance(value, dict):
        return False
    ref = value.get("reference")
    if ref is not None and not (isinstance(ref, str) and 0 < len(ref) <= 2048):
        return False
    return any(value.get(k) for k in ("reference", "identifier", "display"))


UNITLESS_LOINC = {"6301-6", "34714-6"}   # INR


def _quantity_issues(c: _Collector, path: str, quantity: dict, unitless: bool = False) -> None:
    value = quantity.get("value")
    if value is None:
        c.add("error", f"{path}.value", "is missing, so there is no result to use", True)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        c.add("error", f"{path}.value", "isn't a number", True)
        return
    code, unit, system = quantity.get("code"), quantity.get("unit"), quantity.get("system")
    if code is None and unit is None:
        if unitless:
            return
        c.add("error", f"{path}.unit", "has a value but no unit, so it can't be used safely", True)
        return
    if code is not None:
        if not ucum_syntax_ok(code):
            c.add("error", f"{path}.code", "isn't a UCUM unit code", True)
        else:
            canon = ucum_canonical(code)
            if canon is None:
                c.add("warning", f"{path}.code", "isn't a UCUM unit GroundCheckHealth recognises", True)
            elif canon != code:
                c.add("warning", f"{path}.code", f"UCUM codes are case-sensitive; expected {canon}", True)
        if system is None:
            c.add("warning", f"{path}.system", "should be http://unitsofmeasure.org when a code is given", True)
        elif system != UCUM:
            c.add("error", f"{path}.system", "must be http://unitsofmeasure.org for quantity codes", True)
    elif isinstance(unit, str) and ucum_canonical(unit) is None:
        c.add("warning", f"{path}.unit", "has no UCUM code and the unit text isn't recognised", True)


def _status_code(concept) -> str:
    return next((x.get("code", "") for x in _codings(concept) if isinstance(x.get("code"), str)), "")


def _validate_patient(c: _Collector, r: dict, uk: bool) -> None:
    bd = c.single(r, "birthDate", "date", False, relied=True, ms=True)
    gender = c.single(r, "gender", "code", False, relied=True, ms=True)
    if gender is not None and gender not in ("male", "female", "other", "unknown"):
        c.add("error", "gender", "must be male, female, other or unknown", True)
    if isinstance(bd, str) and valid_date(bd) and int(bd[:4]) > _now_year():
        c.add("error", "birthDate", "is in the future", True)
    for name, kind_ok in (("identifier", lambda i: isinstance(i, dict) and (i.get("value") or i.get("system"))),
                          ("name", lambda n: isinstance(n, dict) and any(n.get(k) for k in ("family", "given", "text")))):
        items = c.many(r, name, False)
        if name not in r:
            c.add("error" if not uk or name == "name" else "warning", name, "is required by the profile but missing")
        elif not items:
            c.add("error", name, "is required (1..*) but empty")
        for item in items:
            if not kind_ok(item):
                c.add("error", name, "has an entry with no usable content")
    for key in ("deceasedBoolean", "deceasedDateTime"):
        if key in r and not c.is_kind(r[key], "boolean" if key == "deceasedBoolean" else "dateTime"):
            c.add("error", key, f"isn't a valid {'boolean' if key == 'deceasedBoolean' else 'dateTime'}")


def _now_year() -> int:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).year


def _validate_observation(c: _Collector, r: dict, uk: bool) -> None:
    status = c.single(r, "status", "code", True, relied=True)
    if status is not None and status not in ("registered", "preliminary", "final", "amended", "corrected", "cancelled",
                                              "entered-in-error", "unknown"):
        c.add("error", "status", "isn't an Observation status", True)
    code = c.single(r, "code", "CodeableConcept", True, relied=True)
    if code is not None:
        if not _codings(code):
            c.add("error", "code.coding", "needs at least one coding with a system and code", True)
        for coding in _codings(code):
            if not coding.get("system") or not coding.get("code"):
                c.add("error", "code.coding", "each coding needs a system and a code", True)
    cats = c.many(r, "category", not uk, ms=True)
    for cat in cats:
        if not _is_concept(cat):
            c.add("error", "category", "isn't a CodeableConcept")
    subject = c.single(r, "subject", "Reference", not uk, ms=uk)
    if subject is not None and not _reference_ok(subject):
        c.add("error", "subject", "isn't a usable Reference")
    effective_keys = [k for k in ("effectiveDateTime", "effectivePeriod", "effectiveInstant") if k in r]
    if not effective_keys and "issued" not in r:
        c.add("warning", "effective[x]", "is missing, so the result can't be dated and may be out of date", True)
    for key in effective_keys:
        if key == "effectivePeriod":
            period = r[key]
            if not isinstance(period, dict):
                c.add("error", key, "isn't a Period", True)
                continue
            for end in ("start", "end"):
                if end in period and (problem := datetime_problem(period[end])):
                    c.add("error", f"{key}.{end}", problem, True)
        else:
            problem = datetime_problem(r[key]) if key == "effectiveDateTime" else (
                None if valid_instant(r[key]) else "isn't a valid instant")
            if problem:
                c.add("error", key, problem, True)
    if "issued" in r and not valid_instant(r["issued"]):
        c.add("error", "issued", "isn't a valid instant", True)
    values = [k for k in r if k.startswith("value")]
    if not values and not r.get("dataAbsentReason") and not r.get("component"):
        c.add("warning", "value[x]", "is missing and there is no data-absent reason", True)
    if len(values) > 1:
        c.add("error", "value[x]", "has more than one value[x]", True)
    if "valueQuantity" in r:
        if isinstance(r["valueQuantity"], dict):
            _quantity_issues(c, "valueQuantity", r["valueQuantity"],
                             any(x.get("code") in UNITLESS_LOINC for x in _codings(code)))
        else:
            c.add("error", "valueQuantity", "isn't a Quantity", True)
    if "valueCodeableConcept" in r and not isinstance(r["valueCodeableConcept"], dict):
        c.add("error", "valueCodeableConcept", "isn't a CodeableConcept", True)


def _validate_allergy(c: _Collector, r: dict, uk: bool) -> None:
    code = c.single(r, "code", "CodeableConcept", True, relied=True)
    if code is not None and not _has_code(code):
        c.add("error", "code", "has no code or text, so this allergy can't be named and would be missed", True)
    clinical = c.single(r, "clinicalStatus", "CodeableConcept", False, relied=True, ms=True)
    verification = c.single(r, "verificationStatus", "CodeableConcept", False, relied="verificationStatus" in r, ms=True)
    for name, concept, allowed in (("clinicalStatus", clinical, ("active", "inactive", "resolved")),
                                   ("verificationStatus", verification,
                                    ("unconfirmed", "confirmed", "refuted", "entered-in-error"))):
        if concept is not None and _status_code(concept) not in allowed:
            c.add("error", name, f"code isn't one of {', '.join(allowed)}", True)
    patient = c.single(r, "patient", "Reference", True)
    if patient is not None and not _reference_ok(patient):
        c.add("error", "patient", "isn't a usable Reference")
    for reaction in c.many(r, "reaction", False, ms=True):
        if not isinstance(reaction, dict) or not isinstance(reaction.get("manifestation"), list) or not reaction["manifestation"]:
            c.add("error", "reaction.manifestation", "is required (1..*) in each reaction")


def _validate_medication_request(c: _Collector, r: dict, uk: bool) -> None:
    status = c.single(r, "status", "code", True, relied=True)
    if status is not None and status not in ("active", "on-hold", "cancelled", "completed", "entered-in-error", "stopped",
                                              "draft", "unknown"):
        c.add("error", "status", "isn't a MedicationRequest status", True)
    intent = c.single(r, "intent", "code", True, relied=True)
    if intent is not None and intent not in ("proposal", "plan", "order", "original-order", "reflex-order", "filler-order",
                                              "instance-order", "option"):
        c.add("error", "intent", "isn't a MedicationRequest intent", True)
    has_cc, has_ref = "medicationCodeableConcept" in r, "medicationReference" in r
    if not (has_cc or has_ref):
        c.add("error", "medication[x]", "is required but missing, so the medicine is unknown", True)
    elif has_cc and has_ref:
        c.add("error", "medication[x]", "has both medicationCodeableConcept and medicationReference", True)
    elif has_cc and not _has_code(r["medicationCodeableConcept"]):
        c.add("error", "medicationCodeableConcept", "has no code or text, so the medicine can't be named", True)
    elif has_ref and not _reference_ok(r["medicationReference"]):
        c.add("error", "medicationReference", "isn't a usable Reference", True)
    subject = c.single(r, "subject", "Reference", True)
    if subject is not None and not _reference_ok(subject):
        c.add("error", "subject", "isn't a usable Reference")
    if "authoredOn" in r and (problem := datetime_problem(r["authoredOn"])):
        c.add("error", "authoredOn", problem)
    elif "authoredOn" not in r:
        c.add("warning", "authoredOn", "is a must-support element and is missing")
    if "requester" not in r:
        c.add("warning", "requester", "is a must-support element and is missing")
    if "dosageInstruction" not in r:
        c.add("warning", "dosageInstruction", "is a must-support element and is missing")
    else:
        c.many(r, "dosageInstruction", False)


def _validate_condition(c: _Collector, r: dict, uk: bool) -> None:
    code = c.single(r, "code", "CodeableConcept", True, relied=True)
    if code is not None and not _has_code(code):
        c.add("error", "code", "has no code or text, so this condition can't be named", True)
    clinical = c.single(r, "clinicalStatus", "CodeableConcept", False, relied=True, ms=True)
    verification = c.single(r, "verificationStatus", "CodeableConcept", False, relied="verificationStatus" in r, ms=True)
    for name, concept, allowed in (("clinicalStatus", clinical, ("active", "recurrence", "relapse", "inactive", "remission", "resolved")),
                                   ("verificationStatus", verification,
                                    ("unconfirmed", "provisional", "differential", "confirmed", "refuted", "entered-in-error"))):
        if concept is not None and _status_code(concept) not in allowed:
            c.add("error", name, f"code isn't one of {', '.join(allowed)}", True)
    subject = c.single(r, "subject", "Reference", True)
    if subject is not None and not _reference_ok(subject):
        c.add("error", "subject", "isn't a usable Reference")
    if "category" not in r and not uk:
        c.add("error", "category", "is required by the profile but missing")
    else:
        c.many(r, "category", False)
    for key in ("onsetDateTime", "abatementDateTime", "recordedDate"):
        if key in r and (problem := datetime_problem(r[key])):
            c.add("error", key, problem)


_VALIDATORS = {"Patient": _validate_patient, "Observation": _validate_observation,
               "AllergyIntolerance": _validate_allergy, "MedicationRequest": _validate_medication_request,
               "Condition": _validate_condition}


def validate_resource(resource, profile: str | None = None, expect: str | None = None) -> list[Issue]:
    """Issues in one resource against the US Core or UK Core subset (profile
    "us-core" or "uk-core"; by default from meta.profile). Never raises."""
    rtype = resource.get("resourceType") if isinstance(resource, dict) else None
    if not isinstance(resource, dict):
        return [Issue("error", str(expect or "resource"), "", "isn't a JSON object", True)]
    rtype = rtype if isinstance(rtype, str) else ""
    if expect and rtype != expect:
        return [Issue("error", expect, "resourceType", f"is {rtype or 'missing'}, expected {expect}", True)]
    validator = _VALIDATORS.get(rtype)
    if validator is None:
        return [Issue("error", rtype or "resource", "resourceType", "isn't one of the profiled resource types", False)]
    problem = shape_problem(resource)
    if problem:
        return [Issue("error", rtype, "", f"{problem}, so it was not read", True)]
    profile = profile or profile_of(resource)
    c = _Collector(rtype)
    try:
        _check_common(c, resource)
        validator(c, resource, profile == "uk-core")
    except Exception as exc:  # noqa: BLE001 - validation must never break loading a patient
        c.add("error", "", f"couldn't be validated ({type(exc).__name__})", True)
    return c.issues


def check_warnings(patient, observations, allergies, medications, conditions, profile: str | None = None,
                   limit: int = 12, severities: tuple[str, ...] = ("error", "warning")) -> list[str]:
    """Short warnings for elements the patient checks rely on that are missing
    or wrong, counted per message. No values from the record appear in them."""
    groups = (("Patient", [patient] if patient else []), ("Observation", observations),
              ("AllergyIntolerance", allergies), ("MedicationRequest", medications), ("Condition", conditions))
    found: dict[tuple[str, str, str], int] = {}
    totals: dict[str, int] = {}
    for rtype, resources in groups:
        resources = list(resources)[:MAX_RESOURCES]
        totals[rtype] = len(resources)
        for resource in resources:
            for issue in validate_resource(resource, profile, expect=rtype):
                if issue.relied_on and issue.severity in severities:
                    key = (rtype, issue.path, issue.message)
                    found[key] = found.get(key, 0) + 1
    out = []
    for (rtype, path, message), n in sorted(found.items(), key=lambda kv: (-kv[1], kv[0])):
        who = f"{n} of {totals[rtype]} {rtype} resources" if totals[rtype] > 1 else f"The {rtype}"
        out.append(f"{who}: {path} {message}.")
    if len(out) > limit:
        out = out[:limit] + [f"{len(out) - limit} more data-quality problems were not listed."]
    return out


# --- R4 structure of what is written back -------------------------------------------------------

_CC, _REF, _CODING, _ID = "CodeableConcept", "Reference", "Coding", "Identifier"
_BACKBONE_BASE = {"id": (0, 1, "id"), "extension": (0, "*", "Extension"), "modifierExtension": (0, "*", "Extension")}
_BASE = {"resourceType": (1, 1, "code"), "id": (0, 1, "id"), "meta": (0, 1, "Meta"), "implicitRules": (0, 1, "uri"),
         "language": (0, 1, "code"), "text": (0, 1, "Narrative"), "contained": (0, "*", "Resource"),
         "extension": (0, "*", "Extension"), "modifierExtension": (0, "*", "Extension")}

R4: dict[str, dict] = {
    "Provenance": {**_BASE,
                   "target": (1, "*", _REF), "occurredPeriod": (0, 1, "Period"), "occurredDateTime": (0, 1, "dateTime"),
                   "recorded": (1, 1, "instant"), "policy": (0, "*", "uri"), "location": (0, 1, _REF),
                   "reason": (0, "*", _CC), "activity": (0, 1, _CC), "patient": None,
                   "agent": (1, "*", {**_BACKBONE_BASE, "type": (0, 1, _CC), "role": (0, "*", _CC), "who": (1, 1, _REF),
                                      "onBehalfOf": (0, 1, _REF)}),
                   "entity": (0, "*", {**_BACKBONE_BASE, "role": (1, 1, "code", {"derivation", "revision", "quotation",
                                                                                   "source", "removal"}),
                                       "what": (1, 1, _REF), "agent": (0, "*", "Provenance.agent")}),
                   "signature": (0, "*", "Signature")},
    "AuditEvent": {**_BASE,
                   "type": (1, 1, _CODING), "subtype": (0, "*", _CODING),
                   "action": (0, 1, "code", {"C", "R", "U", "D", "E"}), "period": (0, 1, "Period"),
                   "recorded": (1, 1, "instant"), "outcome": (0, 1, "code", {"0", "4", "8", "12"}),
                   "outcomeDesc": (0, 1, "string"), "purposeOfEvent": (0, "*", _CC),
                   "agent": (1, "*", {**_BACKBONE_BASE, "type": (0, 1, _CC), "role": (0, "*", _CC), "who": (0, 1, _REF),
                                      "altId": (0, 1, "string"), "name": (0, 1, "string"), "requestor": (1, 1, "boolean"),
                                      "location": (0, 1, _REF), "policy": (0, "*", "uri"), "media": (0, 1, _CODING),
                                      "network": (0, 1, {**_BACKBONE_BASE, "address": (0, 1, "string"),
                                                         "type": (0, 1, "code", {"1", "2", "3", "4", "5"})}),
                                      "purposeOfUse": (0, "*", _CC)}),
                   "source": (1, 1, {**_BACKBONE_BASE, "site": (0, 1, "string"), "observer": (1, 1, _REF),
                                     "type": (0, "*", _CODING)}),
                   "entity": (0, "*", {**_BACKBONE_BASE, "what": (0, 1, _REF), "type": (0, 1, _CODING),
                                       "role": (0, 1, _CODING), "lifecycle": (0, 1, _CODING),
                                       "securityLabel": (0, "*", _CODING), "name": (0, 1, "string"),
                                       "description": (0, 1, "string"), "query": (0, 1, "base64Binary"),
                                       "detail": (0, "*", {**_BACKBONE_BASE, "type": (1, 1, "string"),
                                                           "valueString": (0, 1, "string"),
                                                           "valueBase64Binary": (0, 1, "base64Binary")})})},
    "DocumentReference": {**_BASE,
                          "masterIdentifier": (0, 1, _ID), "identifier": (0, "*", _ID), "status": (1, 1, "code", {
                              "current", "superseded", "entered-in-error"}),
                          "docStatus": (0, 1, "code", {"preliminary", "final", "amended", "entered-in-error"}),
                          "type": (0, 1, _CC), "category": (0, "*", _CC), "subject": (0, 1, _REF),
                          "date": (0, 1, "instant"), "author": (0, "*", _REF), "authenticator": (0, 1, _REF),
                          "custodian": (0, 1, _REF), "relatesTo": (0, "*", dict), "description": (0, 1, "string"),
                          "securityLabel": (0, "*", _CC),
                          "content": (1, "*", {**_BACKBONE_BASE, "attachment": (1, 1, "Attachment"), "format": (0, 1, _CODING)}),
                          "context": (0, 1, dict)},
}
R4["Provenance"].pop("patient")
R4["DocumentReference"]["relatesTo"] = (0, "*", dict)
R4["DocumentReference"]["context"] = (0, 1, dict)
_PRIMITIVES = {"string", "code", "id", "uri", "dateTime", "instant", "boolean", "base64Binary"}
_B64 = re.compile(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$")


def _r4_primitive(kind: str, value) -> str | None:
    if kind in ("string", "uri", "id", "code"):
        if not isinstance(value, str) or not value.strip():
            return f"must be a non-empty {kind}"
        if kind == "id" and not ID_RE.match(value):
            return "isn't a valid id"
        if kind == "code" and not CODE_RE.match(value):
            return "isn't a valid code"
        return None
    if kind == "boolean":
        return None if isinstance(value, bool) else "must be true or false"
    if kind == "dateTime":
        return datetime_problem(value)
    if kind == "instant":
        return None if valid_instant(value) else "isn't a valid instant (needs a date, time and time zone)"
    if kind == "base64Binary":
        return None if isinstance(value, str) and _B64.match(value) else "isn't valid base64"
    return None


def _r4_complex(kind: str, value) -> str | None:
    if not isinstance(value, dict):
        return f"must be a {kind} object"
    if kind == _REF and not _reference_ok(value):
        return "needs a reference, identifier or display"
    if kind == _CODING and not (value.get("code") or value.get("display")):
        return "needs a code or display"
    if kind == _CC and not (value.get("coding") or value.get("text")):
        return "needs a coding or text"
    if kind == "Attachment":
        if "contentType" in value and (not isinstance(value["contentType"], str) or "/" not in value["contentType"]):
            return "contentType must be a MIME type"
        if "data" in value and _r4_primitive("base64Binary", value["data"]):
            return "data isn't valid base64"
        if "creation" in value and datetime_problem(value["creation"]):
            return "creation isn't a valid dateTime"
    if kind == _CODING:
        for key in ("system", "code", "display"):
            if key in value and not isinstance(value[key], str):
                return f"{key} must be a string"
    if kind == "Period":
        for key in ("start", "end"):
            if key in value and datetime_problem(value[key]):
                return f"{key} isn't a valid dateTime"
    return None


def _check_element(spec: dict, data: dict, path: str, errors: list[str], root: dict, depth: int = 0) -> None:
    if depth > 6:
        return
    for key in data:
        if key not in spec and not key.startswith("_"):
            errors.append(f"{path}{key}: isn't an element of this resource in R4")
    for key, rule in spec.items():
        if rule is None:
            continue
        low, high, kind, *bind = rule
        bound = bind[0] if bind else None
        present = key in data and data[key] is not None
        many = high == "*"
        if not present:
            if low >= 1:
                errors.append(f"{path}{key}: is required ({low}..{high}) but missing")
            continue
        value = data[key]
        if many != isinstance(value, list) and key != "resourceType":
            errors.append(f"{path}{key}: {'must be an array' if many else 'may occur only once (0..1/1..1), not an array'}")
            continue
        items = value if many else [value]
        if many and low >= 1 and not items:
            errors.append(f"{path}{key}: is required (1..*) but empty")
        for i, item in enumerate(items):
            where = f"{path}{key}{'[%d]' % i if many else ''}"
            if kind == "Provenance.agent":
                kind_spec = R4["Provenance"]["agent"][2]
                if isinstance(item, dict):
                    _check_element(kind_spec, item, where + ".", errors, root, depth + 1)
                continue
            if isinstance(kind, dict):
                if not isinstance(item, dict):
                    errors.append(f"{where}: must be an object")
                else:
                    _check_element(kind, item, where + ".", errors, root, depth + 1)
            elif kind is dict or kind in ("Extension", "Meta", "Narrative", "Resource", "Signature"):
                if not isinstance(item, dict):
                    errors.append(f"{where}: must be an object")
            elif kind in _PRIMITIVES:
                problem = _r4_primitive(kind, item)
                if problem:
                    errors.append(f"{where}: {problem}")
                elif bound and item not in bound:
                    errors.append(f"{where}: {item!r} isn't one of {', '.join(sorted(bound))}")
            else:
                problem = _r4_complex(kind, item)
                if problem:
                    errors.append(f"{where}: {problem}")


def validate_r4(resource) -> list[str]:
    """Problems with a DocumentReference, Provenance or AuditEvent against the
    FHIR R4 structure. An empty list means structurally valid."""
    if not isinstance(resource, dict):
        return ["resource: must be a JSON object"]
    problem = shape_problem(resource)
    if problem:
        return [f"resource: {problem}"]
    rtype = resource.get("resourceType")
    spec = R4.get(rtype) if isinstance(rtype, str) else None
    if spec is None:
        return [f"resourceType: {rtype!r} isn't supported by this validator"]
    errors: list[str] = []
    _check_element({k: v for k, v in spec.items() if k != "resourceType"}, {k: v for k, v in resource.items() if k != "resourceType"},
                   "", errors, resource)
    if rtype == "AuditEvent":
        for i, agent in enumerate(resource.get("agent") or []):
            if isinstance(agent, dict) and not (agent.get("who") or agent.get("altId") or agent.get("name")):
                errors.append(f"agent[{i}]: needs who, altId or name to say who acted")
    if rtype == "Provenance" and isinstance(resource.get("target"), list):
        for i, t in enumerate(resource["target"]):
            if isinstance(t, dict) and not t.get("reference"):
                errors.append(f"target[{i}]: must point at the resource with a reference")
    return errors
