"""Reading a patient from a FHIR R4 server.

Maps what patient-aware checks need (app/patient_checks.py) from standard
FHIR resources:

- Patient: age from birth date, and administrative gender
- Observation: the latest eGFR, serum creatinine, body weight, potassium,
  sodium, INR, platelets, ALT and QTc (by LOINC code, with units converted),
  and pregnancy status
- AllergyIntolerance: active, not refuted
- MedicationRequest and MedicationStatement: active
- Condition: active

Names, identifiers, addresses and contact details are never copied. Lab
results older than FHIR_LAB_MAX_AGE_DAYS are flagged, because an old eGFR can
be dangerously wrong for a dose today.

Only servers in FHIR_OPEN_SERVERS (without authorisation) or a SMART-launched
issuer (with its access token) are contacted."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from urllib.parse import urlsplit

import httpx

from . import config
from .schemas import PatientContext

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"

# kind -> LOINC codes, most specific first
OBSERVATION_CODES: dict[str, tuple[str, ...]] = {
    "egfr": ("62238-1", "98979-8", "33914-3", "48642-3", "48643-1", "50044-7", "69405-9", "77147-7", "88293-6",
             "88294-4", "98980-6"),
    "creatinine": ("2160-0", "38483-4", "14682-9"),
    "weight": ("29463-7", "3141-9"),
    "potassium": ("2823-3", "6298-4"),
    "sodium": ("2951-2", "2947-0"),
    "inr": ("6301-6", "34714-6"),
    "platelets": ("777-3", "26515-7"),
    "alt": ("1742-6", "1743-4"),
    "qtc": ("8636-3",),
    "pregnancy": ("82810-3",),
}
_CODE_TO_KIND = {code: kind for kind, codes in OBSERVATION_CODES.items() for code in codes}
PREGNANT, NOT_PREGNANT = "77386006", "60001007"
# Values outside these ranges are data-entry errors, not results to dose from.
PLAUSIBLE = {"egfr": (1, 200), "creatinine": (5, 3000), "weight": (0.3, 400), "potassium": (1, 10), "sodium": (90, 200),
             "inr": (0.5, 15), "platelets": (1, 2000), "alt": (1, 10000), "qtc": (200, 800)}


class FhirError(Exception):
    """A FHIR server couldn't be read. The message is safe to show."""


def _clip(text: str, n: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def concept_text(concept: dict | None) -> str:
    if not concept:
        return ""
    if concept.get("text"):
        return concept["text"]
    for coding in concept.get("coding", []):
        if coding.get("display"):
            return coding["display"]
    return ""


def _codes(concept: dict | None) -> list[tuple[str, str]]:
    return [(c.get("system", ""), c.get("code", "")) for c in (concept or {}).get("coding", [])]


def _status(resource: dict, field: str) -> str:
    return next((c for _, c in _codes(resource.get(field))), "")


def validate_base_url(url: str, allowed: list[str]) -> str:
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    local = parts.hostname in ("localhost", "127.0.0.1")
    if parts.scheme not in ("https",) and not (local and parts.scheme == "http"):
        raise FhirError("FHIR servers must use HTTPS.")
    if not any(url == a or url.startswith(a + "/") for a in allowed):
        raise FhirError("That FHIR server isn't on this deployment's allowed list.")
    return url


class FhirClient:
    def __init__(self, base_url: str, token: str | None = None, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {"Accept": "application/fhir+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(timeout=15.0, headers=headers, transport=transport, follow_redirects=False)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get(self, path_or_url: str, params: dict | None = None) -> dict:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}/{path_or_url.lstrip('/')}"
        if not (url.startswith(self.base_url + "/") or url.startswith(self.base_url + "?")):
            raise FhirError("The FHIR server pointed to another site.")
        try:
            response = self._http.get(url, params=params)
        except httpx.HTTPError as exc:
            raise FhirError("Couldn't reach the FHIR server.") from exc
        if response.status_code in (401, 403):
            raise FhirError("The FHIR server refused access. Launch GroundCheck from the EHR again.")
        if response.status_code == 404:
            raise FhirError("The FHIR server has no such record.")
        if response.status_code >= 400:
            raise FhirError(f"The FHIR server returned an error ({response.status_code}).")
        try:
            return response.json()
        except ValueError as exc:
            raise FhirError("The FHIR server didn't return FHIR JSON.") from exc

    def search(self, resource: str, params: dict, max_pages: int = 5) -> list[dict]:
        bundle = self.get(resource, {**params, "_count": params.get("_count", 100)})
        found: list[dict] = []
        for _ in range(max_pages):
            found += [e["resource"] for e in bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == resource]
            next_link = next((l["url"] for l in bundle.get("link", []) if l.get("relation") == "next"), None)
            if not next_link:
                break
            bundle = self.get(next_link)
        return found


@dataclass
class EhrPatient:
    context: PatientContext
    source: dict
    observations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _age(birth: str, today: date) -> float | None:
    try:
        born = date.fromisoformat(birth[:10]) if len(birth) >= 10 else date(int(birth[:4]), int(birth[5:7] or 1), 1)
    except ValueError:
        return None
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    if years < 2:
        return round((today - born).days / 365.25, 2)
    return float(years)


def _when(obs: dict) -> datetime | None:
    raw = obs.get("effectiveDateTime") or (obs.get("effectivePeriod") or {}).get("end") or obs.get("issued")
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            when = datetime.fromisoformat(raw[:10])
        except ValueError:
            return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _quantity(kind: str, obs: dict) -> float | None:
    q = obs.get("valueQuantity") or {}
    value, unit = q.get("value"), (q.get("code") or q.get("unit") or "").strip()
    if not isinstance(value, (int, float)):
        return None
    u = unit.lower().replace("μ", "u").replace("µ", "u").replace(" ", "")
    if kind == "creatinine":
        if u == "mg/dl":
            return round(value * 88.42, 1)
        if u == "mmol/l":
            return round(value * 1000, 1)
        return float(value) if u in ("umol/l", "micromol/l") else None
    if kind == "weight":
        if u in ("kg",):
            return float(value)
        if u in ("g",):
            return round(value / 1000, 3)
        if u in ("[lb_av]", "lb", "lbs"):
            return round(value * 0.45359237, 1)
        return None
    if kind == "egfr":
        return float(value) if "min" in u or u == "" else None
    if kind in ("potassium", "sodium"):
        return float(value) if u in ("mmol/l", "meq/l") else None
    if kind == "platelets":
        if u in ("10*3/ul", "10*9/l", "10^9/l", "x10^9/l", "x10*9/l", "10^3/ul", "x10^3/ul", "k/ul", "10*3/mm3", "thou/ul"):
            return float(value)
        return round(value / 1000, 1) if u in ("/ul", "/mm3", "{cells}/ul") else None
    if kind == "alt":
        return float(value) if u in ("u/l", "iu/l") else None
    if kind == "qtc":
        return float(value) if u in ("ms",) else None
    if kind == "inr":
        return float(value)
    return None


def map_patient(patient: dict, observations: list[dict], allergies: list[dict], medications: list[dict],
                conditions: list[dict], source: dict, today: date | None = None,
                resolve_medication=None) -> EhrPatient:
    """Map FHIR resources to a PatientContext. Pure: no network."""
    today = today or datetime.now(timezone.utc).date()
    warnings: list[str] = []
    ctx: dict = {}

    if patient.get("birthDate"):
        age = _age(patient["birthDate"], today)
        if age is not None:
            ctx["age_years"] = age
    if patient.get("gender") in ("female", "male", "other"):
        ctx["sex"] = patient["gender"]
    if patient.get("deceasedBoolean") or patient.get("deceasedDateTime"):
        warnings.append("The record says this patient has died.")

    latest: dict[str, tuple[datetime | None, dict]] = {}
    for obs in observations:
        if obs.get("status") in ("entered-in-error", "cancelled", "preliminary"):
            continue
        kind = next((_CODE_TO_KIND[c] for s, c in _codes(obs.get("code")) if s == LOINC and c in _CODE_TO_KIND), None)
        if kind is None:
            continue
        when = _when(obs)
        best = latest.get(kind)
        if best is None or (when and (best[0] is None or when > best[0])):
            latest[kind] = (when, obs)

    used = []
    labs = {}
    for kind, (when, obs) in latest.items():
        if kind == "pregnancy":
            codes = {c for _, c in _codes(obs.get("valueCodeableConcept"))}
            if PREGNANT in codes:
                ctx["pregnant"] = True
            elif NOT_PREGNANT in codes:
                ctx["pregnant"] = False
            else:
                continue
        else:
            value = _quantity(kind, obs)
            if value is None:
                warnings.append(f"A {kind} result used units GroundCheck can't convert, so it was left out.")
                continue
            low, high = PLAUSIBLE[kind]
            if not low <= value <= high:
                warnings.append(f"The latest {kind} result ({value:g}) isn't a plausible value, so it was left out.")
                continue
            target = {"egfr": "egfr", "creatinine": "creatinine_umol_l", "weight": "weight_kg"}.get(kind)
            if target:
                ctx[target] = value
            else:
                labs[kind] = value
        age_days = (datetime.now(timezone.utc) - when).days if when else None
        if kind != "pregnancy" and (age_days is None or age_days > config.FHIR_LAB_MAX_AGE_DAYS):
            warnings.append(f"The latest {'eGFR' if kind == 'egfr' else kind} is "
                            f"{'undated' if when is None else f'from {when.date().isoformat()}'}, which may be out of date.")
        used.append({"kind": kind, "date": when.date().isoformat() if when else None,
                     "id": f"Observation/{obs.get('id', '')}"})
    if labs:
        ctx["labs"] = labs

    def active_text(resources, text_of, clinical_ok, limit):
        items = []
        for r in resources:
            if not clinical_ok(r):
                continue
            text = text_of(r)
            if text and text.lower() not in (i.lower() for i in items):
                items.append(_clip(text))
        if len(items) > limit:
            warnings.append(f"Only the first {limit} of {len(items)} items were used.")
        return items[:limit]

    ctx["allergies"] = active_text(
        allergies, lambda r: concept_text(r.get("code")),
        lambda r: _status(r, "clinicalStatus") in ("", "active") and _status(r, "verificationStatus") not in ("refuted", "entered-in-error"),
        30)

    def medication_name(r):
        if r.get("medicationCodeableConcept"):
            return concept_text(r["medicationCodeableConcept"])
        ref = (r.get("medicationReference") or {}).get("reference", "")
        if ref.startswith("#"):
            contained = next((c for c in r.get("contained", []) if c.get("id") == ref[1:]), None)
            return concept_text((contained or {}).get("code"))
        if ref and resolve_medication:
            return resolve_medication(ref)
        return (r.get("medicationReference") or {}).get("display", "")

    ctx["medicines"] = active_text(
        medications, medication_name,
        lambda r: r.get("status") in ("active", "on-hold", "intended") and r.get("intent", "order") != "proposal", 40)
    ctx["conditions"] = active_text(
        conditions, lambda r: concept_text(r.get("code")),
        lambda r: _status(r, "clinicalStatus") in ("", "active", "recurrence", "relapse")
        and _status(r, "verificationStatus") not in ("refuted", "entered-in-error"), 40)

    for key in ("allergies", "medicines", "conditions"):
        if not ctx[key]:
            del ctx[key]
    if ctx.get("sex") == "male":
        ctx.pop("pregnant", None)
    context = PatientContext(**ctx)
    return EhrPatient(context=context, source={**source, "observations": used}, warnings=warnings)


def load_patient(client: FhirClient, patient_id: str, source: dict) -> EhrPatient:
    if not patient_id or "/" in patient_id or len(patient_id) > 64:
        raise FhirError("That isn't a valid FHIR patient ID.")
    patient = client.get(f"Patient/{patient_id}")
    if patient.get("resourceType") != "Patient":
        raise FhirError("The FHIR server didn't return a patient.")
    codes = ",".join(f"{LOINC}|{c}" for codes in OBSERVATION_CODES.values() for c in codes)
    observations = client.search("Observation", {"patient": patient_id, "code": codes, "_sort": "-date"}, max_pages=3)
    allergies = client.search("AllergyIntolerance", {"patient": patient_id}, max_pages=2)
    medications = client.search("MedicationRequest", {"patient": patient_id, "status": "active"}, max_pages=2)
    try:
        medications += client.search("MedicationStatement", {"patient": patient_id, "status": "active"}, max_pages=1)
    except FhirError:
        pass  # not every server supports MedicationStatement
    conditions = client.search("Condition", {"patient": patient_id}, max_pages=2)

    cache: dict[str, str] = {}

    def resolve(ref: str) -> str:
        if ref not in cache:
            try:
                cache[ref] = concept_text(client.get(ref).get("code"))
            except FhirError:
                cache[ref] = ""
        return cache[ref]

    return map_patient(patient, observations, allergies, medications, conditions,
                       {**source, "patient": f"Patient/{patient_id}"}, resolve_medication=resolve)
