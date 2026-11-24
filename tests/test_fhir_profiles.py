"""FHIR US Core / UK Core subset validation of the resources patient checks
are built from, and that what it finds reaches the clinician as patient-check
warnings. No network."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import fhir, fhir_profiles as fp  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-043")

FIXTURES = Path(__file__).parent / "fixtures" / "fhir"
US = json.loads((FIXTURES / "us_core_valid.json").read_text())
UK = json.loads((FIXTURES / "uk_core_valid.json").read_text())
CASES = json.loads((FIXTURES / "invalid_cases.json").read_text())
TYPES = {"Patient": "Patient", "Observation": "Observation", "Observation_weight": "Observation",
         "AllergyIntolerance": "AllergyIntolerance", "MedicationRequest": "MedicationRequest", "Condition": "Condition"}
TODAY = date(2026, 11, 24)


def issues(resource, **kw):
    return fp.validate_resource(resource, **kw)


# --- Valid and invalid resources ----------------------------------------------------------------

@pytest.mark.parametrize("name", list(TYPES))
def test_valid_us_core_resources_have_no_issues(name):
    assert issues(US[name]) == []
    assert fp.profile_of(US[name]) == "us-core"


@pytest.mark.parametrize("name", list(TYPES))
def test_valid_uk_core_resources_have_no_issues(name):
    assert fp.profile_of(UK[name]) == "uk-core"
    assert issues(UK[name]) == []


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_each_invalid_fixture_is_reported_at_the_right_element(case):
    found = issues(case["resource"], profile=case.get("profile"))
    matching = [i for i in found if i.path == case["path"] and case["contains"] in i.message]
    assert matching, [str(i) for i in found]
    assert matching[0].severity == case["severity"]
    assert matching[0].relied_on is case["relied_on"]


def test_the_invalid_fixtures_cover_every_profiled_resource_type():
    assert {c["resource"]["resourceType"] for c in CASES} == {"Patient", "Observation", "AllergyIntolerance",
                                                               "MedicationRequest", "Condition"}
    assert len(CASES) >= 50


def test_uk_core_does_not_require_what_us_core_requires():
    r = copy.deepcopy(US["Observation"])
    r.pop("category")
    assert any(i.path == "category" for i in issues(r, profile="us-core"))
    assert not any(i.path == "category" and i.severity == "error" for i in issues(r, profile="uk-core"))
    p = copy.deepcopy(US["Condition"])
    p.pop("category")
    assert any(i.path == "category" for i in issues(p, profile="us-core"))
    assert not any(i.path == "category" and i.severity == "error" for i in issues(p, profile="uk-core"))


def test_a_resource_of_the_wrong_type_or_shape_is_one_clear_error():
    assert [i.message for i in issues(US["Patient"], expect="Observation")][0].startswith("is Patient, expected")
    for junk in (None, "Patient", 7, [], [{"resourceType": "Patient"}], True):
        [issue] = issues(junk, expect="Patient")
        assert issue.severity == "error" and "JSON object" in issue.message
    [issue] = issues({"resourceType": "Encounter"})
    assert "isn't one of the profiled" in issue.message
    [issue] = issues({})
    assert issue.path == "resourceType"
    [issue] = issues({"resourceType": 5})
    assert issue.severity == "error"


def test_unknown_extensions_are_ignored_but_modifier_extensions_are_not():
    r = copy.deepcopy(US["Patient"])
    r["extension"] = [{"url": "http://example.org/unknown", "valueString": "x"},
                      {"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", "extension": []}]
    r["_birthDate"] = {"extension": [{"url": "http://hl7.org/fhir/StructureDefinition/patient-birthTime", "valueDateTime": "1958-03-14T10:00:00Z"}]}
    assert issues(r) == []
    r["modifierExtension"] = [{"url": "http://example.org/unknown-modifier", "valueBoolean": True}]
    [issue] = issues(r)
    assert issue.relied_on and "modifier extension" in issue.message


# --- Dates, time zones and UCUM -------------------------------------------------------------------

@pytest.mark.parametrize("value", ["2026", "2026-11", "2026-11-24", "2024-02-29", "1900-01-01", "0001-01-01".replace("0001", "1001")])
def test_valid_dates(value):
    assert fp.valid_date(value) and fp.valid_datetime(value)


@pytest.mark.parametrize("value", ["", "2026-13-01", "2026-00-10", "2026-11-31", "2023-02-29", "1900-02-29", "2026-1-1",
                                   "26-11-24", "2026/11/24", "20261124", " 2026-11-24", "2026-11-24 ", None, 20261124, True,
                                   "2026-11-24T10:00:00Z", "0000-01-01"])
def test_invalid_dates(value):
    assert not fp.valid_date(value)


@pytest.mark.parametrize("value", ["2026-11-24T10:00:00Z", "2026-11-24T10:00:00+00:00", "2026-11-24T10:00:00-05:00",
                                   "2026-11-24T10:00:00.123456Z", "2026-11-24T23:59:60Z", "2026-11-24T10:00:00+14:00", "2026-11-24"])
def test_valid_datetimes_with_zones(value):
    assert fp.valid_datetime(value)


@pytest.mark.parametrize("value", ["2026-11-24T10:00:00", "2026-11-24T10:00", "2026-11-24T24:00:00Z", "2026-11-24T10:60:00Z",
                                   "2026-11-24T10:00:00+15:00", "2026-11-24T10:00:00+0100", "2026-11-24t10:00:00z",
                                   "2026-02-30T10:00:00Z", "tomorrow"])
def test_invalid_datetimes(value):
    assert not fp.valid_datetime(value)


def test_time_without_a_zone_is_called_out_specifically():
    assert "time zone" in fp.datetime_problem("2026-11-24T10:00:00")
    assert "valid FHIR" in fp.datetime_problem("nonsense")
    assert fp.datetime_problem("2026-11-24T10:00:00Z") is None


def test_instants_need_a_full_timestamp():
    assert fp.valid_instant("2026-11-24T10:00:00Z")
    for bad in ("2026-11-24", "2026-11", "2026-11-24T10:00:00", None, 5):
        assert not fp.valid_instant(bad)


@pytest.mark.parametrize("code,canonical", [("mg/dL", "mg/dL"), ("MG/DL", "mg/dL"), ("umol/L", "umol/L"), ("UMOL/L", "umol/L"),
                                            ("[lb_av]", "[lb_av]"), ("mL/min/{1.73_m2}", "mL/min/{1.73_m2}"),
                                            ("10*3/uL", "10*3/uL"), ("furlongs", None), ("", None), (None, None), (5, None)])
def test_ucum_canonical_case(code, canonical):
    assert fp.ucum_canonical(code) == canonical


@pytest.mark.parametrize("code", ["mg/dL", "10*3/uL", "[lb_av]", "mm[Hg]", "/min", "{beats}/min", "%", "kg/m2"])
def test_ucum_syntax_accepts(code):
    assert fp.ucum_syntax_ok(code)


@pytest.mark.parametrize("code", ["", "mg per dl", "mg/", "x" * 65, None, 3, "mg\ndl", "<script>"])
def test_ucum_syntax_rejects(code):
    assert not fp.ucum_syntax_ok(code)


# --- Hostile input -----------------------------------------------------------------------------

def nested(depth: int):
    root = leaf = {}
    for _ in range(depth):
        leaf["x"] = {}
        leaf = leaf["x"]
    return root


def test_deeply_nested_resources_are_refused_without_recursion_errors():
    for depth in (41, 500, 50_000):
        r = copy.deepcopy(US["Patient"])
        r["extension"] = [{"url": "http://x", "valueString": "a", "extension": [nested(depth)]}]
        [issue] = issues(r)
        assert issue.severity == "error" and "deep" in issue.message and issue.relied_on
    ok = copy.deepcopy(US["Patient"])
    ok["extension"] = [{"url": "http://x", "extension": [nested(20)]}]
    assert issues(ok) == []


def test_huge_strings_and_node_counts_are_refused():
    r = copy.deepcopy(US["Patient"])
    r["name"][0]["family"] = "A" * 2_000_000
    [issue] = issues(r)
    assert "1 MB" in issue.message
    wide = copy.deepcopy(US["Patient"])
    wide["extension"] = [{"url": "http://x"} for _ in range(120_000)]
    [issue] = issues(wide)
    assert "elements" in issue.message
    assert fp.shape_problem({"a": float("nan")}) and fp.shape_problem([float("inf")])
    assert fp.shape_problem({"a": [1, 2, {"b": "ok"}]}) is None


def test_moderately_long_strings_are_not_a_problem():
    r = copy.deepcopy(US["AllergyIntolerance"])
    r["code"]["text"] = "Z" * 500_000
    assert issues(r) == []


def test_a_huge_bundle_is_validated_quickly_and_capped():
    many = [copy.deepcopy(US["Observation"]) for _ in range(5000)]
    for o in many:
        o.pop("status")
    start = time.perf_counter()
    warnings = fp.check_warnings(US["Patient"], many, [], [], [])
    assert time.perf_counter() - start < 20
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{fp.MAX_RESOURCES} of {fp.MAX_RESOURCES} Observation resources: status")


def test_warnings_never_contain_values_from_the_record():
    patient = copy.deepcopy(US["Patient"])
    patient["name"] = [{"family": "Zzyzx-Surname", "given": ["Qwerty-Given"]}]
    patient["identifier"] = [{"value": "SECRET-MRN-77"}]
    patient["birthDate"] = "2999-01-01"
    bad_allergy = {"resourceType": "AllergyIntolerance", "code": {"text": "  "}, "note": [{"text": "Zzyzx-Surname lives at 1 Elm St"}]}
    warnings = fp.check_warnings(patient, [], [bad_allergy], [], [])
    text = " ".join(warnings)
    assert warnings and not any(w in text for w in ("Zzyzx", "Qwerty", "SECRET-MRN", "Elm"))


def test_malicious_strings_in_codes_and_units_stay_inert():
    obs = copy.deepcopy(US["Observation"])
    obs["valueQuantity"]["code"] = "<script>alert(1)</script>"
    obs["code"]["coding"][0]["code"] = "'; DROP TABLE users;--"
    found = issues(obs)
    assert found and all("<script>" not in i.message and "DROP" not in i.message for i in found)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(max_size=20),
                    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=12), c, max_size=5), max_leaves=30))
def test_validation_never_raises_on_arbitrary_json(value):
    fp.validate_resource(value)
    fp.validate_resource(value, expect="Observation")
    if isinstance(value, dict):
        for rtype in ("Patient", "Observation", "AllergyIntolerance", "MedicationRequest", "Condition"):
            fp.validate_resource({**value, "resourceType": rtype})


FIELD_VALUES = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True, allow_infinity=True),
                         st.text(max_size=12), st.lists(st.text(max_size=6), max_size=2),
                         st.fixed_dictionaries({}, optional={"coding": st.lists(st.fixed_dictionaries(
                             {}, optional={"system": st.text(max_size=8), "code": st.text(max_size=8), "display": st.text(max_size=8)}),
                             max_size=2), "text": st.text(max_size=8)}))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.fixed_dictionaries({}, optional={k: FIELD_VALUES for k in (
    "status", "code", "effectiveDateTime", "valueQuantity", "valueCodeableConcept", "subject", "clinicalStatus",
    "verificationStatus", "medicationCodeableConcept", "intent", "birthDate", "gender", "deceasedBoolean")}))
def test_mapping_a_patient_from_garbled_fields_never_raises(fields):
    resources = [{"resourceType": t, **fields} for t in ("Observation", "AllergyIntolerance", "MedicationRequest", "Condition")]
    loaded = fhir.map_patient({"resourceType": "Patient", **fields}, resources[:1], resources[1:2], resources[2:3], resources[3:], {})
    assert loaded.context is not None


# --- What reaches the clinician: patient-check warnings ----------------------------------------------

def rich(**over):
    return {"patient": copy.deepcopy(US["Patient"]), "observations": [copy.deepcopy(US["Observation"])],
            "allergies": [copy.deepcopy(US["AllergyIntolerance"])], "medications": [copy.deepcopy(US["MedicationRequest"])],
            "conditions": [copy.deepcopy(US["Condition"])], **over}


def load(**kw):
    r = rich(**kw)
    return fhir.map_patient(r["patient"], r["observations"], r["allergies"], r["medications"], r["conditions"], {}, today=TODAY)


def recent(o):
    o["effectiveDateTime"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    return o


def test_a_complete_valid_record_produces_no_profile_warnings():
    loaded = load(observations=[recent(copy.deepcopy(US["Observation"]))])
    assert loaded.warnings == []
    assert loaded.context.egfr == 38 and loaded.context.allergies == ["Allergy to penicillin"]


def test_missing_elements_the_checks_rely_on_become_warnings_not_silence():
    patient = copy.deepcopy(US["Patient"])
    del patient["birthDate"]
    obs = recent(copy.deepcopy(US["Observation"]))
    del obs["status"]
    allergy = {"resourceType": "AllergyIntolerance", "patient": {"reference": "Patient/p1"}}
    loaded = load(patient=patient, observations=[obs], allergies=[allergy])
    text = "\n".join(loaded.warnings)
    assert "Patient: birthDate is a must-support element and is missing" in text
    assert "Observation: status is required but missing" in text
    assert "The AllergyIntolerance: code is required but missing" in text
    assert loaded.context.age_years is None and loaded.context.allergies == []


def test_warnings_use_counts_not_one_line_per_resource():
    obs = [recent(copy.deepcopy(US["Observation"])) for _ in range(40)]
    for o in obs[:30]:
        o.pop("effectiveDateTime")
    loaded = load(observations=obs)
    [w] = [w for w in loaded.warnings if "effective[x]" in w]
    assert w.startswith("30 of 40 Observation resources")


def test_irrelevant_profile_gaps_do_not_warn_clinicians():
    patient = copy.deepcopy(US["Patient"])
    del patient["name"], patient["identifier"]
    med = copy.deepcopy(US["MedicationRequest"])
    for k in ("requester", "authoredOn", "dosageInstruction", "subject"):
        del med[k]
    assert load(patient=patient, medications=[med]).warnings == []
    assert any(i.path == "name" for i in issues(patient))


def test_warning_list_is_capped():
    obs = []
    for missing in ("status", "code", "effectiveDateTime", "valueQuantity"):
        o = copy.deepcopy(US["Observation"])
        o.pop(missing)
        obs.append(o)
    full = fp.check_warnings(None, obs, [], [], [])
    assert len(full) >= 4
    out = fp.check_warnings(None, obs, [], [], [], limit=2)
    assert len(out) == 3 and f"{len(full) - 2} more data-quality problems" in out[-1]


def test_profile_check_failures_never_stop_a_patient_loading(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(fp, "check_warnings", boom)
    loaded = load()
    assert loaded.context.egfr == 38
    assert "data quality couldn't be checked" in " ".join(loaded.warnings)


# --- Hardening of the mapper the validation protects ----------------------------------------------------

def test_a_boolean_or_nan_is_never_read_as_a_lab_value():
    for bad in (True, False, float("nan"), float("inf"), "70", None, [70], {"v": 1}):
        obs = copy.deepcopy(US["Observation_weight"])
        obs["valueQuantity"]["value"] = bad
        loaded = fhir.map_patient({"resourceType": "Patient"}, [obs], [], [], [], {})
        assert loaded.context.weight_kg is None, bad


def test_unit_must_be_text():
    obs = copy.deepcopy(US["Observation_weight"])
    obs["valueQuantity"].update(unit=["kg"], code=7)
    assert fhir.map_patient({"resourceType": "Patient"}, [obs], [], [], [], {}).context.weight_kg is None


def test_an_observation_dated_in_the_future_cannot_hide_the_real_latest_result():
    real = recent(copy.deepcopy(US["Observation"]))
    real["valueQuantity"]["value"] = 25
    future = copy.deepcopy(US["Observation"])
    future["valueQuantity"]["value"] = 95
    future["effectiveDateTime"] = (datetime.now(timezone.utc) + timedelta(days=400)).isoformat()
    loaded = fhir.map_patient({"resourceType": "Patient"}, [future, real], [], [], [], {})
    assert loaded.context.egfr == 25
    assert any("dated in the future" in w for w in loaded.warnings)


@pytest.mark.parametrize("birth", ["2999-01-01", "1700-01-01", "not-a-date", "", 19580314, None, ["1958"], "2026-11-25"])
def test_unusable_or_future_birth_dates_do_not_crash_and_warn(birth):
    loaded = fhir.map_patient({"resourceType": "Patient", "birthDate": birth}, [], [], [], [], {}, today=TODAY)
    assert loaded.context.age_years is None
    if birth:
        assert any("birth date" in w for w in loaded.warnings)


def test_age_and_dates_boundaries():
    assert fhir._age("2026-11-24", TODAY) == 0.0
    assert fhir._age("2024-11-24", TODAY) == 2.0
    assert fhir._age("2024-11-25", TODAY) == pytest.approx(2.0, abs=0.01)
    assert fhir._age("1896-11-24", TODAY) == 130.0
    assert fhir._age("1895-11-24", TODAY) is None
    assert fhir._age("2000", TODAY) == 26.0


def test_non_object_entries_in_resource_lists_are_skipped():
    loaded = fhir.map_patient("not a patient", [None, "x", 3, copy.deepcopy(US["Observation_weight"])], [[], "a"], [5], [None], {})
    assert loaded.context.weight_kg == 70


def test_a_server_that_returns_odd_bundles_does_not_crash_search():
    def handler(request):
        return httpx.Response(200, json={"resourceType": "Bundle", "entry": [None, "x", {"resource": "y"}, {"resource": None},
                                                                              {"resource": {"resourceType": "Condition", "id": "1"}},
                                                                              {"resource": {"resourceType": "Observation"}}],
                                          "link": [None, "x", {"relation": "next"}, {"relation": "next", "url": 5}]})

    with fhir.FhirClient("http://localhost:1/fhir", transport=httpx.MockTransport(handler)) as client:
        assert client.search("Condition", {"patient": "p"}) == [{"resourceType": "Condition", "id": "1"}]
    for body in ({"entry": "no"}, {"entry": {"a": 1}}, {"link": "x"}, {}):
        with fhir.FhirClient("http://localhost:1/fhir", transport=httpx.MockTransport(lambda r, b=body: httpx.Response(200, json=b))) as client:
            assert client.search("Condition", {"patient": "p"}) == []
    for body in ([], "x", 5, None):
        with fhir.FhirClient("http://localhost:1/fhir", transport=httpx.MockTransport(lambda r, b=body: httpx.Response(200, json=b))) as client:
            with pytest.raises(fhir.FhirError, match="FHIR JSON"):
                client.search("Condition", {"patient": "p"})


def test_concept_text_and_codes_survive_garbage():
    for junk in (None, "x", 5, [], {"coding": "x"}, {"coding": [None, 5, {"display": 5}]}, {"text": 5}):
        assert fhir.concept_text(junk) == ""
        assert fhir._codes(junk) == [] or all(isinstance(c, tuple) for c in fhir._codes(junk))


def test_out_of_range_values_degrade_with_a_warning_instead_of_failing():
    loaded = fhir.map_patient({"resourceType": "Patient", "gender": "male"}, [], [], [], [], {}, today=TODAY)
    assert loaded.context.sex == "male"
    pregnant = {"resourceType": "Observation", "status": "final", "code": {"coding": [{"system": fhir.LOINC, "code": "82810-3"}]},
                "effectiveDateTime": "2026-11-20T09:30:00Z",
                "valueCodeableConcept": {"coding": [{"system": fhir.SNOMED, "code": fhir.PREGNANT}]}}
    assert fhir.map_patient({"resourceType": "Patient", "gender": "male"}, [pregnant], [], [], [], {}).context.pregnant is None


def test_uk_core_resources_load_with_no_profile_warnings():
    obs = recent(copy.deepcopy(UK["Observation"]))
    loaded = fhir.map_patient(UK["Patient"], [obs], [UK["AllergyIntolerance"]], [UK["MedicationRequest"]], [UK["Condition"]], {},
                              today=TODAY)
    assert loaded.warnings == []
