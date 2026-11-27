"""Red-team round 2: hostile or conflicting patient data over CDS Hooks and FHIR, and orders whose frequency or dose shape
used to slip past the dose checks."""

from __future__ import annotations

import copy
import os
import random
import sys
from datetime import date
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import cds_hooks, fhir  # noqa: E402
from app.patient_checks import times_per_day  # noqa: E402
import test_cds_conformance as t  # noqa: E402

TODAY = date(2026, 11, 27)


def cards(orders, **kw):
    return cds_hooks.cards_for(t.request(orders=orders, **kw), t.SERVICE_ID)["cards"]


def with_dosage(**repeat):
    order = t.draft()
    order["dosageInstruction"][0]["timing"]["repeat"] = repeat
    return order


def codes(found):
    return {c["source"]["topic"]["code"] for c in found}


@pytest.mark.parametrize("repeat", [dict(frequency=2, period=1, periodUnit="d"), dict(frequency=3, period=1, periodUnit="d"),
                                    dict(frequency=4, period=1, periodUnit="d"), dict(frequency=8, period=1, periodUnit="d"),
                                    dict(frequency=1, period=6, periodUnit="h"), dict(frequency=1, period=8, periodUnit="h"),
                                    dict(frequency=1, period=12, periodUnit="h"), dict(frequency=2, period=12, periodUnit="h")])
def test_an_order_repeated_through_the_day_is_checked_against_the_daily_maximum(repeat):
    found = cards([with_dosage(**repeat)])
    assert any(c["indicator"] == "critical" and "maximum" in c["summary"] for c in found), found


def test_once_daily_and_a_split_dose_within_the_maximum_are_not_blocked():
    assert cards([with_dosage(frequency=1, period=1, periodUnit="d")]) == []
    split = t.draft(dose=7.5)
    split["dosageInstruction"][0]["timing"]["repeat"] = dict(frequency=2, period=1, periodUnit="d")
    assert not [c for c in cards([split]) if c["indicator"] == "critical"]


def test_a_second_dosage_instruction_is_checked_too():
    order = t.draft()
    order["dosageInstruction"].append({"doseAndRate": [{"doseQuantity": {"value": 150, "unit": "mg"}}]})
    assert any(c["indicator"] == "critical" for c in cards([order]))


def test_a_dose_in_grams_is_reported_in_grams_not_relabelled_as_mg():
    order = t.draft()
    order["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"] = {"value": 15, "unit": "g"}
    found = [c for c in cards([order]) if c["indicator"] == "critical"]
    assert found and "15000 mg" in found[0]["summary"]


@pytest.mark.parametrize("shape", [
    [{"doseAndRate": [{"doseRange": {"low": {"value": 15, "unit": "mg"}, "high": {"value": 1500, "unit": "mg"}}}]}],
    [{"text": "as directed"}], [],
])
def test_an_order_whose_dose_cannot_be_read_says_so_instead_of_staying_silent(shape):
    order = t.draft()
    order["dosageInstruction"] = shape
    assert "dose_not_checked" in codes(cards([order]))


def test_the_doseRange_notice_is_a_warning_and_a_missing_dose_is_info():
    order = t.draft()
    order["dosageInstruction"] = [{"doseAndRate": [{"doseRange": {"high": {"value": 1500, "unit": "mg"}}}]}]
    assert next(c for c in cards([order]) if c["source"]["topic"]["code"] == "dose_not_checked")["indicator"] == "warning"
    order["dosageInstruction"] = []
    assert next(c for c in cards([order]) if c["source"]["topic"]["code"] == "dose_not_checked")["indicator"] == "info"


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), 1e400, "15", [15], {"v": 1}])
def test_a_dose_that_is_not_a_finite_number_is_not_read_as_a_dose(value):
    order = t.draft()
    order["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] = value
    assert "dose_not_checked" in codes(cards([order]))


def test_hostile_shapes_in_a_request_never_crash():
    HOST = [None, "", [], {}, 0, -1, 1e400, "x" * 3000, [[]], {"a": {"b": [1]}}, True, 10 ** 30, {"resourceType": "Bundle", "entry": "x"},
            {"resourceType": "Bundle", "entry": [None, 1, "s", {"resource": None}, {"resource": []}]},
            [{"resourceType": "MedicationRequest"}], "1970-13-45", {"coding": "x"}, {"coding": [None, 1, {"code": {}, "system": []}]},
            {"text": []}, {"text": {"a": 1}}]

    def paths(o, p=()):
        yield p
        if isinstance(o, dict):
            for k, v in o.items():
                yield from paths(v, p + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from paths(v, p + (i,))

    base = t.request(orders=[t.draft(), t.draft("Warfarin", 5, oid="m2")], allergies=["penicillin"])
    base["prefetch"]["conditions"]["entry"].append({"resource": {"resourceType": "Condition", "code": {"text": "asthma"}}})
    base["context"]["selections"] = ["MedicationRequest/mr1"]
    spots = [p for p in paths(base) if p and p not in [("hook",), ("hookInstance",)]]
    rng = random.Random(2)
    for _ in range(1500):
        body = copy.deepcopy(base)
        for _ in range(rng.choice([1, 1, 2, 3])):
            spot = rng.choice(spots)
            try:
                node = body
                for k in spot[:-1]:
                    node = node[k]
                node[spot[-1]] = copy.deepcopy(rng.choice(HOST))
            except (KeyError, IndexError, TypeError):
                pass
        try:
            cds_hooks.cards_for(body, t.SERVICE_ID)
        except cds_hooks.CdsError:
            pass


# --- FHIR ------------------------------------------------------------------------------------------------

def patient(birth="1970-01-01", sex="male"):
    return {"resourceType": "Patient", "birthDate": birth, "gender": sex}


def lab(code, value, unit, when="2026-11-20T10:00:00Z", oid="o"):
    return {"resourceType": "Observation", "id": oid, "status": "final",
            "code": {"coding": [{"system": fhir.LOINC, "code": code}]}, "effectiveDateTime": when,
            "valueQuantity": {"value": value, "unit": unit}}


def mapped(observations=(), birth="1970-01-01"):
    return fhir.map_patient(patient(birth), list(observations), [], [], [], {}, today=TODAY)


@pytest.mark.parametrize("birth,most", [("2014", 11.0), ("2014-12", 11.0), ("2014-11", 11.0), ("2014-10", 12.0), ("2026", 0.0), ("2026-11", 0.0)])
def test_a_partial_birth_date_never_makes_the_patient_older_than_they_may_be(birth, most):
    assert mapped(birth=birth).context.age_years == most


@pytest.mark.parametrize("order", [(90, 20), (20, 90)])
def test_two_egfr_results_at_the_same_time_with_different_values_are_not_chosen_by_list_order(order):
    loaded = mapped([lab("62238-1", v, "mL/min/1.73m2", oid=f"o{v}") for v in order])
    assert loaded.context.egfr is None
    assert any("same time" in w for w in loaded.warnings) and loaded.serious


def test_two_weights_in_different_units_at_the_same_time_that_disagree_are_not_used():
    loaded = mapped([lab("29463-7", 70, "kg", oid="a"), lab("29463-7", 7, "[lb_av]", oid="b")])
    assert loaded.context.weight_kg is None


def test_the_same_result_twice_and_a_newer_result_still_work():
    same = mapped([lab("29463-7", 70, "kg", oid="a"), lab("29463-7", 154.3236, "[lb_av]", oid="b")])
    assert same.context.weight_kg == 70
    newer = mapped([lab("62238-1", 90, "mL/min/1.73m2", "2026-11-01T00:00:00Z", "a"),
                    lab("62238-1", 25, "mL/min/1.73m2", "2026-11-20T00:00:00Z", "b"),
                    lab("62238-1", 25, "mL/min/1.73m2", "2026-11-20T00:00:00Z", "c")])
    assert newer.context.egfr == 25


def test_the_newest_lb_weight_is_converted():
    assert mapped([lab("29463-7", 154.3236, "[lb_av]")]).context.weight_kg == 70


@pytest.mark.parametrize("code", [{}, [], 5, None])
def test_a_coding_whose_code_is_not_a_string_is_ignored(code):
    obs = lab("62238-1", 90, "mL/min/1.73m2")
    obs["code"]["coding"].append({"system": fhir.LOINC, "code": code})
    obs["code"]["coding"].append({"system": code, "code": "x"})
    mapped([obs])


@pytest.mark.parametrize("text,times", [("15 mg four times daily", 4), ("15 mg 4 times a day", 4), ("15 mg every 6 hours", 4),
                                        ("15 mg bid", 2), ("15 mg t.i.d.", 3), ("15 mg once daily", 1), ("15 mg", None),
                                        ("15 mg every 0 hours", None), ("15 mg 3x daily", 3), ("15 mg twice a day", 2)])
def test_times_per_day(text, times):
    assert times_per_day(text) == times
