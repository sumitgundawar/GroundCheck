"""Patient-aware checks: each formulary rule, patient data validation, the
formulary's own validation, and the pipeline and API with a patient."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import formulary, patient_checks, pipeline, retrieval  # noqa: E402
from app.schemas import Claim, PatientContext  # noqa: E402

ADULT = dict(age_years=50, sex="female", weight_kg=70, egfr=90)


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


def claim(text: str) -> Claim:
    return Claim(text=text, source_ids=["X-1"], grounded=True, grounding_score=0.9)


def codes(query: str, claims: list[Claim], **patient) -> dict[str, str]:
    result = patient_checks.review(query, claims, PatientContext(**patient))
    return {f.code: f.severity for f in result.findings}


# --- Rules ----------------------------------------------------------------------------

def test_no_patient_means_no_checks():
    assert patient_checks.review("What is the dose of Caloradine?", [], None).findings == []
    assert patient_checks.review("What is the dose of Caloradine?", [], PatientContext()).findings == []


def test_questions_without_formulary_medicines_have_no_findings():
    result = patient_checks.review("How is Veltris syndrome monitored?", [claim("Levels are checked.")],
                                   PatientContext(**ADULT, allergies=["quorl"]))
    assert result.findings == [] and result.medicines == []


def test_an_adult_dose_for_a_child_is_blocked_with_the_weight_based_dose():
    found = patient_checks.review("What is the dose of Mendel solution?", [claim("The course is 5 mL once daily.")],
                                  PatientContext(age_years=8, weight_kg=25))
    [finding] = found.findings
    assert (finding.severity, finding.code) == ("block", "paediatric_dose")
    assert "2.5 mL once daily" in finding.message and "0.1 mL/kg × 25 kg" in finding.message


def test_paediatric_doses_are_capped():
    found = patient_checks.review("What is the dose of Mendel solution?", [], PatientContext(age_years=15, weight_kg=90))
    assert "capped at 5 mL" in found.findings[0].message and "5 mL once daily" in found.findings[0].message


def test_weight_based_dosing_labs_and_high_alert():
    found = patient_checks.review("What dose of Vorantil should I give?", [],
                                  PatientContext(**{**ADULT, "weight_kg": 130}, labs={"platelets": 80, "inr": 1.8}))
    by_code = {f.code: f for f in found.findings}
    assert by_code["high_alert"].severity == "warn"
    assert by_code["lab_platelets"].severity == "block" and "80" in by_code["lab_platelets"].message
    assert by_code["lab_inr"].severity == "warn"
    assert "100 mg every 12 hours" in by_code["weight_dose"].message and "capped" in by_code["weight_dose"].message


def test_weight_based_medicines_need_the_weight():
    assert codes("What dose of Vorantil?", [], age_years=50, egfr=90) == {"high_alert": "warn", "missing_data": "block"}


def test_duplicate_classes_and_medicines_already_taken():
    found = codes("Can I start Vorantil?", [], **ADULT, medicines=["Vorantil"])
    assert found["already_taking"] == "info"
    other = formulary.Medicine(name="Plaxetin", classes=["anticoagulants"])
    idx = formulary.index()
    idx.by_name["plaxetin"] = other
    try:
        found = codes("Can I start Vorantil?", [], **ADULT, medicines=["Plaxetin"])
        assert found["interaction_contraindicated"] == "block" and found["duplicate_class"] == "warn"
    finally:
        del idx.by_name["plaxetin"]


def test_doses_above_the_maximum_are_blocked():
    assert codes("What dose of Caloradine?", [claim("Caloradine is given at 30 mg once daily.")], **ADULT) == {
        "dose_above_maximum": "block"}
    assert codes("What dose of Caloradine?", [claim("Caloradine is given at 15 mg once daily.")], **ADULT) == {}
    # Units are converted: 15,000 mcg is 15 mg.
    assert codes("What dose of Caloradine?", [claim("Caloradine: 15000 mcg once daily.")], **ADULT) == {}


def test_kidney_function_from_creatinine_clearance():
    patient = PatientContext(age_years=80, sex="female", weight_kg=60, creatinine_umol_l=150)
    assert patient_checks.creatinine_clearance(patient) == pytest.approx((140 - 80) * 60 / (0.815 * 150) * 0.85, abs=0.1)
    found = patient_checks.review("What dose of Caloradine?", [], patient)
    assert found.derived["creatinine_clearance"] < 30 and found.blocking.code == "renal"
    assert patient_checks.creatinine_clearance(PatientContext(age_years=10, weight_kg=30, creatinine_umol_l=40)) is None


def test_breastfeeding_rules():
    no_data = codes("What dose of Caloradine?", [], **ADULT, breastfeeding=True)
    assert no_data == {"breastfeeding_no_data": "warn"}
    assert codes("What dose of Mendel solution?", [], **ADULT, breastfeeding=True) == {}


def test_informational_questions_warn_instead_of_blocking():
    found = codes("What is Caloradine used for?", [], **ADULT, allergies=["quorl"], medicines=["Tessorin"])
    assert found == {"allergy": "warn", "interaction_contraindicated": "warn"}


# --- Validation ------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"age_years": -1}, {"age_years": 200}, {"weight_kg": 0}, {"egfr": 500}, {"child_pugh": "D"},
    {"sex": "male", "pregnant": True}, {"name": "Jane Doe"}, {"labs": {"cholesterol": 5}},
    {"allergies": ["x" * 81]}, {"medicines": ["a"] * 41},
])
def test_patient_data_is_validated(bad):
    with pytest.raises(ValidationError):
        PatientContext(**bad)


def test_patient_lists_are_tidied():
    assert PatientContext(allergies=["  quorl ", "Quorl", "", "penicillin"]).allergies == ["quorl", "penicillin"]


def test_invalid_formularies_are_refused(tmp_path):
    base = {"name": "Test", "version": "1", "medicines": [{"name": "A"}, {"name": "B", "aliases": ["a"]}]}
    with pytest.raises(ValidationError, match="more than one medicine"):
        formulary.Formulary.model_validate(base)
    bad_rule = {"name": "Test", "version": "1", "medicines": [{"name": "A", "interactions": [
        {"id": "A-1", "severity": "major"}]}]}
    with pytest.raises(ValidationError, match="exactly one"):
        formulary.Formulary.model_validate(bad_rule)
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"name": "Test", "version": "1", "medicines": [{"name": "A", "renal": [
        {"id": "A-R", "egfr_below": 30, "action": "delete"}]}]}))
    with pytest.raises(ValidationError):
        formulary.load(path)


def test_the_demo_formulary_is_valid_and_complete():
    f = formulary.load()
    assert f.synthetic and len(f.medicines) > 100
    assert {"Caloradine", "Mendel solution", "Tessorin", "Vorantil"} <= {m.name for m in f.medicines}


# --- Pipeline and API --------------------------------------------------------------------

def test_the_pipeline_records_patient_checks_and_audits_the_patient():
    response = pipeline.run("What is the standard dose of Caloradine?",
                            patient=PatientContext(**{**ADULT, "egfr": 45}), review=False)
    assert response.decision == "refuse" and response.patient_findings[0].code == "renal_dose"
    step = next(s for s in response.trace if s.name == "patient checks")
    assert step.status == "fail" and "Caloradine" in step.data["medicines"]
    from app import audit
    assert audit.store.get(response.audit_id)["patient"]["egfr"] == 45

    plain = pipeline.run("What is the standard dose of Caloradine?", review=False)
    assert plain.decision == "answer" and plain.patient_findings == []
    assert next(s for s in plain.trace if s.name == "patient checks").status == "skip"


def test_the_api_accepts_and_validates_a_patient():
    from app.main import app

    with TestClient(app) as client:
        ok = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?", "patient": ADULT})
        assert ok.status_code == 200 and ok.json()["decision"] == "answer"
        blocked = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?",
                                                "patient": {**ADULT, "pregnant": True}})
        assert blocked.json()["decision"] == "refuse"
        assert blocked.json()["patient_findings"][0]["code"] == "pregnancy"
        assert client.post("/api/ask", json={"query": "x", "patient": {"name": "Jane"}}).status_code == 422


def test_an_informational_answer_says_when_it_is_only_about_adults():
    """Asking what the first-line medicine is isn't asking to give it, so it's
    still answered — but with a child loaded, the answer says it's for adults."""
    child = PatientContext(age_years=5, weight_kg=18, egfr=100)
    response = pipeline.run("What is the first-line medication for Veltris syndrome?",
                            patient=child, review=False)
    assert response.decision == "answer"
    codes = {f.code for f in response.patient_findings}
    assert "paediatric_not_covered" in codes
    warning = next(f for f in response.patient_findings if f.code == "paediatric_not_covered")
    assert warning.severity == "warn" and "adults" in warning.message

    # An adult gets the same answer with nothing to qualify it.
    grown = pipeline.run("What is the first-line medication for Veltris syndrome?",
                         patient=PatientContext(**ADULT), review=False)
    assert grown.decision == "answer"
    assert "paediatric_not_covered" not in {f.code for f in grown.patient_findings}

    # Asking for the dose is asking to give it, so that is still refused.
    dose = pipeline.run("What is the dose of Caloradine?", patient=child, review=False)
    assert dose.decision == "refuse"


def test_a_twice_daily_maximum_is_a_daily_maximum():
    """Tessorin is 10 mg twice daily, at most 20 mg a day. A single 20 mg dose
    is the whole day's allowance at once, so it is refused — 70 of the demo
    formulary's medicines are dosed this way."""
    assert codes("What dose of Tessorin?", [claim("Tessorin is given at 20 mg twice daily.")],
                 **ADULT) == {"dose_above_maximum": "block"}
    assert codes("What dose of Tessorin?", [claim("Tessorin is given at 10 mg twice daily.")], **ADULT) == {}


def test_interaction_severity_decides_whether_an_answer_stops():
    """Contraindicated blocks; major warns and the answer still stands; the
    severity comes from the formulary, not from the wording of the note."""
    caloradine = [claim("Caloradine is given at 15 mg once daily.")]
    blocked = codes("What dose of Caloradine?", caloradine, **{**ADULT, "medicines": ["Tessorin 10 mg"]})
    assert blocked["interaction_contraindicated"] == "block"

    warned = codes("What dose of Caloradine?", caloradine, **{**ADULT, "medicines": ["Mendel solution 5 mL"]})
    assert warned["interaction_major"] == "warn"
    assert "interaction_contraindicated" not in warned


def test_an_interaction_counts_from_either_medicine():
    """The formulary writes the Caloradine and Mendel solution interaction on
    one entry only. Asking about either one has to find it."""
    from_caloradine = codes("What dose of Caloradine?", [claim("Caloradine is given at 15 mg once daily.")],
                            **{**ADULT, "medicines": ["Mendel solution 5 mL once daily"]})
    from_mendel = codes("What dose of Mendel solution?", [claim("Mendel solution is 5 mL once daily.")],
                        **{**ADULT, "medicines": ["Caloradine 15 mg once daily"]})
    assert from_caloradine.get("interaction_major") == "warn"
    assert from_mendel.get("interaction_major") == "warn"


def test_a_strength_written_onto_the_name_is_still_the_medicine():
    """Electronic records hold "Tessorin10mg" as readily as "Tessorin 10 mg",
    and the rules have to apply either way."""
    for written in ["Tessorin 10 mg", "Tessorin-10mg", "Tessorin10mg", "TESSORIN10MG"]:
        found = codes("What dose of Caloradine?", [claim("Caloradine is given at 15 mg once daily.")],
                      **{**ADULT, "medicines": [written]})
        assert found.get("interaction_contraindicated") == "block", written


def test_a_dose_above_the_weight_based_maximum_is_blocked():
    """Vorantil is dosed by weight, at most 100 mg a dose. For a 70 kg adult
    that is 70 mg, and a stated 200 mg has to stop the answer rather than sit
    beside it as a note."""
    heavy = codes("Is 200 mg of Vorantil right?", [claim("Give 200 mg of Vorantil every 12 hours.")], **ADULT)
    assert heavy.get("dose_above_maximum") == "block"
    right = codes("Is 70 mg of Vorantil right?", [claim("Give 70 mg of Vorantil every 12 hours.")], **ADULT)
    assert right.get("dose_above_maximum") is None and right.get("weight_dose") == "info"


def test_sex_is_required_when_the_kidney_figure_comes_from_creatinine():
    """Cockcroft-Gault multiplies by 0.85 for a woman: the same creatinine puts
    this patient either side of a dose rule, so the sex cannot be assumed."""
    without = codes("What dose of Caloradine?", [claim("Caloradine is given at 15 mg once daily.")],
                    age_years=70, weight_kg=60, creatinine_umol_l=150)
    assert without == {"missing_data": "block"}
    with_sex = codes("What dose of Caloradine?", [claim("Caloradine is given at 15 mg once daily.")],
                     age_years=70, weight_kg=60, creatinine_umol_l=150, sex="female")
    assert "missing_data" not in with_sex


def test_a_moderate_interaction_is_noted_without_stopping_the_answer():
    """Three severities, three outcomes: contraindicated stops an answer, major
    warns, moderate is worth knowing but changes nothing. No medicine in the
    demo formulary carries a moderate rule, so this builds one."""
    idx = formulary.index()
    watched = formulary.Medicine(
        name="Plaxetin", classes=["anticoagulants"],
        interactions=[formulary.InteractionRule(id="PLX-INT-1", with_medicine="Caloradine",
                                                severity="moderate", note="Watch for drowsiness.")])
    idx.by_name["plaxetin"] = watched
    try:
        found = codes("What dose of Caloradine?", [claim("Caloradine is given at 15 mg once daily.")],
                      **{**ADULT, "medicines": ["Plaxetin"]})
        assert found["interaction_moderate"] == "info"
        assert "interaction_contraindicated" not in found and "interaction_major" not in found
    finally:
        del idx.by_name["plaxetin"]


def test_a_per_dose_maximum_accounts_for_how_often_it_is_given():
    """The daily maximum was divided by two only for twice-daily. Three times
    daily fell through to one, so a single dose equal to the whole daily
    maximum raised nothing, even though three of them are three times it. The
    system generates that frequency string itself in cds_hooks."""
    assert patient_checks._DOSES_PER_DAY["three times daily"] == 3
    assert patient_checks._DOSES_PER_DAY["every 8 hours"] == 3
    assert patient_checks._DOSES_PER_DAY["four times daily"] == 4
    assert patient_checks._DOSES_PER_DAY["every 6 hours"] == 4
    assert patient_checks._DOSES_PER_DAY["twice daily"] == 2
    # Anything unlisted stays at once daily, which compares against the full
    # daily maximum rather than a larger one.
    assert patient_checks._DOSES_PER_DAY.get("once daily", 1) == 1


def test_a_thousands_separator_in_a_dose_is_not_truncated():
    """"2,500 mg" matched as 500, five times under the stated dose, so the
    formulary maximum could not fire on it."""
    assert patient_checks._VALUE.findall("Alphamed is given as 2,500 mg once daily.") == [("2,500", "mg")]
    assert patient_checks._VALUE.findall("Give 500 mg daily.") == [("500", "mg")]
    # And the parsed number is the whole one.
    amounts = [(float(v.replace(",", "")), u) for v, u in
               patient_checks._VALUE.findall("Alphamed is given as 2,500 mg once daily.")]
    assert amounts == [(2500.0, "mg")]
    assert patient_checks._exceeds(amounts, 500.0, "mg") == 2500.0
