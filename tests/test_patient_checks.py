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
