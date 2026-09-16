"""De-identification: each kind of identifier is replaced, clinical details a
question depends on are kept, and the pipeline never stores or searches the
original."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import audit, deid, guards_output, pipeline, retrieval  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


def clean(text: str) -> str:
    return deid.deidentify(text, retrieval.is_known_term).text


@pytest.mark.parametrize("text, expected", [
    # Names
    ("Mr John Smith needs a dose", "Mr [NAME] needs a dose"),
    ("reviewed by Dr. Ahmed yesterday", "reviewed by Dr. [NAME] yesterday"),
    ("my patient Jane Doe has a rash", "my patient [NAME] has a rash"),
    ("patient named Priya Patel", "patient named [NAME]"),
    ("Name: Oluwaseun Adeyemi", "Name: [NAME]"),
    # Dates
    ("DOB 12/03/1948", "DOB [DATE]"),
    ("admitted 2024-03-12", "admitted [DATE]"),
    ("since 12.05.2023", "since [DATE]"),
    ("on 3rd March 2024", "on [DATE]"),
    ("on March 3, 2024", "on [DATE]"),
    ("review in Sept 2025", "review in [DATE]"),
    # Numbers and contact details
    ("NHS number 943 476 5919", "NHS number [NHS NUMBER]"),
    ("NHS 9434765919", "NHS [NHS NUMBER]"),
    ("SSN 123-45-6789", "SSN [SSN]"),
    ("MRN: A1234567", "MRN: [ID]"),
    ("hospital number is RX 88123", "hospital number is [ID]"),
    ("call 07700 900123", "call [PHONE]"),
    ("call +44 20 7946 0958", "call [PHONE]"),
    ("call (415) 555-2671", "call [PHONE]"),
    ("email jo.bloggs@nhs.net", "email [EMAIL]"),
    ("see https://ehr.example.org/patient/123", "see [URL]"),
    ("from 192.168.0.14", "from [IP ADDRESS]"),
    ("from 2001:db8::ff00:42:8329", "from [IP ADDRESS]"),
    # Places
    ("lives at 221B Baker Street", "lives at [ADDRESS]"),
    ("London NW1 6XE", "London [POSTCODE]"),
    ("Springfield, IL 62704", "Springfield, IL [ZIP]"),
    # Ages over 89 are generalised; the number that remains still drives checks
    ("dose for a 94 year old", "dose for a 90 year old"),
    ("patient aged 101", "patient aged 90"),
    ("a 97-year-old man", "a 90-year-old man"),
    # Leftover long numbers
    ("ref 4412 998 7765 21", "ref [NUMBER]"),
])
def test_identifiers_are_replaced(text, expected):
    assert clean(text) == expected


@pytest.mark.parametrize("text", [
    "What is the standard dose of Caloradine?",
    "What is the dose of Caloradine for a 5 year old?",
    "What is the dose of Caloradine for an 89 year old?",
    "Can Caloradine be given at 1,000,000 units daily for 14 days?",
    "Give 10 000 000 units of Caloradine?",
    "Is the patient Caloradine dose different in renal failure?",
    "Does Dr Caloradine exist?",
    "What dose for a patient with Veltris syndrome?",
    "Is 5 mg in May too much?",
    "Guidance changed in 2024, what is the dose now?",
    "HbA1c of 7.5 and BP 140/90, what next?",
    "Give 2.5 mg twice a day for 7 days",
    "Patient Presenting With Fever",
])
def test_clinical_details_are_kept(text):
    assert clean(text) == text


def test_nhs_numbers_need_a_valid_checksum():
    assert clean("NHS 943 476 5918") != "NHS [NHS NUMBER]"
    assert deid._nhs_checksum_ok("9434765919")
    assert not deid._nhs_checksum_ok("1111111111")


def test_summary_reports_kinds_never_values():
    result = deid.deidentify("Mr John Smith, DOB 12/03/1948, call 07700 900123", retrieval.is_known_term)
    assert result.summary() == "removed 1 date, 1 name, 1 phone number"
    assert "Smith" not in result.summary()
    assert deid.deidentify("What is the dose?").summary() == "no identifiers found"


def test_placeholders_are_not_treated_as_unknown_topics():
    assert guards_output._salient_terms("Is [NAME] on [DATE] due Caloradine?") == ["caloradine"]


def test_the_pipeline_never_keeps_the_original():
    question = "Mr John Smith, DOB 12/03/1948, NHS 943 476 5919: what is the standard dose of Caloradine?"
    response = pipeline.run(question)
    assert response.decision == "answer"
    record = audit.store.get(response.audit_id)
    stored = str(record)
    for secret in ("John", "Smith", "1948", "943 476 5919"):
        assert secret not in stored
    step = next(s for s in response.trace if s.name == "pii redaction")
    assert step.detail == "removed 1 NHS number, 1 date, 1 name"
