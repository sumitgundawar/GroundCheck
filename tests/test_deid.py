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


@pytest.mark.parametrize("text, expected", [
    ("John Smith has a fever", "[NAME] has a fever"),
    ("Patient: SMITH, John", "Patient: [NAME]"),
    ("mrs o'neil needs caloradine", "mrs [NAME] needs caloradine"),
    ("nurse Kelly Brown checked", "nurse [NAME] checked"),
    ("Sarah, 34, pregnant", "[NAME], 34, pregnant"),
    ("husband Raj called", "husband [NAME] called"),
    ("insurance id ABC123456", "insurance id [ID]"),
    ("card 4111 1111 1111 1111", "card [CARD]"),
    ("SSN 123456789", "SSN [SSN]"),
    ("IBAN GB82 WEST 1234 5698 7654 32", "IBAN [ID]"),
    ("NPI 1234567893", "NPI [ID]"),
])
def test_v2_detectors(text, expected):
    assert clean(text) == expected


@pytest.mark.parametrize("text", [
    "Will Caloradine interact with warfarin?",
    "May I give Caloradine with food?",
    "Grace period for the next Caloradine dose?",
    "Bill the patient after the review",
    "Faith in the guidance",
    "Mr Caloradine dose",
    "What is the dose of 5 mg/kg for 12/12 months?",
    "Can Caloradine be given at 1,000,000 units daily for 14 days?",
    "A 12 year old needs 250 mg twice daily",
    "Give 10 mg/kg over 30 minutes",
    "Wilson disease and Caloradine",
    "SSN-like 123456789 is a dose count",
    "4111 1111 1111 1112 is not a card number checksum",
])
def test_nothing_that_looks_like_a_name_or_number_is_removed_wrongly(text):
    result = clean(text)
    assert "[NAME]" not in result and "[CARD]" not in result and "[SSN]" not in result
    assert result == text or result.replace("[NUMBER]", "") != result  # only the long-number fallback may fire


def test_undashed_ssn_needs_context_and_checksums_are_validated():
    assert deid._ssn_ok("123456789") and not deid._ssn_ok("000123456") and not deid._ssn_ok("900123456")
    assert deid._luhn_ok("4111111111111111") and not deid._luhn_ok("4111111111111112")
    assert deid._iban_ok("GB82 WEST 1234 5698 7654 32") and not deid._iban_ok("GB82 WEST 1234 5698 7654 33")
    assert deid._npi_ok("1234567893") and not deid._npi_ok("1234567890")
    assert clean("SSN 123456789") == "SSN [SSN]"
    assert clean("dose 123456789 times") != "dose [SSN] times"


def test_impossible_dates_are_not_dates():
    assert clean("on 31/02/2024") == "on 31/02/2024"
    assert clean("on 13/05/2024") == "on [DATE]"


def test_entities_have_positions_and_never_values():
    text = "Mr John Smith, DOB 12/03/1948"
    result = deid.deidentify(text, retrieval.is_known_term)
    assert [e.kind for e in result.entities] == ["name", "date"]
    first = result.entities[0]
    assert text[first.start:first.end] == "John Smith" and 0 < first.score <= 1 and first.detector
    assert not any(isinstance(v, str) and "Smith" in v for e in result.entities for v in vars(e).values())


def test_a_known_term_is_never_a_name_even_beside_a_cue():
    assert clean("patient Caloradine Veltris dose") == "patient Caloradine Veltris dose"
    assert clean("Dr Veltris dose") == "Dr Veltris dose"
    assert clean("nurse Quorl checked") == "nurse Quorl checked"


def test_missing_ner_packages_change_nothing(monkeypatch):
    import builtins

    from app import config

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.split(".")[0] in ("presidio_analyzer", "spacy"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    sample = ["John Smith has a fever", "Patient: SMITH, John DOB 03/04/1950", "Sarah, 34, pregnant, caloradine?",
              "What is the standard dose of Caloradine?", "lives at 12 Oak Rd, Leeds LS1 4AP"]
    monkeypatch.setattr(config, "DEID_NER", "off")
    expected = [clean(s) for s in sample]
    monkeypatch.setattr(config, "DEID_NER", "auto")
    monkeypatch.setattr(deid, "_NER_LOADED", False)
    monkeypatch.setattr(deid, "_NER_FN", None)
    monkeypatch.setattr(builtins, "__import__", refuse)
    assert [clean(s) for s in sample] == expected
    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setattr(deid, "_NER_LOADED", False)


def test_ner_proposals_never_override_a_known_term(monkeypatch):
    from app import config

    def fake(text):
        return [("name", text.index("Caloradine"), text.index("Caloradine") + 10, 0.9),
                ("name", text.index("Priya"), text.index("Priya") + 5, 0.6)]

    monkeypatch.setattr(config, "DEID_NER", "on")
    monkeypatch.setattr(deid, "_NER_LOADED", True)
    monkeypatch.setattr(deid, "_NER_FN", fake)
    assert clean("Caloradine for Priya here") == "Caloradine for [NAME] here"
    monkeypatch.setattr(deid, "_NER_LOADED", False)
    monkeypatch.setattr(deid, "_NER_FN", None)


def test_the_gold_sentences_are_found_and_nothing_is_wrongly_removed():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pii_eval", Path(__file__).resolve().parent.parent / "scripts" / "pii_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.evaluate()
    assert result["cases"] >= 200
    assert result["overall"]["recall"] >= 0.95 and result["overall"]["precision"] >= 0.97, result["misses"][:5]


def test_redaction_is_fast_without_ner(monkeypatch):
    import time

    from app import config

    monkeypatch.setattr(config, "DEID_NER", "off")
    text = "Mr John Smith, DOB 12/03/1948, NHS 943 476 5919, lives at 12 Oak Rd Leeds LS1 4AP: what is the dose of Caloradine?"
    clean(text)
    times = []
    for _ in range(100):
        t = time.perf_counter()
        clean(text)
        times.append((time.perf_counter() - t) * 1000)
    assert sorted(times)[94] < 3
