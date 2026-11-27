"""A question that proposes a regimen is compared with the source: "No" when any
part differs, "Yes" when every part matches, a refusal when a part cannot be
read or checked. Applies in extractive and model modes and over /api/v1/ask."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import apikeys, config, guards_input, llm, pipeline, proposal, retrieval  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import LLMAnswer, LLMClaim, PatientContext  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
        retrieval.load_index()


def ask(q, **kw):
    return pipeline.run(q, review=False, **kw)


def verdict(r):
    return r.answer_text.split(" ")[0] if r.decision == "answer" else None


def statuses(r):
    return {d.name: d.status for d in r.proposal_check.dimensions}


# Caloradine: 15 mg once daily, in the morning with food, 14 days, oral.
# Velvalondel: 25 mL twice daily, in the morning with food, 7 days.
# Lembitulex: 50 mcg once daily, in the morning with food, 10 days.

# --- reading a question ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,dose", [
    ("15 mg", ("mass", "15")), ("15mg", ("mass", "15")), ("fifteen milligrams", ("mass", "15")),
    ("0.015 g", ("mass", "15")), ("15000 mcg", ("mass", "15")), ("15000 µg", ("mass", "15")),
    ("0,015 g", ("mass", "15")), ("2,5 mg", ("mass", "2.5")), ("2.5 mg", ("mass", "2.5")),
    ("twenty five mg", ("mass", "25")), ("5 mL", ("volume", "5")), ("five millilitres", ("volume", "5")),
    ("0.005 L", ("volume", "5")), ("a dose of 1.5 mg", ("mass", "1.5")),
])
def test_a_dose_is_read_in_any_spelling_and_unit(text, dose):
    p = proposal.parse(f"Is {text} okay?")
    assert [(q.kind, str(q.value.normalize())) for q in p.dose] == [(dose[0], str(__import__("decimal").Decimal(dose[1]).normalize()))]
    assert not p.ambiguities


@pytest.mark.parametrize("text,rate", [
    ("once daily", 1), ("once a day", 1), ("od", 1), ("qd", 1), ("every 24 hours", 1),
    ("q24h", 1), ("1x daily", 1), ("one time a day", 1), ("once every morning", 1), ("every morning", 1),
    ("nightly", 1), ("nocte", 1), ("every day", 1), ("daily", 1), ("twice daily", 2), ("twice a day", 2), ("two times a day", 2), ("2 times daily", 2),
    ("bd", 2), ("bid", 2), ("bds", 2), ("b.i.d.", 2), ("every 12 hours", 2), ("every twelve hours", 2), ("q12h", 2),
    ("2x/day", 2), ("morning and evening", 2), ("tds", 3), ("tid", 3), ("t.d.s", 3), ("thrice daily", 3),
    ("three times a day", 3), ("every 8 hours", 3), ("q8h", 3), ("8-hourly", 3), ("qid", 4), ("qds", 4),
    ("four times a day", 4), ("every 6 hours", 4), ("q6h", 4), ("every hour", 24), ("hourly", 24),
    ("weekly", Fraction(1, 7)), ("once a week", Fraction(1, 7)), ("twice a week", Fraction(2, 7)),
    ("every other day", Fraction(1, 2)), ("every 2 days", Fraction(1, 2)), ("every 2 weeks", Fraction(1, 14)),
])
def test_a_frequency_is_read_as_doses_a_day(text, rate):
    p = proposal.parse(f"Can I give 15 mg {text}?" if text not in ("every day", "daily") else f"Can I give Caloradine {text}?")
    assert p.freq == [Fraction(rate)], p.ambiguities
    assert not p.ambiguities


@pytest.mark.parametrize("text,days", [
    ("for 14 days", 14), ("for fourteen days", 14), ("for 2 weeks", 14), ("for two weeks", 14), ("for a week", 7),
    ("for 1 week", 7), ("for 48 hours", 2), ("x 7 days", 7), ("a 10-day course", 10), ("for 3 days", 3),
    ("for a fortnight", 14),
])
def test_a_duration_is_read_in_days(text, days):
    p = proposal.parse(f"Is 15 mg {text} okay?")
    assert p.duration == [(Fraction(days), Fraction(days))]


def test_months_and_years_are_ranges_of_days():
    assert proposal.parse("15 mg for 2 months").duration == [(Fraction(56), Fraction(62))]
    assert proposal.parse("15 mg for 1 year").duration == [(Fraction(365), Fraction(366))]


@pytest.mark.parametrize("text,route", [
    ("orally", "oral"), ("by mouth", "oral"), ("PO", "oral"), ("oral", "oral"), ("topically", "topical"),
    ("on the skin", "topical"), ("subcutaneously", "subcutaneous"), ("SC", "subcutaneous"), ("under the skin", "subcutaneous"),
    ("intravenously", "intravenous"), ("IV", "intravenous"), ("intramuscularly", "intramuscular"), ("IM", "intramuscular"),
    ("inhaled", "inhaled"), ("rectally", "rectal"), ("sublingually", "sublingual"),
])
def test_a_route_is_read(text, route):
    assert proposal.parse(f"Is 15 mg {text} okay?").route == [route]


@pytest.mark.parametrize("text", [
    "15 mg/kg", "15 mg/mL", "10-20 mg", "10 to 20 mg", "2 tablets", "one capsule", "half a tablet", "15", "1,000 mg",
    "double the dose", "15 units", "15 mg twise a day", "15 mg at 8am", "once", "15 mg monthly", "an injection",
    "15 mg orally and subcutaneously", "a maximum of 30 mg",
])
def test_what_cannot_be_read_is_flagged_not_guessed(text):
    assert proposal.parse(f"Is {text} okay?").ambiguities


@pytest.mark.parametrize("text", [
    "a maximum of 30 mg a day", "no more than 30 mg per day", "up to 30 mg daily", "at most 30 mg a day",
    "a daily maximum of 30 mg", "30 mg a day at most", "maximum daily dose of 30 mg", "30 mg/day maximum",
    "not to exceed 30 mg per day", "a total of 30 mg per day",
])
def test_a_maximum_per_day_is_read(text):
    p = proposal.parse(f"Is {text} okay?")
    assert [q.shown for q in p.max_daily] == ["30 mg"] and not p.dose and not p.ambiguities, p


def test_a_class_name_that_looks_like_an_abbreviation_is_not_read_as_one():
    assert not proposal.parse("Is it ok to combine it with Tid-blockers?").freq
    assert not proposal.parse("Is it ok to combine it with Bid-blockers?").freq


# --- a regimen compared with the source --------------------------------------------------------------------------

MATCH = [
    "Is 15 mg okay for Caloradine?", "Is 15 mg once daily okay for Caloradine?", "Is 15 mg once a day okay for Caloradine?",
    "Is 15 mg od okay for Caloradine?", "Is 15 mg every 24 hours okay for Caloradine?",
    "Is 0.015 g once daily okay for Caloradine?", "Is 15000 mcg once daily okay for Caloradine?",
    "Is fifteen mg once daily okay for Caloradine?", "Is fifteen mg once daily for fourteen days okay for Caloradine?",
    "Is 15 mg once daily for 2 weeks okay for Caloradine?", "Is Caloradine okay for 14 days?",
    "Is Caloradine okay for two weeks?", "Is Caloradine okay orally?", "Can Caloradine be taken by mouth?",
    "Is 15 mg once daily orally for 14 days okay for Caloradine?",
    "Is 15 mg once daily in the morning with food okay for Caloradine?",
    "Is it okay to give Caloradine in the morning?", "Is it okay to give Caloradine with food?",
    "Is 15 mg a day okay for Caloradine?", "Is 15 mg daily okay for Caloradine?",
    "Is 25 mL twice daily okay for Velvalondel?", "Is 25 mL bd okay for Velvalondel?",
    "Is 25 mL every 12 hours okay for Velvalondel?", "Is 25 mL q12h okay for Velvalondel?",
    "Is 25 mL two times a day for 7 days okay for Velvalondel?", "Is 0.025 L twice daily okay for Velvalondel?",
    "Is 25 mL twice daily for a week okay for Velvalondel?",
    "Is 50 mcg once daily okay for Lembitulex?", "Is 0.05 mg once daily okay for Lembitulex?",
    "Is 50 mcg once daily for 10 days okay for Lembitulex?",
]


@pytest.mark.parametrize("q", MATCH)
def test_a_proposal_that_matches_the_source_is_told_yes(q):
    r = ask(q)
    assert r.decision == "answer", (r.reason_code, r.refused_reason)
    assert r.answer_text.startswith("Yes — "), r.answer_text
    assert r.proposal_check.status == "match" and "mismatch" not in statuses(r).values()
    assert r.claims and all(c.grounded for c in r.claims)
    assert r.claims[0].source_ids and r.claims[0].source_ids[0] in r.answer_text


MISMATCH = [
    ("Is 15 mg twice a day okay for Caloradine?", "frequency"),
    ("Is 15 mg bd okay for Caloradine?", "frequency"), ("Is 15 mg tds okay for Caloradine?", "frequency"),
    ("Is 15 mg q8h okay for Caloradine?", "frequency"), ("Is 15 mg every 8 hours okay for Caloradine?", "frequency"),
    ("Is 15 mg every 12 hours okay for Caloradine?", "frequency"), ("Is 15 mg qid okay for Caloradine?", "frequency"),
    ("Is 15 mg four times a day okay for Caloradine?", "frequency"), ("Is 15 mg weekly okay for Caloradine?", "frequency"),
    ("Is 15 mg every other day okay for Caloradine?", "frequency"), ("Is 15 mg hourly okay for Caloradine?", "frequency"),
    ("Is 15 mg as needed okay for Caloradine?", "frequency"), ("Is 15 mg twice daily okay for Caloradine?", "frequency"),
    ("Is 15 mg once daily for 7 days okay for Caloradine?", "duration"),
    ("Is Caloradine okay for 28 days?", "duration"), ("Is Caloradine okay for a week?", "duration"),
    ("Is Caloradine okay for 3 weeks?", "duration"), ("Is Caloradine okay for 48 hours?", "duration"),
    ("Is Caloradine okay for a fortnight and a half?", None),
    ("Is Caloradine okay for 10 days?", "duration"),
    ("Is 15 mg in the evening okay for Caloradine?", "timing"),
    ("Is Caloradine okay before food?", "timing"), ("Is Caloradine okay after food?", "timing"),

    ("Is 25 mL once daily okay for Velvalondel?", "frequency"), ("Is 25 mL tds okay for Velvalondel?", "frequency"),
    ("Is 25 mL twice daily for 14 days okay for Velvalondel?", "duration"),
    ("Is 50 mcg twice daily okay for Lembitulex?", "frequency"),
    ("Is 50 mcg once daily for 14 days okay for Lembitulex?", "duration"),
]


@pytest.mark.parametrize("q,dimension", [m for m in MISMATCH if m[1]])
def test_a_proposal_that_differs_from_the_source_is_told_no(q, dimension):
    r = ask(q)
    assert r.decision == "answer", (r.reason_code, r.refused_reason)
    assert r.answer_text.startswith("No — the source gives "), r.answer_text
    assert r.proposal_check.status == "mismatch" and statuses(r)[dimension] == "mismatch"
    assert r.claims and all(c.grounded for c in r.claims)


@pytest.mark.parametrize("q,dimension", [
    ("Is 30 mg once daily okay for Caloradine?", "dose"), ("Is 7.5 mg once daily okay for Caloradine?", "dose"),
    ("Is 15 mcg once daily okay for Caloradine?", "dose"), ("Is 15 mL once daily okay for Caloradine?", "dose"),
    ("Is 0.03 g once daily okay for Caloradine?", "dose"), ("Is Caloradine okay intravenously?", "route"),
    ("Is Caloradine okay topically?", "route"), ("Is Caloradine okay subcutaneously?", "route"),
    ("Is 25 mL once daily okay for Velvalondel?", "frequency"), ("Is Caloradine okay at midday?", "timing"),
])
def test_a_wrong_dose_or_route_is_never_answered_without_a_no(q, dimension):
    """Other guards may refuse these first; an answer must lead with No, and the structured check says what differed."""
    r = ask(q)
    if r.decision == "answer":
        assert r.answer_text.startswith("No — ") and r.proposal_check.status == "mismatch"
    elif r.proposal_check is not None:
        assert r.proposal_check.status == "mismatch" and statuses(r)[dimension] == "mismatch"


WRONG_VALUE = [
    ("Is 30 mg once daily okay for Caloradine?", "30 mg", "dose"),
    ("Caloradine is 50 mg once daily, correct?", "50 mg", "dose"),
    ("Is 7.5 mg once daily okay for Caloradine?", "7.5 mg", "dose"),
    ("Is 15 mcg once daily okay for Caloradine?", "15 mcg", "dose"),
    ("Is 15 mL once daily okay for Caloradine?", "15 mL", "dose"),
    ("Is 0.03 g once daily okay for Caloradine?", "30 mg", "dose"),
    ("Is Caloradine okay intravenously?", "intravenous", "route"),
    ("Is Caloradine okay IV?", "intravenous", "route"),
    ("Is Caloradine okay subcut?", "subcutaneous", "route"),
    ("Is Caloradine okay SC?", "subcutaneous", "route"),
    ("Is Caloradine okay IM?", "intramuscular", "route"),
    ("Is Caloradine okay by subcutaneous injection?", "subcutaneous", "route"),
    ("Is Caloradine okay topically?", "topical", "route"),
    ("Is Velvalondel okay rectally?", "rectal", "route"),
    ("Is 30 mg twice daily okay for Caloradine?", "30 mg", "dose"),
    ("Is 25 mL once daily by mouth okay for Velvalondel?", "once daily", "frequency"),
    ("Is 100 mcg once daily okay for Lembitulex?", "100 mcg", "dose"),
]


@pytest.mark.parametrize("q,wrong,dimension", WRONG_VALUE)
def test_a_wrong_dose_or_route_is_answered_no_with_the_sourced_regimen_and_never_restated(q, wrong, dimension):
    r = ask(q)
    assert r.decision == "answer", (r.reason_code, r.refused_reason)
    assert r.answer_text.startswith("No — the source gives "), r.answer_text
    assert r.proposal_check.status == "mismatch" and statuses(r)[dimension] == "mismatch"
    lead, _, rest = r.answer_text.partition(". ")
    given, sep, said = lead.rpartition(", not ")
    assert sep, lead
    assert wrong.lower() not in given.lower() and wrong.lower() not in rest.lower(), r.answer_text
    assert r.claims and all(c.grounded for c in r.claims) and "[" in r.answer_text
    assert all(wrong.lower() not in c.text.lower() for c in r.claims)


@pytest.mark.parametrize("q", [
    "Caloradine is 50 mg once daily.", "Caloradine 50 mg once daily", "Caloradine should be 50 mg once daily",
    "I think Caloradine IV",
])
def test_a_statement_that_is_not_a_question_with_a_wrong_value_is_still_refused(q):
    r = ask(q)
    assert r.decision == "refuse" or r.answer_text.startswith("No — ")
    if r.decision == "answer":
        assert "50 mg" not in r.answer_text.partition(", not ")[0]


@pytest.mark.parametrize("q", [
    "Is 50 mg once daily okay for Caloradine and Velvalondel?", "Is 50 mg once daily okay for Zorbextine?",
    "Is 50 mg once daily okay for a 5 year old on Caloradine?", "Is 50 mg once daily okay for Caloradine XR?",
    "Is 50 mg once daily okay for Caloradine in pregnancy?",
    "Is Caloradine 50 mg once daily okay by injection?", "Is 50 mg once daily okay for Caloradine tablets?",
])
def test_a_wrong_value_is_still_refused_when_the_drug_or_the_term_cannot_be_judged(q):
    r = ask(q)
    assert r.decision == "refuse", r.answer_text


def test_a_patient_the_sources_do_not_cover_still_wins_over_the_no():
    r = ask("Is 50 mg once daily okay for Caloradine?", patient=PatientContext(age_years=5))
    assert r.decision == "refuse"


@pytest.mark.parametrize("q", [
    "Is a maximum of 15 mg a day okay for Caloradine?", "Is a maximum of 30 mg a day okay for Caloradine?",
    "Is no more than 10 mg per day okay for Caloradine?", "Is a maximum of 25 mL a day okay for Velvalondel?",
    "Is a daily maximum of 50 mcg okay for Lembitulex?",
])
def test_a_maximum_daily_dose_is_never_judged_against_the_usual_regimen(q):
    r = ask(q)
    assert r.decision == "refuse", r.answer_text
    assert not (r.proposal_check and r.proposal_check.status in ("match", "mismatch"))


def test_the_verdict_names_what_the_source_gives_and_what_was_proposed():
    r = ask("Is 15 mg twice a day okay for Caloradine?")
    assert r.answer_text.startswith("No — the source gives 15 mg once daily, not 15 mg twice daily. ")
    assert "[CALO-001]" in r.answer_text
    d = {x.name: x for x in r.proposal_check.dimensions}
    assert d["frequency"].documented == "once daily" and d["frequency"].proposed == "twice daily"
    assert d["frequency"].source_id == "CALO-001" and d["dose"].status == "match"
    assert r.proposal_check.drug == "caloradine"


def test_a_match_names_what_was_matched():
    r = ask("Is 0.015 g once a day okay for Caloradine?")
    assert r.answer_text.startswith("Yes — this matches the source: 0.015 g once daily. ")


def test_the_answer_cites_the_sentence_that_states_the_regimen_and_the_route_sentence_for_a_route():
    r = ask("Is Caloradine okay intravenously?")
    assert r.proposal_check is None or r.proposal_check.status == "mismatch"
    r = ask("Can Caloradine be given orally?")
    assert r.decision == "answer" and r.answer_text.startswith("Yes — ") and "oral agent" in r.answer_text


# --- phrasing ---------------------------------------------------------------------------------------------------

HEDGES = ["Is {p} okay for Caloradine?", "Can I give Caloradine {p}?", "Would {p} be fine for Caloradine?",
          "Is it safe to give Caloradine {p}?", "Caloradine {p}, correct?", "Should Caloradine be given {p}?",
          "Could I give Caloradine {p}?", "Is it alright to give Caloradine {p}?", "Am I allowed to give Caloradine {p}?",
          "Quick question: is Caloradine {p} appropriate?", "What if I give Caloradine {p}?",
          "My colleague suggests Caloradine {p}. Is that right?", "Hi, would it be ok to prescribe Caloradine {p}?",
          "Is giving Caloradine {p} recommended?", "please can i give caloradine {p}"]


@pytest.mark.parametrize("template", HEDGES)
def test_hedged_phrasing_gets_the_same_verdict(template):
    good = ask(template.format(p="15 mg once daily"))
    bad = ask(template.format(p="15 mg twice daily"))
    assert good.decision == "answer" and good.answer_text.startswith("Yes"), good.refused_reason
    assert bad.decision == "answer" and bad.answer_text.startswith("No"), bad.refused_reason


NEGATED = ["Isn't {p} fine for Caloradine?", "Is it not okay to give Caloradine {p}?", "Shouldn't Caloradine be given {p}?",
           "Can't I give Caloradine {p}?", "Wouldn't Caloradine {p} be ok?", "Isn't it correct to give Caloradine {p}?"]


@pytest.mark.parametrize("template", NEGATED)
def test_a_negated_question_is_refused_because_a_yes_or_no_would_be_ambiguous(template):
    for p in ("15 mg once daily", "15 mg twice daily"):
        r = ask(template.format(p=p))
        assert r.decision == "refuse" and r.reason_code == "proposal_unverifiable", r.answer_text
        assert "phrased negatively" in r.refused_reason


@pytest.mark.parametrize("template", ["Is {p} too much for Caloradine?", "Is it dangerous to give Caloradine {p}?",
                                      "Is {p} of Caloradine an overdose?", "Would {p} be harmful for Caloradine?",
                                      "Is {p} too little for Caloradine?", "Is it wrong to give Caloradine {p}?"])
@pytest.mark.parametrize("p", ["15 mg once daily", "15 mg twice daily"])
def test_a_question_about_harm_is_not_answered_with_a_yes_or_no(template, p):
    r = ask(template.format(p=p))
    assert r.decision == "refuse" and r.reason_code in ("proposal_unverifiable", "term_not_covered")


def test_a_plain_question_is_untouched():
    for q in ["What is the standard dose of Caloradine?", "How often should Caloradine be given?",
              "What is the usual course length of Caloradine?", "How is Caloradine given?"]:
        r = ask(q)
        assert r.proposal_check is None and r.decision == "answer", q


def test_a_question_that_asks_for_a_value_and_names_none_is_untouched():
    r = ask("What is the dose of Caloradine for adults?")
    assert r.proposal_check is None


# --- what cannot be checked is refused --------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "Is 15 of Caloradine okay?", "Is 2 tablets of Caloradine okay?", "Is one tablet of Caloradine okay twice daily?",
    "Is 0.5 mg/kg of Caloradine okay?", "Is 1,000 mg of Caloradine okay?", "Is it ok to double the dose of Caloradine?",
    "Is 15 mg of Caloradine at 8am okay?",
    "Is 10 to 20 mg of Caloradine okay?", "Is 15 mg of Caloradine okay by injection?",
    "Is 15 mg of Caloradine okay monthly?", "Is a maximum of 30 mg of Caloradine okay?",
    "Is 15 mg of Caloradine orally and subcutaneously okay?", "Is 15 mg bd or od okay for Caloradine?",
    "Is 15 mg twice daily at 8am okay for Caloradine?",
    "Is 15 mg of Caloradine and 5 mL of Mendel solution okay?",
    "Is 25 mL twice daily of Velvalondel and 15 mg once daily of Caloradine okay?",
])
def test_an_unverifiable_proposal_is_refused_with_its_own_code(q):
    r = ask(q)
    assert r.decision == "refuse", r.answer_text
    assert r.reason_code in ("proposal_unverifiable", "stated_value_unsupported", "form_unsupported",
                             "term_not_covered"), r.reason_code
    if r.reason_code == "proposal_unverifiable":
        assert r.proposal_check.status == "unverifiable" and r.explanation.summary
        assert any(d.reason for d in r.proposal_check.dimensions)


def test_an_unverifiable_proposal_with_a_mismatch_elsewhere_is_still_refused():
    r = ask("Is 15 mg twice daily at 8am okay for Caloradine?")
    assert r.decision == "refuse" and r.reason_code == "proposal_unverifiable"
    assert statuses(r)["frequency"] == "mismatch" and statuses(r)["timing"] == "unverifiable"


@pytest.mark.parametrize("q", ["Is 15 mg of Caloradin okay once daily?", "Is 15 mg of Kaloradine okay once daily?",
                               "Is 15 mg of Zorbextine okay twice daily?", "Is 15 mg okay twice daily?",
                               "Is 15 mg of Veltris syndrome okay?"])
def test_an_unknown_or_missing_medicine_is_never_answered(q):
    assert ask(q).decision == "refuse"


def test_a_misspelt_medicine_that_gets_past_coverage_is_still_caught():
    out = proposal.evaluate("Is 15 mg of Caloradin okay once daily?")
    assert out.check.status == "unverifiable"
    assert "did you mean" in out.message


def test_two_medicines_are_refused_with_the_proposal_code():
    r = ask("Is 15 mg of Caloradine and 5 mL of Mendel solution okay?")
    assert r.reason_code == "proposal_unverifiable" and "more than one medicine" in r.refused_reason


def test_the_sources_contradicting_themselves_is_unverifiable_not_a_guess():
    # Simkasestal: "25 mL once daily, taken twice daily with water"
    r = ask("Is 25 mL once daily okay for Simkasestal?")
    assert r.decision == "refuse" and r.reason_code == "proposal_unverifiable"
    assert "conflicting frequencies" in r.refused_reason
    r = ask("Is 25 mL okay for Simkasestal?")
    assert r.decision == "answer" and r.answer_text.startswith("Yes")
    r = ask("Is Simkasestal okay for 14 days?")
    assert r.decision == "answer" and r.answer_text.startswith("Yes")
    r = ask("Is Simkasestal okay for 10 days?")
    assert r.decision == "answer" and r.answer_text.startswith("No")


def test_a_source_with_two_times_of_day_is_unverifiable_for_a_time_of_day():
    # Rodbimizane: "once every morning, after the evening meal"
    r = ask("Is 75 mL in the morning okay for Rodbimizane?")
    assert r.decision == "refuse" and "two different timings" in r.refused_reason


def test_two_doses_or_two_frequencies_in_one_question_are_unverifiable():
    for q in ("Is 15 mg or 30 mg once daily okay for Caloradine?", "Is 15 mg bd or od okay for Caloradine?",
              "Is 15 mg for 7 days or 14 days okay for Caloradine?"):
        out = proposal.evaluate(q)
        assert out.check.status == "unverifiable", q


def test_a_misspelt_dosing_word_is_never_matched_to_the_source():
    out = proposal.evaluate("Is 15 mg of Caloradine twise a day okay?")
    assert out.check.status == "unverifiable" and "misspelt" in out.message


def test_a_missing_dimension_is_unverifiable():
    doc = proposal.documented("caloradine")
    doc.parsed.duration.clear()
    doc.parsed.duration_shown.clear()
    reading = proposal.read("Is Caloradine okay for 14 days?")
    dims = proposal.compare(reading, doc)
    assert dims[0].status == "unverifiable" and "no duration" in dims[0].reason
    doc = proposal.documented("caloradine")
    doc.parsed.route.clear()
    dims = proposal.compare(proposal.read("Is Caloradine okay orally?"), doc)
    assert dims[0].status == "unverifiable" and "no route" in dims[0].reason


def test_a_dose_the_sources_do_not_pair_with_a_daily_total_is_unverifiable():
    doc = proposal.documented("caloradine")
    doc.parsed.freq.clear()
    dims = proposal.compare(proposal.read("Is a maximum of 15 mg a day okay for Caloradine?"), doc)
    assert dims[0].status == "unverifiable"


def test_a_stated_maximum_is_compared_only_when_the_source_gives_one():
    from decimal import Decimal
    doc = proposal.documented("caloradine")
    doc.max_stated = True
    doc.parsed.max_daily = [proposal.Quantity(Decimal(30), "mass", "30 mg")]
    assert proposal.compare(proposal.read("Is a maximum of 30 mg a day okay for Caloradine?"), doc)[0].status == "match"
    assert proposal.compare(proposal.read("Is a maximum of 40 mg a day okay for Caloradine?"), doc)[0].status == "mismatch"


def test_one_dose_or_a_total_for_the_day_when_the_two_readings_disagree_is_unverifiable():
    # Vuspemoxine: 40 mg twice daily. "80 mg a day" is a dose of 80 mg, or a total of 80: the two readings differ.
    out = proposal.evaluate("Is 80 mg a day okay for Vuspemoxine?")
    assert out.check.status == "unverifiable"
    assert "one dose a day or a total" in out.message
    # Caloradine: once daily, so one dose a day and a total are the same
    assert proposal.evaluate("Is 15 mg a day okay for Caloradine?").check.status == "match"
    assert proposal.evaluate("Is 30 mg a day okay for Caloradine?").check.status == "mismatch"


# --- patient context --------------------------------------------------------------------------------------------

def test_a_matching_adult_regimen_is_refused_for_a_child():
    for q in ("Is 15 mg once daily okay for Caloradine?", "Is 15 mg twice daily okay for Caloradine?"):
        r = ask(q, patient=PatientContext(age_years=8, weight_kg=25))
        assert r.decision == "refuse" and r.reason_code == "patient_unsafe"
        assert r.proposal_check is not None


def test_the_patient_checks_still_apply_to_a_proposal():
    r = ask("Is 15 mg once daily okay for Caloradine?", patient=PatientContext(age_years=50, weight_kg=80, egfr=95))
    assert r.decision == "answer" and r.answer_text.startswith("Yes") and r.patient_findings is not None
    r = ask("Is 15 mg once daily okay for Caloradine?", patient=PatientContext(age_years=50, weight_kg=80, egfr=30))
    assert r.decision == "refuse" and r.reason_code == "patient_unsafe"


# --- extractive, model and the HTTP API -------------------------------------------------------------------------

def _pretend_model(monkeypatch, claim_text):
    calls = []

    def generate(query, sources, temperature=0.0):
        calls.append(query)
        return LLMAnswer(insufficient_context=False, claims=[LLMClaim(text=claim_text, source_ids=["CALO-001"])])

    monkeypatch.setattr(llm, "active_provider", lambda: {"kind": "cloud", "model": "m", "provider": "p"})
    monkeypatch.setattr(llm, "generate_llm", generate)
    monkeypatch.setattr(llm, "judge_claim", lambda *a, **k: None)
    return calls


def test_a_model_cannot_change_the_verdict(monkeypatch):
    calls = _pretend_model(monkeypatch, "Caloradine is given as 15 mg twice daily.")
    r = ask("Is 15 mg twice a day okay for Caloradine?")
    assert r.decision == "answer" and r.answer_text.startswith("No — the source gives 15 mg once daily")
    assert "twice daily." not in r.claims[0].text and not calls and r.llm_used is False
    r = ask("Is 15 mg once a day okay for Caloradine?")
    assert r.answer_text.startswith("Yes — ")


def test_a_plain_question_still_uses_the_model(monkeypatch):
    calls = _pretend_model(monkeypatch, "The standard adult regimen is 15 mg once daily, taken in the morning with food, "
                                        "for an initial course of 14 days.")
    r = ask("What is the standard dose of Caloradine?")
    assert calls and r.llm_used is True and r.proposal_check is None


@pytest.fixture()
def client(database):
    guards_input.key_limiter._by_client.clear()
    made = apikeys.create("proposal", ["ask", "verify", "usage", "audit:read"])
    return TestClient(app), {"Authorization": "Bearer " + made["key"]}


def test_the_v1_api_returns_the_verdict_and_the_structured_check(client):
    http, headers = client
    r = http.post("/api/v1/ask", json={"question": "Is 15 mg twice a day okay for Caloradine?"}, headers=headers).json()
    assert r["decision"] == "answer" and r["answer_text"].startswith("No — the source gives 15 mg once daily")
    check = r["proposal_check"]
    assert check["status"] == "mismatch" and check["drug"] == "caloradine"
    freq = next(d for d in check["dimensions"] if d["name"] == "frequency")
    assert freq == {"name": "frequency", "status": "mismatch", "proposed": "twice daily", "documented": "once daily",
                    "source_id": "CALO-001", "reason": None, "code": "differs"}
    r = http.post("/api/v1/ask", json={"question": "Is 15 mg once daily okay for Caloradine?"}, headers=headers).json()
    assert r["proposal_check"]["status"] == "match"
    r = http.post("/api/v1/ask", json={"question": "Is 15 of Caloradine okay?"}, headers=headers).json()
    assert r["decision"] == "refuse" and r["reason_code"] == "proposal_unverifiable"
    assert r["proposal_check"]["status"] == "unverifiable"
    r = http.post("/api/v1/ask", json={"question": "What is the standard dose of Caloradine?"}, headers=headers).json()
    assert r["proposal_check"] is None
    r = http.post("/api/v1/ask", json={"question": "Is 15 mg once daily okay for Caloradine?",
                                       "patient": {"age_years": 8, "weight_kg": 25}}, headers=headers).json()
    assert r["decision"] == "refuse" and r["reason_code"] == "patient_unsafe"


def test_the_audit_record_keeps_the_check(client):
    http, headers = client
    r = http.post("/api/v1/ask", json={"question": "Is 15 mg twice a day okay for Caloradine?"}, headers=headers).json()
    stored = http.get(f"/api/v1/audit/{r['audit_id']}", headers=headers)
    assert stored.status_code == 200


def test_the_trace_shows_the_proposal_step_only_when_there_is_a_proposal():
    names = [t.name for t in ask("Is 15 mg twice a day okay for Caloradine?").trace]
    assert names.index("source coverage") < names.index("proposal check") < names.index("generate")
    assert "proposal check" not in [t.name for t in ask("What is the standard dose of Caloradine?").trace]
    r = ask("Is 15 of Caloradine okay?")
    assert [t.name for t in r.trace if t.status == "fail"][0] == "proposal check"
    assert [t.name for t in r.trace][-1] == "decision"


def test_the_cached_answer_keeps_the_check():
    pipeline.clear_answer_cache()
    first = pipeline.run("Is 15 mg twice a day okay for Caloradine?", review=True)
    again = pipeline.run("Is 15 mg twice a day okay for Caloradine?", review=True)
    assert first.proposal_check == again.proposal_check and again.answer_text.startswith("No")


def test_decimal_commas_are_read_as_decimals_before_any_guard():
    r = ask("Is 0,015 g once daily okay for Caloradine?")
    assert r.decision == "answer" and r.answer_text.startswith("Yes")
    r = ask("Is 0,5 mg once daily okay for Caloradine?")
    assert r.decision == "refuse" or r.answer_text.startswith("No")
    assert pipeline._decimal_commas("Is 2,5 mg of X ok, or 1,5 g?") == "Is 2.5 mg of X ok, or 1.5 g?"
    assert pipeline._decimal_commas("on 1,000 mg") == "on 1,000 mg"
    assert pipeline._decimal_commas("Is it ok by mouth?") == "Is it ok orally?"


# --- the labelled cases -----------------------------------------------------------------------------------------

def _evaluator():
    spec = importlib.util.spec_from_file_location("proposal_eval", ROOT / "scripts" / "proposal_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_labelled_cases_have_no_wrong_answer():
    cases = json.loads((ROOT / "eval" / "proposal_cases.json").read_text())["cases"]
    assert len(cases) >= 300
    assert {c["family"] for c in cases} == {"tuned", "heldout"}
    ev = _evaluator()
    bad = []
    step = max(1, len(cases) // 250)
    for c in cases[::step]:
        patient = PatientContext(**c["patient"]) if c.get("patient") else None
        outcome, failure = ev.judge(c, ask(c["query"], patient=patient))
        if failure:
            bad.append((c["id"], c["query"], failure))
    assert not bad, bad[:5]


@pytest.mark.parametrize("q", ["Is 15 mg once daily without food okay for Caloradine?",
                               "Is no more than 15 mg per day okay for Caloradine?"])
def test_without_food_and_no_more_than_are_not_negations(q):
    assert "phrased negatively" not in (ask(q).refused_reason or "")


def test_a_dose_question_that_states_a_regimen_gets_the_verdict_on_it():
    assert ask("What is the dose of Caloradine 15 mg twice daily?").answer_text.startswith("No — the source gives 15 mg once daily")
    assert ask("What is the dose of Caloradine 15 mg once daily?").answer_text.startswith("Yes — ")
    assert ask("What is the dose of Caloradine?").proposal_check is None


@pytest.mark.parametrize("q", ["Is Caloradine okay for at least 14 days?", "Is Caloradine okay for up to 14 days?",
                               "Is Caloradine okay for 14 days or more?", "Is Caloradine okay for 14+ days?"])
def test_an_open_ended_duration_is_never_a_match(q):
    r = ask(q)
    assert r.decision == "refuse" and r.reason_code == "proposal_unverifiable"
