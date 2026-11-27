"""Round 2 defects in proposal.py, pipeline.py and the output guards, now fixed. Each name is the QA2 number in
.tracker/v3_progress.md."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, knowledge, llm, pipeline, retrieval  # noqa: E402
from app.schemas import LLMAnswer, LLMClaim  # noqa: E402
from test_tenant_documents import upload, world  # noqa: E402,F401

RIGHT = "The standard adult regimen is 15 mg once daily, taken in the morning with food, for an initial course of 14 days."


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
        retrieval.load_index()


def ask(q):
    pipeline.clear_answer_cache()
    return pipeline.run(q, review=False)


def says_yes(r):
    return r.decision == "answer" and r.answer_text.startswith("Yes")


@pytest.mark.parametrize("q", [
    "Is Caloradine 15 mg once daily not correct?", "Isn't Caloradine 15 mg once daily?", "Is Caloradine 15 mg once daily NOT the right dose?",
    "Is it true that Caloradine is not 15 mg once daily?", "Is Caloradine never 15 mg once daily?",
    "Is Caloradine 15 mg once daily no longer correct?", "Is Caloradine 15 mg once daily false?",
    "Is Caloradine 15 mg once daily contraindicated?", "Is Caloradine 15 mg once daily correct, or not?"])
def test_qa2_1_negated_proposals_are_not_confirmed(q):
    assert not says_yes(ask(q)), ask(q).answer_text


def test_qa2_1_spanish_negation(monkeypatch):
    monkeypatch.setattr(config, "SUPPORTED_LANGUAGES", "en,es")
    r = ask("¿No es correcto Caloradine 15 mg una vez al día?")
    assert not says_yes(r) and not r.answer_text.startswith("Sí")


@pytest.mark.parametrize("q", ["Is Caloradine 15 mg once daily for 14 days or longer correct?",
                               "Is Caloradine 15 mg once daily for at least 14 days correct?"])
def test_qa2_2_open_ended_durations_are_not_confirmed(q):
    assert not says_yes(ask(q))


def _model(monkeypatch, claims):
    monkeypatch.setattr(llm, "active_provider", lambda: {"name": "x", "kind": "cloud", "model": "m"})
    monkeypatch.setattr(llm, "generate_llm", lambda *a, **k: LLMAnswer(insufficient_context=False, claims=claims))
    pipeline.clear_answer_cache()
    return pipeline.run("What is the standard dose of Caloradine?", review=False)


def test_qa2_3_an_invented_citation_is_never_shown(monkeypatch):
    r = _model(monkeypatch, [LLMClaim(text=RIGHT, source_ids=["CALO-001", "NOPE-1"])])
    assert r.decision == "refuse" or "NOPE-1" not in r.answer_text


def test_qa2_4_bidi_controls_in_a_model_claim_do_not_pass(monkeypatch):
    r = _model(monkeypatch, [LLMClaim(text=RIGHT.replace("15 mg", "‮15 mg"), source_ids=["CALO-001"])])
    assert r.decision == "refuse" or "‮" not in r.answer_text


def test_qa2_5_conflicting_approved_documents_are_not_answered_as_one(world):
    made, client = world
    for name, text in (("Zentavil Adults", "Zentavil is 10 mg once daily."), ("Zentavil Clinic", "Zentavil is 500 mg once daily.")):
        body = f"# {name}\n\n## Dosage and Administration\n\n{text}\n".encode()
        source = upload(client, "group-admin", name, body).json()["source"]["id"]
        knowledge.review(source, "approve", None, allow_self_approval=True)
    knowledge.rebuild_index("two documents")
    r = ask("What is the dose of Zentavil?")
    assert r.decision == "refuse"
