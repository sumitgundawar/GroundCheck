"""Round 2: the same question must get the same decision however it arrives (/api/ask, /api/v1/ask, the SDK client,
a Spanish translation, a cached or fresh run, a model's text or the extractive text)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sdk" / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from app import apikeys, config, guards_input, llm, pipeline, retrieval  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import LLMAnswer, LLMClaim  # noqa: E402
from test_sdk_live import client, server  # noqa: E402,F401

CASES = json.loads((ROOT / "eval" / "golden_es.json").read_text(encoding="utf-8"))["cases"]
GOLDEN = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
QUESTIONS = ["What is the standard dose of Caloradine?", "What is the standard dose of Quelvarin?", "Is Caloradine 15 mg twice daily correct?",
             "Is Caloradine 15 mg once daily correct?", "What are the side effects of Caloradine?", "Is Caloradine safe in pregnancy?",
             "What is the dose of Caloradine for a child?", "Caloradine 150 mg once daily?", "Caloradine and Orrin-blockers together?"]


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
        retrieval.load_index()


def shape(r) -> tuple:
    body = r if isinstance(r, dict) else r.model_dump()
    return (body["decision"], body.get("reason_code"), [c["text"] if isinstance(c, dict) else c.text for c in body.get("claims") or []])


@pytest.fixture()
def api(database):
    guards_input.key_limiter._by_client.clear()
    pipeline.clear_answer_cache()
    return TestClient(app)


def test_every_route_and_a_cache_hit_give_the_same_answer(api):
    key = {"Authorization": "Bearer " + apikeys.create("qa2", ["ask"])["key"]}
    for q in QUESTIONS:
        pipeline.clear_answer_cache()
        fresh = shape(pipeline.run(q, review=False))
        cached = shape(pipeline.run(q, review=False))
        legacy = shape(api.post("/api/ask", json={"query": q}).json())
        v1 = shape(api.post("/api/v1/ask", json={"question": q}, headers=key).json())
        assert fresh == cached == legacy == v1, q


def test_the_sdk_client_matches_the_service(server):
    c, _ = client(server)
    for q in QUESTIONS:
        pipeline.clear_answer_cache()
        ask = c.ask(q)
        expected = pipeline.run(q, review=False)
        assert (ask.decision, ask.reason_code, [c.text for c in ask.claims]) == shape(expected), q


def test_spanish_never_answers_what_english_refuses_and_is_answered_only_when_english_is(monkeypatch):
    monkeypatch.setattr(config, "SUPPORTED_LANGUAGES", "en,es")
    pipeline.clear_answer_cache()
    for case in CASES[::2]:
        es, en = pipeline.run(case["query"], review=False), pipeline.run(case["en"], review=False)
        if es.decision == "answer":
            assert en.decision == "answer", case
        if case["expect"] == "refuse":
            assert es.decision == "refuse", case


def test_a_model_that_returns_the_source_sentences_gives_the_extractive_decision(monkeypatch):
    for q in QUESTIONS:
        pipeline.clear_answer_cache()
        extractive = pipeline.run(q, review=False)
        if extractive.decision != "answer" or extractive.proposal_check is not None:
            continue
        monkeypatch.setattr(llm, "active_provider", lambda: {"name": "x", "kind": "cloud", "model": "m"})
        monkeypatch.setattr(llm, "generate_llm", lambda *a, e=extractive, **k: LLMAnswer(
            insufficient_context=False, claims=[LLMClaim(text=c.text, source_ids=c.source_ids) for c in e.claims]))
        pipeline.clear_answer_cache()
        assert pipeline.run(q, review=False).decision == "answer", q
        monkeypatch.undo()
