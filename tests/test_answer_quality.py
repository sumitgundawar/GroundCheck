"""The extractive answer cites the passages that answer the question, not every
passage retrieved with it."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm, pipeline, retrieval  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


def _sections(response):
    return {retrieval.record_for(sid)["section"] for c in response.claims for sid in c.source_ids}


def test_a_side_effects_question_cites_only_adverse_reactions_or_contraindications():
    r = pipeline.run("What are the side effects of Varloraprex?", review=False)
    assert r.decision == "answer"
    assert _sections(r) <= {"Adverse Reactions", "Contraindications"}


def test_a_symptoms_question_cites_no_monitoring_passage():
    r = pipeline.run("What are the symptoms of Velgoleus Syndrome?", review=False)
    assert r.decision == "answer"
    assert "Monitoring" not in _sections(r) and "Symptoms" in _sections(r)


def test_a_dose_question_cites_the_dosage_passage_even_when_retrieval_ranks_it_below_the_cut():
    r = pipeline.run("How is Velgoleus Syndrome treated?", review=False)
    assert "Treatment" in _sections(r)


@pytest.mark.parametrize("text, sections", [
    ("What is the dose of Caloradine?", {"Dosage and Administration"}),
    ("What are the side effects of X?", {"Adverse Reactions"}),
    ("What should not be taken with X?", {"Drug Interactions"}),
    ("When should X not be used?", {"Contraindications"}),
    ("How is X monitored?", {"Monitoring"}),
    ("What are the symptoms of X?", {"Symptoms"}),
    ("What is X used for?", {"Indications and Usage"}),
    ("What causes X?", {"Causes"}),
    ("What is X?", {"Information"}),
    ("Tell me a joke", set()),
])
def test_the_intent_names_the_sections_that_answer(text, sections):
    assert llm.classify_intent(text) == sections


def test_a_question_with_two_parts_is_split_and_needs_a_sentence_for_each():
    parts = llm.split_question("What is the dose of Caloradine, and can it be combined with Mendel solution?")
    assert len(parts) == 2
    assert llm.split_question("What is the dose of Caloradine?") == ["What is the dose of Caloradine?"]
    r = pipeline.run("What is the dose of Caloradine, and can it be combined with Mendel solution?", review=False)
    assert r.decision == "answer" and {"CALO-001", "INTR-001"} <= {sid for c in r.claims for sid in c.source_ids}


def test_at_most_three_claims_each_with_relevance_and_where_it_was_found():
    r = pipeline.run("What is the first-line management of Veltris syndrome?", review=False)
    assert 1 <= len(r.claims) <= 3
    for claim in r.claims:
        text = retrieval.corpus_text_for(claim.source_ids[0])
        assert 0 < claim.relevance <= 1 and claim.support is not None
        assert text[claim.support.start:claim.support.end] == claim.support.sentence
        assert claim.support.sentence in claim.text
    grounding = next(s for s in r.trace if s.name == "grounding check")
    assert all("relevance" in row for row in grounding.data["claims"])


def test_a_passage_cited_from_beyond_the_retrieval_cut_is_listed_among_the_sources():
    r = pipeline.run("How is Velgoleus Syndrome treated?", review=False)
    cited = {sid for c in r.claims for sid in c.source_ids}
    assert cited <= {s.id for s in r.sources}


def test_the_answer_quality_targets_hold_on_a_sample():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "answer_quality_eval", Path(__file__).resolve().parent.parent / "scripts" / "answer_quality_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    spec.loader.exec_module(module)
    result = module.evaluate(sample=120)
    assert result["claim_precision"] >= 0.90 and result["section_recall"] >= 0.95 and result["claims_per_answer"] <= 2.0
