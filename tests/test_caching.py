"""Caches and machine-aware sizing: the same question isn't computed twice,
a cached answer is never stale or shared across patients, sites or test runs,
and batch sizes follow the machine's real limits."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, knowledge, pipeline, resources, retrieval  # noqa: E402
from app.schemas import PatientContext, Settings  # noqa: E402

ANSWERED = "What is the standard dose of Caloradine?"


@pytest.fixture(autouse=True)
def fresh():
    pipeline.clear_answer_cache()
    retrieval.clear_caches()
    yield
    pipeline.clear_answer_cache()


def test_the_same_question_is_answered_from_the_cache(database):
    first = pipeline.run(ANSWERED)
    second = pipeline.run(ANSWERED)
    assert second.answer_text == first.answer_text and second.decision == first.decision
    assert [c.text for c in second.claims] == [c.text for c in first.claims]
    # Still its own request: a new audit record, with the question in it.
    assert second.audit_id != first.audit_id
    from app import audit

    assert audit.store.get(second.audit_id)["redacted_query"] == ANSWERED
    assert pipeline.answer_cache_stats()["answers"] == 1


def test_different_settings_are_not_the_same_question():
    pipeline.run(ANSWERED)
    pipeline.run(ANSWERED, Settings(top_k=8))
    assert pipeline.answer_cache_stats()["answers"] == 2


def test_a_patient_is_never_answered_from_the_cache():
    pipeline.run(ANSWERED)
    with_patient = pipeline.run(ANSWERED, patient=PatientContext(age_years=70, weight_kg=70, egfr=20))
    assert with_patient.decision == "refuse"          # kidney function, not the cached answer
    assert pipeline.answer_cache_stats()["answers"] == 1


def test_test_runs_neither_use_nor_fill_the_cache():
    pipeline.run(ANSWERED, review=False)
    assert pipeline.answer_cache_stats()["answers"] == 0


def test_changing_the_documents_drops_cached_answers(monkeypatch):
    pipeline.run(ANSWERED)
    before = retrieval.index_version()
    retrieval.load_index()                             # as a release going live does
    assert retrieval.index_version() > before
    assert retrieval.cache_stats() == {"questions": 0, "passages": 0}
    assert pipeline._cached_answer(pipeline._cache_key(ANSWERED, Settings())) is None


def test_turning_the_cache_off(monkeypatch):
    monkeypatch.setattr(config, "ANSWER_CACHE_SECONDS", 0)
    pipeline.run(ANSWERED)
    pipeline.run(ANSWERED)
    assert pipeline.answer_cache_stats()["answers"] == 0


def test_sentences_and_questions_are_embedded_once():
    pipeline.run(ANSWERED)
    stats = retrieval.cache_stats()
    assert stats["questions"] >= 1 and stats["passages"] >= 1
    pipeline.clear_answer_cache()
    pipeline.run(ANSWERED)                             # same passages, no new embeddings
    assert retrieval.cache_stats()["passages"] == stats["passages"]


def test_sizes_follow_the_machine(monkeypatch):
    summary = resources.summary()
    assert summary["cpus"] >= 0.5 and summary["memory_gb"] > 0
    assert summary["accelerator"] in ("cpu", "mps", "cuda")
    assert 32 <= summary["embed_batch"] <= 1024
    assert 32 <= summary["imaging_batch"] <= 2048
    assert 1 <= summary["web_workers"] <= 16
    # A small container gets small numbers.
    monkeypatch.setattr(resources, "cpus", lambda: 2.0)
    monkeypatch.setattr(resources, "memory_bytes", lambda: 2 * resources.GB)
    monkeypatch.setattr(resources, "accelerator", lambda: "cpu")
    monkeypatch.setattr(resources, "accelerator_memory_bytes", lambda: 2 * resources.GB)
    assert resources.recommended_workers() == 1
    assert resources.volume_cache_size() == 1
    assert resources.embed_batch_size() <= 128
    assert resources.patch_batch_size(64) >= 32


@pytest.mark.parametrize("name, value", [("EMBED_BATCH_SIZE", "17"), ("IMAGING_BATCH_SIZE", "23"),
                                         ("WEB_WORKERS", "3"), ("IMAGING_CACHE_SERIES", "5")])
def test_an_operator_can_override_any_size(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    chosen = {"EMBED_BATCH_SIZE": resources.embed_batch_size, "IMAGING_BATCH_SIZE": lambda: resources.patch_batch_size(64),
              "WEB_WORKERS": resources.recommended_workers, "IMAGING_CACHE_SERIES": resources.volume_cache_size}[name]
    assert chosen() == int(value)
