"""Retrieval tests: the keyword index, hybrid ranking, and the retrieval gate's
use of the best cosine score. Offline; no model calls beyond local embeddings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
# Deterministic: never draft with a cloud or a selected local model.
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import pipeline, retrieval  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _index_ready():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
        retrieval.load_index()


# --- Keyword index ---------------------------------------------------------

def test_tokenize_lowercases_and_keeps_hyphenated_words():
    assert retrieval.tokenize("Orrin-blockers, 15 mg!") == ["orrin-blockers", "mg"]


def test_rare_name_outweighs_common_phrase():
    # Two documents share the common phrase; only one has the rare name. With
    # squared IDF the rare name must decide the ranking.
    docs = [f"filler document number {i} about side effects" for i in range(40)]
    docs.append("Remdodaprex may cause mild drowsiness")
    index = retrieval.LexicalIndex(docs)
    scores = index.scores("What are the side effects of Remdodaprex?")
    assert int(scores.argmax()) == len(docs) - 1


def test_unknown_terms_score_zero():
    index = retrieval.LexicalIndex(["Caloradine is started at 15 mg once daily."])
    assert float(index.scores("Zalortin").max()) == 0.0


# --- Hybrid search ---------------------------------------------------------

# Regression cases: with embeddings alone, these names were confused with
# look-alike drug names and the questions were wrongly refused.
@pytest.mark.parametrize("query, prefix", [
    ("What are the side effects of Remdodaprex?", "REMD-"),
    ("What are the side effects of Lennedizane?", "LENN-"),
    ("What are the side effects of Messenestal?", "MESS-"),
    ("What are the side effects of Monfatirin?", "MONF-"),
    ("What are the side effects of Kormalizane?", "KORM-"),
    ("How is Nolleseen Dysregulation monitored?", "NOLL-"),
])
def test_hybrid_search_finds_the_named_entity(query, prefix):
    results = retrieval.search(query, 4, hybrid=True)
    assert results[0][0]["id"].startswith(prefix)
    assert pipeline.run(query).decision == "answer"


def test_hybrid_scores_are_cosine_similarities():
    query = "What is the standard dose of Caloradine?"
    hybrid = dict((r["id"], s) for r, s in retrieval.search(query, 4, hybrid=True))
    embedding = dict((r["id"], s) for r, s in retrieval.search(query, 40, hybrid=False))
    for doc_id, score in hybrid.items():
        if doc_id in embedding:
            assert score == pytest.approx(embedding[doc_id], abs=1e-4)


def test_embedding_only_search_is_still_available():
    results = retrieval.search("What is the standard dose of Caloradine?", 4, hybrid=False)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_gate_uses_best_score_not_first_result():
    r = pipeline.run("What are the side effects of Remdodaprex?")
    gate = next(s for s in r.trace if s.name == "retrieval gate")
    assert gate.data["best_score"] == pytest.approx(max(s.score for s in r.sources), abs=1e-4)


def test_embeddings_are_safe_for_concurrent_requests():
    """The model runs on the configured device (the CPU by default), and
    embedding from many threads at once neither crashes nor changes results."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from app import config, retrieval

    model = retrieval.get_model()
    assert str(model.device).startswith(config.EMBED_DEVICE)
    queries = [f"What is the standard dose of drug {i}?" for i in range(40)]
    expected = retrieval.embed(queries)
    with ThreadPoolExecutor(max_workers=16) as pool:
        got = list(pool.map(lambda q: retrieval.embed([q])[0], queries))
    assert np.allclose(np.stack(got), expected, atol=1e-5)
