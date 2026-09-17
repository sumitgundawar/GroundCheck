"""Vector store tests. The local store and Qdrant (embedded, in a temp folder) must behave
identically: same neighbours, same scores, same stored vectors, and the same
hybrid retrieval results on the real corpus."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, retrieval, vectorstore  # noqa: E402


def _random_unit_vectors(n: int, dim: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _store(kind: str, tmp_path: Path) -> vectorstore.VectorStore:
    if kind == "local":
        return vectorstore.LocalStore(index_dir=tmp_path / "local")
    return vectorstore.QdrantStore(url="", path=str(tmp_path / "qdrant"), collection="test")


@pytest.mark.parametrize("kind", ["local", "qdrant"])
def test_store_round_trip_and_search(kind, tmp_path):
    vectors = _random_unit_vectors(200, 16)
    store = _store(kind, tmp_path)
    store.build(vectors, [{"doc_id": f"D{i}"} for i in range(len(vectors))])
    assert store.count() == 200

    query = vectors[42]
    hits = store.search(query, 5)
    assert hits[0][0] == 42
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    exact = np.argsort(-(vectors @ query))[:5].tolist()
    assert [row for row, _ in hits] == exact

    rows = [7, 199, 0]
    assert np.allclose(store.vectors(rows), vectors[rows], atol=1e-5)
    assert store.vectors().shape == (200, 16)


@pytest.mark.parametrize("kind", ["local", "qdrant"])
def test_rebuilding_replaces_the_contents(kind, tmp_path):
    store = _store(kind, tmp_path)
    store.build(_random_unit_vectors(50, 8, seed=1), [{}] * 50)
    store.build(_random_unit_vectors(30, 8, seed=2), [{}] * 30)
    assert store.count() == 30


def test_loading_a_missing_store_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        vectorstore.LocalStore(index_dir=tmp_path / "nothing").load()
    with pytest.raises(FileNotFoundError):
        vectorstore.QdrantStore(url="", path=str(tmp_path / "q"), collection="absent").load()


def test_faiss_name_still_selects_the_local_store():
    assert isinstance(vectorstore.create("faiss"), vectorstore.LocalStore)


def test_unknown_store_name_is_refused():
    with pytest.raises(ValueError, match="local or qdrant"):
        vectorstore.create("elasticsearch")


def test_hybrid_retrieval_is_identical_on_local_and_qdrant(tmp_path, monkeypatch):
    """Build a Qdrant copy of the real index and compare results query by query."""
    retrieval.load_index()
    live = retrieval._state()
    vectors = live.store.vectors()
    qdrant = vectorstore.QdrantStore(url="", path=str(tmp_path / "corpus"), collection="corpus")
    qdrant.build(vectors, [{} for _ in range(len(vectors))])

    queries = [
        "What is the standard dose of Caloradine?",
        "What are the side effects of Remdodaprex?",
        "How is Nolleseen Dysregulation monitored?",
        "If conservative management fails for Veltris syndrome, which medication is used and at what dose?",
        "How should I treat a broken arm at home?",
    ]
    for query in queries:
        with_local = [(r["id"], round(s, 4)) for r, s in retrieval.search(query, 4, hybrid=True)]
        with retrieval.using(retrieval.make_state(qdrant, live.metadata)):
            with_qdrant = [(r["id"], round(s, 4)) for r, s in retrieval.search(query, 4, hybrid=True)]
        assert with_local == with_qdrant, query


def test_metadata_and_store_must_agree(tmp_path, monkeypatch):
    store = vectorstore.LocalStore(index_dir=tmp_path)
    store.build(_random_unit_vectors(3, 8), [{}] * 3)
    (tmp_path / "metadata.json").write_text("[]")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(vectorstore, "create", lambda name=None: vectorstore.LocalStore(index_dir=tmp_path))
    with pytest.raises(FileNotFoundError, match="build_index"):
        retrieval.load_index()
    monkeypatch.undo()
    retrieval.load_index()
