"""Embeddings and FAISS retrieval.

The embedding model is loaded once and reused. The FAISS index is built ahead of
time by scripts/build_index.py and loaded at app startup. Vectors are
L2-normalised, so the inner-product index returns cosine similarity directly."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from . import config

# A parallel list of corpus records, populated when the index is loaded.
_metadata: list[dict] = []
_index: faiss.Index | None = None


@lru_cache(maxsize=1)
def get_model():
    """Load the sentence-transformers model once. Imported lazily so that
    modules which only need the data (for example the config or schema tests)
    do not pay the import cost."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalise a list of texts. Returns a float32 matrix."""
    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def load_corpus() -> list[dict]:
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_index() -> None:
    """Embed every corpus text and persist a FAISS index plus a metadata list."""
    corpus = load_corpus()
    texts = [record["text"] for record in corpus]
    vectors = embed(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine on normalised vectors
    index.add(vectors)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "corpus.faiss"))

    metadata = [
        {
            "id": r["id"],
            "title": r["title"],
            "topic": r["topic"],
            "section": r.get("section", ""),
            "kind": classify_kind(r),
            "text": r["text"],
        }
        for r in corpus
    ]
    with open(config.INDEX_DIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


# Sections that identify a drug-label record versus a disease record. Used to
# tag each document with a coarse "kind" for the source cards and corpus map.
_DRUG_SECTIONS = {
    "Indications and Usage", "Dosage and Administration", "Contraindications",
    "Drug Interactions", "Adverse Reactions", "Use in Specific Populations",
}


def classify_kind(record: dict) -> str:
    """Coarse category for the source cards and corpus map. An explicit kind on
    the record wins (markers and procedures set their own); otherwise infer."""
    if record.get("kind"):
        return record["kind"]
    if record.get("topic") in ("safety", "interactions"):
        return "reference"
    if record.get("section") in _DRUG_SECTIONS:
        return "drug"
    return "condition"


def load_index() -> None:
    """Load the prebuilt index and metadata into module state."""
    global _index, _metadata
    index_path = config.INDEX_DIR / "corpus.faiss"
    meta_path = config.INDEX_DIR / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Index not found. Run scripts/build_index.py before starting the app."
        )
    _index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as fh:
        _metadata = json.load(fh)


def _ensure_loaded() -> None:
    if _index is None or not _metadata:
        load_index()


def search(query: str, k: int | None = None) -> list[tuple[dict, float]]:
    """Return up to k (record, cosine_score) pairs, highest score first."""
    _ensure_loaded()
    assert _index is not None
    k = k or config.TOP_K
    query_vec = embed([query])
    scores, idxs = _index.search(query_vec, min(k, len(_metadata)))
    results: list[tuple[dict, float]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        # Cosine on normalised vectors is already in [-1, 1]; clamp to [0, 1]
        # for a clean similarity score in the UI.
        results.append((_metadata[idx], float(max(0.0, min(1.0, score)))))
    return results


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts or [text.strip()]


def best_sentence(query: str, passage_text: str) -> str:
    """Pick the sentence of a passage most similar to the query. Used by the
    extractive fallback so claims stay short and on-topic."""
    sentences = split_sentences(passage_text)
    if len(sentences) == 1:
        return sentences[0]
    vectors = embed([query] + sentences)
    query_vec = vectors[0]
    sentence_vecs = vectors[1:]
    sims = sentence_vecs @ query_vec
    return sentences[int(np.argmax(sims))]


def corpus_text_for(source_id: str) -> str | None:
    _ensure_loaded()
    for record in _metadata:
        if record["id"] == source_id:
            return record["text"]
    return None


def all_metadata() -> list[dict]:
    _ensure_loaded()
    return list(_metadata)


# --------------------------------------------------------------------------
# Corpus map: a 2D PCA projection of every document embedding, plus aggregate
# statistics. Computed once and cached so the endpoint is instant.
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def corpus_projection(dims: int = 3) -> list[dict]:
    """Project all document vectors to 2D or 3D with PCA for the corpus map. The
    flat index stores the original (normalised) vectors, so we reconstruct them
    rather than re-embedding. Each axis is scaled independently into [-1, 1].
    Returns x, y, and (for 3D) z, so the frontend can render either."""
    _ensure_loaded()
    assert _index is not None
    n = _index.ntotal
    vectors = _index.reconstruct_n(0, n)  # (n, dim), already L2-normalised

    centred = vectors - vectors.mean(axis=0, keepdims=True)
    # Top principal directions via SVD on the centred matrix.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    k = max(2, min(3, dims))
    coords = centred @ vt[:k].T  # (n, k)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    scaled = 2.0 * (coords - mins) / span - 1.0

    points = []
    for record, row in zip(_metadata, scaled):
        p = {
            "id": record["id"],
            "kind": record.get("kind", "condition"),
            "title": record.get("title", ""),
            "section": record.get("section", ""),
            "x": round(float(row[0]), 4),
            "y": round(float(row[1]), 4),
        }
        if k >= 3:
            p["z"] = round(float(row[2]), 4)
        points.append(p)
    return points


def corpus_stats() -> dict:
    """Aggregate counts for the corpus map header tiles."""
    _ensure_loaded()
    by_kind: dict[str, int] = {}
    by_section: dict[str, int] = {}
    topics: set[str] = set()
    for r in _metadata:
        by_kind[r.get("kind", "condition")] = by_kind.get(r.get("kind", "condition"), 0) + 1
        section = r.get("section") or "Unlabelled"
        by_section[section] = by_section.get(section, 0) + 1
        topics.add(r.get("topic", ""))
    # A topic, with its kind and document count, for the clickable topics list.
    topic_kind: dict[str, str] = {}
    topic_count: dict[str, int] = {}
    for r in _metadata:
        t = r.get("topic", "")
        topic_kind[t] = r.get("kind", "condition")
        topic_count[t] = topic_count.get(t, 0) + 1
    topic_list = [
        {"topic": t, "kind": topic_kind[t], "count": topic_count[t]}
        for t in sorted(topics)
    ]
    return {
        "total": len(_metadata),
        "topics": len(topics),
        "by_kind": by_kind,
        "by_section": dict(sorted(by_section.items(), key=lambda kv: -kv[1])),
        "topic_list": topic_list,
    }
