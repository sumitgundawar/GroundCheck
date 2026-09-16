"""Embeddings and hybrid retrieval.

The embedding model is loaded once and reused. Document vectors are built ahead
of time by scripts/build_index.py into the configured vector store (see
app/vectorstore.py) and loaded at app startup. Vectors are L2-normalised, so an
inner product is cosine similarity.

Retrieval is hybrid by default. Embeddings capture meaning but blur rare,
look-alike names (invented or real drug names), so a BM25 keyword index runs
alongside and each passage is ranked by a weighted blend of the two scores.
The score reported for each passage is always its cosine similarity, so the
retrieval gate keeps its meaning."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config, vectorstore

# A parallel list of corpus records, populated when the index is loaded. Row i
# of the vector store is _metadata[i].
_metadata: list[dict] = []
_store: "vectorstore.VectorStore | None" = None
_not_before: np.ndarray = np.zeros(0)
_not_after: np.ndarray = np.zeros(0)
_lexical: "LexicalIndex | None" = None


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


def load_demo_corpus() -> list[dict]:
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus() -> list[dict]:
    """Everything the index is built from: the demo corpus (unless
    INCLUDE_DEMO_CORPUS is false) and every approved imported document."""
    from . import knowledge  # imported here to avoid a cycle

    demo = load_demo_corpus() if config.INCLUDE_DEMO_CORPUS else []
    return demo + knowledge.approved_records()


def build_index() -> None:
    """Embed the corpus into the configured vector store and write the
    metadata that maps store rows to passages. Embeddings are cached, so a
    rebuild only embeds new text."""
    from . import knowledge

    knowledge.rebuild_index()


def write_index(records: list[dict], vectors: np.ndarray) -> None:
    """Replace the stored index with these records and their vectors."""
    metadata = []
    for r in records:
        entry = {
            "id": r["id"],
            "title": r["title"],
            "topic": r["topic"],
            "section": r.get("section", ""),
            "kind": classify_kind(r),
            "text": r["text"],
        }
        for extra in ("source_id", "source_version", "effective_from", "expires_on"):
            if r.get(extra) is not None:
                entry[extra] = r[extra]
        metadata.append(entry)

    store = vectorstore.create()
    store.build(vectors, [{"doc_id": m["id"], "title": m["title"]} for m in metadata])
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.INDEX_DIR / "metadata.tmp.json"
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp.replace(config.INDEX_DIR / "metadata.json")


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
    """Open the vector store and load the metadata and keyword index."""
    global _store, _metadata, _lexical
    meta_path = config.INDEX_DIR / "metadata.json"
    store = vectorstore.create()
    try:
        store.load()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Index not found ({exc}). Run scripts/build_index.py before starting the app."
        ) from exc
    if not meta_path.exists():
        raise FileNotFoundError("Index metadata not found. Run scripts/build_index.py before starting the app.")
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if store.count() != len(metadata):
        raise FileNotFoundError(
            f"The {store.name} store has {store.count()} vectors but the metadata lists "
            f"{len(metadata)} passages. Run scripts/build_index.py again."
        )
    global _not_before, _not_after
    _store, _metadata = store, metadata
    _lexical = LexicalIndex([f"{r['title']} {r['text']}" for r in _metadata])
    # Effective and expiry dates as timestamps (NaN when unset), so search can
    # skip documents that aren't in force without a rebuild.
    _not_before = np.array([_timestamp(r.get("effective_from")) for r in metadata], dtype="float64")
    _not_after = np.array([_timestamp(r.get("expires_on")) for r in metadata], dtype="float64")


def _ensure_loaded() -> None:
    if _store is None or not _metadata:
        load_index()


_TOKEN = re.compile(r"[a-z][a-z0-9'-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LexicalIndex:
    """A small BM25 index over the corpus, kept in memory.

    BM25 weights a term by how rare it is across the corpus, so a question that
    names "Remdodaprex" strongly prefers the passages that contain that exact
    name, even when an embedding model sees several similar names as close."""

    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(documents)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(self.n, dtype="float32")
        for i, doc in enumerate(documents):
            counts = Counter(tokenize(doc))
            self.lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.avg_length = float(self.lengths.mean()) if self.n else 1.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5)) if df else 0.0

    def scores(self, query: str) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0, dtype="float32")
        """BM25 with squared IDF. Plain BM25 lets two common words ("side
        effects") outweigh one rare name ("Remdodaprex"), so every drug's
        side-effect page scores alike. Squaring IDF makes distinctive terms,
        which are usually the names a question is about, dominate."""
        scores = np.zeros(self.n, dtype="float32")
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term) ** 2
            for doc, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc] / self.avg_length)
                scores[doc] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


def _timestamp(iso: str | None) -> float:
    if not iso:
        return float("nan")
    return datetime.fromisoformat(iso).timestamp()


def _in_force(rows: np.ndarray) -> np.ndarray:
    """Which rows' documents are within their effective and expiry dates now."""
    now = time.time()
    before, after = _not_before[rows], _not_after[rows]
    return (np.isnan(before) | (before <= now)) & (np.isnan(after) | (after > now))


def _cosine(value: float) -> float:
    # Cosine on normalised vectors is in [-1, 1]; clamp to [0, 1] for a clean
    # similarity score in the UI and the retrieval gate.
    return float(max(0.0, min(1.0, value)))


def search(query: str, k: int | None = None,
           hybrid: bool | None = None) -> list[tuple[dict, float]]:
    """Return up to k (record, cosine_score) pairs.

    Embedding-only search is ordered by cosine score. Hybrid search ranks by a
    blend of cosine score and keyword score, so the first result is not always
    the highest-scoring one; callers that need the best score should take the
    maximum."""
    _ensure_loaded()
    assert _store is not None and _lexical is not None
    k = k or config.TOP_K
    hybrid = config.HYBRID_RETRIEVAL if hybrid is None else hybrid
    query_vec = embed([query])[0]

    if not hybrid:
        hits = _store.search(query_vec, k * 3)
        rows = np.array([row for row, _ in hits], dtype=int)
        keep = _in_force(rows) if len(rows) else np.zeros(0, dtype=bool)
        return [(_metadata[row], _cosine(score)) for (row, score), ok in zip(hits, keep) if ok][:k]

    # Candidates: the nearest passages by meaning and the best by keywords.
    # Every store can answer this, and at corpus scale it matches a full scan.
    depth = max(_HYBRID_DEPTH, k * 25)
    nearest = dict(_store.search(query_vec, depth))
    keyword = _lexical.scores(query)
    by_keyword = [int(i) for i in np.argsort(-keyword)[:depth] if keyword[i] > 0]
    missing = [row for row in by_keyword if row not in nearest]
    if missing:
        for row, vec in zip(missing, _store.vectors(missing)):
            nearest[row] = float(np.dot(vec, query_vec))

    rows = np.fromiter(nearest.keys(), dtype=int)
    cosines = np.fromiter(nearest.values(), dtype="float32")
    keep = _in_force(rows)
    rows, cosines = rows[keep], cosines[keep]
    if len(rows) == 0:
        return []
    peak = float(keyword.max())
    keyword_norm = keyword[rows] / peak if peak > 0 else keyword[rows]
    alpha = config.HYBRID_ALPHA
    blended = alpha * np.clip(cosines, 0.0, 1.0) + (1.0 - alpha) * keyword_norm

    # Ties break on cosine score, then corpus order, so results are stable.
    order = np.lexsort((rows, -cosines, -blended))[:k]
    return [(_metadata[int(rows[i])], _cosine(cosines[i])) for i in order]


# How many candidates hybrid search takes from each of embedding and keyword
# search before blending.
_HYBRID_DEPTH = 100


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


@lru_cache(maxsize=1)
def _topic_names() -> dict[str, str]:
    """Each corpus topic keyed by its distinctive first word, for example
    "caloradine" or "veltris". Reference topics are generic and skipped."""
    names: dict[str, str] = {}
    for record in all_metadata():
        if record.get("kind") == "reference":
            continue
        topic = record.get("topic", "").lower()
        words = tokenize(topic)
        if words and len(words[0]) >= 4:
            names.setdefault(words[0], topic)
    return names


def topics_mentioned(query: str) -> dict[str, str]:
    """Corpus topics a question names, as {name word: topic}."""
    names = _topic_names()
    return {w: names[w] for w in set(tokenize(query)) if w in names}


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
    if not _metadata:
        return []
    assert _store is not None
    vectors = _store.vectors()  # (n, dim), already L2-normalised

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
