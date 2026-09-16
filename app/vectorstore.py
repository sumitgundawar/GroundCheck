"""Vector stores: where document embeddings live and are searched.

Retrieval talks to a small interface, so the embeddings can live in:

- local, the default: exact search in-process, vectors in index/vectors.npy
- Qdrant: a vector database, either a server (QDRANT_URL) or embedded on
  disk (QDRANT_PATH)

Choose with VECTOR_STORE=local|qdrant. Every store holds L2-normalised
vectors and answers with cosine similarity, and each vector is identified by
its row in index/metadata.json, which stays the source of passage text and
titles for every store."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from . import config


class VectorStore(Protocol):
    name: str

    def build(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        """Replace the stored vectors. Row i of vectors is document i."""

    def load(self) -> None:
        """Open an existing store. Raises FileNotFoundError if there's nothing to open."""

    def count(self) -> int: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """The k nearest documents as (row, cosine similarity), best first."""

    def vectors(self, rows: list[int] | None = None) -> np.ndarray:
        """Stored vectors for the given rows, in that order, or all rows."""


class LocalStore:
    """Exact cosine search with NumPy, over vectors saved in index/vectors.npy.

    A single matrix multiplication scores every passage, which is exact and
    fast well beyond this corpus's size (a million 384-dimension vectors take
    about 1.5 GB of memory). For larger collections, or several app
    instances sharing one index, use Qdrant.

    This replaces FAISS. faiss-cpu and PyTorch each bundle an OpenMP runtime,
    and on macOS the process aborts when both start one (OMP Error #15).
    NumPy's matrix multiplication doesn't use OpenMP."""

    name = "local"

    def __init__(self, index_dir=None):
        self.index_dir = index_dir or config.INDEX_DIR
        self._matrix: np.ndarray | None = None

    @property
    def _path(self):
        return self.index_dir / "vectors.npy"

    def build(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        matrix = np.ascontiguousarray(vectors, dtype="float32")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp.npy")
        np.save(tmp, matrix)
        tmp.replace(self._path)  # atomic, so a reader never sees half a file
        self._matrix = matrix

    def load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"No vectors at {self._path}.")
        self._matrix = np.load(self._path)

    def count(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.count())
        if k <= 0:
            return []
        scores = self._matrix @ np.asarray(query, dtype="float32")
        # Stable order for equal scores: lower row first.
        top = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(int(r), float(scores[r])) for r in top]

    def vectors(self, rows: list[int] | None = None) -> np.ndarray:
        assert self._matrix is not None
        return self._matrix if rows is None else self._matrix[rows]


class QdrantStore:
    """Qdrant, as a server or embedded. Points are keyed by document row."""

    name = "qdrant"
    _BATCH = 256

    def __init__(self, url: str | None = None, path: str | None = None,
                 collection: str | None = None, api_key: str | None = None):
        from qdrant_client import QdrantClient

        url = url if url is not None else config.QDRANT_URL
        path = path if path is not None else config.QDRANT_PATH
        self.collection = collection or config.QDRANT_COLLECTION
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or config.QDRANT_API_KEY or None)
        else:
            self.client = QdrantClient(path=path or str(config.INDEX_DIR / "qdrant"))

    def build(self, vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
        )
        for start in range(0, len(vectors), self._BATCH):
            end = min(start + self._BATCH, len(vectors))
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
                for i in range(start, end)
            ], wait=True)

    def load(self) -> None:
        if not self.client.collection_exists(self.collection):
            raise FileNotFoundError(f"No Qdrant collection named {self.collection}.")

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        hits = self.client.query_points(self.collection, query=np.asarray(query).tolist(),
                                        limit=k, with_payload=False).points
        return [(int(h.id), float(h.score)) for h in hits]

    def vectors(self, rows: list[int] | None = None) -> np.ndarray:
        if rows is None:
            rows = list(range(self.count()))
        if not rows:
            return np.zeros((0, 0), dtype="float32")
        found = {}
        for start in range(0, len(rows), self._BATCH):
            for point in self.client.retrieve(self.collection, ids=rows[start:start + self._BATCH],
                                              with_vectors=True, with_payload=False):
                found[int(point.id)] = point.vector
        return np.asarray([found[r] for r in rows], dtype="float32")


def create(name: str | None = None) -> VectorStore:
    name = (name or config.VECTOR_STORE).lower()
    if name in ("local", "faiss"):  # "faiss" kept for existing configurations
        return LocalStore()
    if name == "qdrant":
        return QdrantStore()
    raise ValueError(f"Unknown VECTOR_STORE {name!r}. Use local or qdrant.")
