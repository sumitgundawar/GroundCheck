"""The organisation's knowledge: imported documents, their approval, and the
search index built from them.

Lifecycle of a document:

  upload   parsed and chunked, stored as version N with status "pending".
           Nothing pending can be cited.
  approve  a second person approves it (uploaders can't approve their own
           when accounts are required). The previously approved version of
           the same document is retired, and the index is rebuilt.
  reject   stays out of the index, with the reviewer's note.
  retire   removed from the index, kept for the record.

An approved version is cited only between its effective and expiry dates.

The search index is built from the demo corpus (unless INCLUDE_DEMO_CORPUS is
false) plus every approved chunk. Embeddings are cached by the text's hash,
so a rebuild only embeds text that is new."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select

from . import config, db, documents
from .db import Source, SourceChunk, utcnow

log = logging.getLogger("groundcheck.knowledge")

STATUSES = ("pending", "approved", "rejected", "retired")


class KnowledgeError(ValueError):
    """A document operation was refused. The message is safe to show."""


# --- Document keys ---------------------------------------------------------

def document_key(title: str) -> str:
    """A stable key for a document across versions, from its title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:64] or "document"


def _chunk_id(source_id: int, position: int) -> str:
    return f"DOC{source_id}-{position + 1:03d}"


# --- Import and review -----------------------------------------------------

def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def import_document(filename: str, data: bytes, *, title: str | None = None, owner: str = "",
                    effective_from: datetime | None = None, expires_on: datetime | None = None,
                    uploaded_by: int | None = None, max_chars: int = 1000) -> dict:
    """Parse, chunk and store a document as a pending version."""
    parsed = documents.parse(filename, data)
    title = (title or parsed.title).strip()[:300] or parsed.title
    effective_from, expires_on = _as_utc(effective_from), _as_utc(expires_on)
    if effective_from and expires_on and expires_on <= effective_from:
        raise KnowledgeError("The expiry date must be after the effective date.")
    chunks = documents.chunk(parsed, max_chars=max_chars)
    digest = hashlib.sha256(data).hexdigest()
    key = document_key(title)

    with db.session() as s:
        duplicate = s.scalar(select(Source).where(
            Source.document_key == key, Source.content_sha256 == digest,
            Source.status.in_(("pending", "approved"))))
        if duplicate is not None:
            raise KnowledgeError(f"This exact file is already version {duplicate.version} of {duplicate.title}.")
        latest = s.scalar(select(func.max(Source.version)).where(Source.document_key == key)) or 0
        source = Source(
            document_key=key, version=latest + 1, title=title, status="pending",
            filename=(filename or "")[:255], media_type=parsed.media_type, content_sha256=digest,
            owner=owner.strip()[:200], effective_from=effective_from, expires_on=expires_on,
            uploaded_by=uploaded_by, review_note="",
        )
        s.add(source)
        s.flush()
        for c in chunks:
            s.add(SourceChunk(source_id=source.id, position=c.position, chunk_id=_chunk_id(source.id, c.position),
                              section=c.section[:300], text=c.text))
        s.flush()
        return _summary(source, len(chunks))


def _summary(source: Source, chunk_count: int | None = None) -> dict:
    return {
        "id": source.id,
        "document_key": source.document_key,
        "version": source.version,
        "title": source.title,
        "status": source.status,
        "filename": source.filename,
        "media_type": source.media_type,
        "owner": source.owner,
        "effective_from": source.effective_from.isoformat() if source.effective_from else None,
        "expires_on": source.expires_on.isoformat() if source.expires_on else None,
        "uploaded_by": source.uploaded_by,
        "uploaded_at": source.uploaded_at.isoformat() if source.uploaded_at else None,
        "reviewed_by": source.reviewed_by,
        "reviewed_at": source.reviewed_at.isoformat() if source.reviewed_at else None,
        "review_note": source.review_note,
        "chunks": chunk_count if chunk_count is not None else len(source.chunks),
        "in_force": in_force(source),
        "evaluation": source.evaluation,
    }


def in_force(source: Source, now: datetime | None = None) -> bool:
    now = now or utcnow()
    return (source.status == "approved"
            and (source.effective_from is None or source.effective_from <= now)
            and (source.expires_on is None or source.expires_on > now))


def list_sources() -> list[dict]:
    with db.session() as s:
        counts = dict(s.execute(select(SourceChunk.source_id, func.count(SourceChunk.id))
                                .group_by(SourceChunk.source_id)).all())
        rows = s.scalars(select(Source).order_by(Source.title, Source.version.desc())).all()
        return [_summary(r, counts.get(r.id, 0)) for r in rows]


def get_source(source_id: int) -> dict:
    with db.session() as s:
        source = s.get(Source, source_id)
        if source is None:
            raise KnowledgeError("No such document.")
        summary = _summary(source)
        summary["sections"] = [{"chunk_id": c.chunk_id, "section": c.section, "text": c.text} for c in source.chunks]
        return summary


def review(source_id: int, decision: str, reviewer_id: int | None, note: str = "",
           allow_self_approval: bool = False) -> dict:
    """Approve, reject or retire a document version."""
    if decision not in ("approve", "reject", "retire"):
        raise KnowledgeError("Choose approve, reject or retire.")
    with db.session() as s:
        source = s.get(Source, source_id)
        if source is None:
            raise KnowledgeError("No such document.")
        if decision in ("approve", "reject") and source.status != "pending":
            raise KnowledgeError(f"Only pending documents can be {decision}d; this one is {source.status}.")
        if decision == "retire" and source.status != "approved":
            raise KnowledgeError("Only approved documents can be retired.")
        if (decision == "approve" and not allow_self_approval and reviewer_id is not None
                and source.uploaded_by == reviewer_id):
            raise KnowledgeError("Someone other than the uploader must approve a document.")

        now = utcnow()
        if decision == "approve":
            for previous in s.scalars(select(Source).where(
                    Source.document_key == source.document_key, Source.status == "approved",
                    Source.id != source.id)):
                previous.status = "retired"
                previous.reviewed_at = now
                previous.review_note = f"Replaced by version {source.version}."
            source.status = "approved"
        elif decision == "reject":
            source.status = "rejected"
        else:
            source.status = "retired"
        source.reviewed_by = reviewer_id
        source.reviewed_at = now
        source.review_note = note.strip()[:2000] or source.review_note
        return _summary(source)


# --- Corpus ----------------------------------------------------------------

def approved_records(now: datetime | None = None) -> list[dict]:
    """Chunks of every approved document, as corpus records. Effective and
    expiry dates are carried along, so retrieval can skip a document that goes
    out of force between index rebuilds."""
    if not db.ready():
        return []
    records = []
    with db.session() as s:
        rows = s.execute(
            select(SourceChunk, Source)
            .join(Source, SourceChunk.source_id == Source.id)
            .where(Source.status == "approved")
            .order_by(Source.id, SourceChunk.position)
        ).all()
        for chunk, source in rows:
            records.append({
                "id": chunk.chunk_id,
                "title": f"{source.title}: {chunk.section}" if chunk.section != source.title else source.title,
                "topic": source.title.lower(),
                "section": chunk.section,
                "kind": "organisation",
                "text": chunk.text,
                "source_id": source.id,
                "source_version": source.version,
                "document_key": source.document_key,
                "document_title": source.title,
                "effective_from": source.effective_from.isoformat() if source.effective_from else None,
                "expires_on": source.expires_on.isoformat() if source.expires_on else None,
            })
    return records


# --- Index -----------------------------------------------------------------

@dataclass
class IndexStatus:
    state: str = "idle"          # idle, running, failed
    started_at: str | None = None
    finished_at: str | None = None
    documents: int = 0
    embedded: int = 0
    reused: int = 0
    error: str | None = None
    release: dict | None = None  # the release the last rebuild made


_status = IndexStatus()
_index_lock = threading.Lock()


def index_status() -> dict:
    return dict(_status.__dict__)


def _cache_path():
    return config.INDEX_DIR / "embedding_cache.npz"


def _load_cache() -> dict[str, np.ndarray]:
    try:
        with np.load(_cache_path()) as data:
            return {k: data[k] for k in data.files}
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path().with_suffix(".tmp.npz")
    np.savez(tmp, **cache)
    tmp.replace(_cache_path())


def _text_key(text: str) -> str:
    return "t" + hashlib.sha256(f"{config.EMBED_MODEL}\n{text}".encode("utf-8")).hexdigest()[:40]


def rebuild_index(reason: str = "Index rebuilt", user_id: int | None = None, user_name: str = "GroundCheck") -> dict:
    """Rebuild the search index from the demo corpus and approved documents.
    With a database, the result is a release that is checked before it goes
    live (app/releases.py); without one, it's swapped in directly. Only one
    rebuild runs at a time."""
    from . import retrieval  # imported here to avoid a cycle

    if not _index_lock.acquire(blocking=False):
        raise KnowledgeError("The index is already being rebuilt.")
    try:
        _status.state, _status.error = "running", None
        _status.started_at, _status.finished_at = utcnow().isoformat(), None
        records = retrieval.load_corpus()
        cache = _load_cache()
        keys = [_text_key(r["text"]) for r in records]
        missing = sorted({k for k in keys if k not in cache})
        if missing:
            texts = {k: r["text"] for k, r in zip(keys, records) if k in set(missing)}
            vectors = retrieval.embed([texts[k] for k in missing])
            cache.update(zip(missing, vectors))
            _save_cache({k: cache[k] for k in set(keys)})
        if keys:
            vectors = np.stack([cache[k] for k in keys]).astype("float32")
        else:
            # Nothing to search yet: every question will be refused.
            dim = retrieval.embedding_dimension()
            vectors = np.zeros((0, dim), dtype="float32")
        _status.release = None
        if db.ready():
            from . import releases

            _status.release = releases.create(records, vectors, reason, user_id, user_name)
        else:
            retrieval.write_index(records, vectors)
            retrieval.load_index()
        _status.documents, _status.embedded, _status.reused = len(records), len(missing), len(records) - len(missing)
        _status.state = "idle"
        return index_status()
    except Exception as exc:
        _status.state, _status.error = "failed", str(exc)
        log.exception("Index rebuild failed")
        raise
    finally:
        _status.finished_at = utcnow().isoformat()
        _index_lock.release()


def rebuild_index_in_background(reason: str = "Index rebuilt", user_id: int | None = None,
                                user_name: str = "GroundCheck") -> None:
    def run():
        try:
            rebuild_index(reason, user_id, user_name)
        except Exception:  # noqa: BLE001 - recorded in the status
            pass
    threading.Thread(target=run, name="index-rebuild", daemon=True).start()


# --- Evaluation per document -------------------------------------------------

_INVENTED = "Quelvarin"


def evaluate_source(source_id: int, limit: int = 10) -> dict:
    """Generate test questions from an approved document and run them.

    For each section (up to limit): an answerable question built from its
    heading, which must be answered citing this document, and the same
    question about an invented document name, which must be refused. The
    results are saved on the document."""
    from . import pipeline

    with db.session() as s:
        source = s.get(Source, source_id)
        if source is None:
            raise KnowledgeError("No such document.")
        if not in_force(source):
            raise KnowledgeError("Only approved documents in force can be evaluated.")
        chunk_ids = {c.chunk_id for c in source.chunks}
        sections = []
        for c in source.chunks:
            if c.section not in {x[0] for x in sections}:
                sections.append((c.section, c.chunk_id))
        title = source.title

    cases = []
    for section, _ in sections[:limit]:
        topic = re.sub(r"^\d+(\.\d+)*\.?\s+", "", section).strip() or title
        cases.append({"query": f"According to the {title}, what does it say about {topic.lower()}?",
                      "expect": "answer"})
        cases.append({"query": f"According to the {_INVENTED} guideline, what does it say about {topic.lower()}?",
                      "expect": "refuse"})

    started = time.time()
    rows = []
    settings = None
    for case in cases:
        response = pipeline.run(case["query"], settings, client_id="document-eval", review=False)
        cited = sorted({sid for claim in response.claims for sid in claim.source_ids})
        ok = response.decision == case["expect"]
        if case["expect"] == "answer" and ok:
            ok = any(sid in chunk_ids for sid in cited)
        rows.append({**case, "got": response.decision, "cited": cited, "ok": ok,
                     "refused_reason": response.refused_reason})

    result = {
        "ran_at": utcnow().isoformat(),
        "total": len(rows),
        "passed": sum(r["ok"] for r in rows),
        "unsafe_answers": sum(r["expect"] == "refuse" and r["got"] == "answer" for r in rows),
        "seconds": round(time.time() - started, 1),
        "cases": rows,
    }
    with db.session() as s:
        s.get(Source, source_id).evaluation = json.loads(json.dumps(result))
    return result
