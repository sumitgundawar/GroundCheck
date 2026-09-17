"""Knowledge releases: every change to what answers are grounded in is
checked before it goes live, and can be rolled back in one step.

A release is a snapshot of the search index (passages and their vectors)
with a manifest: which approved documents it contains, and fingerprints of
its content and of the formulary. Rebuilding the index, for example after a
document is approved or retired, makes a candidate release. The candidate is
checked against the live index's safety tests while live questions keep
using the current index:

- every must-refuse question in the golden set, and every must-refuse
  patient scenario, when the demo corpus is included
- every test question added from a review

The check fails if any question that must be refused is answered. Answerable
questions that are now refused are reported, but don't fail the check,
because refusing is the safe direction. A candidate that passes goes live
straight away, or waits for someone to promote it when RELEASE_AUTO_PROMOTE
is false. One that fails never goes live, and raises an alert.

Rolling back makes an earlier release live again from its snapshot. The last
RELEASES_KEEP releases are kept, and always the live one and the one before.
Checks run in extractive mode and record nothing in the audit trail."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from . import config, db
from .db import Release

log = logging.getLogger("groundcheck.releases")
_lock = threading.Lock()


class ReleaseError(ValueError):
    """The request can't be carried out. The message is safe to show."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


GOLDEN_PATH = config.ROOT_DIR / "eval" / "golden.json"
PATIENT_PATH = config.ROOT_DIR / "eval" / "patient_cases.json"


def _dir(number: int) -> Path:
    root = Path(config.RELEASES_DIR) if config.RELEASES_DIR else config.INDEX_DIR / "releases"
    return root / f"R{number}"


def _fingerprint(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for r in records:
        digest.update(json.dumps([r["id"], r.get("title"), r.get("text"), r.get("effective_from"),
                                  r.get("expires_on")], ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _formulary_fingerprint() -> str:
    try:
        return hashlib.sha256(Path(config.FORMULARY_PATH).read_bytes()).hexdigest()
    except OSError:
        return ""


def _documents(records: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in records:
        if r.get("source_id") is not None:
            key = r.get("document_key") or f"source-{r['source_id']}"
            seen.setdefault(key, {"key": key, "source_id": int(r["source_id"]), "version": r.get("source_version"),
                                  "title": r.get("document_title") or r.get("title", "")})
    return sorted(seen.values(), key=lambda d: d["title"].lower())


def _summary(row: Release) -> dict:
    return {
        "id": row.id, "name": f"R{row.number}", "number": row.number, "status": row.status, "reason": row.reason,
        "created_at": row.created_at.isoformat(), "created_by": row.created_by_name,
        "passages": row.passages, "demo_passages": row.demo_passages, "documents": row.documents,
        "content_sha256": row.content_sha256, "formulary_sha256": row.formulary_sha256,
        "check": row.check, "live_at": row.live_at.isoformat() if row.live_at else None,
        "live_by": row.live_by_name or None,
        "snapshot": _dir(row.number).is_dir(),
    }


def _live(s) -> Release | None:
    return s.scalar(select(Release).where(Release.status == "live"))


# ---------------------------------------------------------------- snapshots

def _write_snapshot(number: int, records: list[dict], vectors: np.ndarray) -> None:
    target = _dir(number)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    np.save(tmp / "vectors.npy", np.ascontiguousarray(vectors, dtype="float32"))
    shutil.rmtree(target, ignore_errors=True)
    tmp.replace(target)


def _read_snapshot(number: int) -> tuple[list[dict], np.ndarray]:
    folder = _dir(number)
    try:
        records = json.loads((folder / "records.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy")
    except OSError as exc:
        raise ReleaseError(f"The snapshot of R{number} is missing, so it can't be used.") from exc
    return records, vectors


def _prune(s) -> None:
    rows = s.scalars(select(Release).order_by(Release.number.desc())).all()
    live = next((r for r in rows if r.status == "live"), None)
    keep = {r.number for r in rows[:max(2, config.RELEASES_KEEP)]}
    if live:
        keep.add(live.number)
        before = next((r for r in rows if r.number < live.number and r.status in ("retired", "rolled_back")), None)
        if before:
            keep.add(before.number)
    for r in rows:
        if r.number not in keep and _dir(r.number).is_dir():
            shutil.rmtree(_dir(r.number), ignore_errors=True)


# ---------------------------------------------------------------- the check

def _load_cases(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def check(records: list[dict], vectors: np.ndarray) -> dict:
    """Run the safety tests against an index that isn't live."""
    from . import governance, pipeline, retrieval
    from .schemas import PatientContext

    started = time.monotonic()
    state = retrieval.state_from(records, vectors)
    demo = any(r.get("source_id") is None for r in records)
    groups: dict[str, dict] = {}
    unsafe: list[dict] = []
    over_refused: list[str] = []

    def record(group: str, query: str, expect: str, got: str) -> None:
        g = groups.setdefault(group, {"total": 0, "passed": 0, "unsafe": 0, "over_refused": 0})
        g["total"] += 1
        g["passed"] += int(got == expect)
        if expect == "refuse" and got == "answer":
            g["unsafe"] += 1
            if len(unsafe) < 20:
                unsafe.append({"group": group, "query": query})
        elif expect == "answer" and got == "refuse":
            g["over_refused"] += 1
            if len(over_refused) < 10:
                over_refused.append(query)

    with retrieval.using(state):
        if demo:
            golden = _load_cases(GOLDEN_PATH)
            for case in golden:
                if case["expect"] == "refuse" or config.RELEASE_CHECK_ANSWERABLE:
                    record("golden", case["query"], case["expect"], pipeline.run(case["query"], check_only=True).decision)
            for case in _load_cases(PATIENT_PATH):
                if case["expect"] != "refuse":
                    continue
                got = pipeline.run(case["query"], patient=PatientContext(**case["patient"]), check_only=True).decision
                record("patient", case["query"], case["expect"], got)
        if db.ready():
            for case in governance.list_eval_cases():
                record("review_tests", case["query"], case["expect"],
                       pipeline.run(case["query"], check_only=True).decision)
    total_unsafe = sum(g["unsafe"] for g in groups.values())
    return {"passed": total_unsafe == 0, "unsafe": total_unsafe, "groups": groups, "unsafe_examples": unsafe,
            "over_refused_examples": over_refused, "seconds": round(time.monotonic() - started, 1),
            "checked_at": _now().isoformat()}


# ---------------------------------------------------------------- lifecycle

def _go_live(s, row: Release, records: list[dict], vectors: np.ndarray, user_id: int | None, name: str,
             previous_status: str = "retired") -> None:
    from . import retrieval

    current = _live(s)
    retrieval.write_index(records, vectors)
    retrieval.load_index()
    if current is not None and current.id != row.id:
        current.status = previous_status
    row.status, row.live_at, row.live_by, row.live_by_name = "live", _now(), user_id, name[:200]


def create(records: list[dict], vectors: np.ndarray, reason: str, user_id: int | None = None,
           name: str = "GroundCheck") -> dict:
    """Make a candidate from a rebuilt index, check it, and put it live if it
    passes (and promotion is automatic)."""
    with _lock:
        with db.session() as s:
            number = (s.scalar(select(func.max(Release.number))) or 0) + 1
            fingerprint = _fingerprint(records)
            live = _live(s)
            if live is not None and live.content_sha256 == fingerprint and live.formulary_sha256 == _formulary_fingerprint():
                # Nothing changed: no new release, but make sure this instance serves it.
                from . import retrieval

                retrieval.write_index(records, vectors)
                retrieval.load_index()
                return {**_summary(live), "unchanged": True}
            _write_snapshot(number, records, vectors)
            row = Release(number=number, status="checking", reason=reason[:300], created_by=user_id,
                          created_by_name=name[:200], passages=len(records),
                          demo_passages=sum(1 for r in records if r.get("source_id") is None),
                          documents=_documents(records), content_sha256=fingerprint,
                          formulary_sha256=_formulary_fingerprint())
            s.add(row)
            s.flush()
            release_id = row.id
        try:
            result = check(records, vectors) if config.RELEASE_CHECKS else \
                {"passed": True, "skipped": True, "checked_at": _now().isoformat()}
        except Exception as exc:  # noqa: BLE001 - a check that can't run is a failed check
            log.exception("Release check failed to run")
            result = {"passed": False, "error": f"The check couldn't run: {type(exc).__name__}.",
                      "checked_at": _now().isoformat()}
        with db.session() as s:
            row = s.get(Release, release_id)
            row.check = result
            if not result["passed"]:
                row.status = "failed"
            elif config.RELEASE_AUTO_PROMOTE or s.scalar(select(func.count(Release.id)).where(Release.status == "live")) == 0:
                _go_live(s, row, records, vectors, user_id, name)
            else:
                row.status = "ready"
            _prune(s)
            return _summary(row)


def promote(release_id: int, user_id: int | None, name: str) -> dict:
    """Put a checked release live: a candidate waiting for promotion, or an
    earlier release (a rollback)."""
    with _lock, db.session() as s:
        row = s.get(Release, release_id)
        if row is None:
            raise LookupError(release_id)
        if row.status == "live":
            raise ReleaseError(f"R{row.number} is already live.")
        if row.status in ("checking", "failed"):
            raise ReleaseError(f"R{row.number} didn't pass its check, so it can't go live.")
        records, vectors = _read_snapshot(row.number)
        live = _live(s)
        rolling_back = live is not None and row.number < live.number
        _go_live(s, row, records, vectors, user_id, name, previous_status="rolled_back" if rolling_back else "retired")
        return {**_summary(row), "rolled_back": rolling_back}


def rollback(user_id: int | None, name: str) -> dict:
    """One step back: the most recent release before the live one that was live."""
    with db.session() as s:
        live = _live(s)
        if live is None:
            raise ReleaseError("Nothing is live yet.")
        previous = s.scalar(select(Release).where(Release.number < live.number, Release.live_at.is_not(None),
                                                  Release.status.in_(("retired", "rolled_back")))
                            .order_by(Release.number.desc()))
        if previous is None:
            raise ReleaseError("There's no earlier release to go back to.")
        previous_id = previous.id
    return promote(previous_id, user_id, name)


def baseline() -> dict | None:
    """At startup, record the index already in use as the first live release,
    so there's always something to roll back to."""
    from . import retrieval

    with db.session() as s:
        if s.scalar(select(func.count(Release.id))):
            return None
    state = retrieval._state()
    records, vectors = list(state.metadata), state.store.vectors()
    with _lock, db.session() as s:
        if s.scalar(select(func.count(Release.id))):
            return None
        _write_snapshot(1, records, vectors)
        row = Release(number=1, status="live", reason="The index in use when releases began", created_by_name="GroundCheck",
                      passages=len(records), demo_passages=sum(1 for r in records if r.get("source_id") is None),
                      documents=_documents(records), content_sha256=_fingerprint(records),
                      formulary_sha256=_formulary_fingerprint(), check={"passed": True, "skipped": True},
                      live_at=_now(), live_by_name="GroundCheck")
        s.add(row)
        s.flush()
        return _summary(row)


def list_releases() -> dict:
    with db.session() as s:
        rows = s.scalars(select(Release).order_by(Release.number.desc()).limit(50)).all()
        live = _live(s)
        out = [_summary(r) for r in rows]
    live_docs = {d["key"]: d for d in (live.documents if live else [])}
    for r in out:
        docs = {d["key"]: d for d in r["documents"]}
        r["changes"] = {
            "added": [d["title"] for k, d in docs.items() if k not in live_docs],
            "removed": [d["title"] for k, d in live_docs.items() if k not in docs],
            "updated": [d["title"] for k, d in docs.items() if k in live_docs and live_docs[k]["version"] != d["version"]],
        } if live and r["id"] != live.id else None
    return {"releases": out, "auto_promote": config.RELEASE_AUTO_PROMOTE, "checks": config.RELEASE_CHECKS,
            "keep": config.RELEASES_KEEP}
