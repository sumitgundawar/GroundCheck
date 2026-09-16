"""Retention: deleting records once they're older than an organisation keeps them.

- Audit records older than AUDIT_RETENTION_DAYS are deleted, oldest first and
  in chain order. The chain's anchor moves to the last one deleted, so the
  records that remain still verify (app/integrity.py).
- Resolved review cases older than REVIEW_RETENTION_DAYS are deleted with
  their timelines. Open cases are never deleted. Tests added from a case stay.
- Expired sign-in sessions are deleted.

0 days, the default, keeps records for ever. Retention periods for clinical
records are set by law and local policy (often eight years or more), so set
them with your information governance lead. Nothing is deleted until
retention is run, from the dashboard or `python -m app.cli retention --apply`,
and every run is recorded.

The JSONL file used when the database is unavailable isn't covered; rotate it
with the system's log rotation."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, func, select, update

from . import auth, config, db, integrity
from .db import AuditChain, AuditRecord, RetentionRun, ReviewCase


def _cutoffs(now: datetime) -> tuple[datetime | None, datetime | None]:
    audit = now - timedelta(days=config.AUDIT_RETENTION_DAYS) if config.AUDIT_RETENTION_DAYS > 0 else None
    review = now - timedelta(days=config.REVIEW_RETENTION_DAYS) if config.REVIEW_RETENTION_DAYS > 0 else None
    return audit, review


def _audit_boundary(s, cutoff: datetime) -> int | None:
    """The highest chain number among records older than the cutoff. Everything
    up to it is deleted, so the remaining chain has no holes."""
    return s.scalar(select(func.max(AuditRecord.seq)).where(AuditRecord.created_at < cutoff))


def plan(now: datetime | None = None) -> dict:
    """What running retention now would delete, without deleting anything."""
    now = now or db.utcnow()
    audit_cutoff, review_cutoff = _cutoffs(now)
    with db.session() as s:
        audit_count = 0
        if audit_cutoff is not None:
            boundary = _audit_boundary(s, audit_cutoff)
            condition = AuditRecord.created_at < audit_cutoff
            if boundary is not None:
                condition = (AuditRecord.seq <= boundary) | (AuditRecord.seq.is_(None) & condition)
            else:
                condition = AuditRecord.seq.is_(None) & condition
            audit_count = s.scalar(select(func.count(AuditRecord.id)).where(condition)) or 0
        review_count = 0
        if review_cutoff is not None:
            review_count = s.scalar(select(func.count(ReviewCase.id)).where(
                ReviewCase.status != "open", ReviewCase.resolved_at < review_cutoff)) or 0
        last = s.scalars(select(RetentionRun).order_by(RetentionRun.id.desc()).limit(1)).first()
        return {
            "audit_retention_days": config.AUDIT_RETENTION_DAYS,
            "review_retention_days": config.REVIEW_RETENTION_DAYS,
            "audit_cutoff": audit_cutoff.isoformat() if audit_cutoff else None,
            "review_cutoff": review_cutoff.isoformat() if review_cutoff else None,
            "audit_records_to_delete": audit_count,
            "review_cases_to_delete": review_count,
            "last_run": _run_summary(last) if last else None,
        }


def _run_summary(run: RetentionRun) -> dict:
    return {"ran_at": run.ran_at.isoformat(), "ran_by": run.ran_by, "audit_deleted": run.audit_deleted,
            "reviews_deleted": run.reviews_deleted, "sessions_deleted": run.sessions_deleted,
            "anchor_seq": run.anchor_seq}


def apply(user_id: int | None = None, now: datetime | None = None) -> dict:
    now = now or db.utcnow()
    audit_cutoff, review_cutoff = _cutoffs(now)
    audit_deleted = reviews_deleted = 0
    with integrity.write_lock, db.session() as s:
        # Lock the chain head first, as appends do, so nothing is chained
        # between choosing the boundary and moving the anchor.
        s.execute(update(AuditChain).where(AuditChain.id == 1).values(updated_at=now))
        chain = s.get(AuditChain, 1)
        if audit_cutoff is not None:
            boundary = _audit_boundary(s, audit_cutoff)
            if boundary is not None and boundary > chain.anchor_seq:
                last = s.scalar(select(AuditRecord).where(AuditRecord.seq == boundary))
                chain.anchor_seq, chain.anchor_hash = boundary, last.entry_hash
                audit_deleted += s.execute(delete(AuditRecord).where(AuditRecord.seq <= boundary)).rowcount or 0
            audit_deleted += s.execute(delete(AuditRecord).where(
                AuditRecord.seq.is_(None), AuditRecord.created_at < audit_cutoff)).rowcount or 0
        if review_cutoff is not None:
            cases = s.scalars(select(ReviewCase).where(
                ReviewCase.status != "open", ReviewCase.resolved_at < review_cutoff)).all()
            for case in cases:
                s.delete(case)  # its events go with it
            reviews_deleted = len(cases)
        run = RetentionRun(ran_at=now, ran_by=user_id, audit_cutoff=audit_cutoff, audit_deleted=audit_deleted,
                           anchor_seq=chain.anchor_seq, review_cutoff=review_cutoff,
                           reviews_deleted=reviews_deleted)
        s.add(run)
    sessions_deleted = auth.purge_expired_sessions()
    with db.session() as s:
        s.execute(update(RetentionRun).where(RetentionRun.id == run.id).values(sessions_deleted=sessions_deleted))
    run.sessions_deleted = sessions_deleted
    return _run_summary(run)
