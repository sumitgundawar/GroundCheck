"""Clinician feedback on answers: a thumbs up or down, and for a thumbs down an
optional reason and comment.

Feedback lives in its own table. The audit record it refers to is signed and
hash-chained and is never touched. One person gives one rating per answer; a
second submission replaces the first. A comment is run through the identifier
removal used on questions before it is stored, and is dropped if that fails.
Marking an answer "wrong" also opens a high-priority review case, once."""

from __future__ import annotations

import logging
import threading

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from . import db, governance, guards_input
from .db import AnswerFeedback, AuditRecord, utcnow

log = logging.getLogger("groundcheck.feedback")
_case_lock = threading.Lock()

RATINGS = ("up", "down")
REASONS = ("not_useful", "incomplete", "wrong", "too_slow")
MAX_COMMENT = 500


class FeedbackError(ValueError):
    """The feedback was refused. The message is safe to show."""


def clean_comment(comment: str | None) -> str:
    """The comment with identifiers replaced by placeholders. Empty if it can't
    be checked: an unchecked comment is never stored."""
    text = (comment or "").strip()
    if not text:
        return ""
    if len(text) > MAX_COMMENT:
        raise FeedbackError(f"Keep the comment under {MAX_COMMENT} characters.")
    try:
        return guards_input.deidentify(text).text
    except Exception:  # noqa: BLE001
        log.exception("Couldn't remove identifiers from a feedback comment; it was not stored")
        return ""


def _out(row: AnswerFeedback) -> dict:
    return {"audit_id": row.audit_id, "rating": row.rating, "reason": row.reason, "comment": row.comment,
            "case_id": row.case_id, "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat()}


def _voter(user_id: int | None) -> str:
    return f"u{user_id}" if user_id is not None else "anon"


def get_mine(audit_id: str, user_id: int | None) -> dict | None:
    with db.session() as s:
        row = s.scalar(select(AnswerFeedback).where(AnswerFeedback.audit_id == audit_id,
                                                    AnswerFeedback.voter == _voter(user_id)))
        return _out(row) if row else None


def submit(audit_id: str, rating: str, reason: str | None, comment: str | None, user_id: int | None,
           site_scope: int | None = None, own_only: bool = False) -> dict:
    """Record or replace this person's feedback on an answer.

    site_scope limits the answers someone can see to their site's; own_only
    limits them to ones they asked themselves."""
    if rating not in RATINGS:
        raise FeedbackError("Rating must be up or down.")
    reason = (reason or "").strip()
    if reason and reason not in REASONS:
        raise FeedbackError("Reason must be one of: " + ", ".join(REASONS) + ".")
    if rating == "up" and reason:
        raise FeedbackError("A reason is only for a thumbs down.")
    text = clean_comment(comment)
    if rating == "up" and text:
        raise FeedbackError("A comment is only for a thumbs down.")

    with db.session() as s:
        record = s.scalar(select(AuditRecord).where(AuditRecord.audit_id == audit_id))
        if (record is None or record.decision == "event"
                or (site_scope is not None and record.site_id != site_scope)
                or (own_only and record.user_id != user_id)):
            raise FeedbackError("That answer isn't in the audit trail.")
        decision, site_id = record.decision, record.site_id

    now = utcnow()
    voter = _voter(user_id)
    for attempt in range(2):
        try:
            with db.session() as s:
                row = s.scalar(select(AnswerFeedback).where(AnswerFeedback.audit_id == audit_id,
                                                            AnswerFeedback.voter == voter))
                if row is None:
                    row = AnswerFeedback(audit_id=audit_id, voter=voter, user_id=user_id, site_id=site_id,
                                         rating=rating, reason=reason, comment=text, created_at=now, updated_at=now)
                    s.add(row)
                    s.flush()
                else:
                    row.rating, row.reason, row.comment, row.updated_at = rating, reason, text, now
                needs_case = reason == "wrong" and decision == "answer" and row.case_id is None
                out = _out(row)
                row_id = row.id
            break
        except IntegrityError:
            if attempt:  # two submissions raced; the second pass updates the first
                raise
    if needs_case:
        # Under a lock, and re-reading the row, so submissions that race open one case between them.
        with _case_lock:
            with db.session() as s:
                existing = s.get(AnswerFeedback, row_id).case_id
            if existing is not None:
                out["case_id"] = existing
                return out
            note = "A clinician marked this answer as wrong." + (f" {text}" if text else "")
            try:
                case = governance.flag_answer(audit_id, note, user_id, site_scope)
            except governance.GovernanceError:
                return out
            with db.session() as s:
                s.get(AnswerFeedback, row_id).case_id = case["id"]
            out["case_id"] = case["id"]
    return out
