"""Clinical review and governance.

- Review queue. Every refusal opens a review case; asking the same question
  again while it's open adds to that case instead of opening another.
  Clinicians can flag an answer that looks wrong, which opens a high-priority
  case. Cases are assigned, have a due date, escalate when overdue, and keep a
  timeline of everything done to them.
- Closing the loop. Resolving a case records the outcome. "Add as a test"
  turns the question into a permanent evaluation case with the decision the
  reviewer says is correct, so the same mistake is caught from then on.
- Hazard log. What could go wrong, scored by severity and likelihood before
  and after its controls, in the shape clinical safety standards (DCB0129,
  ISO 14971) expect.
- Reports and a safety case document for governance meetings and releases.

Nothing here decides an answer. It records and routes what the pipeline did."""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from . import __version__, config, db, encryption
from .db import AuditRecord, EvalCase, Hazard, ReviewCase, ReviewEvent, Source, User, utcnow

OUTCOMES = {
    "add_test": "Added as a permanent test question",
    "add_source": "The sources need a new or updated document",
    "correct_refusal": "The refusal was correct",
    "guard_issue": "A check behaved wrongly and needs fixing",
    "no_action": "No action needed",
}
RISK_LEVELS = ((1, 4, "low"), (5, 9, "medium"), (10, 16, "high"), (17, 25, "very high"))


class GovernanceError(ValueError):
    """A review or governance operation was refused. The message is safe to show."""


# --- Helpers ---------------------------------------------------------------

def _key(query: str) -> str:
    normalised = re.sub(r"\s+", " ", query.strip().lower())
    return encryption.keyring().lookup_hash(normalised)


def categorise(reason: str) -> str:
    """A short, stable category for a refusal reason, for grouping in reports."""
    r = (reason or "").lower()
    rules = (
        ("in the question does not appear", "stated dose not in sources"),
        ("no trusted source covers", "patient group not covered"),
        ("does not appear in any trusted source", "term not in sources"),
        ("states a maximum", "dose limit not stated"),
        ("states a minimum", "dose limit not stated"),
        ("must not be combined", "combination not allowed"),
        ("no sufficiently relevant source", "no relevant source"),
        ("could not be grounded", "claim not grounded"),
        ("not supported by any source", "dose not in sources"),
        ("enough information", "not enough information"),
        ("input guard", "blocked input"),
        ("rate limit", "rate limited"),
    )
    for needle, category in rules:
        if needle in r:
            return category
    return "other"


def _user_names(s, ids) -> dict[int, str]:
    ids = {i for i in ids if i}
    if not ids:
        return {}
    return {u.id: (u.name or u.email) for u in s.scalars(select(User).where(User.id.in_(ids)))}


def _event(s, case: ReviewCase, action: str, user_id: int | None = None, note: str = "") -> None:
    s.add(ReviewEvent(case_id=case.id, action=action, user_id=user_id, note=note.strip()[:4000]))


def _overdue(case: ReviewCase, now: datetime) -> bool:
    return case.status == "open" and case.due_at <= now


def _case_summary(case: ReviewCase, names: dict[int, str], now: datetime) -> dict:
    return {
        "id": case.id,
        "kind": case.kind,
        "status": case.status,
        "priority": case.priority,
        "query": case.query,
        "reason": case.reason,
        "reason_category": case.reason_category,
        "occurrences": case.occurrences,
        "first_audit_id": case.first_audit_id,
        "last_audit_id": case.last_audit_id,
        "created_at": case.created_at.isoformat(),
        "last_seen_at": case.last_seen_at.isoformat(),
        "due_at": case.due_at.isoformat(),
        "overdue": _overdue(case, now),
        "escalated": case.escalated_at is not None,
        "assigned_to": case.assigned_to,
        "assigned_to_name": names.get(case.assigned_to),
        "flagged_by_name": names.get(case.flagged_by),
        "resolved_at": case.resolved_at.isoformat() if case.resolved_at else None,
        "resolved_by_name": names.get(case.resolved_by),
        "outcome": case.outcome,
        "outcome_label": OUTCOMES.get(case.outcome, ""),
        "outcome_note": case.outcome_note,
    }


# --- Creating cases ------------------------------------------------------------

def record_refusal(audit_id: str, query: str, reason: str, site_id: int | None = None) -> None:
    """Open a review case for a refusal, or add to the open case for the same
    question. Never raises: a failure here must not affect the answer."""
    if not config.REVIEW_QUEUE or not db.ready() or not query.strip():
        return
    category = categorise(reason)
    if category in ("rate limited",):
        return
    try:
        key = _key(query)
        now = utcnow()
        with db.session() as s:
            case = s.scalar(select(ReviewCase).where(
                ReviewCase.query_key == key, ReviewCase.kind == "refusal", ReviewCase.status == "open",
                ReviewCase.site_id.is_(None) if site_id is None else ReviewCase.site_id == site_id))
            if case is not None:
                case.occurrences += 1
                case.last_seen_at = now
                case.last_audit_id = audit_id
                return
            case = ReviewCase(
                kind="refusal", priority="normal", query=query.strip()[:2000], query_key=key,
                reason=reason or "", reason_category=category, first_audit_id=audit_id,
                last_audit_id=audit_id, created_at=now, last_seen_at=now,
                due_at=now + timedelta(hours=config.REVIEW_SLA_HOURS), outcome_note="", site_id=site_id,
            )
            s.add(case)
            s.flush()
            _event(s, case, "opened", note=f"Refused: {reason}")
    except Exception:  # noqa: BLE001
        import logging

        logging.getLogger("groundcheck.governance").exception("Couldn't record a review case for %s", audit_id)


def flag_answer(audit_id: str, note: str, user_id: int | None, site_scope: int | None = None) -> dict:
    """Flag an answer as possibly wrong. Opens a high-priority case."""
    note = (note or "").strip()
    if len(note) < 5:
        raise GovernanceError("Say briefly what looks wrong, so a reviewer can check it.")
    with db.session() as s:
        record = s.scalar(select(AuditRecord).where(AuditRecord.audit_id == audit_id))
        if record is None or (site_scope is not None and record.site_id != site_scope):
            raise GovernanceError("That answer isn't in the audit trail.")
        if record.decision != "answer":
            raise GovernanceError("Only answers can be flagged. Refusals are reviewed automatically.")
        now = utcnow()
        case = ReviewCase(
            kind="flagged", priority="high", query=record.query, query_key=_key(record.query),
            reason=note[:2000], reason_category="flagged answer", first_audit_id=audit_id,
            last_audit_id=audit_id, created_at=now, last_seen_at=now,
            due_at=now + timedelta(hours=config.FLAGGED_SLA_HOURS), flagged_by=user_id, outcome_note="",
            site_id=record.site_id,
        )
        s.add(case)
        s.flush()
        _event(s, case, "flagged", user_id, note)
        return _case_summary(case, _user_names(s, [user_id]), now)


# --- Working the queue ---------------------------------------------------------

def escalate_overdue(now: datetime | None = None) -> int:
    """Mark open cases past their due date as escalated and high priority."""
    now = now or utcnow()
    with db.session() as s:
        cases = s.scalars(select(ReviewCase).where(
            ReviewCase.status == "open", ReviewCase.due_at <= now, ReviewCase.escalated_at.is_(None))).all()
        for case in cases:
            case.escalated_at = now
            case.priority = "high"
            _event(s, case, "escalated", note="Past its due date.")
        return len(cases)


def _scoped(q, site_id: int | None):
    return q if site_id is None else q.where(ReviewCase.site_id == site_id)


def _case(s, case_id: int, site_id: int | None) -> ReviewCase:
    case = s.get(ReviewCase, case_id)
    if case is None or (site_id is not None and case.site_id != site_id):
        raise GovernanceError("No such case.")
    return case


def list_cases(status: str = "open", assigned_to: int | None = None, overdue_only: bool = False,
               limit: int = 200, site_id: int | None = None) -> dict:
    escalate_overdue()
    now = utcnow()
    with db.session() as s:
        q = _scoped(select(ReviewCase), site_id)
        if status != "all":
            q = q.where(ReviewCase.status == status)
        if assigned_to is not None:
            q = q.where(ReviewCase.assigned_to == assigned_to)
        if overdue_only:
            q = q.where(ReviewCase.status == "open", ReviewCase.due_at <= now)
        # High priority first, then the oldest due date.
        q = q.order_by((ReviewCase.priority == "high").desc(), ReviewCase.due_at).limit(limit)
        cases = s.scalars(q).all()
        names = _user_names(s, [c.assigned_to for c in cases] + [c.flagged_by for c in cases]
                            + [c.resolved_by for c in cases])
        counts = dict(s.execute(_scoped(select(ReviewCase.status, func.count(ReviewCase.id)), site_id)
                                .group_by(ReviewCase.status)).all())
        overdue = s.scalar(_scoped(select(func.count(ReviewCase.id)), site_id).where(
            ReviewCase.status == "open", ReviewCase.due_at <= now)) or 0
        return {
            "cases": [_case_summary(c, names, now) for c in cases],
            "counts": {"open": counts.get("open", 0), "resolved": counts.get("resolved", 0),
                       "dismissed": counts.get("dismissed", 0), "overdue": overdue},
            "outcomes": OUTCOMES,
        }


def get_case(case_id: int, site_id: int | None = None) -> dict:
    now = utcnow()
    with db.session() as s:
        case = _case(s, case_id, site_id)
        names = _user_names(s, [case.assigned_to, case.flagged_by, case.resolved_by]
                            + [e.user_id for e in case.events])
        summary = _case_summary(case, names, now)
        summary["events"] = [{"at": e.at.isoformat(), "action": e.action, "user": names.get(e.user_id),
                              "note": e.note} for e in case.events]
        record = s.scalar(select(AuditRecord).where(AuditRecord.audit_id == case.last_audit_id))
        if record is not None:
            response = record.record.get("response", {})
            summary["answer_text"] = response.get("answer_text")
            summary["sources"] = [{"id": x.get("id"), "title": x.get("title"), "score": x.get("score")}
                                  for x in response.get("sources", [])]
        return summary


def assign(case_id: int, assignee_id: int | None, acting_user_id: int | None, site_id: int | None = None) -> dict:
    with db.session() as s:
        case = _case(s, case_id, site_id)
        if case.status != "open":
            raise GovernanceError("Only open cases can be assigned.")
        if assignee_id is not None:
            user = s.get(User, assignee_id)
            if user is None or not user.is_active or user.role == "clinician" or \
                    (user.site_id is not None and user.site_id != case.site_id):
                raise GovernanceError("Assign cases to an active reviewer or admin at this site.")
        case.assigned_to = assignee_id
        names = _user_names(s, [assignee_id])
        _event(s, case, "assigned", acting_user_id, f"Assigned to {names.get(assignee_id, 'nobody')}.")
    return get_case(case_id, site_id)


def comment(case_id: int, note: str, user_id: int | None, site_id: int | None = None) -> dict:
    if not (note or "").strip():
        raise GovernanceError("Write a comment first.")
    with db.session() as s:
        case = _case(s, case_id, site_id)
        _event(s, case, "commented", user_id, note)
    return get_case(case_id, site_id)


def resolve(case_id: int, outcome: str, note: str, user_id: int | None,
            expected_decision: str | None = None, site_id: int | None = None) -> dict:
    """Close a case with an outcome. "add_test" also adds the question to the
    evaluation, with the decision the reviewer says is correct."""
    if outcome not in OUTCOMES:
        raise GovernanceError("Choose an outcome.")
    if outcome == "add_test" and expected_decision not in ("answer", "refuse"):
        raise GovernanceError("Say whether the correct decision is to answer or to refuse.")
    with db.session() as s:
        case = _case(s, case_id, site_id)
        if case.status != "open":
            raise GovernanceError("This case is already closed.")
        now = utcnow()
        case.status = "resolved"
        case.outcome = outcome
        case.outcome_note = (note or "").strip()[:4000]
        case.resolved_at = now
        case.resolved_by = user_id
        _event(s, case, "resolved", user_id, f"{OUTCOMES[outcome]}. {case.outcome_note}".strip())
        if outcome == "add_test":
            s.add(EvalCase(query=case.query, expect=expected_decision, note=case.outcome_note,
                           from_case_id=case.id, created_by=user_id))
    return get_case(case_id, site_id)


def reopen(case_id: int, note: str, user_id: int | None, site_id: int | None = None) -> dict:
    with db.session() as s:
        case = _case(s, case_id, site_id)
        if case.status == "open":
            raise GovernanceError("This case is already open.")
        case.status = "open"
        case.resolved_at = None
        case.resolved_by = None
        case.due_at = utcnow() + timedelta(hours=config.REVIEW_SLA_HOURS)
        case.escalated_at = None
        _event(s, case, "reopened", user_id, note)
    return get_case(case_id, site_id)


# --- Evaluation cases from reviews -------------------------------------------

def list_eval_cases() -> list[dict]:
    with db.session() as s:
        return [{"id": c.id, "query": c.query, "expect": c.expect, "note": c.note,
                 "from_case_id": c.from_case_id, "created_at": c.created_at.isoformat()}
                for c in s.scalars(select(EvalCase).order_by(EvalCase.id))]


def run_eval_cases() -> dict:
    """Run every evaluation case added from reviews through the pipeline."""
    from . import pipeline

    rows = []
    for case in list_eval_cases():
        got = pipeline.run(case["query"], client_id="review-eval", review=False).decision
        rows.append({**case, "got": got, "ok": got == case["expect"]})
    return {
        "total": len(rows),
        "passed": sum(r["ok"] for r in rows),
        "unsafe_answers": sum(r["expect"] == "refuse" and r["got"] == "answer" for r in rows),
        "cases": rows,
    }


# --- Hazard log --------------------------------------------------------------

def risk(severity: int, likelihood: int) -> dict:
    score = severity * likelihood
    level = next(name for low, high, name in RISK_LEVELS if low <= score <= high)
    return {"score": score, "level": level}


def _check_scale(name: str, value) -> int:
    if not isinstance(value, int) or not 1 <= value <= 5:
        raise GovernanceError(f"{name} must be a whole number from 1 to 5.")
    return value


def _hazard_summary(h: Hazard) -> dict:
    return {
        "id": h.id, "title": h.title, "cause": h.cause, "effect": h.effect,
        "severity": h.severity, "likelihood": h.likelihood, "initial_risk": risk(h.severity, h.likelihood),
        "controls": h.controls, "residual_severity": h.residual_severity,
        "residual_likelihood": h.residual_likelihood,
        "residual_risk": risk(h.residual_severity, h.residual_likelihood),
        "status": h.status, "owner": h.owner, "related_case_id": h.related_case_id,
        "created_at": h.created_at.isoformat(), "updated_at": h.updated_at.isoformat(),
    }


def save_hazard(data: dict, user_id: int | None, hazard_id: int | None = None) -> dict:
    title = (data.get("title") or "").strip()
    if not title:
        raise GovernanceError("Give the hazard a title.")
    fields = {
        "title": title[:200],
        "cause": (data.get("cause") or "").strip(),
        "effect": (data.get("effect") or "").strip(),
        "severity": _check_scale("Severity", data.get("severity")),
        "likelihood": _check_scale("Likelihood", data.get("likelihood")),
        "controls": (data.get("controls") or "").strip(),
        "residual_severity": _check_scale("Residual severity", data.get("residual_severity")),
        "residual_likelihood": _check_scale("Residual likelihood", data.get("residual_likelihood")),
        "status": data.get("status") or "open",
        "owner": (data.get("owner") or "").strip()[:200],
        "related_case_id": data.get("related_case_id"),
    }
    if fields["status"] not in ("open", "mitigated", "closed"):
        raise GovernanceError("Status must be open, mitigated or closed.")
    if fields["residual_severity"] * fields["residual_likelihood"] > fields["severity"] * fields["likelihood"]:
        raise GovernanceError("Controls can't make the residual risk higher than the initial risk.")
    with db.session() as s:
        if hazard_id is None:
            hazard = Hazard(created_by=user_id, **fields)
            s.add(hazard)
        else:
            hazard = s.get(Hazard, hazard_id)
            if hazard is None:
                raise GovernanceError("No such hazard.")
            for k, v in fields.items():
                setattr(hazard, k, v)
            hazard.updated_at = utcnow()
        s.flush()
        return _hazard_summary(hazard)


def list_hazards() -> list[dict]:
    with db.session() as s:
        hazards = s.scalars(select(Hazard)).all()
        summaries = [_hazard_summary(h) for h in hazards]
    return sorted(summaries, key=lambda h: (-h["residual_risk"]["score"], h["id"]))


# --- Reports -----------------------------------------------------------------

def report(days: int = 30, site_id: int | None = None) -> dict:
    """Usage, refusals, overrides, review performance and documents for a period."""
    if not 1 <= days <= 3660:
        raise GovernanceError("Choose a period of 1 to 3,660 days.")
    now = utcnow()
    since = now - timedelta(days=days)
    audit_q = select(AuditRecord).where(AuditRecord.created_at >= since)
    if site_id is not None:
        audit_q = audit_q.where(AuditRecord.site_id == site_id)
    with db.session() as s:
        records = [r for r in s.scalars(audit_q) if not r.record.get("test_run")]
        answered = sum(r.decision == "answer" for r in records)
        refused = len(records) - answered
        reasons: dict[str, int] = {}
        overrides = 0
        model_drafted = sum(bool(r.llm_used) for r in records)
        for r in records:
            if r.decision == "refuse":
                category = categorise(r.record.get("response", {}).get("refused_reason") or "")
                reasons[category] = reasons.get(category, 0) + 1
            settings = r.record.get("settings") or {}
            if any(settings.get(k) is False for k in ("enable_pii_redaction", "enable_injection_guard",
                                                      "enable_coverage_guard", "enable_grounding_guard",
                                                      "enable_dosage_guard")):
                overrides += 1

        cases = s.scalars(_scoped(select(ReviewCase), site_id).where(ReviewCase.created_at >= since)).all()
        resolved = [c for c in cases if c.resolved_at]
        hours = [(c.resolved_at - c.created_at).total_seconds() / 3600 for c in resolved]
        on_time = sum(c.resolved_at <= c.due_at for c in resolved)
        outcomes: dict[str, int] = {}
        for c in resolved:
            outcomes[c.outcome] = outcomes.get(c.outcome, 0) + 1

        approved_docs = s.scalar(select(func.count(Source.id)).where(
            Source.status == "approved", Source.reviewed_at >= since)) or 0
        retired_docs = s.scalar(select(func.count(Source.id)).where(
            Source.status == "retired", Source.reviewed_at >= since)) or 0

        return {
            "period_days": days,
            "from": since.isoformat(),
            "to": now.isoformat(),
            "questions": {
                "total": len(records), "answered": answered, "refused": refused,
                "refusal_rate": round(refused / len(records), 3) if records else None,
                "refusal_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
                "drafted_by_a_model": model_drafted,
                "with_a_check_switched_off": overrides,
            },
            "reviews": {
                "opened": len(cases),
                "flagged_answers": sum(c.kind == "flagged" for c in cases),
                "resolved": len(resolved),
                "resolved_on_time": on_time,
                "escalated": sum(c.escalated_at is not None for c in cases),
                "median_hours_to_resolve": round(statistics.median(hours), 1) if hours else None,
                "outcomes": outcomes,
                "open_now": s.scalar(_scoped(select(func.count(ReviewCase.id)), site_id)
                                     .where(ReviewCase.status == "open")) or 0,
                "overdue_now": s.scalar(_scoped(select(func.count(ReviewCase.id)), site_id).where(
                    ReviewCase.status == "open", ReviewCase.due_at <= now)) or 0,
            },
            "documents": {"approved": approved_docs, "retired": retired_docs},
        }


def usage(days: int = 30, site_id: int | None = None) -> dict:
    """Day-by-day use for the usage dashboard: questions answered and refused,
    response times, what drafted answers, the most-cited sources, who asks, and
    how often identifiers were removed. Test runs are left out."""
    if not 1 <= days <= 366:
        raise GovernanceError("Choose a period of 1 to 366 days.")
    now = utcnow()
    today = now.date()
    start = today - timedelta(days=days - 1)
    since = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    daily = {start + timedelta(days=i): {"answered": 0, "refused": 0} for i in range(days)}
    latencies: list[int] = []
    drafted = {"Extractive": 0, "Local model": 0, "Cloud model": 0}
    cited: dict[str, dict] = {}
    by_user: dict[int, int] = {}
    deidentified = 0
    reasons: dict[str, int] = {}
    with db.session() as s:
        usage_q = select(AuditRecord).where(AuditRecord.created_at >= since)
        if site_id is not None:
            usage_q = usage_q.where(AuditRecord.site_id == site_id)
        records = [r for r in s.scalars(usage_q) if not r.record.get("test_run")]
        for r in records:
            day = daily.get(r.created_at.date())
            if day is not None:
                day["answered" if r.decision == "answer" else "refused"] += 1
            latencies.append(r.total_ms)
            provider = r.record.get("provider") or {}
            kind = provider.get("kind") if r.llm_used else None
            drafted["Local model" if kind == "local" else "Cloud model" if r.llm_used else "Extractive"] += 1
            response = r.record.get("response", {})
            if r.decision == "answer":
                titles = {x.get("id"): x.get("title") for x in response.get("sources", [])}
                for claim in response.get("claims", []):
                    for sid in set(claim.get("source_ids", [])):
                        entry = cited.setdefault(sid, {"id": sid, "title": titles.get(sid) or sid, "citations": 0})
                        entry["citations"] += 1
            else:
                category = categorise(response.get("refused_reason") or "")
                reasons[category] = reasons.get(category, 0) + 1
            if r.user_id:
                by_user[r.user_id] = by_user.get(r.user_id, 0) + 1
            if any(step.get("name") == "pii redaction" and step.get("status") == "warn"
                   for step in response.get("trace", [])):
                deidentified += 1
        names = _user_names(s, by_user.keys())
        open_cases = s.scalar(_scoped(select(func.count(ReviewCase.id)), site_id)
                              .where(ReviewCase.status == "open")) or 0
        overdue = s.scalar(_scoped(select(func.count(ReviewCase.id)), site_id).where(
            ReviewCase.status == "open", ReviewCase.due_at <= now)) or 0

    latencies.sort()

    def percentile(p: float) -> int | None:
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))]

    total = len(records)
    refused = sum(d["refused"] for d in daily.values())
    return {
        "period_days": days,
        "from": start.isoformat(),
        "to": today.isoformat(),
        "totals": {
            "questions": total,
            "answered": total - refused,
            "refused": refused,
            "refusal_rate": round(refused / total, 3) if total else None,
            "median_ms": percentile(0.5),
            "p95_ms": percentile(0.95),
            "deidentified": deidentified,
            "open_reviews": open_cases,
            "overdue_reviews": overdue,
        },
        "daily": [{"date": d.isoformat(), **v} for d, v in daily.items()],
        "drafted_by": drafted,
        "refusal_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "top_sources": sorted(cited.values(), key=lambda e: -e["citations"])[:8],
        "by_user": sorted(({"name": names.get(uid, f"Account {uid}"), "questions": n} for uid, n in by_user.items()),
                          key=lambda e: -e["questions"])[:8],
    }


def surveillance_markdown(days: int = 90, site_id: int | None = None) -> str:
    """A post-market surveillance report for the period: what people asked, what
    was refused and why, what reviewers found, incidents and their harm,
    operational alerts, and which knowledge releases went live. Regulators and
    quality systems expect this kind of summary at regular intervals; it
    gathers the evidence, and a qualified person signs it off."""
    from . import incidents as incidents_module
    from .db import Alert, Incident, Release

    r = report(days, site_id)
    use = usage(min(days, 366), site_id)
    now = utcnow()
    since = now - timedelta(days=days)
    with db.session() as s:
        incident_rows = s.scalars(select(Incident).where(Incident.reported_at >= since, *(
            [Incident.site_id == site_id] if site_id is not None else []))).all()
        by_harm = {k: sum(i.harm == k for i in incident_rows) for k in incidents_module.HARM}
        by_category = {}
        for i in incident_rows:
            by_category[i.category] = by_category.get(i.category, 0) + 1
        open_incidents = [i for i in incident_rows if i.status != "closed"]
        reportable = [i for i in incident_rows if i.category == "data_protection" or i.harm in ("severe", "death")]
        alerts = s.scalars(select(Alert).where(Alert.first_seen >= since).order_by(Alert.first_seen)).all()
        releases = s.scalars(select(Release).where(Release.created_at >= since).order_by(Release.number)).all()
        alert_counts: dict[str, int] = {}
        for a in alerts:
            alert_counts[a.rule] = alert_counts.get(a.rule, 0) + 1
    q, rv = r["questions"], r["reviews"]
    lines = [
        "# GroundCheckHealth post-market surveillance report",
        "",
        f"Version {__version__}. Generated {now:%Y-%m-%d %H:%M} UTC. Period: {since:%Y-%m-%d} to {now:%Y-%m-%d} "
        f"({days} days).",
        "",
        "This report gathers what the software did in use. It supports, and does not replace, the surveillance",
        "your quality system requires. GroundCheckHealth is not a certified medical device.",
        "",
        "## Use",
        "",
        f"- Questions: {q['total']} ({q['answered']} answered, {q['refused']} refused"
        + (f", {q['refusal_rate']:.0%} refused" if q["refusal_rate"] is not None else "") + ")",
        f"- Drafted by a model: {q['drafted_by_a_model']}; asked with a check switched off: {q['with_a_check_switched_off']}",
        f"- Median response time: {use['totals'].get('median_ms', '–')} ms; 95th percentile: {use['totals'].get('p95_ms', '–')} ms",
        f"- Questions with identifiers removed before storage: {use['totals'].get('deidentified', 0)}",
        "",
        "## Why answers were refused",
        "",
    ]
    if q["refusal_reasons"]:
        lines += ["| Reason | Refusals |", "| --- | --- |"]
        lines += [f"| {reason} | {count} |" for reason, count in q["refusal_reasons"].items()]
    else:
        lines.append("No refusals in this period.")
    lines += [
        "",
        "## Clinical review",
        "",
        f"- Cases opened: {rv['opened']} ({rv['flagged_answers']} flagged answers); resolved: {rv['resolved']}, "
        f"on time: {rv['resolved_on_time']}",
        f"- Escalated: {rv['escalated']}; open now: {rv['open_now']}; overdue now: {rv['overdue_now']}",
        f"- Median time to resolve: {rv['median_hours_to_resolve'] if rv['median_hours_to_resolve'] is not None else '–'} hours",
        "- Outcomes: " + (", ".join(f"{OUTCOMES.get(k, k)}: {v}" for k, v in rv["outcomes"].items()) or "none"),
        "",
        "## Incidents",
        "",
        f"- Reported: {len(incident_rows)}; still open: {len(open_incidents)}",
        "- Harm: " + (", ".join(f"{incidents_module.HARM[k]}: {v}" for k, v in by_harm.items() if v) or "none reported"),
        "- Concerning: " + (", ".join(f"{incidents_module.CATEGORIES[k]}: {v}" for k, v in by_category.items()) or "none"),
        f"- Needing a decision on reporting to a regulator: {len(reportable)}",
        "",
    ]
    if incident_rows:
        lines += ["| Reference | Concerns | Harm | Status | External report |", "| --- | --- | --- | --- | --- |"]
        for i in incident_rows:
            lines.append("| " + " | ".join([
                incidents_module.reference(i.id, i.reported_at), incidents_module.CATEGORIES[i.category],
                incidents_module.HARM[i.harm], i.status,
                (i.external_reference or "none recorded").replace("|", "/")[:80],
            ]) + " |")
        lines.append("")
    lines += [
        "## Operational alerts",
        "",
        ("- " + "\n- ".join(f"{rule}: {count}" for rule, count in sorted(alert_counts.items()))) if alert_counts
        else "No alerts fired in this period.",
        "",
        "## Knowledge releases",
        "",
    ]
    if releases:
        lines += ["| Release | Reason | Passages | Check | Status |", "| --- | --- | --- | --- | --- |"]
        for rel in releases:
            check = rel.check or {}
            verdict = "not checked" if check.get("skipped") else ("passed" if check.get("passed") else
                                                                 f"blocked, {check.get('unsafe', 0)} unsafe")
            lines.append(f"| R{rel.number} | {rel.reason.replace('|', '/')} | {rel.passages} | {verdict} | {rel.status} |")
    else:
        lines.append("No releases in this period.")
    lines += [
        "",
        "## Actions for this period",
        "",
        "- [ ] Every incident reviewed, and reportable ones decided on",
        "- [ ] Refusal reasons reviewed for documents that need writing or updating",
        "- [ ] Overdue review cases cleared",
        "- [ ] Hazard log reviewed against what happened",
        "- [ ] Thresholds and the evaluation re-checked if use has changed",
        "",
        "## Sign-off",
        "",
        "Quality or clinical safety lead: ____________________  Date: __________",
        "",
    ]
    return "\n".join(lines)


def safety_case_markdown(days: int = 30, site_id: int | None = None) -> str:
    """A clinical safety case summary for this release, as Markdown."""
    r = report(days, site_id)
    hazards = list_hazards()
    try:
        evaluation = json.loads(config.EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        evaluation = None
    review_tests = run_eval_cases() if db.ready() else None
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # A document shaped like a deliverable gets filed like one. If the evidence
    # underneath it is invented, that has to be the first thing on the page and
    # it has to be impossible to read past — not a line near the bottom.
    from . import formulary

    synthetic = formulary.load().synthetic
    lines = [
        f"# GroundCheckHealth clinical safety case summary",
        "",
        f"Version {__version__}. Generated {now}. Period: last {days} days.",
        "",
    ]
    if synthetic:
        lines += [
            "> ## THIS IS NOT A SAFETY CASE",
            ">",
            "> Every figure below was produced against a **synthetic demonstration corpus and",
            "> formulary**: the medicines, conditions and doses in it are invented and do not",
            "> describe any real medicine. Nothing here is evidence about the safety of this",
            "> software on real clinical content, and no part of it should be quoted, filed or",
            "> signed as though it were.",
            ">",
            "> To produce a document that means anything, point FORMULARY_PATH at your own",
            "> licensed formulary, load your own approved documents, and re-run it.",
            "",
        ]
    lines += [
        "This summary supports, and does not replace, a clinical safety case signed off by a",
        "qualified clinical safety officer. GroundCheckHealth is not a certified medical device.",
        "",
        "## Evaluation",
        "",
    ]
    if evaluation:
        adv = evaluation.get("adversarial", {})
        lines += [
            f"- Golden set: {evaluation.get('passed')} of {evaluation.get('total')} correct",
            f"- Must refuse: {evaluation.get('must_refuse_correct')} of {evaluation.get('must_refuse_total')} refused",
            f"- Adversarial probes: {adv.get('passed')} of {adv.get('total')}",
        ]
    else:
        lines.append("- The golden-set evaluation hasn't been run on this deployment.")
    if review_tests and review_tests["total"]:
        lines.append(f"- Tests added from reviews: {review_tests['passed']} of {review_tests['total']} "
                     f"correct, {review_tests['unsafe_answers']} unsafe answers")
    q, rv = r["questions"], r["reviews"]
    lines += [
        "",
        "## Use and review",
        "",
        f"- Questions: {q['total']} ({q['answered']} answered, {q['refused']} refused)",
        f"- Questions asked with a check switched off: {q['with_a_check_switched_off']}",
        f"- Review cases opened: {rv['opened']} ({rv['flagged_answers']} flagged answers)",
        f"- Resolved: {rv['resolved']}, on time: {rv['resolved_on_time']}, escalated: {rv['escalated']}",
        f"- Open now: {rv['open_now']}, overdue now: {rv['overdue_now']}",
        "",
        "## Hazard log",
        "",
    ]
    if hazards:
        lines += ["| ID | Hazard | Initial risk | Controls | Residual risk | Status | Owner |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for h in hazards:
            cells = [
                f"H{h['id']}", h["title"],
                f"{h['initial_risk']['score']} ({h['initial_risk']['level']})",
                h["controls"].replace("\n", " ") or "None recorded",
                f"{h['residual_risk']['score']} ({h['residual_risk']['level']})",
                h["status"], h["owner"],
            ]
            lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    else:
        lines.append("No hazards recorded.")
    lines += ["", "## Sign-off", ""]
    lines += (["There is nothing here to sign. Produce this document against your own",
               "formulary and your own approved documents first.", ""]
              if synthetic else
              ["Clinical safety officer: ____________________  Date: __________", ""])
    return "\n".join(lines)
