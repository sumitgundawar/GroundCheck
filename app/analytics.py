"""Adoption and value metrics, and the pilot report built from them.

Everything is counted over whole UTC days: a period of `days` ends with today
and starts at 00:00 UTC `days - 1` days ago, or runs between two dates given
inclusive. A record at 23:59:59 UTC belongs to that day and one at 00:00:00
to the next. Evaluation runs (`test_run`), administrative event rows and
records marked `demo_seed` are never counted. A site's people see only that
site's numbers; documents are shared across sites and so are their counts.

The pilot report prints no question text. The one place a question's words
appear, the document-gap list, takes single terms the sources lacked from
questions in which no identifier was found, and drops any term that looks like
a name, a number or a placeholder."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select

from . import __version__, config, db, feedback, governance
from .db import AnswerFeedback, AuditRecord, EvalCase, Incident, ReviewCase, Source, utcnow

MAX_DAYS = 3660
_TERM = re.compile(r"^[a-z][a-z-]{2,39}$")


def _is_counted(record: AuditRecord) -> bool:
    r = record.record or {}
    return record.decision != "event" and not r.get("test_run") and not r.get("demo_seed")


def window(days: int | None = None, start: date | None = None, end: date | None = None,
           now: datetime | None = None) -> tuple[date, date, datetime, datetime]:
    """(first day, last day, from, until) with `until` exclusive."""
    now = now or utcnow()
    if start is not None or end is not None:
        if start is None or end is None:
            raise governance.GovernanceError("Give both a start and an end date.")
        if end < start:
            raise governance.GovernanceError("The end date is before the start date.")
        if (end - start).days + 1 > MAX_DAYS:
            raise governance.GovernanceError(f"Choose a period of up to {MAX_DAYS:,} days.")
    else:
        if days is None or not 1 <= days <= MAX_DAYS:
            raise governance.GovernanceError(f"Choose a period of 1 to {MAX_DAYS:,} days.")
        end = now.astimezone(timezone.utc).date()
        start = end - timedelta(days=days - 1)
    since = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    until = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start, end, since, until


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _rate(n: int, total: int) -> float | None:
    return round(n / total, 4) if total else None


def adoption_metrics(days: int | None = 30, site_id: int | None = None, *, start: date | None = None,
                     end: date | None = None, now: datetime | None = None) -> dict:
    first, last, since, until = window(days, start, end, now)
    now = now or utcnow()
    last_7_from = max(since, until - timedelta(days=7))
    with db.session() as s:
        q = select(AuditRecord).where(AuditRecord.created_at >= since, AuditRecord.created_at < until,
                                      AuditRecord.decision != "event")
        if site_id is not None:
            q = q.where(AuditRecord.site_id == site_id)
        records = [r for r in s.scalars(q) if _is_counted(r)]

        answered = sum(r.decision == "answer" for r in records)
        refused = len(records) - answered
        reasons: Counter = Counter()
        by_user: Counter = Counter()
        week_users: dict[date, set[int]] = {}
        week_questions: Counter = Counter()
        recent_users: set[int] = set()
        via_api = 0
        for r in records:
            if r.decision == "refuse":
                response = (r.record or {}).get("response", {})
                reasons[governance.categorise(response.get("refused_reason") or "",
                                              response.get("reason_code"))] += 1
            week = _monday(r.created_at.date())
            week_questions[week] += 1
            if r.api_key_id is not None:
                via_api += 1
            if r.user_id is not None:
                by_user[r.user_id] += 1
                week_users.setdefault(week, set()).add(r.user_id)
                if r.created_at >= last_7_from:
                    recent_users.add(r.user_id)

        weeks = []
        week = _monday(first)
        while week <= last:
            weeks.append({"week_start": week.isoformat(), "questions": week_questions.get(week, 0),
                          "active_clinicians": len(week_users.get(week, ()))})
            week += timedelta(days=7)

        fq = select(AnswerFeedback).where(AnswerFeedback.created_at >= since, AnswerFeedback.created_at < until)
        if site_id is not None:
            fq = fq.where(AnswerFeedback.site_id == site_id)
        rows = list(s.scalars(fq))
        ignored = set()
        if rows:
            ids = {r.audit_id for r in rows}
            for a in s.scalars(select(AuditRecord).where(AuditRecord.audit_id.in_(ids))):
                if not _is_counted(a):
                    ignored.add(a.audit_id)
        rows = [r for r in rows if r.audit_id not in ignored]
        down = [r for r in rows if r.rating == "down"]
        fb_reasons = Counter(r.reason or "none" for r in down)

        def scoped(query, column=ReviewCase.site_id):
            return query if site_id is None else query.where(column == site_id)

        opened = s.scalars(scoped(select(ReviewCase).where(ReviewCase.created_at >= since,
                                                           ReviewCase.created_at < until))).all()
        closed = s.scalars(scoped(select(ReviewCase).where(ReviewCase.resolved_at >= since,
                                                           ReviewCase.resolved_at < until))).all()
        hours = [(c.resolved_at - c.created_at).total_seconds() / 3600 for c in closed]

        tests = select(EvalCase).where(EvalCase.created_at >= since, EvalCase.created_at < until)
        tests_added = [t for t in s.scalars(tests)]
        if site_id is not None:
            case_sites = {c.id: c.site_id for c in s.scalars(select(ReviewCase).where(
                ReviewCase.id.in_({t.from_case_id for t in tests_added if t.from_case_id}) )) } if tests_added else {}
            tests_added = [t for t in tests_added if t.from_case_id and case_sites.get(t.from_case_id) == site_id]

        approved = s.scalars(select(Source).where(Source.status == "approved", Source.reviewed_at >= since,
                                                  Source.reviewed_at < until)).all()
        open_now = len(s.scalars(scoped(select(ReviewCase).where(ReviewCase.status == "open"))).all())

    active = len(by_user)
    return {
        "period_days": (last - first).days + 1,
        "from": first.isoformat(),
        "to": last.isoformat(),
        "timezone": "UTC",
        "generated_at": now.isoformat(),
        "clinicians": {
            "active_in_period": active,
            "weekly_active": len(recent_users),
            "questions_per_active_clinician": round(sum(by_user.values()) / active, 2) if active else None,
            "weekly": weeks,
        },
        "questions": {
            "total": len(records), "answered": answered, "refused": refused,
            "answer_rate": _rate(answered, len(records)), "refusal_rate": _rate(refused, len(records)),
            "via_api": via_api,
            "refusal_by_reason": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "feedback": {
            "total": len(rows), "up": sum(r.rating == "up" for r in rows), "down": len(down),
            "positive_rate": _rate(sum(r.rating == "up" for r in rows), len(rows)),
            "by_reason": dict(sorted(fb_reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "reviews": {
            "opened": len(opened), "resolved": len(closed), "open_now": open_now,
            "median_hours_to_close": round(statistics.median(hours), 2) if hours else None,
        },
        "tests_added": len(tests_added),
        "documents_approved": len(approved),
    }


def document_gaps(start: date, end: date, site_id: int | None = None, limit: int = 15,
                  now: datetime | None = None) -> dict:
    """Terms the sources lacked (`term_not_covered` refusals), grouped, with how
    many questions each blocked. Terms from a question in which an identifier
    was found are withheld, and so is any term that isn't a plain lower-case word."""
    _, _, since, until = window(start=start, end=end, now=now)
    counts: Counter = Counter()
    withheld = 0
    with db.session() as s:
        q = select(AuditRecord).where(AuditRecord.created_at >= since, AuditRecord.created_at < until,
                                      AuditRecord.decision == "refuse")
        if site_id is not None:
            q = q.where(AuditRecord.site_id == site_id)
        for r in s.scalars(q):
            if not _is_counted(r):
                continue
            response = (r.record or {}).get("response", {})
            if response.get("reason_code") != "term_not_covered":
                continue
            terms = {t.strip().lower() for t in (response.get("explanation") or {}).get("missing_terms", [])}
            if (response.get("redaction") or {}).get("kinds"):
                withheld += len(terms) or 1
                continue
            for term in terms:
                if _TERM.match(term) and feedback.clean_comment(term) == term:
                    counts[term] += 1
                elif term:
                    withheld += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"terms": [{"term": t, "questions": n} for t, n in ranked[:limit]],
            "distinct_terms": len(ranked), "withheld": withheld}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value) -> str:
    return "n/a" if value is None else f"{value}"


def pilot_report_markdown(days: int | None = 30, site_id: int | None = None, *, start: date | None = None,
                          end: date | None = None, site_name: str = "", now: datetime | None = None) -> str:
    """A report for one site over a period, as Markdown. Deterministic: the same
    data and the same `now` give the same bytes."""
    from . import formulary

    now = now or utcnow()
    m = adoption_metrics(days, site_id, start=start, end=end, now=now)
    first, last = date.fromisoformat(m["from"]), date.fromisoformat(m["to"])
    gaps = document_gaps(first, last, site_id, now=now)
    _, _, since, until = window(start=first, end=last, now=now)
    with db.session() as s:
        iq = select(Incident).where(Incident.reported_at >= since, Incident.reported_at < until)
        if site_id is not None:
            iq = iq.where(Incident.site_id == site_id)
        incidents = Counter(i.harm for i in s.scalars(iq))
        flagged = len(s.scalars(select(ReviewCase).where(
            ReviewCase.kind == "flagged", ReviewCase.created_at >= since, ReviewCase.created_at < until,
            *([ReviewCase.site_id == site_id] if site_id is not None else []))).all())
    c, q, f, rv = m["clinicians"], m["questions"], m["feedback"], m["reviews"]
    lines = [f"# GroundCheckHealth pilot report", "",
             f"Site: {site_name or ('all sites' if site_id is None else f'site {site_id}')}. "
             f"Period: {m['from']} to {m['to']} ({m['period_days']} days, UTC). "
             f"Version {__version__}. Generated {now.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.", ""]
    if formulary.load().synthetic:
        lines += ["> ## DEMONSTRATION DATA",
                  ">",
                  "> This installation uses the synthetic demonstration corpus and formulary. The",
                  "> figures describe use of a demonstration, not of real clinical content.", ""]
    lines += ["This report counts use and review activity. It is not evidence of clinical safety or",
              "effectiveness, and GroundCheckHealth is not a certified medical device. It contains no",
              "question text and no patient identifiers.", "",
              "## Usage", "",
              f"- Questions: {q['total']} ({q['answered']} answered, {q['refused']} refused)",
              f"- Answer rate: {_pct(q['answer_rate'])}; refusal rate: {_pct(q['refusal_rate'])}",
              f"- Active clinicians in the period: {c['active_in_period']}; in the last 7 days of it: {c['weekly_active']}",
              f"- Questions per active clinician: {_num(c['questions_per_active_clinician'])}",
              f"- Questions through the API: {q['via_api']}", "",
              "| Week starting | Questions | Active clinicians |", "| --- | --- | --- |"]
    lines += [f"| {w['week_start']} | {w['questions']} | {w['active_clinicians']} |" for w in c["weekly"]]
    lines += ["", "## Safety outcomes", "",
              f"- Answers flagged by clinicians: {flagged}",
              f"- Feedback: {f['total']} ratings, {f['up']} up, {f['down']} down (positive {_pct(f['positive_rate'])})"]
    lines += [f"  - down because {k.replace('_', ' ')}: {n}" for k, n in f["by_reason"].items()]
    lines += [f"- Incidents reported: {sum(incidents.values())}"]
    lines += [f"  - harm {harm}: {n}" for harm, n in sorted(incidents.items())]
    lines += ["", "## Review loop", "",
              f"- Review cases opened: {rv['opened']}; resolved: {rv['resolved']}; open now: {rv['open_now']}",
              f"- Median hours to close: {_num(rv['median_hours_to_close'])}",
              f"- Questions added as permanent tests: {m['tests_added']}",
              f"- Documents approved: {m['documents_approved']}", "",
              "## Top refusal reasons", ""]
    lines += ([f"{i}. {reason.capitalize()}: {n}" for i, (reason, n) in enumerate(q["refusal_by_reason"].items(), 1)]
              or ["No refusals in the period."])
    lines += ["", "## Document gaps", "",
              "Terms that questions used and the approved documents did not cover, most frequent first.", ""]
    lines += ([f"- {g['term']}: {g['questions']} question{'' if g['questions'] == 1 else 's'}" for g in gaps["terms"]]
              or ["No gaps recorded in the period."])
    if gaps["withheld"]:
        lines += ["", f"{gaps['withheld']} term(s) were withheld because they could be identifying."]
    lines.append("")
    return "\n".join(lines)
