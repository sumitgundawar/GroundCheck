"""Operational monitoring: metrics for Prometheus, and alerts.

Metrics (`render()`, served at /metrics) are counted in this process:
requests by route and status with latency, questions by decision with
latency, and imaging analyses by outcome. Gauges read from the database when
scraped: open and overdue review cases, and firing alerts. Behind a load
balancer, Prometheus scrapes every instance and sums them.

Alerts are evaluated from the database, so every instance reaches the same
answer, every ALERT_INTERVAL_SECONDS and on demand:

- refusal_rate: the last hour's refusal rate is well above the last week's
- latency: the last hour's 95th-percentile response time is too slow
- retrieval_drift: the best retrieval scores of the last day have shifted
  from the last month's (population stability index), a sign that questions
  are moving away from the documents
- review_overdue: review cases past their due date
- documents_expiring: approved documents expiring soon, or expired
- audit_chain: the tamper-evident audit trail doesn't verify
- server_errors: this instance returned many 5xx responses recently
- imaging_failures: imaging analyses failed in the last day
- release_blocked: the newest knowledge release failed its safety check

An alert fires once, stays firing while its condition holds, and resolves
when it clears. Firing and resolving are posted to ALERT_WEBHOOK_URL as
JSON with a "text" field, which Slack and Microsoft Teams incoming webhooks
accept."""

from __future__ import annotations

import bisect
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import httpx
from sqlalchemy import func, select

from . import config, db

log = logging.getLogger("groundcheck.monitoring")

# Tests replace this with a mock transport; None uses the network.
transport: httpx.BaseTransport | None = None

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
_lock = threading.Lock()
_counters: dict[tuple, float] = {}
_histograms: dict[tuple, list] = {}
_recent_requests: deque = deque(maxlen=20000)   # (monotonic time, status) for server_errors
_last_chain_check: dict = {"at": 0.0, "result": None}
_started = time.time()

SEVERITY_RANK = {"warning": 1, "critical": 2}


# ---------------------------------------------------------------- metrics

def _inc(name: str, labels: tuple = (), value: float = 1.0) -> None:
    with _lock:
        _counters[(name, labels)] = _counters.get((name, labels), 0.0) + value


def _observe(name: str, labels: tuple, seconds: float) -> None:
    with _lock:
        entry = _histograms.setdefault((name, labels), [[0] * len(LATENCY_BUCKETS), 0, 0.0])
        index = bisect.bisect_left(LATENCY_BUCKETS, seconds)
        if index < len(LATENCY_BUCKETS):
            entry[0][index] += 1
        entry[1] += 1
        entry[2] += seconds


def observe_request(method: str, route: str, status: int, seconds: float) -> None:
    labels = (("method", method), ("route", route), ("status", str(status)))
    _inc("groundcheck_http_requests_total", labels)
    _observe("groundcheck_http_request_duration_seconds", (("route", route),), seconds)
    with _lock:
        _recent_requests.append((time.monotonic(), status))


def observe_answer(decision: str, milliseconds: int, drafted_by: str) -> None:
    _inc("groundcheck_questions_total", (("decision", decision), ("drafted_by", drafted_by)))
    _observe("groundcheck_question_duration_seconds", (("decision", decision),), milliseconds / 1000)


def observe_stages(decision: str, trace) -> None:
    """Which stage stopped the run, and how long each one took.

    The pipeline already works both of these out for the trace it shows the
    user, but neither left the process, so Prometheus could say how often the
    system refused and never which check was doing the refusing — the question
    an operator actually asks after a release changes the answer rate.
    """
    for step in trace:
        # A stage that was never reached reports 0 ms. Counting those would
        # pull every stage's latency toward zero in proportion to how often
        # the run stopped earlier, so only stages that actually ran are timed.
        if step.name != "decision" and step.status != "skip":
            _observe("groundcheck_pipeline_stage_seconds", (("stage", step.name),), (step.ms or 0) / 1000)
    if decision != "answer":
        stopped = next((s.name for s in trace if s.status == "fail" and s.name != "decision"), "unknown")
        _inc("groundcheck_refusals_total", (("stage", stopped),))


def observe_imaging(status: str) -> None:
    _inc("groundcheck_imaging_analyses_total", (("status", status),))


def reset() -> None:
    """Forget in-process metrics (tests)."""
    with _lock:
        _counters.clear()
        _histograms.clear()
        _recent_requests.clear()
    _last_chain_check.update(at=0.0, result=None)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(pairs: tuple, extra: tuple = ()) -> str:
    items = [*pairs, *extra]
    return "{" + ",".join(f'{k}="{_escape(str(v))}"' for k, v in items) + "}" if items else ""


HELP = {
    "groundcheck_http_requests_total": ("counter", "HTTP requests handled by this instance."),
    "groundcheck_http_request_duration_seconds": ("histogram", "HTTP request duration."),
    "groundcheck_questions_total": ("counter", "Questions answered or refused."),
    "groundcheck_question_duration_seconds": ("histogram", "Time to answer or refuse a question."),
    "groundcheck_imaging_analyses_total": ("counter", "Imaging analyses by outcome."),
    "groundcheck_refusals_total": ("counter", "Refusals by the stage that stopped the run."),
    "groundcheck_pipeline_stage_seconds": ("histogram", "Time spent in each pipeline stage."),
}


def render() -> str:
    lines: list[str] = []
    with _lock:
        counters = dict(_counters)
        histograms = {k: (list(v[0]), v[1], v[2]) for k, v in _histograms.items()}
    for name, (kind, text) in HELP.items():
        rows_c = [(labels, value) for (n, labels), value in counters.items() if n == name]
        rows_h = [(labels, value) for (n, labels), value in histograms.items() if n == name]
        if not rows_c and not rows_h:
            continue
        lines += [f"# HELP {name} {text}", f"# TYPE {name} {kind}"]
        for labels, value in sorted(rows_c):
            lines.append(f"{name}{_labels(labels)} {value:g}")
        for labels, (buckets, count, total) in sorted(rows_h):
            running = 0
            for bound, n in zip(LATENCY_BUCKETS, buckets):
                running += n
                lines.append(f'{name}_bucket{_labels(labels, (("le", f"{bound:g}"),))} {running}')
            lines.append(f'{name}_bucket{_labels(labels, (("le", "+Inf"),))} {count}')
            lines.append(f"{name}_sum{_labels(labels)} {total:.6f}")
            lines.append(f"{name}_count{_labels(labels)} {count}")
    gauges = _gauges()
    for name, (text, value) in gauges.items():
        lines += [f"# HELP {name} {text}", f"# TYPE {name} gauge"]
        if isinstance(value, dict):
            for label, v in sorted(value.items()):
                lines.append(f'{name}{{severity="{label}"}} {v}')
        else:
            lines.append(f"{name} {value}")
    lines += ["# HELP groundcheck_build_info GroundCheckHealth version.", "# TYPE groundcheck_build_info gauge",
              f'groundcheck_build_info{{version="{config.VERSION}"}} 1',
              "# HELP groundcheck_process_start_time_seconds When this instance started.",
              "# TYPE groundcheck_process_start_time_seconds gauge",
              f"groundcheck_process_start_time_seconds {_started:.0f}"]
    return "\n".join(lines) + "\n"


def _gauges() -> dict:
    from . import audit
    from .db import Alert, ReviewCase

    # Emitted whatever the database is doing. Every other gauge here needs a
    # working database to compute, so they all disappeared at exactly the
    # moment something had gone wrong -- leaving the one state an operator
    # most needs to alert on indistinguishable from a quiet instance.
    backend = audit.store.backend()
    configured = bool(config.DATABASE_URL)
    recording = int(backend == "database" or (backend == "file" and not configured))
    base = {"groundcheck_recording": (
        "1 when questions are written to a durable audit trail, 0 when the instance "
        "is answering without recording anything.", recording)}
    if not db.ready():
        return base
    now = db.utcnow()
    with db.session() as s:
        open_cases = s.scalar(select(func.count(ReviewCase.id)).where(ReviewCase.status == "open")) or 0
        overdue = s.scalar(select(func.count(ReviewCase.id)).where(
            ReviewCase.status == "open", ReviewCase.due_at <= now)) or 0
        firing = dict(s.execute(select(Alert.severity, func.count(Alert.id))
                                .where(Alert.status == "firing").group_by(Alert.severity)).all())
    return {
        **base,
        "groundcheck_review_cases_open": ("Open review cases.", open_cases),
        "groundcheck_review_cases_overdue": ("Open review cases past their due date.", overdue),
        "groundcheck_alerts_firing": ("Alerts firing, by severity.",
                                      {k: firing.get(k, 0) for k in SEVERITY_RANK}),
    }


# ---------------------------------------------------------------- alert rules

def _psi(baseline: list[float], recent: list[float], bins: int = 10) -> float:
    """Population stability index of recent against baseline, with bins at
    the baseline's deciles. Under 0.1 is stable, over 0.25 a real shift."""
    ordered = sorted(baseline)
    edges = sorted({ordered[int(len(ordered) * i / bins)] for i in range(1, bins)})

    def shares(values: list[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[bisect.bisect_right(edges, v)] += 1
        return [max(c / len(values), 1e-4) for c in counts]

    b, r = shares(baseline), shares(recent)
    return float(sum((ri - bi) * math.log(ri / bi) for bi, ri in zip(b, r)))


def _questions(s, since: datetime, until: datetime | None = None):
    from .db import AuditRecord

    q = select(AuditRecord.decision, AuditRecord.total_ms, AuditRecord.top_score).where(
        AuditRecord.created_at >= since, AuditRecord.test_run.is_(False))
    if until is not None:
        q = q.where(AuditRecord.created_at < until)
    return s.execute(q).all()


def _rule_refusal_rate(s, now) -> dict | None:
    window = timedelta(minutes=config.ALERT_WINDOW_MINUTES)
    recent = _questions(s, now - window)
    baseline = _questions(s, now - timedelta(days=7), now - window)
    if len(recent) < config.ALERT_MIN_QUESTIONS or len(baseline) < config.ALERT_MIN_QUESTIONS * 5:
        return None
    rate = sum(r.decision == "refuse" for r in recent) / len(recent)
    base = sum(r.decision == "refuse" for r in baseline) / len(baseline)
    margin = max(config.ALERT_REFUSAL_RISE, 3 * math.sqrt(max(base * (1 - base), 1e-4) / len(recent)))
    if rate - base < margin:
        return None
    return {"severity": "critical" if rate - base >= 2 * margin else "warning",
            "title": "Refusals are much higher than usual",
            "detail": f"{rate:.0%} of {len(recent)} questions in the last {config.ALERT_WINDOW_MINUTES} minutes were "
                      f"refused, against {base:.0%} over the previous week. Check for a document that expired or was "
                      "retired, a search index problem, or a new kind of question.",
            "value": round(rate, 4)}


def _rule_latency(s, now) -> dict | None:
    recent = _questions(s, now - timedelta(minutes=config.ALERT_WINDOW_MINUTES))
    if len(recent) < 10:
        return None
    times = sorted(r.total_ms for r in recent)
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    if p95 <= config.ALERT_LATENCY_P95_MS:
        return None
    return {"severity": "critical" if p95 > 2 * config.ALERT_LATENCY_P95_MS else "warning",
            "title": "Answers are slow",
            "detail": f"95% of questions in the last {config.ALERT_WINDOW_MINUTES} minutes took up to "
                      f"{p95 / 1000:.1f} s, above the {config.ALERT_LATENCY_P95_MS / 1000:.1f} s limit. Check the "
                      "language model provider, the local model's hardware, and the database.",
            "value": p95}


def _rule_retrieval_drift(s, now) -> dict | None:
    recent = [r.top_score for r in _questions(s, now - timedelta(days=1)) if r.top_score is not None]
    baseline = [r.top_score for r in _questions(s, now - timedelta(days=30), now - timedelta(days=1))
                if r.top_score is not None]
    if len(recent) < 50 or len(baseline) < 200:
        return None
    psi = _psi(baseline, recent)
    if psi < config.ALERT_DRIFT_PSI:
        return None
    lower = sorted(recent)[len(recent) // 2] < sorted(baseline)[len(baseline) // 2]
    return {"severity": "warning", "title": "Questions are drifting from the documents",
            "detail": f"The best search scores of the last day have shifted from the last month's "
                      f"(stability index {psi:.2f}, alert at {config.ALERT_DRIFT_PSI:.2f})"
                      f"{', mostly lower' if lower else ''}. People may be asking about topics the approved "
                      "documents don't cover. Look at recent refusals in the review queue.",
            "value": round(psi, 4)}


def _rule_review_overdue(s, now) -> dict | None:
    from .db import ReviewCase

    overdue = s.scalar(select(func.count(ReviewCase.id)).where(
        ReviewCase.status == "open", ReviewCase.due_at <= now)) or 0
    if not overdue:
        return None
    return {"severity": "critical" if overdue >= config.ALERT_OVERDUE_CRITICAL else "warning",
            "title": "Review cases are overdue",
            "detail": f"{overdue} review case{'s are' if overdue != 1 else ' is'} past the due date. Assign or "
                      "resolve them on the Review page.",
            "value": overdue}


def _rule_documents_expiring(s, now) -> dict | None:
    from .db import Source

    soon = now + timedelta(days=config.ALERT_DOCUMENT_EXPIRY_DAYS)
    rows = s.execute(select(Source.title, Source.expires_on).where(
        Source.status == "approved", Source.expires_on.is_not(None), Source.expires_on <= soon)).all()
    if not rows:
        return None
    expired = [r for r in rows if r.expires_on <= now]
    names = ", ".join(r.title for r in sorted(rows, key=lambda r: r.expires_on)[:3])
    more = f" and {len(rows) - 3} more" if len(rows) > 3 else ""
    return {"severity": "critical" if expired else "warning",
            "title": "Documents have expired" if expired else "Documents expire soon",
            "detail": (f"{len(expired)} approved document{'s have' if len(expired) != 1 else ' has'} expired and "
                       "can no longer be cited" if expired else
                       f"{len(rows)} approved document{'s expire' if len(rows) != 1 else ' expires'} within "
                       f"{config.ALERT_DOCUMENT_EXPIRY_DAYS} days") +
                      f": {names}{more}. Upload a current version on the Documents page.",
            "value": len(rows)}


def _rule_audit_chain(s, now) -> dict | None:
    from . import integrity

    if time.monotonic() - _last_chain_check["at"] > config.ALERT_CHAIN_CHECK_MINUTES * 60 or \
            _last_chain_check["result"] is None:
        _last_chain_check.update(at=time.monotonic(), result=integrity.verify(max_problems=3))
    result = _last_chain_check["result"]
    if result.get("ok"):
        return None
    first = (result.get("problems") or [{}])[0]
    return {"severity": "critical", "title": "The audit trail doesn't verify",
            "detail": result.get("error") or f"The tamper-evident audit chain is broken"
                      f"{' at record ' + str(first.get('seq')) if first.get('seq') is not None else ''}: "
                      f"{first.get('problem', 'records were changed or removed')}. Treat this as a security "
                      "incident.",
            "value": 0}


def _rule_server_errors(s, now) -> dict | None:
    cutoff = time.monotonic() - 15 * 60
    with _lock:
        recent = [status for at, status in _recent_requests if at >= cutoff]
    if len(recent) < 20:
        return None
    share = sum(status >= 500 for status in recent) / len(recent)
    if share < 0.05:
        return None
    return {"severity": "critical", "title": "Server errors on this instance",
            "detail": f"{share:.0%} of {len(recent)} requests in the last 15 minutes failed with a server error. "
                      "Check the application logs.",
            "value": round(share, 4)}


def _rule_imaging_failures(s, now) -> dict | None:
    from .db import ImagingAnalysis

    failed = s.scalar(select(func.count(ImagingAnalysis.id)).where(
        ImagingAnalysis.status == "failed", ImagingAnalysis.finished_at >= now - timedelta(days=1))) or 0
    if not failed:
        return None
    return {"severity": "warning", "title": "Imaging analyses failed",
            "detail": f"{failed} imaging analys{'es' if failed != 1 else 'is'} failed in the last day. The series "
                      "page shows each error.",
            "value": failed}


def _rule_release_blocked(s, now) -> dict | None:
    from .db import Release

    latest = s.scalar(select(Release).order_by(Release.number.desc()))
    if latest is None or latest.status != "failed":
        return None
    unsafe = (latest.check or {}).get("unsafe")
    return {"severity": "critical", "title": "A knowledge release failed its safety check",
            "detail": f"R{latest.number} ({latest.reason}) " + (f"answered {unsafe} question{'s' if unsafe != 1 else ''} "
                      "that must be refused" if unsafe else "couldn't be checked") +
                      ", so it didn't go live and answers still come from the live release. See Releases.",
            "value": unsafe or 0}


RULES = {
    "refusal_rate": _rule_refusal_rate,
    "latency": _rule_latency,
    "retrieval_drift": _rule_retrieval_drift,
    "review_overdue": _rule_review_overdue,
    "documents_expiring": _rule_documents_expiring,
    "audit_chain": _rule_audit_chain,
    "server_errors": _rule_server_errors,
    "imaging_failures": _rule_imaging_failures,
    "release_blocked": _rule_release_blocked,
}


def _notify(alert, event: str) -> bool:
    if not config.ALERT_WEBHOOK_URL:
        return False
    word = "Resolved" if event == "resolved" else alert.severity.capitalize()
    text = f"GroundCheckHealth {word}: {alert.title}. {alert.detail if event != 'resolved' else ''}".strip()
    payload = {"text": text, "alert": {"id": alert.id, "rule": alert.rule, "severity": alert.severity,
                                       "status": alert.status, "title": alert.title, "detail": alert.detail,
                                       "first_seen": alert.first_seen.isoformat(),
                                       "event": event, "instance": config.INSTANCE_NAME}}
    try:
        with httpx.Client(timeout=5.0, transport=transport, follow_redirects=False) as client:
            response = client.post(config.ALERT_WEBHOOK_URL, json=payload)
        return response.status_code < 300
    except httpx.HTTPError as exc:
        log.warning("Couldn't post alert %s to the webhook: %s", alert.rule, exc)
        return False


def evaluate() -> dict:
    """Run every rule, update alerts, and notify on changes."""
    from .db import Alert

    now = db.utcnow()
    changes = {"fired": [], "resolved": [], "errors": []}
    with db.session() as s:
        firing = {a.rule: a for a in s.scalars(select(Alert).where(Alert.status == "firing"))}
    for rule, check in RULES.items():
        if rule in config.ALERT_DISABLED_RULES:
            continue
        try:
            with db.session() as s:
                found = check(s, now)
        except Exception as exc:  # noqa: BLE001 - one broken rule mustn't stop the others
            log.exception("Alert rule %s failed", rule)
            changes["errors"].append({"rule": rule, "error": type(exc).__name__})
            continue
        with db.session() as s:
            current = s.get(Alert, firing[rule].id) if rule in firing else None
            if found and current is None:
                current = Alert(rule=rule, severity=found["severity"], title=found["title"], detail=found["detail"],
                                value=found.get("value"), first_seen=now, last_seen=now, status="firing")
                s.add(current)
                s.flush()
                changes["fired"].append(rule)
                current.notified = _notify(current, "fired")
            elif found and current is not None:
                escalated = SEVERITY_RANK[found["severity"]] > SEVERITY_RANK[current.severity]
                current.severity, current.title, current.detail = found["severity"], found["title"], found["detail"]
                current.value, current.last_seen = found.get("value"), now
                if escalated:
                    current.acknowledged_at, current.acknowledged_by = None, None
                    _notify(current, "escalated")
            elif not found and current is not None:
                current.status, current.resolved_at = "resolved", now
                changes["resolved"].append(rule)
                _notify(current, "resolved")
    return changes


def _instance() -> dict:
    """What this instance is running on, and what it chose from that."""
    from . import pipeline, resources, retrieval

    caches = retrieval.cache_stats()
    return {
        "name": config.INSTANCE_NAME,
        "version": config.VERSION,
        **resources.summary(),
        "answers_cached": pipeline.answer_cache_stats()["answers"],
        "answer_cache_seconds": config.ANSWER_CACHE_SECONDS,
        "questions_cached": caches["questions"],
        "passages_cached": caches["passages"],
    }


def _alert(a) -> dict:
    return {"id": a.id, "rule": a.rule, "severity": a.severity, "status": a.status, "title": a.title,
            "detail": a.detail, "value": a.value, "first_seen": a.first_seen.isoformat(),
            "last_seen": a.last_seen.isoformat(), "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
            "acknowledged_by": a.acknowledged_by_name, "notified": a.notified}


def acknowledge(alert_id: int, user_id: int | None, name: str) -> dict:
    from .db import Alert

    with db.session() as s:
        alert = s.get(Alert, alert_id)
        if alert is None:
            raise LookupError(alert_id)
        if alert.status == "firing" and alert.acknowledged_at is None:
            alert.acknowledged_at, alert.acknowledged_by, alert.acknowledged_by_name = db.utcnow(), user_id, name[:200]
        return _alert(alert)


def overview() -> dict:
    """What the Monitoring page shows."""
    from .db import Alert

    now = db.utcnow()
    with db.session() as s:
        firing = s.scalars(select(Alert).where(Alert.status == "firing").order_by(Alert.first_seen.desc())).all()
        resolved = s.scalars(select(Alert).where(Alert.status == "resolved")
                             .order_by(Alert.resolved_at.desc()).limit(20)).all()
        hour = _questions(s, now - timedelta(minutes=config.ALERT_WINDOW_MINUTES))
        week = _questions(s, now - timedelta(days=7))
        firing_out = sorted((_alert(a) for a in firing), key=lambda a: (-SEVERITY_RANK[a["severity"]], a["first_seen"]))
        resolved_out = [_alert(a) for a in resolved]
    times = sorted(r.total_ms for r in hour)
    cutoff = time.monotonic() - 15 * 60
    with _lock:
        requests = [status for at, status in _recent_requests if at >= cutoff]
    return {
        "firing": firing_out,
        "resolved": resolved_out,
        "now": {
            "window_minutes": config.ALERT_WINDOW_MINUTES,
            "questions": len(hour),
            "refusal_rate": (sum(r.decision == "refuse" for r in hour) / len(hour)) if hour else None,
            "baseline_refusal_rate": (sum(r.decision == "refuse" for r in week) / len(week)) if week else None,
            "p95_ms": times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))] if times else None,
            "requests_15m": len(requests),
            "server_error_rate_15m": (sum(s_ >= 500 for s_ in requests) / len(requests)) if requests else None,
            "uptime_seconds": round(time.time() - _started),
        },
        "rules": [
            {"rule": "refusal_rate", "name": "Refusal rate", "limit": f"{config.ALERT_REFUSAL_RISE:.0%} above the last week, over {config.ALERT_WINDOW_MINUTES} min with at least {config.ALERT_MIN_QUESTIONS} questions"},
            {"rule": "latency", "name": "Response time", "limit": f"95th percentile above {config.ALERT_LATENCY_P95_MS / 1000:.1f} s over {config.ALERT_WINDOW_MINUTES} min"},
            {"rule": "retrieval_drift", "name": "Question drift", "limit": f"Stability index of search scores at least {config.ALERT_DRIFT_PSI:.2f}, last day against last month"},
            {"rule": "review_overdue", "name": "Overdue reviews", "limit": f"Any overdue; critical at {config.ALERT_OVERDUE_CRITICAL}"},
            {"rule": "documents_expiring", "name": "Document expiry", "limit": f"Expiring within {config.ALERT_DOCUMENT_EXPIRY_DAYS} days; critical once expired"},
            {"rule": "audit_chain", "name": "Audit trail", "limit": f"Any break, checked every {config.ALERT_CHAIN_CHECK_MINUTES} min"},
            {"rule": "server_errors", "name": "Server errors", "limit": "5% of requests in 15 min, on this instance"},
            {"rule": "imaging_failures", "name": "Imaging failures", "limit": "Any failed analysis in the last day"},
            {"rule": "release_blocked", "name": "Knowledge releases", "limit": "The newest release failed its safety check"},
        ],
        "instance": _instance(),
        "disabled": sorted(config.ALERT_DISABLED_RULES),
        "webhook": bool(config.ALERT_WEBHOOK_URL),
        "interval_seconds": config.ALERT_INTERVAL_SECONDS,
    }
