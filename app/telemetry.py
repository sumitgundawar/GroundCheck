"""Opt-in anonymous install telemetry. Off by default.

With TELEMETRY unset or off, sending does nothing: no database write, no socket
(the admin page's sample only counts this week's requests locally). With TELEMETRY=on and TELEMETRY_URL set, once a week the install
posts the fields in FIELDS and nothing else: a random id, the version, the
database kind, the kind of model that drafts answers, and counts of the week's
requests bucketed into ranges. No question, answer, user, site, host name,
address, document or setting value is ever included. A failed post is dropped
without a message to the user and tried again at the next check."""

from __future__ import annotations

import logging
import uuid
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from . import config, db, llm
from .db import AppSetting, AuditRecord, utcnow

log = logging.getLogger("groundcheck.telemetry")

SCHEMA = 1
FIELDS = {
    "schema": "Version of this payload format (an integer).",
    "install_id": "A random identifier generated on first send. Not derived from anything about the machine or organisation.",
    "version": "The GroundCheckHealth version, for example 1.0.0.",
    "db_kind": "The database engine: sqlite, postgresql or mysql.",
    "llm_provider_kind": "How answers are drafted: extractive (no model), local, or cloud. Not the provider, model or address.",
    "week_start": "The Monday (UTC date) of the week the counts cover.",
    "requests_bucket": "Questions asked that week, as a range: 0, 1-9, 10-99, 100-999, 1000-9999 or 10000+.",
    "answers_bucket": "Of those, answered, as a range.",
    "refusals_bucket": "Of those, refused, as a range.",
}
BUCKETS = ((0, "0"), (9, "1-9"), (99, "10-99"), (999, "100-999"), (9999, "1000-9999"))
TOP_BUCKET = "10000+"
ID_KEY, SENT_KEY = "telemetry_install_id", "telemetry_last_sent"


def enabled() -> bool:
    return config.TELEMETRY == "on"


def bucket(n: int) -> str:
    for limit, label in BUCKETS:
        if n <= limit:
            return label
    return TOP_BUCKET


def _setting(s, key: str) -> str | None:
    row = s.get(AppSetting, key)
    return row.value if row else None


def _put(s, key: str, value: str) -> None:
    row = s.get(AppSetting, key)
    if row is None:
        s.add(AppSetting(key=key, value=value))
    else:
        row.value = value


def _kind() -> str:
    provider = llm.active_provider()
    return "extractive" if provider is None else provider["kind"]


def payload(install_id: str, now: datetime | None = None) -> dict:
    """What would be sent, for the given id. Reads only counts."""
    now = (now or utcnow()).astimezone(timezone.utc)
    week = now.date() - timedelta(days=now.date().weekday())
    since = datetime(week.year, week.month, week.day, tzinfo=timezone.utc)
    requests = answers = 0
    if db.ready():
        with db.session() as s:
            for r in s.scalars(select(AuditRecord).where(AuditRecord.created_at >= since,
                                                         AuditRecord.decision != "event")):
                if (r.record or {}).get("test_run"):
                    continue
                requests += 1
                answers += r.decision == "answer"
    return {"schema": SCHEMA, "install_id": install_id, "version": config.VERSION,
            "db_kind": db.engine().dialect.name if db.ready() else "none", "llm_provider_kind": _kind(),
            "week_start": week.isoformat(), "requests_bucket": bucket(requests),
            "answers_bucket": bucket(answers), "refusals_bucket": bucket(requests - answers)}


def status() -> dict:
    """For the admin page: whether it is on, where it would send, every field
    with its meaning, and a sample payload with a placeholder id. Touches
    nothing when off, apart from counting this week's requests for the sample."""
    return {"enabled": enabled(), "url": config.TELEMETRY_URL or None, "interval_days": config.TELEMETRY_INTERVAL_DAYS,
            "fields": FIELDS, "sample": payload("00000000000000000000000000000000"),
            "how_to_enable": "Set TELEMETRY=on and TELEMETRY_URL, then restart. It is off by default."}


def _valid_url(url: str) -> bool:
    """HTTPS to any host, or plain HTTP to this machine for testing."""
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1"))


def send_if_due(now: datetime | None = None) -> dict:
    """Send this week's payload if telemetry is on and one is due. Never raises."""
    if not enabled():
        return {"sent": False, "reason": "off"}
    if not config.TELEMETRY_URL or not _valid_url(config.TELEMETRY_URL):
        return {"sent": False, "reason": "no endpoint"}
    if not db.ready():
        return {"sent": False, "reason": "no database"}
    now = (now or utcnow()).astimezone(timezone.utc)
    try:
        with db.try_advisory_lock("telemetry", 7301) as held:
            if not held:
                return {"sent": False, "reason": "another replica is sending"}
            with db.session() as s:
                last = _setting(s, SENT_KEY)
                if last and now - datetime.fromisoformat(last) < timedelta(days=config.TELEMETRY_INTERVAL_DAYS):
                    return {"sent": False, "reason": "not due"}
                install_id = _setting(s, ID_KEY)
                if not install_id:
                    install_id = uuid.uuid4().hex
                    _put(s, ID_KEY, install_id)
            body = payload(install_id, now)
            try:
                response = httpx.post(config.TELEMETRY_URL, json=body, timeout=5.0, follow_redirects=False)
                response.raise_for_status()
            except Exception:  # noqa: BLE001 - telemetry must never trouble the user
                log.debug("Telemetry post failed", exc_info=False)
                return {"sent": False, "reason": "network"}
            with db.session() as s:
                _put(s, SENT_KEY, now.isoformat())
            return {"sent": True, "payload": body}
    except Exception:  # noqa: BLE001
        return {"sent": False, "reason": "error"}
