"""Usage metering: how many requests, per day, site, API key, endpoint and
decision. Counts only. No question or answer text is stored here.

`decision` is what the endpoint decided (answer, refuse, pass, fail, review),
or `error` for a request that was refused before it was evaluated (bad input,
wrong scope) and `throttled` for one stopped by a rate limit or quota.
Billable requests are the ones that were evaluated: every decision except
error and throttled.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from . import db
from .db import ApiKey, Site, UsageDaily

log = logging.getLogger("groundcheck.metering")

DECISIONS = ("answer", "refuse", "pass", "fail", "review", "error", "throttled")
UNBILLED = ("error", "throttled")
GROUPS = ("key", "day", "site", "endpoint", "decision")
FIELD = {"key": "key_id", "day": "day", "site": "site_id", "endpoint": "endpoint", "decision": "decision"}
MAX_DAYS = 366


class UsageError(ValueError):
    """The usage query was invalid. The message is safe to show."""


def _today() -> str:
    return db.utcnow().date().isoformat()


def record(endpoint: str, decision: str, site_id: int | None, api_key_id: int | None, n: int = 1) -> None:
    """Add to today's count. Never raises: metering must not fail a request."""
    try:
        day, site, key = _today(), site_id or 0, api_key_id or 0
        for _ in range(3):
            with db.session() as s:
                where = (UsageDaily.day == day, UsageDaily.site_id == site, UsageDaily.api_key_id == key,
                         UsageDaily.endpoint == endpoint, UsageDaily.decision == decision)
                if s.execute(update(UsageDaily).where(*where).values(count=UsageDaily.count + n)).rowcount:
                    return
                try:
                    with s.begin_nested():
                        s.add(UsageDaily(day=day, site_id=site, api_key_id=key, endpoint=endpoint,
                                         decision=decision, count=n))
                    return
                except IntegrityError:
                    continue
        log.error("Couldn't meter %s %s", endpoint, decision)
    except Exception as exc:  # noqa: BLE001
        log.error("Couldn't meter %s %s: %s", endpoint, decision, exc)


def _parse_day(value: str | date | None, name: str) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise UsageError(f"'{name}' must be a date like 2026-11-24.") from exc


def usage(group_by: str = "key", start: str | date | None = None, end: str | date | None = None,
          site_scope: int | None = None) -> dict:
    """Request counts between two dates (inclusive, UTC; default the last 30
    days), grouped by key, day, site, endpoint or decision. With site_scope,
    only that site's rows."""
    if group_by not in GROUPS:
        raise UsageError(f"group_by must be one of: {', '.join(GROUPS)}.")
    last = _parse_day(end, "to") or db.utcnow().date()
    first = _parse_day(start, "from") or last - timedelta(days=29)
    if first > last:
        raise UsageError("'from' must not be after 'to'.")
    if (last - first).days >= MAX_DAYS:
        raise UsageError(f"Choose a period of at most {MAX_DAYS} days.")
    column = {"key": UsageDaily.api_key_id, "day": UsageDaily.day, "site": UsageDaily.site_id,
              "endpoint": UsageDaily.endpoint, "decision": UsageDaily.decision}[group_by]
    with db.session() as s:
        q = (select(column, UsageDaily.decision, func.sum(UsageDaily.count))
             .where(UsageDaily.day >= first.isoformat(), UsageDaily.day <= last.isoformat())
             .group_by(column, UsageDaily.decision))
        if site_scope is not None:
            q = q.where(UsageDaily.site_id == site_scope)
        grouped: dict = {}
        for label, decision, count in s.execute(q):
            grouped.setdefault(label, {})[decision] = int(count)
        names: dict = {}
        if group_by == "key":
            for k in s.scalars(select(ApiKey).where(ApiKey.id.in_([x for x in grouped if x]))):
                names[k.id] = (k.name, "gck_" + k.prefix)
        elif group_by == "site":
            for site in s.scalars(select(Site).where(Site.id.in_([x for x in grouped if x]))):
                names[site.id] = (site.name, site.key)
    rows = []
    for label in sorted(grouped, key=lambda x: (isinstance(x, str), x)):
        decisions = grouped[label]
        row = {FIELD[group_by]: label, "requests": sum(decisions.values()),
               "billable": sum(v for d, v in decisions.items() if d not in UNBILLED), "decisions": decisions}
        if group_by == "key":
            row["name"], row["prefix"] = names.get(label, ("", ""))
        elif group_by == "site":
            row["name"], row["site_key"] = names.get(label, ("", ""))
        rows.append(row)
    return {"group_by": group_by, "from": first.isoformat(), "to": last.isoformat(), "rows": rows,
            "total": sum(r["requests"] for r in rows), "billable": sum(r["billable"] for r in rows)}


def _safe(value) -> str:
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def to_csv(report: dict) -> str:
    """The usage report as CSV: one row per group, a column per decision."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    label = report["group_by"]
    writer.writerow([FIELD[label], "name", "requests", "billable", *DECISIONS])
    for row in report["rows"]:
        writer.writerow([_safe(row[FIELD[label]]), _safe(row.get("name", "")), row["requests"], row["billable"],
                         *[row["decisions"].get(d, 0) for d in DECISIONS]])
    return out.getvalue()
