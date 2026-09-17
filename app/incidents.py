"""Incident reporting: when something went wrong, or nearly did.

Anyone signed in can report an incident, linked to what it concerns: an
answer or refusal (its audit ID), a review case, an alert, an imaging series,
or a hazard in the hazard log. Reviewers investigate it: they grade the harm,
take ownership, record the root cause and the actions taken, and close it.
Closing needs both. Every change is kept in the incident's timeline.

Harm is graded as in NHS patient safety reporting: none (including near
misses), low, moderate, severe, death. Two kinds of incident come with a
deadline shown on the incident, because regulators set one:

- a personal data breach: 72 hours from becoming aware of it to notify the
  data protection authority, when the breach is notifiable (UK and EU GDPR
  article 33)
- severe harm or death involving the software: a report to the medical
  device regulator may be required (for example the MHRA in the UK or the
  FDA's MedWatch in the US), which the organisation must decide

Descriptions, root causes, actions and notes are encrypted when
DATA_ENCRYPTION_KEYS is set. Don't enter patient names or identifiers."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from . import db
from .db import Alert, Hazard, ImagingSeries, Incident, IncidentEvent, ReviewCase, User

CATEGORIES = {
    "answer": "An answer or refusal",
    "patient_check": "A patient safety check",
    "imaging": "Imaging",
    "ehr": "EHR integration",
    "data_protection": "Data protection or security",
    "availability": "Outage or slowness",
    "other": "Something else",
}
HARM = {"none": "No harm or near miss", "low": "Low harm", "moderate": "Moderate harm",
        "severe": "Severe harm", "death": "Death"}
STATUSES = ("open", "investigating", "closed")
BREACH_HOURS = 72


class IncidentError(ValueError):
    """The request can't be carried out. The message is safe to show."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def reference(incident_id: int, reported_at: datetime) -> str:
    return f"INC-{reported_at:%Y}-{incident_id:04d}"


def _names(s, ids) -> dict[int, str]:
    ids = {i for i in ids if i}
    return {u.id: (u.name or u.email) for u in s.scalars(select(User).where(User.id.in_(ids)))} if ids else {}


def _deadlines(i: Incident) -> list[dict]:
    out = []
    if i.category == "data_protection":
        due = i.aware_at + timedelta(hours=BREACH_HOURS)
        out.append({"kind": "data_protection", "due_at": due.isoformat(), "done": bool(i.external_reference),
                    "text": "If this personal data breach is notifiable, tell the data protection authority within "
                            "72 hours of becoming aware of it."})
    if i.harm in ("severe", "death"):
        out.append({"kind": "device_regulator", "due_at": None, "done": bool(i.external_reference),
                    "text": "Severe harm or death: decide whether to report to the medical device regulator, such as "
                            "the MHRA or the FDA's MedWatch, and record the reference."})
    return out


def _summary(i: Incident, names: dict[int, str]) -> dict:
    return {
        "id": i.id, "reference": reference(i.id, i.reported_at), "title": i.title, "category": i.category,
        "category_label": CATEGORIES[i.category], "harm": i.harm, "harm_label": HARM[i.harm], "status": i.status,
        "reported_at": i.reported_at.isoformat(), "occurred_at": i.occurred_at.isoformat() if i.occurred_at else None,
        "reported_by": i.reporter_name, "owner_id": i.owner_id, "owner": names.get(i.owner_id),
        "updated_at": i.updated_at.isoformat(), "closed_at": i.closed_at.isoformat() if i.closed_at else None,
        "deadlines": _deadlines(i),
    }


def _detail(s, i: Incident) -> dict:
    events = s.scalars(select(IncidentEvent).where(IncidentEvent.incident_id == i.id).order_by(IncidentEvent.id)).all()
    names = _names(s, [i.owner_id])
    return {
        **_summary(i, names),
        "description": i.description, "root_cause": i.root_cause, "actions": i.actions,
        "external_reference": i.external_reference,
        "links": {"audit_id": i.audit_id, "review_case_id": i.review_case_id, "alert_id": i.alert_id,
                  "imaging_series_id": i.imaging_series_id, "hazard_id": i.hazard_id},
        "timeline": [{"at": e.at.isoformat(), "by": e.user_name, "action": e.action, "note": e.note} for e in events],
    }


def _text(value, name: str, limit: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise IncidentError(f"Add {name}.")
    if len(text) > limit:
        raise IncidentError(f"The {name} is too long: up to {limit:,} characters.")
    return text


def _when(value, name: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise IncidentError(f"The {name} isn't a date and time.") from exc
    parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    if parsed > _now() + timedelta(minutes=5):
        raise IncidentError(f"The {name} can't be in the future.")
    return parsed


def _check_links(s, data: dict) -> dict:
    links = {}
    for key, model, label in (("review_case_id", ReviewCase, "review case"), ("alert_id", Alert, "alert"),
                              ("imaging_series_id", ImagingSeries, "imaging series"), ("hazard_id", Hazard, "hazard")):
        value = data.get(key)
        if value in (None, ""):
            links[key] = None
            continue
        if s.get(model, int(value)) is None:
            raise IncidentError(f"There's no {label} {value}.")
        links[key] = int(value)
    audit = str(data.get("audit_id") or "").strip()
    if audit:
        from .db import AuditRecord

        if s.scalar(select(AuditRecord.id).where(AuditRecord.audit_id == audit)) is None:
            raise IncidentError(f"There's no audit record {audit}.")
    links["audit_id"] = audit or None
    return links


def report(data: dict, user_id: int | None, user_name: str) -> dict:
    category = data.get("category")
    if category not in CATEGORIES:
        raise IncidentError("Choose what the incident concerns.")
    harm = data.get("harm") or "none"
    if harm not in HARM:
        raise IncidentError("Choose the level of harm.")
    title = _text(data.get("title"), "a short title", 200, required=True)
    description = _text(data.get("description"), "a description of what happened", 20000, required=True)
    occurred_at = _when(data.get("occurred_at"), "time it happened")
    now = _now()
    with db.session() as s:
        links = _check_links(s, data)
        incident = Incident(title=title, category=category, harm=harm, status="open", description=description,
                            occurred_at=occurred_at, reported_at=now, aware_at=now, reported_by=user_id,
                            reporter_name=user_name[:200], updated_at=now, **links)
        s.add(incident)
        s.flush()
        s.add(IncidentEvent(incident_id=incident.id, user_id=user_id, user_name=user_name[:200], action="reported",
                            note=f"{HARM[harm]}. {CATEGORIES[category]}."))
        return _detail(s, incident)


def list_incidents(status: str = "", category: str = "", reporter_id: int | None = None) -> dict:
    with db.session() as s:
        q = select(Incident).order_by(Incident.reported_at.desc())
        if status:
            if status not in STATUSES and status != "active":
                raise IncidentError("Unknown status.")
            q = q.where(Incident.status != "closed") if status == "active" else q.where(Incident.status == status)
        if category:
            q = q.where(Incident.category == category)
        if reporter_id is not None:
            q = q.where(Incident.reported_by == reporter_id)
        rows = s.scalars(q.limit(500)).all()
        names = _names(s, [r.owner_id for r in rows])
        counts = dict(s.execute(select(Incident.status, func.count(Incident.id)).group_by(Incident.status)).all())
        serious = s.scalar(select(func.count(Incident.id)).where(
            Incident.status != "closed", Incident.harm.in_(("moderate", "severe", "death")))) or 0
        return {"incidents": [_summary(r, names) for r in rows],
                "counts": {k: counts.get(k, 0) for k in STATUSES}, "serious_active": serious,
                "categories": CATEGORIES, "harm": HARM}


def get(incident_id: int, reporter_id: int | None = None) -> dict:
    with db.session() as s:
        incident = s.get(Incident, incident_id)
        if incident is None or (reporter_id is not None and incident.reported_by != reporter_id):
            raise LookupError(incident_id)
        return _detail(s, incident)


UPDATABLE = ("status", "harm", "category", "owner_id", "root_cause", "actions", "external_reference",
             "hazard_id", "aware_at")


def update(incident_id: int, changes: dict, user_id: int | None, user_name: str) -> dict:
    with db.session() as s:
        incident = s.get(Incident, incident_id)
        if incident is None:
            raise LookupError(incident_id)
        notes = []
        if "harm" in changes and changes["harm"] != incident.harm:
            if changes["harm"] not in HARM:
                raise IncidentError("Choose the level of harm.")
            notes.append(f"Harm: {HARM[incident.harm]} to {HARM[changes['harm']]}.")
            incident.harm = changes["harm"]
        if "category" in changes and changes["category"] != incident.category:
            if changes["category"] not in CATEGORIES:
                raise IncidentError("Choose what the incident concerns.")
            notes.append(f"Concerns: {CATEGORIES[changes['category']]}.")
            incident.category = changes["category"]
        if "owner_id" in changes and changes["owner_id"] != incident.owner_id:
            owner = s.get(User, int(changes["owner_id"])) if changes["owner_id"] else None
            if changes["owner_id"] and owner is None:
                raise IncidentError("That person doesn't have an account.")
            incident.owner_id = owner.id if owner else None
            notes.append(f"Owner: {(owner.name or owner.email) if owner else 'nobody'}.")
        for field, label, limit in (("root_cause", "Root cause", 20000), ("actions", "Actions", 20000),
                                    ("external_reference", "External report", 500)):
            if field in changes:
                value = _text(changes[field], label.lower(), limit)
                if value != getattr(incident, field):
                    setattr(incident, field, value)
                    notes.append(f"{label} updated.")
        if "hazard_id" in changes:
            hazard = _check_links(s, {"hazard_id": changes["hazard_id"]})["hazard_id"]
            if hazard != incident.hazard_id:
                incident.hazard_id = hazard
                notes.append(f"Linked to hazard {hazard}." if hazard else "Hazard link removed.")
        if "aware_at" in changes and changes["aware_at"]:
            aware = _when(changes["aware_at"], "time the organisation became aware")
            if aware != incident.aware_at:
                incident.aware_at = aware
                notes.append("Time of becoming aware updated.")
        if "status" in changes and changes["status"] != incident.status:
            status = changes["status"]
            if status not in STATUSES:
                raise IncidentError("Unknown status.")
            if status == "closed" and not (incident.root_cause and incident.actions):
                raise IncidentError("Record the root cause and the actions taken before closing the incident.")
            if status == "closed" and any(not d["done"] for d in _deadlines(incident)):
                raise IncidentError("Record the external report reference, or 'Not reportable' with the reason, "
                                    "before closing.")
            notes.append({"open": "Reopened.", "investigating": "Investigation started.", "closed": "Closed."}[status])
            incident.status = status
            incident.closed_at = _now() if status == "closed" else None
        if notes:
            incident.updated_at = _now()
            s.add(IncidentEvent(incident_id=incident.id, user_id=user_id, user_name=user_name[:200],
                                action="updated", note=" ".join(notes)))
        s.flush()
        return _detail(s, incident)


def comment(incident_id: int, note: str, user_id: int | None, user_name: str, reporter_only: bool = False) -> dict:
    note = _text(note, "comment", 4000, required=True)
    with db.session() as s:
        incident = s.get(Incident, incident_id)
        if incident is None or (reporter_only and incident.reported_by != user_id):
            raise LookupError(incident_id)
        incident.updated_at = _now()
        s.add(IncidentEvent(incident_id=incident.id, user_id=user_id, user_name=user_name[:200], action="comment",
                            note=note))
        s.flush()
        return _detail(s, incident)


def export_csv() -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Reference", "Title", "Concerns", "Harm", "Status", "Occurred", "Reported", "Reported by",
                     "Owner", "Closed", "Root cause", "Actions", "External report", "Audit ID", "Review case",
                     "Alert", "Imaging series", "Hazard"])
    with db.session() as s:
        rows = s.scalars(select(Incident).order_by(Incident.reported_at)).all()
        names = _names(s, [r.owner_id for r in rows])
        for i in rows:
            writer.writerow([reference(i.id, i.reported_at), i.title, CATEGORIES[i.category], HARM[i.harm], i.status,
                             i.occurred_at.isoformat() if i.occurred_at else "", i.reported_at.isoformat(),
                             i.reporter_name, names.get(i.owner_id, ""), i.closed_at.isoformat() if i.closed_at else "",
                             i.root_cause, i.actions, i.external_reference, i.audit_id or "", i.review_case_id or "",
                             i.alert_id or "", i.imaging_series_id or "", i.hazard_id or ""])
    return buffer.getvalue()
