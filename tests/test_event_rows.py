"""Administrative events (integrity.record_event) are chained into the audit
trail but are not questions. Every reader that counts, lists or looks up
questions must skip them, even if an event row were stored as an ordinary
(non-test) record with no response."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import (audit, governance, incidents, integrity, metering, monitoring, pipeline,  # noqa: E402
                 retrieval)
from app import db  # noqa: E402
from app.main import app  # noqa: E402

Q = "What is the standard dose of Caloradine?"
REFUSED = "What is the standard dose of Quelvarin?"


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def trail(database):
    """Two questions, then events: one the way record_event writes them, and
    one stored as though it were an ordinary question record."""
    pipeline.run(Q)
    pipeline.run(REFUSED)
    events = [integrity.record_event("api_key.created", {"key_id": 1}, None),
              integrity.record_event("release.promoted", {"n": 2}, None)]
    sneaky = "feedfeedfeedfeed"
    with integrity.write_lock, db.session() as s:
        integrity.append(s, db.AuditRecord(audit_id=sneaky, decision="event", query="sneaky", total_ms=0,
                                           llm_used=False, test_run=False, record={"audit_id": sneaky, "event": "x"}))
    return events + [sneaky]


def test_events_are_in_the_chain_and_the_chain_verifies(trail):
    result = integrity.verify()
    assert result["ok"] is True and result["checked"] >= 5
    with db.session() as s:
        from sqlalchemy import func, select

        assert s.scalar(select(func.count(db.AuditRecord.id)).where(db.AuditRecord.decision == "event")) == 3


def test_the_audit_store_never_lists_counts_or_returns_them(trail):
    assert audit.store.count() == 2
    listed = audit.store.recent(50)
    assert len(listed) == 2 and {r["decision"] for r in listed} == {"answer", "refuse"}
    for event_id in trail:
        assert audit.store.get(event_id) is None


def test_usage_and_report_ignore_events(trail):
    report = governance.report(30)["questions"]
    assert (report["total"], report["answered"], report["refused"]) == (2, 1, 1)
    usage = governance.usage(30)["totals"]
    assert (usage["questions"], usage["answered"], usage["refused"]) == (2, 1, 1)
    assert sum(d["answered"] + d["refused"] for d in governance.usage(30)["daily"]) == 2


def test_monitoring_ignores_events(trail):
    with db.session() as s:
        rows = monitoring._questions(s, db.utcnow() - timedelta(days=1))
    assert len(rows) == 2
    assert all(r.decision in ("answer", "refuse") for r in rows)


def test_review_and_incident_links_cannot_point_at_an_event(trail):
    for event_id in trail:
        with pytest.raises(governance.GovernanceError):
            governance.flag_answer(event_id, "this looks wrong to me", None)
        with pytest.raises(incidents.IncidentError, match="no audit record"):
            with db.session() as s:
                incidents._check_links(s, {"audit_id": event_id}, None)


def test_http_audit_views_skip_events(trail):
    client = TestClient(app)
    listing = client.get("/api/audit").json()
    assert listing["count"] == 2 and {r["decision"] for r in listing["recent"]} == {"answer", "refuse"}
    for event_id in trail:
        assert client.get(f"/api/audit/{event_id}").status_code == 404
    assert client.get("/api/usage?days=30").json()["totals"]["questions"] == 2
    assert client.get("/api/governance/report?days=30").json()["questions"]["total"] == 2


def test_metering_has_no_row_for_an_event(trail):
    assert metering.usage("endpoint")["total"] == 0
