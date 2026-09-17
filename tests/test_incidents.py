"""Incident reporting: reporting with links, investigation and closing
rules, regulator deadlines, the timeline, encryption, CSV export, and who can
see and change what."""

from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, config, db, encryption, incidents, pipeline  # noqa: E402

PASSWORD = "correct horse battery staple"


@pytest.fixture()
def people(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    from app.main import app

    users = {
        "clinician": auth.create_user("clin@example.org", PASSWORD, role="clinician", name="Dr Clin"),
        "other": auth.create_user("other@example.org", PASSWORD, role="clinician", name="Dr Other"),
        "reviewer": auth.create_user("rev@example.org", PASSWORD, role="reviewer", name="Dr Rev"),
    }

    def client_for(who: str) -> TestClient:
        client = TestClient(app)
        assert client.post("/api/auth/login", json={"email": f"{ {'clinician': 'clin', 'other': 'other', 'reviewer': 'rev'}[who] }@example.org",
                                                    "password": PASSWORD}).status_code == 200
        return client

    return users, client_for


def test_report_investigate_and_close(people):
    users, client_for = people
    answer = pipeline.run("What is the standard dose of Caloradine?")
    clinician, reviewer = client_for("clinician"), client_for("reviewer")

    bad = clinician.post("/api/incidents", json={"title": "x", "category": "nope", "description": "y"})
    assert bad.status_code == 400 and "concerns" in bad.json()["detail"]
    assert clinician.post("/api/incidents", json={"title": "x", "category": "answer", "description": "y",
                                                  "audit_id": "ffffffff"}).json()["detail"] == "There's no audit record ffffffff."
    future = clinician.post("/api/incidents", json={"title": "x", "category": "answer", "description": "y",
                                                    "occurred_at": "2999-01-01T00:00:00"})
    assert "future" in future.json()["detail"]

    created = clinician.post("/api/incidents", json={
        "title": "Dose given for the wrong formulation", "category": "answer", "harm": "low",
        "description": "The answer quoted the tablet dose when the question was about the solution.",
        "audit_id": answer.audit_id}).json()["incident"]
    assert created["reference"].startswith("INC-") and created["status"] == "open"
    assert created["reported_by"] == "Dr Clin" and created["links"]["audit_id"] == answer.audit_id
    assert created["timeline"][0]["action"] == "reported"

    # A clinician sees only their own incidents and can't manage them.
    assert [i["id"] for i in clinician.get("/api/incidents").json()["incidents"]] == [created["id"]]
    other = client_for("other")
    assert other.get("/api/incidents").json()["incidents"] == []
    assert other.get(f"/api/incidents/{created['id']}").status_code == 404
    assert clinician.patch(f"/api/incidents/{created['id']}", json={"status": "closed"}).status_code == 403
    assert clinician.post(f"/api/incidents/{created['id']}/comments", json={"note": "Happened twice."}).status_code == 200

    listing = reviewer.get("/api/incidents").json()
    assert listing["can_manage"] and listing["counts"]["open"] == 1
    assert {p["name"] for p in listing["people"]} == {"Dr Rev"}

    started = reviewer.patch(f"/api/incidents/{created['id']}", json={
        "status": "investigating", "owner_id": users["reviewer"].id, "harm": "moderate"}).json()["incident"]
    assert started["owner"] == "Dr Rev" and started["harm"] == "moderate"
    assert "Investigation started." in started["timeline"][-1]["note"]
    refused = reviewer.patch(f"/api/incidents/{created['id']}", json={"status": "closed"})
    assert "root cause" in refused.json()["detail"]
    closed = reviewer.patch(f"/api/incidents/{created['id']}", json={
        "root_cause": "The two formulations share a section heading in the source.",
        "actions": "Source split into two sections; test question added.", "status": "closed"}).json()["incident"]
    assert closed["status"] == "closed" and closed["closed_at"]
    assert [e["action"] for e in closed["timeline"]] == ["reported", "comment", "updated", "updated"]
    assert reviewer.patch(f"/api/incidents/{created['id']}", json={"secret": 1}).status_code == 422

    rows = list(csv.reader(io.StringIO(reviewer.get("/api/incidents/export.csv").text)))
    assert rows[0][0] == "Reference" and rows[1][1] == "Dose given for the wrong formulation"
    assert clinician.get("/api/incidents/export.csv").status_code == 403


def test_regulator_deadlines_must_be_answered_before_closing(people):
    _, client_for = people
    reviewer = client_for("reviewer")
    breach = reviewer.post("/api/incidents", json={
        "title": "Export emailed to the wrong address", "category": "data_protection", "harm": "none",
        "description": "A usage export with user names was sent outside the organisation."}).json()["incident"]
    [deadline] = breach["deadlines"]
    assert deadline["kind"] == "data_protection" and not deadline["done"] and deadline["due_at"]
    severe = reviewer.post("/api/incidents", json={"title": "Missed interaction", "category": "patient_check",
                                                  "harm": "severe", "description": "..."}).json()["incident"]
    assert [d["kind"] for d in severe["deadlines"]] == ["device_regulator"]

    reviewer.patch(f"/api/incidents/{breach['id']}", json={"root_cause": "Wrong recipient.", "actions": "Recalled."})
    blocked = reviewer.patch(f"/api/incidents/{breach['id']}", json={"status": "closed"})
    assert "external report reference" in blocked.json()["detail"]
    done = reviewer.patch(f"/api/incidents/{breach['id']}", json={
        "external_reference": "Not reportable: no personal data of patients; recorded by the DPO.",
        "status": "closed"}).json()["incident"]
    assert done["status"] == "closed" and done["deadlines"][0]["done"]
    active = reviewer.get("/api/incidents?status=active").json()
    assert [i["title"] for i in active["incidents"]] == ["Missed interaction"] and active["serious_active"] == 1


def test_incident_text_is_encrypted_at_rest(people, monkeypatch):
    _, client_for = people
    monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", encryption.generate_key())
    encryption.reset()
    try:
        reviewer = client_for("reviewer")
        made = reviewer.post("/api/incidents", json={"title": "Wrong series analysed", "category": "imaging",
                                                    "description": "Model run on the previous study."}).json()["incident"]
        from sqlalchemy import text

        with db.engine().connect() as conn:
            stored = conn.execute(text("SELECT title, description FROM incidents WHERE id = :i"), {"i": made["id"]}).one()
        assert all(value.startswith("gcenc:v1:") for value in stored)
        assert incidents.get(made["id"])["title"] == "Wrong series analysed"
    finally:
        monkeypatch.undo()
        encryption.reset()
