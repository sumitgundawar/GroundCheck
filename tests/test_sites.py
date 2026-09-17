"""Sites: people at one hospital see only their hospital's questions,
reviews, incidents and imaging; group-wide people see all; site admins
manage only their own site's people."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, config, db, sites, sso  # noqa: E402

PASSWORD = "correct horse battery staple"
REFUSED = "What is the standard dose of Zyntrafen?"


@pytest.fixture()
def group(database, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    monkeypatch.setattr(config, "IMAGING_DIR", tmp_path / "imaging")
    from app.main import app

    north, south = sites.create("north", "North General"), sites.create("south", "South Community")
    people = {
        "group-admin": ("admin", None), "north-admin": ("admin", north["id"]),
        "north-rev": ("reviewer", north["id"]), "north-clin": ("clinician", north["id"]),
        "south-rev": ("reviewer", south["id"]), "south-clin": ("clinician", south["id"]),
    }
    for email, (role, site_id) in people.items():
        auth.create_user(f"{email}@example.org", PASSWORD, role=role, name=email, site_id=site_id)
    clients = {}

    def client(who: str) -> TestClient:
        if who not in clients:
            c = TestClient(app)
            assert c.post("/api/auth/login", json={"email": f"{who}@example.org", "password": PASSWORD}).status_code == 200
            clients[who] = c
        return clients[who]

    return north, south, client


def test_questions_reviews_usage_and_audit_stay_within_a_site(group):
    north, south, client = group
    north_answer = client("north-clin").post("/api/ask", json={"query": REFUSED}).json()
    south_answer = client("south-clin").post("/api/ask", json={"query": REFUSED}).json()
    assert north_answer["decision"] == south_answer["decision"] == "refuse"

    # The same question at two sites is two cases.
    north_cases = client("north-rev").get("/api/reviews").json()
    south_cases = client("south-rev").get("/api/reviews").json()
    assert len(north_cases["cases"]) == len(south_cases["cases"]) == 1
    assert north_cases["cases"][0]["id"] != south_cases["cases"][0]["id"]
    assert {r["name"] for r in north_cases["reviewers"]} == {"north-rev", "north-admin", "group-admin"}
    south_case = south_cases["cases"][0]["id"]
    assert client("north-rev").get(f"/api/reviews/{south_case}").json()["detail"] == "No such case."
    assert client("north-rev").post(f"/api/reviews/{south_case}/comment", json={"note": "x"}).status_code == 400
    assert len(client("group-admin").get("/api/reviews").json()["cases"]) == 2

    # A reviewer can't be assigned another site's case.
    south_rev = next(u["id"] for u in client("group-admin").get("/api/users").json()["users"] if u["name"] == "south-rev")
    refused = client("north-rev").post(f"/api/reviews/{north_cases['cases'][0]['id']}/assign", json={"user_id": south_rev})
    assert "at this site" in refused.json()["detail"]

    assert client("north-rev").get("/api/usage").json()["totals"]["questions"] == 1
    assert client("group-admin").get("/api/usage").json()["totals"]["questions"] == 2
    assert client("north-rev").get("/api/governance/report").json()["questions"]["total"] == 1

    assert [r["audit_id"] for r in client("north-rev").get("/api/audit").json()["recent"]] == [north_answer["audit_id"]]
    assert client("north-rev").get(f"/api/audit/{south_answer['audit_id']}").status_code == 404
    assert client("group-admin").get(f"/api/audit/{south_answer['audit_id']}").status_code == 200


def test_incidents_and_imaging_stay_within_a_site(group):
    from tests.test_imaging import ct_series

    north, south, client = group
    made = client("south-clin").post("/api/incidents", json={
        "title": "Wrong dose shown", "category": "answer", "description": "Seen at South."}).json()["incident"]
    assert client("south-rev").get("/api/incidents").json()["incidents"][0]["id"] == made["id"]
    assert client("north-rev").get("/api/incidents").json()["incidents"] == []
    assert client("north-rev").get(f"/api/incidents/{made['id']}").status_code == 404
    assert client("north-rev").patch(f"/api/incidents/{made['id']}", json={"harm": "low"}).status_code == 404
    assert len(client("group-admin").get("/api/incidents").json()["incidents"]) == 1
    assert "Wrong dose shown" not in client("north-rev").get("/api/incidents/export.csv").text

    files = [("files", (name, data, "application/dicom")) for name, data in ct_series()]
    added = client("north-clin").post("/api/imaging/upload", files=files).json()["added"][0]
    assert client("south-clin").get("/api/imaging").json()["series"] == []
    assert client("south-clin").get(f"/api/imaging/series/{added['id']}").status_code == 404
    assert client("south-clin").get(f"/api/imaging/series/{added['id']}/slices/0.png").status_code == 404
    assert "another site" in client("south-clin").post("/api/imaging/upload", files=files).json()["detail"]
    report = client("north-clin").post(f"/api/imaging/series/{added['id']}/reports", json={
        "findings": "", "impression": "Normal.", "agreement": "not_used", "sign": True}).json()["report"]
    assert client("south-clin").get(f"/api/imaging/reports/{report['id']}/sr.dcm").status_code == 404
    assert client("north-clin").get(f"/api/imaging/reports/{report['id']}/sr.dcm").status_code == 200


def test_site_admins_manage_only_their_site(group):
    north, south, client = group
    north_admin = client("north-admin")
    listed = north_admin.get("/api/users").json()
    assert {u["name"] for u in listed["users"]} == {"north-admin", "north-rev", "north-clin"}
    created = north_admin.post("/api/users", json={"email": "new@example.org", "password": PASSWORD,
                                                   "role": "clinician", "site_id": south["id"]}).json()["user"]
    assert created["site_id"] == north["id"]
    south_clin = next(u["id"] for u in client("group-admin").get("/api/users").json()["users"] if u["name"] == "south-clin")
    assert north_admin.patch(f"/api/users/{south_clin}", json={"role": "reviewer"}).status_code == 404
    assert north_admin.patch(f"/api/users/{created['id']}", json={"site_id": None}).status_code == 403
    assert north_admin.post("/api/sites", json={"key": "east", "name": "East"}).status_code == 403

    group_admin = client("group-admin")
    assert group_admin.post("/api/sites", json={"key": "East Wing!", "name": "East"}).status_code == 400
    east = group_admin.post("/api/sites", json={"key": "east", "name": "East Clinic"}).json()["site"]
    moved = group_admin.patch(f"/api/users/{created['id']}", json={"site_id": east["id"]}).json()["user"]
    assert moved["site_id"] == east["id"]
    assert group_admin.patch(f"/api/users/{created['id']}", json={"site_id": 999}).status_code == 400
    me = client("north-rev").get("/api/auth/me").json()
    assert me["user"]["site_name"] == "North General" and me["sites"] is True


def test_the_site_comes_from_the_identity_provider_when_configured(group, monkeypatch):
    north, _, _ = group
    monkeypatch.setattr(config, "OIDC_SITE_CLAIM", "site")
    monkeypatch.setattr(config, "OIDC_ALL_SITES_VALUE", "group")
    with db.session() as s:
        assert sso._site_from_claims(s, {"site": "north"}) == north["id"]
        assert sso._site_from_claims(s, {"site": ["north"]}) == north["id"]
        assert sso._site_from_claims(s, {"site": "group"}) is None
        for claims in ({}, {"site": "nowhere"}, {"site": ["north", "south"]}):
            with pytest.raises(sso.SsoError, match="site"):
                sso._site_from_claims(s, claims)


def test_a_move_between_sites_takes_effect(group):
    north, south, client = group
    from app import sites

    group_admin = client("group-admin")
    person = next(u["id"] for u in group_admin.get("/api/users").json()["users"] if u["name"] == "north-clin")
    assert sites.of_user(person) == north["id"]
    group_admin.patch(f"/api/users/{person}", json={"site_id": south["id"]})
    assert sites.of_user(person) == south["id"]     # not the remembered site
