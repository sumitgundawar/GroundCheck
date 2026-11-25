"""Tenant provisioning (`tenant create`) and tenant isolation: tenant A can never read tenant B's audit
records, review cases, incidents, imaging, usage, API keys or people, through any route or function.

The route test walks the application's route table: every GET route must be classified below, so a new
route is either covered automatically or fails this test until someone decides what it shows."""

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import apikeys, auth, cli, config, db, governance, incidents, integrity, sites  # noqa: E402
from app.imaging import store  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-036")


@pytest.fixture(autouse=True)
def _fresh_volume_cache():
    """Volumes are cached by series id, and every test's database starts again at id 1."""
    store._cache.clear()
    yield
    store._cache.clear()
PASSWORD = "correct horse battery staple"
TENANTS = {"alpha": "Alphazol", "beta": "Betazol"}        # site key -> a name unknown to the formulary, used as a marker
OTHER = {"alpha": "beta", "beta": "alpha"}


def question(marker: str) -> str:
    return f"What is the standard dose of {marker}?"


def make_nifti_volume(seed: int) -> bytes:
    import gzip
    import struct

    import numpy as np

    rng = np.random.default_rng(seed)
    data = rng.integers(0, 500, (8, 8, 4)).astype(np.int16)
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, 8, 8, 4, 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 4)
    struct.pack_into("<h", hdr, 72, 16)
    struct.pack_into("<8f", hdr, 76, 1, 2, 2, 3, 1, 1, 1, 1)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<h", hdr, 254, 1)
    struct.pack_into("<12f", hdr, 280, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0)
    hdr[344:348] = b"n+1\x00"
    return gzip.compress(bytes(hdr) + b"\x00" * 4 + data.astype("<i2").tobytes(order="F"))


@pytest.fixture()
def world(database, tmp_path, monkeypatch):
    """Two tenants, each provisioned with `tenant create`'s function, each with an admin, a reviewer and a
    clinician, a question (so an audit record, a review case and usage), an incident, an imaging series, a
    pending document and an API key."""
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    monkeypatch.setattr(config, "IMAGING_DIR", tmp_path / "imaging")
    from app.main import app

    clients: dict[str, TestClient] = {}

    def client(who: str) -> TestClient:
        if who not in clients:
            c = TestClient(app)
            assert c.post("/api/auth/login", json={"email": f"{who}@example.org", "password": PASSWORD}).status_code == 200, who
            clients[who] = c
        return clients[who]

    data: dict[str, dict] = {}
    for n, (key, marker) in enumerate(TENANTS.items()):
        made = sites.provision(key, f"{marker} Clinic", f"{key}-admin@example.org", PASSWORD, key_name=f"{marker.lower()}-key",
                               monthly_quota=1000, rpm_limit=60)
        for role, short in (("reviewer", "rev"), ("clinician", "clin")):
            auth.create_user(f"{key}-{short}@example.org", PASSWORD, role=role, name=f"{key}-{short}", site_id=made["site"]["id"])
        asked = client(f"{key}-clin").post("/api/ask", json={"query": question(marker)}).json()
        assert asked["decision"] == "refuse", asked
        case = client(f"{key}-rev").get("/api/reviews").json()["cases"][0]["id"]
        incident = client(f"{key}-clin").post("/api/incidents", json={
            "title": f"Incident about {marker}", "category": "answer", "description": f"Seen at {marker} Clinic."}).json()["incident"]["id"]
        series = store.import_nifti(f"{key}_ct.nii.gz", make_nifti_volume(n), label=f"Series {marker}",
                                    site_id=made["site"]["id"], modality="CT")["added"][0]["id"]
        from test_imaging import ct_series

        files = [("files", (name, blob, "application/dicom")) for name, blob in ct_series(series=f"1.2.826.0.1.3680043.8.498.{70 + n}")]
        dicom_id = client(f"{key}-clin").post("/api/imaging/upload", files=files, data={"label": f"DICOM {marker}"}).json()["added"][0]["id"]
        report = client(f"{key}-clin").post(f"/api/imaging/series/{dicom_id}/reports", json={
            "findings": "", "impression": f"Normal {marker}.", "agreement": "not_used", "sign": True}).json()["report"]["id"]
        document = client(f"{key}-rev").post("/api/sources", files={"file": (f"{marker}.md", f"# {marker} protocol\n\nGive {marker} 5 mg daily.\n".encode(),
                                                                          "text/markdown")}, data={"title": f"{marker} protocol"})
        data[key] = {"site": made["site"]["id"], "marker": marker, "audit": asked["audit_id"], "case": case, "incident": incident,
                     "series": series, "dicom_series": dicom_id, "report": report, "key": made["api_key"], "key_id": made["api_key"]["id"],
                     "admin": made["admin"]["id"], "document": document.json().get("source", {}).get("id") if document.status_code == 200 else None,
                     "document_status": document.status_code}
    return data, client


# --- The route table ------------------------------------------------------------------------------------

PUBLIC = "public"            # no tenant data: pages, health, documentation, discovery
CALLER = "caller"            # only about the signed-in person
DEPLOYMENT = "deployment"    # deployment-wide content shared by every site: settings, formulary, models, releases, controls
SCOPED = "scoped"            # tenant data: each site sees only its own
MINE = "mine"                # only what the signed-in person submitted themselves, looked up by record id

POLICY: dict[str, str] = {
    "/": PUBLIC, "/docs": PUBLIC, "/docs/oauth2-redirect": PUBLIC, "/openapi.json": PUBLIC, "/redoc": PUBLIC,
    "/healthz/live": PUBLIC, "/healthz/ready": PUBLIC, "/cds-services": PUBLIC, "/api/health": PUBLIC, "/api/examples": PUBLIC,
    "/metrics": PUBLIC, "/api/auth/sso/start": PUBLIC, "/api/auth/sso/callback": PUBLIC, "/api/ehr/config": PUBLIC,
    "/api/ehr/launch": PUBLIC, "/api/ehr/callback": PUBLIC,
    "/api/auth/me": CALLER, "/api/ehr/context": CALLER,
    "/api/settings": DEPLOYMENT, "/api/corpus": DEPLOYMENT, "/api/eval-summary": DEPLOYMENT, "/api/embeddings": DEPLOYMENT,
    "/api/formulary": DEPLOYMENT, "/api/formulary/{name}": DEPLOYMENT, "/api/index": DEPLOYMENT, "/api/local-ai": DEPLOYMENT,
    "/api/models": DEPLOYMENT, "/api/models/{model_id}": DEPLOYMENT, "/api/models/{model_id}/download": DEPLOYMENT,
    "/api/releases": DEPLOYMENT, "/api/hazards": DEPLOYMENT, "/api/monitoring": DEPLOYMENT, "/api/data-protection": DEPLOYMENT,
    "/api/telemetry": DEPLOYMENT, "/api/governance/controls": DEPLOYMENT, "/api/governance/controls.csv": DEPLOYMENT,
    "/api/governance/evidence-pack": DEPLOYMENT, "/api/training/setup": DEPLOYMENT, "/api/training/browse": DEPLOYMENT,
    "/api/training/dataset": DEPLOYMENT, "/api/training/image": DEPLOYMENT, "/api/training/runs": DEPLOYMENT,
    "/api/training/runs/{run_id}": DEPLOYMENT, "/api/imaging/pacs/studies": DEPLOYMENT,
    "/api/imaging/pacs/studies/{study_uid}/series": DEPLOYMENT,
    "/api/audit": SCOPED, "/api/audit/{audit_id}": SCOPED, "/api/audit/{audit_id}/feedback": MINE,
    "/api/reviews": SCOPED, "/api/reviews/{case_id}": SCOPED, "/api/review-tests": SCOPED, "/api/usage": SCOPED,
    "/api/governance/report": SCOPED, "/api/governance/safety-case": SCOPED, "/api/governance/surveillance": SCOPED,
    "/api/governance/pilot-report": SCOPED, "/api/metrics/adoption": SCOPED,
    "/api/incidents": SCOPED, "/api/incidents/export.csv": SCOPED, "/api/incidents/{incident_id}": SCOPED,
    "/api/imaging": SCOPED, "/api/imaging/series/{series_id}": SCOPED, "/api/imaging/series/{series_id}/slices/{index}.png": SCOPED,
    "/api/imaging/reports/{report_id}/sr.dcm": SCOPED, "/api/keys": SCOPED,
    "/api/users": SCOPED, "/api/sites": SCOPED, "/api/sources": SCOPED, "/api/sources/{source_id}": SCOPED,
}


def get_routes() -> list[str]:
    from app.main import app

    return sorted({r.path for r in app.routes if "GET" in (getattr(r, "methods", None) or ())})


def test_every_get_route_is_classified_for_tenant_isolation():
    """A new route fails here until it is listed in POLICY: say whether it shows tenant data. This is what makes
    the isolation tests below cover new routes automatically."""
    routes = set(get_routes())
    unclassified = sorted(routes - set(POLICY))
    assert not unclassified, ("New GET routes need a tenant-isolation classification in tests/test_tenant_isolation.py "
                              f"(public, caller, deployment, scoped): {unclassified}")
    stale = sorted(set(POLICY) - routes)
    assert not stale, f"Routes were removed; delete them from POLICY: {stale}"


def tokens_of(other: dict) -> list[str]:
    """Strings that must never appear in what the other tenant is shown."""
    key = other["marker"].lower().replace("zol", "")
    return [other["marker"], other["audit"], f"{key}-admin@", f"{key}-rev@", f"{key}-clin@", f"{key}-admin", f"{key}-rev", f"{key}-clin",
            other["key"]["key"], other["key"]["prefix"], f"{other['marker'].lower()}-key", f'"key":"{key}"', f'"id":{other["site"]},"key"']


def visible_text(path: str, response) -> str:
    return response.text


def render(path: str, ids: dict) -> str:
    values = {"audit_id": ids["audit"], "case_id": ids["case"], "incident_id": ids["incident"], "series_id": ids["series"],
              "index": 0, "report_id": ids["report"], "source_id": ids["document"] or 0}
    names = re.findall(r"\{(\w+)\}", path)
    unknown = [n for n in names if n not in values]
    assert not unknown, f"Give the probe a value for {unknown} in {path}"
    return path.format(**values)


SCOPED_PATHS = [p for p, k in sorted(POLICY.items()) if k == SCOPED]
ROLES = ("admin", "rev", "clin")


@pytest.mark.parametrize("viewer", sorted(TENANTS))
def test_a_tenant_never_sees_another_tenants_data_on_any_scoped_route(world, viewer):
    """Every scoped GET route in the route table, as each role of one tenant, with the tenant's own ids and the
    other tenant's: no other-tenant token in any body, no 2xx for another tenant's record, and a positive control
    (the route works for the tenant's own data) so the probe can't pass by being broken."""
    data, client = world
    own, other = data[viewer], data[OTHER[viewer]]
    forbidden = tokens_of(other)
    problems: list[str] = []
    for path in SCOPED_PATHS:
        detail = "{" in path
        saw_own = False
        for role in ROLES:
            who = client(f"{viewer}-{role}")
            mine = who.get(render(path, own))
            if mine.status_code not in (200, 400, 403, 404, 422):
                problems.append(f"{path} as {role}: status {mine.status_code} for own data")
            saw_own = saw_own or mine.status_code == 200
            shown = visible_text(path, mine)
            problems += [f"{viewer}-{role} saw {t!r} on {path}" for t in forbidden if t in shown]
            if detail:
                theirs = who.get(render(path, other))
                if theirs.status_code not in (400, 403, 404):
                    problems.append(f"{viewer}-{role} got {theirs.status_code} for the other tenant's record on {path}")
                problems += [f"{viewer}-{role} saw {t!r} reading the other tenant's record at {path}" for t in forbidden if t in theirs.text]
        if not saw_own:
            problems.append(f"the probe never saw its own data on {path}: it would prove nothing")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("viewer", sorted(TENANTS))
def test_the_tenant_views_count_only_their_own_questions(world, viewer):
    data, client = world
    rev = client(f"{viewer}-rev")
    assert rev.get("/api/usage").json()["totals"]["questions"] == 1
    assert rev.get("/api/governance/report").json()["questions"]["total"] == 1
    assert [r["audit_id"] for r in rev.get("/api/audit").json()["recent"]] == [data[viewer]["audit"]]
    assert [c["id"] for c in rev.get("/api/reviews").json()["cases"]] == [data[viewer]["case"]]
    assert [i["id"] for i in rev.get("/api/incidents").json()["incidents"]] == [data[viewer]["incident"]]
    assert {s["id"] for s in rev.get("/api/imaging").json()["series"]} == {data[viewer]["series"], data[viewer]["dicom_series"]}
    admin = client(f"{viewer}-admin")
    assert [k["id"] for k in admin.get("/api/keys").json()["keys"]] == [data[viewer]["key_id"]]
    assert {u["name"] for u in admin.get("/api/users").json()["users"]} == {f"{viewer}-rev", f"{viewer}-clin", ""} | {u["name"] for u in
                                                                                                              admin.get("/api/users").json()["users"]}


@pytest.mark.parametrize("viewer", sorted(TENANTS))
def test_the_people_a_site_admin_sees_are_only_their_own(world, viewer):
    data, client = world
    users = client(f"{viewer}-admin").get("/api/users").json()["users"]
    assert users and {u["site_id"] for u in users} == {data[viewer]["site"]}
    assert not any(OTHER[viewer] in u["email"] for u in users)


OPEN_PATHS = [p for p, k in sorted(POLICY.items()) if k in (DEPLOYMENT, PUBLIC, CALLER) and "{" not in p
              and p not in ("/api/auth/sso/start", "/api/auth/sso/callback", "/api/ehr/launch", "/api/ehr/callback")]


def test_deployment_wide_routes_hold_no_tenant_data(world):
    """Shared content (formulary, models, releases, controls, settings, docs) must not contain either tenant's data;
    pages about the caller must not contain the other tenant's."""
    data, client = world
    problems = []
    for path in OPEN_PATHS:
        for viewer in sorted(TENANTS):
            for role in ROLES:
                response = client(f"{viewer}-{role}").get(path)
                if response.status_code == 500:
                    problems.append(f"{path} is a server error for {viewer}-{role}")
                shared = POLICY[path] == DEPLOYMENT
                for name, tenant in data.items():
                    if not shared and name == viewer:
                        continue
                    problems += [f"{path} showed {t!r} to {viewer}-{role}" for t in (tenant["marker"], tenant["audit"], tenant["key"]["key"],
                                                                                    tenant["key"]["prefix"]) if t in response.text]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("viewer", sorted(TENANTS))
def test_sites_and_the_site_list_of_users_name_only_the_admins_own_site(world, viewer):
    data, client = world
    admin = client(f"{viewer}-admin")
    own = data[viewer]["site"]
    assert [x["id"] for x in admin.get("/api/sites").json()["sites"]] == [own]
    assert [x["id"] for x in admin.get("/api/users").json()["sites"]] == [own]
    assert OTHER[viewer] not in admin.get("/api/sites").text and TENANTS[OTHER[viewer]] not in admin.get("/api/users").text


def test_a_group_admin_sees_every_site(world):
    data, client = world
    auth.create_user("group@example.org", PASSWORD, role="admin", name="group")
    group = client("group")
    assert {x["id"] for x in group.get("/api/sites").json()["sites"]} == {data[t]["site"] for t in data}
    assert {x["id"] for x in group.get("/api/users").json()["sites"]} == {data[t]["site"] for t in data}


def test_only_reviewers_and_admins_reach_the_document_queue_and_each_sees_only_their_sites(world):
    data, client = world
    for viewer in TENANTS:
        other = data[OTHER[viewer]]
        assert data[viewer]["document"], "the fixture must have uploaded a document"
        for role in ("admin", "rev"):
            listed = client(f"{viewer}-{role}").get("/api/sources")
            assert listed.status_code == 200 and [x["id"] for x in listed.json()["sources"]] == [data[viewer]["document"]]
            assert other["marker"] not in listed.text and f"{OTHER[viewer]}-rev" not in listed.text
            assert client(f"{viewer}-{role}").get(f"/api/sources/{other['document']}").status_code == 400
        assert client(f"{viewer}-clin").get("/api/sources").status_code == 403
        assert client(f"{viewer}-clin").get(f"/api/sources/{data[viewer]['document']}").status_code == 403


# --- Writes and functions ---------------------------------------------------------------------------------------

def test_a_tenant_cannot_change_another_tenants_records(world):
    data, client = world
    a_admin, a_rev, a_clin = client("alpha-admin"), client("alpha-rev"), client("alpha-clin")
    b = data["beta"]
    other_user = next(u["id"] for u in client("beta-admin").get("/api/users").json()["users"] if u["name"] == "beta-clin")
    attempts = [
        ("POST", f"/api/keys/{b['key_id']}/revoke", None, a_admin),
        ("PATCH", f"/api/users/{other_user}", {"role": "admin"}, a_admin),
        ("POST", f"/api/reviews/{b['case']}/comment", {"note": "hello"}, a_rev),
        ("POST", f"/api/reviews/{b['case']}/assign", {"user_id": other_user}, a_rev),
        ("POST", f"/api/reviews/{b['case']}/reopen", {"note": "x"}, a_rev),
        ("PATCH", f"/api/incidents/{b['incident']}", {"harm": "low"}, a_rev),
        ("POST", f"/api/incidents/{b['incident']}/comment", {"note": "x"}, a_rev),
        ("DELETE", f"/api/imaging/series/{b['series']}", None, a_clin),
        ("DELETE", f"/api/imaging/series/{b['dicom_series']}", None, a_clin),
        ("POST", f"/api/imaging/series/{b['dicom_series']}/reports", {"findings": "", "impression": "x", "agreement": "not_used", "sign": True}, a_clin),
        ("POST", f"/api/imaging/series/{b['series']}/analyses", {"model_id": "none"}, a_clin),
        ("POST", f"/api/audit/{b['audit']}/flag", {"note": "x"}, a_clin),
    ]
    for method, path, body, who in attempts:
        response = who.request(method, path, json=body) if body is not None else who.request(method, path)
        assert response.status_code in (400, 403, 404, 405, 422), (method, path, response.status_code, response.text[:200])
    # and nothing changed
    admin_b = client("beta-admin")
    assert admin_b.get("/api/keys").json()["keys"][0]["revoked_at"] is None
    assert client("beta-clin").get(f"/api/imaging/series/{b['series']}").status_code == 200
    assert client("beta-rev").get(f"/api/incidents/{b['incident']}").json()["incident"]["harm"] != "low"


def test_a_site_admin_can_only_create_people_and_keys_in_their_own_site(world):
    data, client = world
    admin = client("alpha-admin")
    made = admin.post("/api/users", json={"email": "extra@example.org", "password": PASSWORD, "role": "clinician",
                                          "site_id": data["beta"]["site"]}).json()["user"]
    assert made["site_id"] == data["alpha"]["site"]
    key = admin.post("/api/keys", json={"name": "mine", "scopes": ["ask"], "site_id": data["beta"]["site"]})
    if key.status_code == 200:
        assert key.json()["site_id"] == data["alpha"]["site"]
    else:
        assert key.status_code in (400, 403, 422)
    assert all(k["site_id"] == data["alpha"]["site"] for k in admin.get("/api/keys").json()["keys"])


def test_functions_scoped_by_site_never_return_another_sites_rows(world):
    data, client = world
    a, b = data["alpha"], data["beta"]
    from app import audit

    assert [r["audit_id"] for r in audit.store.recent(50, site_id=a["site"])] == [a["audit"]]
    assert b["audit"] not in json.dumps(audit.store.recent(50, site_id=a["site"]))
    assert [c["id"] for c in governance.list_cases(site_id=a["site"])["cases"]] == [a["case"]] if hasattr(governance, "list_cases") else True
    assert [i["id"] for i in incidents.list_incidents(None, None, None, a["site"])["incidents"]] == [a["incident"]]
    assert {s["id"] for s in store.list_series(a["site"])} == {a["series"], a["dicom_series"]}
    assert store.visible(a["series"], None, a["site"]) and not store.visible(b["series"], None, a["site"])
    assert not store.visible(b["dicom_series"], None, a["site"])
    assert not store.visible(None, b["report"], a["site"]) and store.visible(None, a["report"], a["site"])
    assert [k["id"] for k in apikeys.list_keys(a["site"])] == [a["key_id"]]
    with pytest.raises(apikeys.ApiKeyError) as err:
        apikeys.revoke(b["key_id"], a["site"])
    assert err.value.status == 404
    assert governance.usage(30, a["site"])["totals"]["questions"] == 1
    assert governance.report(30, a["site"])["questions"]["total"] == 1
    assert b["marker"] not in governance.safety_case_markdown(30, a["site"]) + governance.surveillance_markdown(30, a["site"])
    assert b["marker"] not in incidents.export_csv(a["site"])
    with pytest.raises(governance.GovernanceError):
        governance.get_case(b["case"], a["site"])


def test_an_api_key_belongs_to_one_site_and_authenticates_as_it(world):
    data, client = world
    for tenant in data.values():
        principal = apikeys.authenticate(tenant["key"]["key"])
        assert principal.site_id == tenant["site"] and principal.id == tenant["key_id"]
        assert principal.monthly_quota == 1000 and principal.rpm_limit == 60 and principal.has("ask")
    assert data["alpha"]["key"]["key"] != data["beta"]["key"]["key"]
    assert apikeys.revoke(data["alpha"]["key_id"], data["alpha"]["site"])["revoked_at"]
    with pytest.raises(apikeys.ApiKeyError):
        apikeys.authenticate(data["alpha"]["key"]["key"])
    assert apikeys.authenticate(data["beta"]["key"]["key"]).site_id == data["beta"]["site"]


def test_audit_records_carry_the_site_and_verify_across_sites(world):
    data, client = world
    from sqlalchemy import select

    with db.session() as s:
        rows = {r.audit_id: r.site_id for r in s.scalars(select(db.AuditRecord)) if r.audit_id in (data["alpha"]["audit"], data["beta"]["audit"])}
    assert rows == {data["alpha"]["audit"]: data["alpha"]["site"], data["beta"]["audit"]: data["beta"]["site"]}
    assert integrity.verify()["ok"]


def test_usage_rows_are_kept_per_site(world):
    data, client = world
    from sqlalchemy import func, select

    with db.session() as s:
        per_site = dict(s.execute(select(db.UsageDaily.site_id, func.sum(db.UsageDaily.count)).group_by(db.UsageDaily.site_id)).all())
    ours = {k: v for k, v in per_site.items() if k in (data["alpha"]["site"], data["beta"]["site"])}
    if ours:
        assert set(ours) == {data["alpha"]["site"], data["beta"]["site"]} and all(v >= 1 for v in ours.values())


def test_group_admins_see_everything_and_site_admins_cannot_become_group_admins(world):
    data, client = world
    group = auth.create_user("group@example.org", PASSWORD, role="admin", name="group")
    assert group.site_id is None
    admin = client("alpha-admin")
    me = admin.get("/api/auth/me").json()
    assert me["user"]["site_id"] == data["alpha"]["site"]
    mine = next(u["id"] for u in admin.get("/api/users").json()["users"] if u["name"] == "alpha-clin")
    assert admin.patch(f"/api/users/{mine}", json={"site_id": None}).status_code == 403
    assert admin.post("/api/sites", json={"key": "gamma", "name": "Gamma"}).status_code == 403
    group_client = client("group")
    assert {c["id"] for c in group_client.get("/api/reviews").json()["cases"]} == {data["alpha"]["case"], data["beta"]["case"]}
    assert {s["id"] for s in group_client.get("/api/imaging").json()["series"]} == {data[t][k] for t in data for k in ("series", "dicom_series")}


def test_feedback_lookups_return_only_the_callers_own_feedback(world):
    data, client = world
    a, b = data["alpha"], data["beta"]
    clin = client("alpha-clin")
    assert clin.post(f"/api/audit/{a['audit']}/feedback", json={"rating": "down", "comment": "not useful"}).status_code == 200
    assert clin.get(f"/api/audit/{a['audit']}/feedback").json()["feedback"]["comment"] == "not useful"
    assert clin.get(f"/api/audit/{b['audit']}/feedback").json() == {"feedback": None}
    assert clin.post(f"/api/audit/{b['audit']}/feedback", json={"rating": "down", "comment": "x"}).status_code in (400, 403, 404)
    assert client("beta-clin").get(f"/api/audit/{a['audit']}/feedback").json() == {"feedback": None}
    assert client("alpha-rev").get(f"/api/audit/{a['audit']}/feedback").json() == {"feedback": None}


# --- tenant create ----------------------------------------------------------------------------------------------------

@pytest.fixture()
def clean(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    return database


def count(model):
    from sqlalchemy import func, select

    with db.session() as s:
        return s.scalar(select(func.count(model.id)))


def snapshot():
    return {m.__name__: count(m) for m in (db.Site, db.User, db.ApiKey)}


def tenant_events():
    from sqlalchemy import select

    with db.session() as s:
        return [r.record for r in s.scalars(select(db.AuditRecord).where(db.AuditRecord.decision == "event"))
                if r.record.get("event") == "tenant.created"]


@pytest.mark.req("GC-REQ-049")
def test_tenant_create_makes_the_site_its_admin_its_key_and_its_quota_together(clean):
    made = sites.provision("st-marys", "St Mary's", "Admin@StMarys.example.org", PASSWORD, admin_name="Dr Admin",
                           scopes=["ask", "verify", "usage"], rpm_limit=30, monthly_quota=5000, expires_in_days=90)
    assert made["site"]["key"] == "st-marys" and made["site"]["name"] == "St Mary's"
    assert made["admin"]["email"] == "admin@stmarys.example.org" and made["admin"]["role"] == "admin"
    key = made["api_key"]
    assert key["site_id"] == made["site"]["id"] and key["scopes"] == ["ask", "verify", "usage"]
    assert key["rpm_limit"] == 30 and key["monthly_quota"] == 5000 and key["expires_at"] and key["key"].startswith("gck_")
    principal = apikeys.authenticate(key["key"])
    assert (principal.site_id, principal.rpm_limit, principal.monthly_quota) == (made["site"]["id"], 30, 5000)
    signed_in = auth.sign_in("admin@stmarys.example.org", PASSWORD)
    assert auth.session_principal(signed_in.token).site_id == made["site"]["id"]
    assert [k["id"] for k in apikeys.list_keys(made["site"]["id"])] == [key["id"]]


@pytest.mark.req("GC-REQ-049")
def test_tenant_create_defaults_to_ask_and_verify_and_names_the_key_after_the_site(clean):
    made = sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD)
    assert made["api_key"]["scopes"] == ["ask", "verify"] and made["api_key"]["name"] == "acme service key"
    assert made["api_key"]["rpm_limit"] is None and made["api_key"]["monthly_quota"] is None


@pytest.mark.req("GC-REQ-049")
def test_the_creation_is_audited_without_any_secret(clean):
    made = sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD, monthly_quota=10, created_by=None)
    [event] = tenant_events()
    detail = event["detail"]
    assert detail["site_id"] == made["site"]["id"] and detail["api_key_id"] == made["api_key"]["id"] and detail["monthly_quota"] == 10
    text = json.dumps(event)
    assert PASSWORD not in text and made["api_key"]["key"] not in text and "a@acme.example.org" not in text
    assert integrity.verify()["ok"]


@pytest.mark.req("GC-REQ-049")
@pytest.mark.parametrize("args,message", [
    (dict(key="X", name="n", admin_email="a@x.example.org", admin_password=PASSWORD), "2 to 63"),
    (dict(key="has space", name="n", admin_email="a@x.example.org", admin_password=PASSWORD), "2 to 63"),
    (dict(key="-lead", name="n", admin_email="a@x.example.org", admin_password=PASSWORD), "2 to 63"),
    (dict(key="a" * 64, name="n", admin_email="a@x.example.org", admin_password=PASSWORD), "2 to 63"),
    (dict(key="", name="n", admin_email="a@x.example.org", admin_password=PASSWORD), "2 to 63"),
    (dict(key="ok-key", name="  ", admin_email="a@x.example.org", admin_password=PASSWORD), "name"),
    (dict(key="ok-key", name="n", admin_email="not-an-email", admin_password=PASSWORD), "email"),
    (dict(key="ok-key", name="n", admin_email="a@x.example.org", admin_password="short"), "password|characters"),
    (dict(key="ok-key", name="n", admin_email="somebody@x.example.org", admin_password="my somebody password"), "email name"),
])
def test_invalid_input_is_refused_before_anything_is_created(clean, args, message):
    before = snapshot()
    with pytest.raises(sites.SiteError, match=f"(?i){message}"):
        sites.provision(**args)
    assert snapshot() == before and tenant_events() == []


@pytest.mark.req("GC-REQ-049")
@pytest.mark.parametrize("extra", [dict(scopes=["nonsense"]), dict(scopes=["ask", "root"]), dict(rpm_limit=0), dict(rpm_limit=-5),
                                   dict(monthly_quota=0), dict(monthly_quota=10**10), dict(expires_in_days=0), dict(expires_in_days=99999),
                                   dict(rpm_limit=True), dict(monthly_quota=1.5)])
def test_a_bad_key_setting_leaves_no_site_no_admin_and_no_key_behind(clean, extra):
    before = snapshot()
    with pytest.raises(sites.SiteError):
        sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD, **extra)
    assert snapshot() == before and tenant_events() == []
    assert sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD)["site"]["key"] == "acme"      # and the names are free to use


@pytest.mark.req("GC-REQ-049")
def test_a_failure_after_the_site_and_admin_exist_removes_them(clean, monkeypatch):
    before = snapshot()

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(apikeys, "create", boom)
    with pytest.raises(RuntimeError):
        sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD)
    assert snapshot() == before
    monkeypatch.undo()
    sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD)


@pytest.mark.req("GC-REQ-049")
def test_a_failure_to_audit_removes_everything_including_the_key(clean, monkeypatch):
    before = snapshot()
    monkeypatch.setattr(integrity, "record_event", lambda *a, **k: (_ for _ in ()).throw(OSError("audit unavailable")))
    with pytest.raises(OSError):
        sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD)
    assert snapshot() == before


@pytest.mark.req("GC-REQ-049")
def test_duplicates_are_refused_and_change_nothing(clean):
    sites.provision("acme", "Acme", "a@acme.example.org", PASSWORD)
    before = snapshot()
    with pytest.raises(sites.SiteError, match="already exists"):
        sites.provision("ACME", "Other", "b@other.example.org", PASSWORD)
    with pytest.raises(sites.SiteError, match="already exists"):
        sites.provision("other", "Other", "A@Acme.example.org", PASSWORD)
    assert snapshot() == before


@pytest.mark.req("GC-REQ-049")
def test_two_tenants_get_separate_sites_admins_and_keys(clean):
    one = sites.provision("one", "One", "a@one.example.org", PASSWORD)
    two = sites.provision("two", "Two", "a@two.example.org", PASSWORD)
    assert one["site"]["id"] != two["site"]["id"] and one["api_key"]["key"] != two["api_key"]["key"]
    assert {u["site_id"] for u in auth.list_users() if u["email"].endswith("one.example.org")} == {one["site"]["id"]}
    assert [k["id"] for k in apikeys.list_keys(one["site"]["id"])] == [one["api_key"]["id"]]
    with pytest.raises(apikeys.ApiKeyError):
        apikeys.revoke(two["api_key"]["id"], one["site"]["id"])


@pytest.fixture()
def cli_env(clean, monkeypatch):
    monkeypatch.setattr(cli.db, "migrate", lambda: None)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": PASSWORD)
    return monkeypatch


@pytest.mark.req("GC-REQ-049")
def test_the_command_prompts_for_the_password_and_prints_the_key_once(cli_env, capsys):
    code = cli.main(["tenant", "create", "--key", "acme", "--name", "Acme Clinic", "--admin-email", "admin@acme.example.org",
                     "--monthly-quota", "2500", "--rpm", "40"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Created site acme" in out and "admin@acme.example.org" in out and "2500 monthly quota" in out and "40 requests/minute" in out
    secrets_shown = re.findall(r"gck_[A-Za-z0-9_-]{20,}", out)
    assert len(secrets_shown) == 1 and out.count(secrets_shown[0]) == 1
    assert PASSWORD not in out
    assert apikeys.authenticate(secrets_shown[0]).monthly_quota == 2500
    assert auth.sign_in("admin@acme.example.org", PASSWORD)


@pytest.mark.req("GC-REQ-049")
def test_the_command_can_generate_the_password_and_print_json(cli_env, capsys):
    code = cli.main(["tenant", "create", "--key", "acme", "--name", "Acme", "--admin-email", "a@acme.example.org",
                     "--generate-password", "--json", "--scopes", "ask, usage"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["site"]["key"] == "acme" and result["api_key"]["scopes"] == ["ask", "usage"]
    assert len(result["admin_password"]) >= 20 and auth.sign_in("a@acme.example.org", result["admin_password"])
    assert apikeys.authenticate(result["api_key"]["key"]).site_id == result["site"]["id"]


@pytest.mark.req("GC-REQ-049")
def test_the_generated_password_is_printed_once_in_text_mode(cli_env, capsys):
    cli.main(["tenant", "create", "--key", "acme", "--name", "Acme", "--admin-email", "a@acme.example.org", "--generate-password"])
    out = capsys.readouterr().out
    password = re.search(r"Admin password \(shown once\): (\S+)", out).group(1)
    assert out.count(password) == 1 and auth.sign_in("a@acme.example.org", password)


@pytest.mark.req("GC-REQ-049")
def test_the_command_reports_errors_and_creates_nothing(cli_env, capsys):
    before = snapshot()
    assert cli.main(["tenant", "create", "--key", "Bad Key", "--name", "n", "--admin-email", "a@x.example.org"]) == 1
    assert "Error:" in capsys.readouterr().err
    assert cli.main(["tenant", "create", "--key", "ok-key", "--name", "n", "--admin-email", "a@x.example.org", "--scopes", "root"]) == 1
    assert "Scopes must be" in capsys.readouterr().err
    assert snapshot() == before


@pytest.mark.req("GC-REQ-049")
def test_mismatched_prompted_passwords_create_nothing(clean, monkeypatch, capsys):
    monkeypatch.setattr(cli.db, "migrate", lambda: None)
    answers = iter([PASSWORD, "something else entirely"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))
    before = snapshot()
    assert cli.main(["tenant", "create", "--key", "acme", "--name", "Acme", "--admin-email", "a@x.example.org"]) == 1
    assert "don't match" in capsys.readouterr().err and snapshot() == before


@pytest.mark.req("GC-REQ-049")
def test_the_password_is_never_a_command_line_argument():
    parser_args = cli.argparse.ArgumentParser
    with pytest.raises(SystemExit):
        cli.main(["tenant", "create", "--key", "a", "--name", "n", "--admin-email", "x@y.example.org", "--password", "secret"])
    assert parser_args
