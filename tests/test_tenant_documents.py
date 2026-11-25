"""Documents, approval queues and retrieval per site: a site's own documents are answered from, listed and
approved only at that site; documents shared by the group (no site) are visible everywhere; nothing a site
uploads is ever cited to, listed for, or counted for another site."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import apikeys, auth, config, db, knowledge, pipeline, releases, retrieval, sites  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-036")

PASSWORD = "correct horse battery staple"


def protocol(name: str, minutes: int = 60) -> bytes:
    return (f"# {name} Fever Protocol\n\n## First Hour Actions\n\nGive the first dose of {name} within {minutes} minutes of "
            f"recognition.\n").encode()


def ask_text(name: str) -> str:
    return f"When should the first dose of {name} be given?"


@pytest.fixture()
def world(database, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "INCLUDE_DEMO_CORPUS", False)
    monkeypatch.setattr(config, "VECTOR_STORE", "local")
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    monkeypatch.setattr(knowledge, "rebuild_index_in_background", lambda *a, **k: None)
    pipeline.clear_answer_cache()
    from app.main import app

    made = {}
    for key in ("alpha", "beta"):
        site = sites.provision(key, f"{key.title()} Clinic", f"{key}-admin@example.org", PASSWORD, scopes=["ask", "verify"])
        for role, short in (("reviewer", "rev"), ("clinician", "clin"), ("admin", "boss")):
            auth.create_user(f"{key}-{short}@example.org", PASSWORD, role=role, name=f"{key}-{short}", site_id=site["site"]["id"])
        made[key] = site
    auth.create_user("group-admin@example.org", PASSWORD, role="admin", name="group")
    auth.create_user("group-clin@example.org", PASSWORD, role="clinician", name="groupclin")
    clients: dict[str, TestClient] = {}

    def client(who: str) -> TestClient:
        if who not in clients:
            c = TestClient(app)
            assert c.post("/api/auth/login", json={"email": f"{who}@example.org", "password": PASSWORD}).status_code == 200, who
            clients[who] = c
        return clients[who]

    yield made, client
    monkeypatch.undo()
    pipeline.clear_answer_cache()
    db.reset_engine()
    retrieval.load_index()


def upload(client, who, name, data=None, **form):
    return client(who).post("/api/sources", files={"file": (f"{name}.md", data or protocol(name), "text/markdown")},
                            data={"title": f"{name} Fever Protocol", **form})


def approve(client, who, source_id):
    response = client(who).post(f"/api/sources/{source_id}/approve", json={"note": ""})
    assert response.status_code == 200, response.text
    return response


def publish():
    knowledge.rebuild_index("Approved Alphazol Fever Protocol, Betazol Fever Protocol and Gammazol Fever Protocol")


@pytest.fixture()
def docs(world):
    """alpha, beta and the group each have one approved document, and the index holds all three."""
    made, client = world
    ids = {}
    for name, uploader, approver in (("Alphazol", "alpha-rev", "alpha-boss"), ("Betazol", "beta-rev", "beta-boss"),
                                     ("Gammazol", "group-admin", "group-admin")):
        response = upload(client, uploader, name)
        assert response.status_code == 200, response.text
        ids[name] = response.json()["source"]["id"]
        if approver == uploader:
            knowledge.review(ids[name], "approve", None, allow_self_approval=True)
        else:
            approve(client, approver, ids[name])
    publish()
    return made, client, ids


# --- Migration ------------------------------------------------------------------------------------------

def test_migration_0014_adds_the_site_column_and_existing_documents_are_shared(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'm.db'}")
    db.reset_engine()
    cfg = Config(str(config.ROOT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(config.ROOT_DIR / "migrations"))
    try:
        with db.engine().begin() as c:
            cfg.attributes["connection"] = c
            command.upgrade(cfg, "0013")
            assert "site_id" not in {x["name"] for x in inspect(c).get_columns("sources")}
            c.execute(text("INSERT INTO sources (document_key, version, title, status, filename, media_type, content_sha256, owner, "
                           "uploaded_at, review_note) VALUES ('old', 1, 'Old', 'approved', 'o.md', 'text/markdown', 'x', '', "
                           "'2026-01-01 00:00:00', '')"))
            command.upgrade(cfg, "head")
            assert "site_id" in {x["name"] for x in inspect(c).get_columns("sources")}
            assert c.execute(text("SELECT site_id FROM sources WHERE document_key='old'")).scalar() is None
            assert "ix_sources_site_id" in {i["name"] for i in inspect(c).get_indexes("sources")}
            command.downgrade(cfg, "0013")
            assert "site_id" not in {x["name"] for x in inspect(c).get_columns("sources")}
            assert c.execute(text("SELECT count(*) FROM sources")).scalar() == 1
            command.upgrade(cfg, "head")
            assert c.execute(text("SELECT site_id FROM sources WHERE document_key='old'")).scalar() is None
    finally:
        db.reset_engine()


# --- Upload, list, read ----------------------------------------------------------------------------------

def test_a_document_belongs_to_the_site_of_whoever_uploads_it(world):
    made, client = world
    a, b = made["alpha"]["site"]["id"], made["beta"]["site"]["id"]
    assert upload(client, "alpha-rev", "Alphazol").json()["source"]["site_id"] == a
    assert upload(client, "alpha-boss", "Alphazol2").json()["source"]["site_id"] == a
    # A site user cannot upload into another site, whatever the form says.
    assert upload(client, "alpha-rev", "Sneaky", site_id=str(b)).json()["source"]["site_id"] == a
    # The group can upload for a named site, or for everyone.
    assert upload(client, "group-admin", "Betazol", site_id=str(b)).json()["source"]["site_id"] == b
    assert upload(client, "group-admin", "Shared").json()["source"]["site_id"] is None
    assert upload(client, "group-admin", "Nowhere", site_id="9999").status_code == 400
    assert upload(client, "group-admin", "Junk", site_id="abc").status_code == 400


@pytest.mark.parametrize("viewer", ["alpha", "beta"])
def test_each_site_lists_its_own_documents_and_the_groups(docs, viewer):
    made, client, ids = docs
    other = "beta" if viewer == "alpha" else "alpha"
    mine, theirs = f"{viewer.title()}zol", f"{other.title()}zol"
    for role in ("rev", "boss"):
        listed = client(f"{viewer}-{role}").get("/api/sources").json()["sources"]
        assert {s["title"] for s in listed} == {f"{mine} Fever Protocol", "Gammazol Fever Protocol"}
        assert ids[theirs] not in {s["id"] for s in listed} and theirs not in str(listed)
        assert client(f"{viewer}-{role}").get(f"/api/sources/{ids[mine]}").status_code == 200
        assert client(f"{viewer}-{role}").get(f"/api/sources/{ids['Gammazol']}").status_code == 200
        gone = client(f"{viewer}-{role}").get(f"/api/sources/{ids[theirs]}")
        assert gone.status_code == 400 and theirs not in gone.text
    assert client(f"{viewer}-clin").get("/api/sources").status_code == 403


def test_the_group_lists_every_document(docs):
    made, client, ids = docs
    listed = client("group-admin").get("/api/sources").json()["sources"]
    assert {s["id"] for s in listed} == set(ids.values())
    assert {s["site_id"] for s in listed} == {made["alpha"]["site"]["id"], made["beta"]["site"]["id"], None}


def test_a_missing_document_and_another_sites_document_look_the_same(docs):
    made, client, ids = docs
    missing = client("alpha-rev").get("/api/sources/987654")
    hidden = client("alpha-rev").get(f"/api/sources/{ids['Betazol']}")
    assert (missing.status_code, missing.json()) == (hidden.status_code, hidden.json())


# --- Approval queues -------------------------------------------------------------------------------------

@pytest.mark.parametrize("action", ["approve", "reject", "retire", "evaluate"])
def test_a_site_admin_cannot_decide_or_evaluate_another_sites_document(docs, action):
    made, client, ids = docs
    pending = upload(client, "beta-rev", "Pending").json()["source"]["id"]
    for target in (ids["Betazol"], pending):
        before = knowledge.get_source(target)["status"]
        response = client("alpha-boss").post(f"/api/sources/{target}/{action}", json={"note": "x"})
        assert response.status_code == 400 and "Betazol" not in response.text and "Pending" not in response.text
        assert knowledge.get_source(target)["status"] == before


@pytest.mark.parametrize("action", ["approve", "reject", "retire", "evaluate"])
def test_a_site_admin_cannot_decide_the_groups_shared_documents(docs, action):
    made, client, ids = docs
    shared_pending = upload(client, "group-admin", "Sharedpending").json()["source"]["id"]
    for target in (ids["Gammazol"], shared_pending):
        before = knowledge.get_source(target)["status"]
        assert client("alpha-boss").post(f"/api/sources/{target}/{action}", json={"note": "x"}).status_code == 400
        assert knowledge.get_source(target)["status"] == before


def test_reviewers_still_cannot_decide_documents(docs):
    made, client, ids = docs
    pending = upload(client, "alpha-rev", "Pend").json()["source"]["id"]
    assert client("alpha-rev").post(f"/api/sources/{pending}/approve", json={"note": ""}).status_code == 403


def test_the_group_decides_any_documents_and_a_site_admin_their_own(docs):
    made, client, ids = docs
    own = upload(client, "alpha-rev", "Own").json()["source"]["id"]
    assert client("alpha-boss").post(f"/api/sources/{own}/reject", json={"note": "no"}).json()["source"]["status"] == "rejected"
    theirs = upload(client, "beta-rev", "Theirs").json()["source"]["id"]
    assert client("group-admin").post(f"/api/sources/{theirs}/approve", json={"note": ""}).json()["source"]["status"] == "approved"
    shared = upload(client, "group-admin", "Sharednew").json()["source"]["id"]
    assert client("group-admin").post(f"/api/sources/{shared}/reject", json={"note": "no"}).json()["source"]["status"] == "rejected"


def test_same_title_at_two_sites_has_independent_versions_duplicates_and_approvals(world):
    made, client = world
    a = upload(client, "alpha-rev", "Shared", protocol("Alphazol")).json()["source"]
    b = upload(client, "beta-rev", "Shared", protocol("Alphazol")).json()["source"]    # the very same file, at another site
    assert (a["version"], b["version"]) == (1, 1) and a["document_key"] == b["document_key"]
    again = upload(client, "alpha-rev", "Shared", protocol("Alphazol"))
    assert again.status_code == 400 and "already version 1" in again.json()["detail"]
    approve(client, "alpha-boss", a["id"])
    approve(client, "beta-boss", b["id"])
    a2 = upload(client, "alpha-rev", "Shared", protocol("Alphazol", 30)).json()["source"]
    assert a2["version"] == 2
    approve(client, "alpha-boss", a2["id"])
    status = {s["id"]: s["status"] for s in client("group-admin").get("/api/sources").json()["sources"]}
    assert status == {a["id"]: "retired", a2["id"]: "approved", b["id"]: "approved"}
    b2 = upload(client, "beta-rev", "Shared", protocol("Alphazol", 20)).json()["source"]
    assert b2["version"] == 2


def test_an_exact_duplicate_does_not_reveal_that_another_site_uploaded_it(world):
    made, client = world
    upload(client, "alpha-rev", "Secretdoc", protocol("Alphazol"))
    assert upload(client, "beta-rev", "Secretdoc", protocol("Alphazol")).status_code == 200


# --- Answers ---------------------------------------------------------------------------------------------

def cited(response):
    return {sid for claim in response.claims for sid in claim.source_ids}


def test_a_site_is_answered_from_its_own_documents_and_the_groups_only(docs):
    made, client, ids = docs
    for site, name, other in (("alpha", "Alphazol", "Betazol"), ("beta", "Betazol", "Alphazol")):
        who = made[site]["site"]["id"]
        with retrieval.site_scope(who):
            own = pipeline.run(ask_text(name), client_id="t", user_id=None)
            shared = pipeline.run(ask_text("Gammazol"), client_id="t", user_id=None)
            theirs = pipeline.run(ask_text(other), client_id="t", user_id=None)
        assert own.decision == "answer" and any(s.startswith(f"DOC{ids[name]}-") for s in cited(own))
        assert shared.decision == "answer" and any(s.startswith(f"DOC{ids['Gammazol']}-") for s in cited(shared))
        assert theirs.decision == "refuse" and cited(theirs) == set()
        assert other not in theirs.answer_text and not any(other in str(s) for s in theirs.sources)


def test_someone_with_no_site_is_answered_from_shared_documents_only(docs):
    made, client, ids = docs
    with retrieval.site_scope(None):
        assert pipeline.run(ask_text("Gammazol"), client_id="t").decision == "answer"
        assert pipeline.run(ask_text("Alphazol"), client_id="t").decision == "refuse"
        assert pipeline.run(ask_text("Betazol"), client_id="t").decision == "refuse"


def test_a_search_outside_any_scope_sees_only_shared_documents(docs):
    made, client, ids = docs
    hits = retrieval.search("first dose within minutes of recognition", k=10)
    assert {r.get("site_id") for r, _ in hits} == {None}


def test_over_http_each_roles_answers_are_scoped(docs):
    made, client, ids = docs
    for who, name, other in (("alpha-clin", "Alphazol", "Betazol"), ("beta-clin", "Betazol", "Alphazol")):
        mine = client(who).post("/api/ask", json={"query": ask_text(name)}).json()
        assert mine["decision"] == "answer", mine
        theirs = client(who).post("/api/ask", json={"query": ask_text(other)})
        assert theirs.status_code == 200 and theirs.json()["decision"] == "refuse"
        assert not any(x["id"].startswith(f"DOC{ids[other]}-") for x in theirs.json()["sources"])
        assert other not in str([x["title"] for x in theirs.json()["sources"]]) and "claims" in theirs.json()
    assert client("group-clin").post("/api/ask", json={"query": ask_text("Alphazol")}).json()["decision"] == "refuse"
    assert client("group-clin").post("/api/ask", json={"query": ask_text("Gammazol")}).json()["decision"] == "answer"


def test_an_answer_cached_for_one_site_is_never_served_to_another(docs):
    made, client, ids = docs
    config_before = config.ANSWER_CACHE_SECONDS
    config.ANSWER_CACHE_SECONDS = 600
    try:
        pipeline.clear_answer_cache()
        question = ask_text("Alphazol")
        first = client("alpha-clin").post("/api/ask", json={"query": question}).json()
        again = client("alpha-clin").post("/api/ask", json={"query": question}).json()
        assert first["decision"] == again["decision"] == "answer"
        assert any(t["name"] == "remembered answer" for t in again["trace"])
        other = client("beta-clin").post("/api/ask", json={"query": question}).json()
        assert other["decision"] == "refuse" and not any(t["name"] == "remembered answer" for t in other["trace"])
        assert client("group-clin").post("/api/ask", json={"query": question}).json()["decision"] == "refuse"
    finally:
        config.ANSWER_CACHE_SECONDS = config_before
        pipeline.clear_answer_cache()


def test_a_refusal_for_one_site_is_not_reused_for_a_site_that_can_answer(docs):
    made, client, ids = docs
    before = config.ANSWER_CACHE_SECONDS
    config.ANSWER_CACHE_SECONDS = 600
    try:
        pipeline.clear_answer_cache()
        question = ask_text("Betazol")
        assert client("alpha-clin").post("/api/ask", json={"query": question}).json()["decision"] == "refuse"
        assert client("beta-clin").post("/api/ask", json={"query": question}).json()["decision"] == "answer"
    finally:
        config.ANSWER_CACHE_SECONDS = before
        pipeline.clear_answer_cache()


def test_the_vocabulary_a_site_can_see_is_its_own(docs):
    made, client, ids = docs
    a, b = made["alpha"]["site"]["id"], made["beta"]["site"]["id"]
    with retrieval.site_scope(a):
        assert retrieval.is_known_term("alphazol") and not retrieval.is_known_term("betazol")
        assert retrieval.document_frequency("alphazol") > 0 and retrieval.document_frequency("betazol") == 0
        assert retrieval.record_for(f"DOC{ids['Betazol']}-001") is None and retrieval.record_for(f"DOC{ids['Alphazol']}-001")
        assert retrieval.corpus_text_for(f"DOC{ids['Betazol']}-001") is None
        assert "betazol" not in retrieval.topics_mentioned("Betazol dose") and "gammazol" in retrieval.topics_mentioned("Gammazol dose")
        assert not [r for r in retrieval.all_metadata() if r.get("site_id") not in (None, a)]
        assert retrieval.corpus_size() == 2
    with retrieval.site_scope(b):
        assert retrieval.is_known_term("betazol") and not retrieval.is_known_term("alphazol")
    with retrieval.site_scope(None):
        assert not retrieval.is_known_term("alphazol") and not retrieval.is_known_term("betazol") and retrieval.is_known_term("gammazol")
        assert retrieval.corpus_size() == 1
    assert retrieval.passage_count(None) == 3 and retrieval.passage_count(a) == 2 and retrieval.passage_count(b) == 2


def test_the_cache_key_changes_with_the_site(docs):
    made, client, ids = docs
    versions = set()
    for scope in (made["alpha"]["site"]["id"], made["beta"]["site"]["id"], None):
        with retrieval.site_scope(scope):
            versions.add(retrieval.index_version())
    assert len(versions) == 3


def test_without_site_documents_the_index_is_one_shared_view(world):
    made, client = world
    shared = upload(client, "group-admin", "Gammazol").json()["source"]["id"]
    knowledge.review(shared, "approve", None, allow_self_approval=True)
    publish()
    versions = set()
    for scope in (made["alpha"]["site"]["id"], None):
        with retrieval.site_scope(scope):
            versions.add(retrieval.index_version())
            assert pipeline.run(ask_text("Gammazol"), client_id="t").decision == "answer"
    assert len(versions) == 1


def test_a_document_evaluation_runs_in_its_own_sites_scope(docs):
    made, client, ids = docs
    result = client("alpha-boss").post(f"/api/sources/{ids['Alphazol']}/evaluate", json={})
    assert result.status_code == 200 and result.json()["evaluation"]["unsafe_answers"] == 0
    assert result.json()["evaluation"]["passed"] == result.json()["evaluation"]["total"] > 0
    shared = client("group-admin").post(f"/api/sources/{ids['Gammazol']}/evaluate", json={})
    assert shared.status_code == 200 and shared.json()["evaluation"]["passed"] == shared.json()["evaluation"]["total"]


def test_a_release_check_does_not_search_site_documents(docs):
    made, client, ids = docs
    live = {r["id"] for r in retrieval.all_metadata()}
    assert live == {"DOC%d-001" % ids["Gammazol"]}
    assert len(retrieval._state().metadata) == 3


# --- API keys ----------------------------------------------------------------------------------------------

def test_api_keys_answer_and_verify_from_their_sites_documents_only(docs):
    made, client, ids = docs
    from fastapi.testclient import TestClient
    from app.main import app

    raw = TestClient(app)

    def call(key, path, body):
        return raw.post(f"/api/v1/{path}", json=body, headers={"Authorization": f"Bearer {key}"})

    ka, kb = made["alpha"]["api_key"]["key"], made["beta"]["api_key"]["key"]
    assert call(ka, "ask", {"question": ask_text("Alphazol")}).json()["decision"] == "answer"
    assert call(kb, "ask", {"question": ask_text("Alphazol")}).json()["decision"] == "refuse"
    assert call(kb, "ask", {"question": ask_text("Betazol")}).json()["decision"] == "answer"
    chunk = f"DOC{ids['Alphazol']}-001"
    ok = call(ka, "verify", {"question": ask_text("Alphazol"), "answer": "Give the first dose of Alphazol within 60 minutes of recognition.",
                             "sources": [chunk]})
    assert ok.status_code == 200, ok.text
    blocked = call(kb, "verify", {"question": ask_text("Alphazol"), "answer": "Give the first dose of Alphazol within 60 minutes of recognition.",
                                  "sources": [chunk]})
    assert blocked.status_code == 422 and blocked.json()["error"]["code"] == "unknown_source"
    free = call(kb, "verify", {"question": ask_text("Alphazol"), "answer": "Give the first dose of Alphazol within 60 minutes of recognition."})
    assert free.status_code == 200 and free.json()["decision"] != "pass"


# --- Releases and other deployment-wide actions -----------------------------------------------------------------

def test_releases_do_not_name_another_sites_documents_or_people(docs):
    made, client, ids = docs
    listing = {w: client(w).get("/api/releases") for w in ("alpha-rev", "alpha-boss", "beta-rev", "group-admin")}
    assert all(r.status_code == 200 for r in listing.values())
    for who, other in (("alpha-rev", "Betazol"), ("alpha-boss", "Betazol"), ("beta-rev", "Alphazol")):
        body = listing[who].text
        assert other not in body and "group-admin" not in body and "GroundCheckHealth" not in body
        assert not any(d["title"].startswith(other) for r in listing[who].json()["releases"] for d in r["documents"])
        assert all(r["created_by"] == "" for r in listing[who].json()["releases"])
    group = listing["group-admin"].json()["releases"]
    titles = {d["title"] for r in group for d in r["documents"]}
    assert {"Alphazol Fever Protocol", "Betazol Fever Protocol", "Gammazol Fever Protocol"} <= titles
    mine = listing["alpha-rev"].json()["releases"]
    assert any(d["title"] == "Alphazol Fever Protocol" for r in mine for d in r["documents"])
    assert any(d["title"] == "Gammazol Fever Protocol" for r in mine for d in r["documents"])


def test_a_release_reason_naming_a_hidden_document_is_withheld(docs):
    made, client, ids = docs
    out = releases.list_releases(made["alpha"]["site"]["id"], everything=False)
    assert all("Betazol" not in r["reason"] for r in out["releases"])
    full = releases.list_releases()
    assert any("Betazol" in r["reason"] for r in full["releases"])


def test_a_site_admin_cannot_change_group_wide_settings(docs):
    made, client, ids = docs
    boss = client("alpha-boss")
    hazard = {"title": "t", "severity": 1, "likelihood": 1, "residual_severity": 1, "residual_likelihood": 1}
    attempts = [("POST", "/api/releases/1/promote", None), ("POST", "/api/releases/rollback", None),
                ("POST", "/api/retention/run", None), ("POST", "/api/local-ai/select", {"model": None}),
                ("POST", "/api/local-ai/pull", {"model": "x"}), ("POST", "/api/hazards", hazard),
                ("PUT", "/api/hazards/1", hazard), ("POST", "/api/training/runs", {"name": "n", "dataset": "d", "device": "cpu"}),
                ("POST", "/api/training/runs/1/cancel", None), ("POST", "/api/training/runs/1/resume", None),
                ("DELETE", "/api/models/none", None), ("PUT", "/api/models/none/imaging", {"modality": "CT"}),
                ("DELETE", "/api/models/none/imaging", None)]
    for method, path, body in attempts:
        response = boss.request(method, path, json=body) if body is not None else boss.request(method, path)
        assert response.status_code == 403 and "every site" in response.text.lower(), (method, path, response.status_code, response.text[:120])


def test_the_audit_chain_covers_every_site_so_only_the_group_verifies_it(docs):
    made, client, ids = docs
    for who in ("alpha-rev", "alpha-boss", "beta-rev"):
        assert client(who).post("/api/audit/verify").status_code == 403
    assert client("group-admin").post("/api/audit/verify").status_code == 200


def test_index_counts_cover_only_what_the_caller_may_search(docs):
    made, client, ids = docs
    assert client("alpha-rev").get("/api/index").json()["passages"] == 2
    assert client("beta-boss").get("/api/index").json()["passages"] == 2
    assert client("group-admin").get("/api/index").json()["passages"] == 3


def test_retiring_a_sites_document_removes_it_for_that_site_only(docs):
    made, client, ids = docs
    assert client("alpha-boss").post(f"/api/sources/{ids['Alphazol']}/retire", json={"note": ""}).status_code == 200
    publish()
    with retrieval.site_scope(made["alpha"]["site"]["id"]):
        assert pipeline.run(ask_text("Alphazol"), client_id="t").decision == "refuse"
        assert pipeline.run(ask_text("Gammazol"), client_id="t").decision == "answer"
    with retrieval.site_scope(made["beta"]["site"]["id"]):
        assert pipeline.run(ask_text("Betazol"), client_id="t").decision == "answer"


def test_a_group_document_moved_to_nobody_stays_shared_after_a_site_is_removed(docs):
    made, client, ids = docs
    with db.session() as s:
        s.execute(text("UPDATE sources SET site_id = NULL WHERE id = :i"), {"i": ids["Alphazol"]})
    publish()
    with retrieval.site_scope(made["beta"]["site"]["id"]):
        assert pipeline.run(ask_text("Alphazol"), client_id="t").decision == "answer"


def test_an_answer_from_another_sites_key_cannot_be_saved_to_the_record(docs, monkeypatch):
    made, client, ids = docs
    from fastapi.testclient import TestClient

    from app import smart
    from app.main import app

    monkeypatch.setattr(smart, "write_note", lambda *a, **k: {"reference": "DocumentReference/doc-1", "text": "note"})
    asked = TestClient(app).post("/api/v1/ask", json={"question": ask_text("Alphazol")},
                                 headers={"Authorization": f"Bearer {made['alpha']['api_key']['key']}"}).json()
    audit_id = asked["audit_id"]
    assert client("beta-clin").post("/api/ehr/notes", json={"audit_id": audit_id}).status_code == 404
    assert client("group-clin").post("/api/ehr/notes", json={"audit_id": audit_id}).status_code == 200
    assert client("alpha-clin").post("/api/ehr/notes", json={"audit_id": audit_id}).status_code == 200
