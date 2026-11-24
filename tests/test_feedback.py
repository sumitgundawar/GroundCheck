"""Clinician feedback on answers: who may give it, one rating per person per
answer, the review case a "wrong" opens, identifiers removed from comments,
and the signed audit record staying exactly as it was."""

from __future__ import annotations

import conftest  # noqa: F401

import os
import sys
import threading
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select, text  # noqa: E402

from app import auth, config, db, feedback, governance, guards_input, integrity, retrieval, sites  # noqa: E402

PASSWORD = "correct horse battery staple"
ANSWERED = "What is the standard dose of Caloradine?"
REFUSED = "What is the standard dose of Zyntrafen?"


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def world(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    from app.main import app

    north, south = sites.create("north", "North"), sites.create("south", "South")
    for email, role, site in (("clin-a", "clinician", north["id"]), ("clin-b", "clinician", north["id"]),
                              ("rev", "reviewer", north["id"]), ("south-clin", "clinician", south["id"]),
                              ("admin", "admin", None)):
        auth.create_user(f"{email}@example.org", PASSWORD, role=role, name=email, site_id=site)
    clients = {}

    def client(who):
        if who not in clients:
            c = TestClient(app)
            assert c.post("/api/auth/login", json={"email": f"{who}@example.org", "password": PASSWORD}).status_code == 200
            clients[who] = c
        return clients[who]

    return client


def _ask(client, who, query=ANSWERED):
    return client(who).post("/api/ask", json={"query": query}).json()["audit_id"]


def _post(client, who, audit_id, **body):
    return client(who).post(f"/api/audit/{audit_id}/feedback", json=body)


def _rows():
    with db.session() as s:
        return list(s.scalars(select(db.AnswerFeedback).order_by(db.AnswerFeedback.id)))


# --- Who may give feedback ------------------------------------------------------

def test_feedback_needs_a_signed_in_user(world):
    audit_id = _ask(world, "clin-a")
    from app.main import app

    with TestClient(app) as anonymous:
        assert anonymous.post(f"/api/audit/{audit_id}/feedback", json={"rating": "up"}).status_code == 401
        assert anonymous.get(f"/api/audit/{audit_id}/feedback").status_code == 401
    assert _rows() == []


def test_a_clinician_rates_their_own_answer(world):
    audit_id = _ask(world, "clin-a")
    r = _post(world, "clin-a", audit_id, rating="up")
    assert r.status_code == 200, r.text
    body = r.json()["feedback"]
    assert body["rating"] == "up" and body["reason"] == "" and body["case_id"] is None
    assert world("clin-a").get(f"/api/audit/{audit_id}/feedback").json()["feedback"]["rating"] == "up"
    assert world("clin-b").get(f"/api/audit/{audit_id}/feedback").json()["feedback"] is None


def test_a_clinician_cannot_rate_someone_elses_answer_but_a_reviewer_can(world):
    audit_id = _ask(world, "clin-a")
    assert _post(world, "clin-b", audit_id, rating="up").status_code == 400
    assert _post(world, "rev", audit_id, rating="up").status_code == 200
    assert [r.voter for r in _rows()] == [f"u{auth.list_users()[2]['id']}"]


def test_feedback_cannot_cross_sites(world):
    audit_id = _ask(world, "clin-a")
    assert _post(world, "south-clin", audit_id, rating="down", reason="wrong").status_code == 400
    assert _rows() == [] and governance.list_cases("all")["cases"] == []


# --- Validation -----------------------------------------------------------------

@pytest.mark.parametrize("audit_id", ["", "0000000000000000", "nope", "x" * 40, "../../etc/passwd", "1; DROP TABLE answer_feedback"])
def test_an_unknown_audit_id_is_refused(world, audit_id):
    r = _post(world, "clin-a", audit_id or "-", rating="up")
    assert r.status_code in (400, 404, 405) and _rows() == []
    if r.status_code == 400:
        assert "audit trail" in r.json()["detail"]


def test_an_administrative_event_row_is_not_an_answer(world):
    event_id = integrity.record_event("test event", {"x": 1})
    assert event_id
    assert _post(world, "admin", event_id, rating="up").status_code == 400
    assert _rows() == []


@pytest.mark.parametrize("body", [
    {"rating": "sideways"}, {"rating": ""}, {"rating": "UP"}, {}, {"rating": "down", "reason": "rude"},
    {"rating": "up", "reason": "wrong"}, {"rating": "up", "comment": "great"},
    {"rating": "down", "comment": "x" * 501},
])
def test_invalid_feedback_is_refused_and_stores_nothing(world, body):
    audit_id = _ask(world, "clin-a")
    assert _post(world, "clin-a", audit_id, **body).status_code in (400, 422)
    assert _rows() == []


def test_a_down_rating_may_have_no_reason_and_each_reason_is_stored(world):
    for reason in ("", "not_useful", "incomplete", "too_slow"):
        audit_id = _ask(world, "clin-a")
        r = _post(world, "clin-a", audit_id, rating="down", reason=reason)
        assert r.status_code == 200 and r.json()["feedback"]["reason"] == reason
    assert governance.list_cases("all")["cases"] == []


# --- One per person per answer -------------------------------------------------------

def test_a_second_submission_updates_the_first(world):
    audit_id = _ask(world, "clin-a")
    first = _post(world, "clin-a", audit_id, rating="up").json()["feedback"]
    second = _post(world, "clin-a", audit_id, rating="down", reason="incomplete", comment="Missing the renal advice.").json()["feedback"]
    rows = _rows()
    assert len(rows) == 1
    assert (rows[0].rating, rows[0].reason, rows[0].comment) == ("down", "incomplete", "Missing the renal advice.")
    assert second["created_at"] == first["created_at"] and second["updated_at"] >= first["updated_at"]
    third = _post(world, "clin-a", audit_id, rating="up").json()["feedback"]
    assert third["reason"] == "" and third["comment"] == "" and len(_rows()) == 1


def test_two_people_give_two_ratings_on_one_answer(world):
    audit_id = _ask(world, "clin-a")
    _post(world, "clin-a", audit_id, rating="up")
    _post(world, "rev", audit_id, rating="down", reason="not_useful")
    assert sorted(r.rating for r in _rows()) == ["down", "up"]


def test_the_same_person_on_two_answers_gives_two_ratings(world):
    a, b = _ask(world, "clin-a"), _ask(world, "clin-a")
    _post(world, "clin-a", a, rating="up")
    _post(world, "clin-a", b, rating="up")
    assert len(_rows()) == 2


def test_simultaneous_submissions_by_one_person_leave_one_row(world):
    audit_id = _ask(world, "clin-a")
    client = world("clin-a")
    codes = []

    def go(i):
        r = client.post(f"/api/audit/{audit_id}/feedback", json={"rating": "down", "reason": "wrong", "comment": f"try {i}"})
        codes.append(r.status_code)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 8
    assert len(_rows()) == 1
    assert len(governance.list_cases("all")["cases"]) == 1  # one case however many raced


def test_the_open_demo_has_one_anonymous_slot_per_answer(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    from app.main import app

    with TestClient(app) as c:
        audit_id = c.post("/api/ask", json={"query": ANSWERED}).json()["audit_id"]
        assert c.post(f"/api/audit/{audit_id}/feedback", json={"rating": "up"}).status_code == 200
        assert c.post(f"/api/audit/{audit_id}/feedback", json={"rating": "down"}).status_code == 200
    rows = _rows()
    assert len(rows) == 1 and rows[0].voter == "anon" and rows[0].rating == "down" and rows[0].user_id is None


# --- "Wrong" opens a flagged case ---------------------------------------------------------

def test_wrong_opens_one_high_priority_case_linked_to_the_feedback(world):
    audit_id = _ask(world, "clin-a")
    r = _post(world, "clin-a", audit_id, rating="down", reason="wrong", comment="The dose is for children.")
    case_id = r.json()["feedback"]["case_id"]
    assert case_id
    case = governance.get_case(case_id)
    assert case["kind"] == "flagged" and case["priority"] == "high" and case["status"] == "open"
    assert case["first_audit_id"] == audit_id and "The dose is for children." in case["reason"]
    # Saying it again, or changing the comment, does not open another case.
    again = _post(world, "clin-a", audit_id, rating="down", reason="wrong", comment="Still wrong.").json()["feedback"]
    assert again["case_id"] == case_id and len(governance.list_cases("all")["cases"]) == 1
    # A different clinician agreeing opens their own case: two people, two flags.
    other = _post(world, "rev", audit_id, rating="down", reason="wrong").json()["feedback"]
    assert other["case_id"] not in (None, case_id) and len(governance.list_cases("all")["cases"]) == 2


def test_wrong_after_another_reason_opens_the_case_then(world):
    audit_id = _ask(world, "clin-a")
    assert _post(world, "clin-a", audit_id, rating="down", reason="incomplete").json()["feedback"]["case_id"] is None
    assert governance.list_cases("all")["cases"] == []
    assert _post(world, "clin-a", audit_id, rating="down", reason="wrong").json()["feedback"]["case_id"]
    assert len(governance.list_cases("all")["cases"]) == 1


def test_the_case_appears_in_the_reviewers_queue_with_the_site(world):
    audit_id = _ask(world, "clin-a")
    _post(world, "clin-a", audit_id, rating="down", reason="wrong")
    queue = world("rev").get("/api/reviews").json()
    assert [(c["kind"], c["priority"]) for c in queue["cases"]] == [("flagged", "high")]
    assert world("south-clin").get("/api/audit").status_code == 200
    assert governance.list_cases("open", site_id=sites.list_sites()[1]["id"])["cases"] == []


def test_wrong_on_a_refusal_is_kept_but_opens_no_case(world):
    audit_id = _ask(world, "clin-a", REFUSED)
    before = len(governance.list_cases("all")["cases"])
    r = _post(world, "clin-a", audit_id, rating="down", reason="wrong")
    assert r.status_code == 200 and r.json()["feedback"]["case_id"] is None
    assert len(governance.list_cases("all")["cases"]) == before and len(_rows()) == 1


# --- Identifiers never survive in a comment ---------------------------------------------

SENSITIVE = ["943 476 5919", "John Smith", "john.smith@example.org", "07700 900123"]


def _all_stored_text():
    with db.engine().connect() as conn:
        out = []
        for table in ("answer_feedback", "review_cases", "review_events"):
            for row in conn.execute(text(f"SELECT * FROM {table}")).all():
                out.append(" ".join(str(v) for v in row))
    return " ".join(out)


def test_identifiers_in_a_comment_are_removed_everywhere(world):
    audit_id = _ask(world, "clin-a")
    comment = "Mr John Smith, NHS 943 476 5919, john.smith@example.org, phone 07700 900123 got the wrong dose."
    r = _post(world, "clin-a", audit_id, rating="down", reason="wrong", comment=comment)
    assert r.status_code == 200
    stored = r.json()["feedback"]["comment"]
    assert stored and all(s not in stored for s in SENSITIVE)
    assert "wrong dose" in stored
    everything = _all_stored_text() + str(world("clin-a").get(f"/api/audit/{audit_id}/feedback").json())
    everything += str(world("rev").get("/api/reviews").json())
    for secret in SENSITIVE + ["9434765919"]:
        assert secret not in everything


def test_a_comment_that_cannot_be_checked_is_not_stored(world, monkeypatch):
    audit_id = _ask(world, "clin-a")

    def broken(_text):
        raise RuntimeError("detector down")

    monkeypatch.setattr(guards_input, "deidentify", broken)
    r = _post(world, "clin-a", audit_id, rating="down", reason="not_useful", comment="Patient John Smith NHS 943 476 5919")
    assert r.status_code == 200 and r.json()["feedback"]["comment"] == ""
    assert "943" not in _all_stored_text() and "Smith" not in _all_stored_text()


def test_a_plain_clinical_comment_is_kept_as_written(world):
    audit_id = _ask(world, "clin-a")
    text_in = "It left out the renal adjustment."
    assert _post(world, "clin-a", audit_id, rating="down", reason="incomplete", comment=text_in).json()["feedback"]["comment"] == text_in


def test_a_blank_comment_is_stored_empty(world):
    audit_id = _ask(world, "clin-a")
    assert _post(world, "clin-a", audit_id, rating="down", comment="   ").json()["feedback"]["comment"] == ""


def test_comments_are_encrypted_at_rest_when_keys_are_set(world, monkeypatch):
    from app import encryption

    key = encryption.generate_key() if hasattr(encryption, "generate_key") else None
    if key is None:
        pytest.skip("no key generator")
    monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", key)
    encryption.reset()
    try:
        audit_id = _ask(world, "clin-a")
        _post(world, "clin-a", audit_id, rating="down", reason="incomplete", comment="Needs the renal advice.")
        with db.engine().connect() as conn:
            raw = conn.execute(text("SELECT comment FROM answer_feedback")).scalar()
        assert "renal" not in raw
        assert _rows()[0].comment == "Needs the renal advice."
    finally:
        monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", "")
        encryption.reset()


# --- The signed audit record is never touched ----------------------------------------------

def test_feedback_never_alters_the_signed_audit_record(world):
    audit_id = _ask(world, "clin-a")
    _ask(world, "clin-a", REFUSED)

    def snapshot():
        with db.session() as s:
            r = s.scalar(select(db.AuditRecord).where(db.AuditRecord.audit_id == audit_id))
            return (r.seq, r.prev_hash, r.entry_hash, r.chain_alg, r.decision, r.query, r.record, r.user_id, r.site_id)

    before, api_before, head_before = snapshot(), world("clin-a").get(f"/api/audit/{audit_id}").content, integrity.head()
    for body in ({"rating": "up"}, {"rating": "down", "reason": "wrong", "comment": "No."}, {"rating": "down", "reason": "too_slow"}):
        assert _post(world, "clin-a", audit_id, **body).status_code == 200
    assert snapshot() == before
    assert world("clin-a").get(f"/api/audit/{audit_id}").content == api_before
    result = integrity.verify()
    assert result["ok"] and result["problem_count"] == 0
    # Feedback adds no rows to the chain; flagging a case did not either.
    assert integrity.head() == head_before


def test_feedback_leaves_no_trace_in_the_answers_own_record(world):
    audit_id = _ask(world, "clin-a")
    _post(world, "clin-a", audit_id, rating="down", reason="wrong", comment="Wrong.")
    record = world("clin-a").get(f"/api/audit/{audit_id}").json()
    assert "feedback" not in str(record).lower()


def test_deleting_a_user_keeps_their_feedback_under_an_anonymous_account(world):
    audit_id = _ask(world, "clin-b")
    _post(world, "clin-b", audit_id, rating="up")
    uid = [u for u in auth.list_users() if u["email"].startswith("clin-b")][0]["id"]
    auth.delete_user(uid)
    rows = _rows()
    assert len(rows) == 1 and rows[0].user_id == uid
    assert not any("clin-b" in u["email"] for u in auth.list_users())
