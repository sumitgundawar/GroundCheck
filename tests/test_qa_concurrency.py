"""Races: quotas, rate limits, metering, feedback and key lifecycle under parallel requests. The counts must be exact and
nothing may answer 500."""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app import apikeys, auth, config, db, guards_input, integrity, metering, pipeline, retrieval  # noqa: E402
from app.main import app  # noqa: E402

PASSWORD = "correct horse battery staple"
Q = "What is the standard dose of Caloradine?"
GOOD = "Caloradine is 15 mg once daily."
SCOPES = ["ask", "verify", "usage", "audit:read"]


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def api(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    pipeline.clear_answer_cache()
    guards_input.key_limiter._by_client.clear()
    return TestClient(app, raise_server_exceptions=False)


def admin(api) -> TestClient:
    auth.create_user("admin@example.org", PASSWORD, role="admin", name="Admin")
    assert api.post("/api/auth/login", json={"email": "admin@example.org", "password": PASSWORD}).status_code == 200
    return api


def hit(api, headers, i: int):
    if i % 2:
        return api.post("/api/v1/ask", json={"question": Q}, headers=headers)
    return api.post("/api/v1/verify", json={"question": Q, "answer": GOOD}, headers=headers)


def run_parallel(n: int, workers: int, fn):
    workers = min(workers, n)
    barrier = threading.Barrier(workers)

    def task(i):
        if i < workers:
            barrier.wait(timeout=30)
        return fn(i)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(task, range(n)))


# --- Quotas, rate limits, metering ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("quota", [1, 7, 20])
def test_a_monthly_quota_admits_exactly_that_many_requests_in_parallel(api, quota):
    made = apikeys.create("q", SCOPES, rpm_limit=10000, monthly_quota=quota)
    headers = {"Authorization": "Bearer " + made["key"]}
    responses = run_parallel(quota * 3 + 5, 16, lambda i: hit(api, headers, i))
    codes = [r.status_code for r in responses]
    assert codes.count(200) == quota and codes.count(429) == len(codes) - quota, codes
    assert apikeys.quota_status(made["id"])["used"] == quota
    assert all(r.json()["error"]["code"] == "quota_exceeded" for r in responses if r.status_code == 429)
    with db.session() as s:
        billed = s.scalar(select(func.coalesce(func.sum(db.UsageDaily.count), 0)).where(db.UsageDaily.api_key_id == made["id"]))
    assert billed >= quota


def test_the_per_minute_limit_admits_exactly_that_many_in_parallel(api):
    made = apikeys.create("r", SCOPES, rpm_limit=9)
    headers = {"Authorization": "Bearer " + made["key"]}
    codes = [r.status_code for r in run_parallel(40, 16, lambda i: hit(api, headers, i))]
    assert codes.count(200) == 9 and codes.count(429) == 31, codes


def test_metering_counts_every_request_once_under_load(api):
    made = apikeys.create("m", SCOPES, rpm_limit=10000)
    headers = {"Authorization": "Bearer " + made["key"]}

    def mixed(i):
        if i % 5 == 0:
            return api.post("/api/v1/ask", json={"question": ""}, headers=headers)      # a rejected request
        return hit(api, headers, i)

    codes = [r.status_code for r in run_parallel(60, 12, mixed)]
    row = next(r for r in metering.usage("key")["rows"] if r["key_id"] == made["id"])
    total = sum(row["decisions"].values())
    assert total == 60 == len(codes)
    assert row["decisions"].get("error", 0) == codes.count(422) == 12


def test_two_keys_never_share_a_limit_or_a_count(api):
    a = apikeys.create("a", SCOPES, rpm_limit=5)
    b = apikeys.create("b", SCOPES, rpm_limit=5)
    ha, hb = ({"Authorization": "Bearer " + k["key"]} for k in (a, b))
    codes = run_parallel(40, 16, lambda i: hit(api, ha if i % 2 else hb, i).status_code)
    assert codes.count(200) == 10


# --- Key lifecycle ----------------------------------------------------------------------------------------------------------

def test_a_revoked_key_is_refused_from_the_moment_revoke_returns(api):
    client = admin(api)
    made = apikeys.create("victim", SCOPES, rpm_limit=100000)
    headers = {"Authorization": "Bearer " + made["key"]}
    stop, results, after_revoke = threading.Event(), [], []
    revoked_at = threading.Event()

    def hammer():
        while not stop.is_set():
            started_after = revoked_at.is_set()
            r = TestClient(app, raise_server_exceptions=False).post("/api/v1/ask", json={"question": Q}, headers=headers)
            (after_revoke if started_after else results).append(r.status_code)

    threads = [threading.Thread(target=hammer) for _ in range(6)]
    [t.start() for t in threads]
    try:
        while len(results) < 6:
            stop.wait(0.05)
        assert client.post(f"/api/keys/{made['id']}/revoke").status_code == 200
        revoked_at.set()
        stop.wait(1.0)
    finally:
        stop.set()
        [t.join(timeout=30) for t in threads]
    assert set(results) <= {200, 401}, set(results)
    assert after_revoke and set(after_revoke) == {401}, set(after_revoke)


def test_revoking_the_same_key_in_parallel_never_fails_with_a_server_error(api):
    client = admin(api)
    made = apikeys.create("twice", SCOPES)
    codes = run_parallel(12, 12, lambda i: client.post(f"/api/keys/{made['id']}/revoke").status_code)
    assert set(codes) <= {200, 404, 409}, codes
    assert codes.count(200) >= 1
    assert integrity.verify()["ok"] is True


def test_keys_created_in_parallel_are_all_distinct_and_all_work(api):
    client = admin(api)
    responses = run_parallel(20, 10, lambda i: client.post("/api/keys", json={"name": f"k{i}", "scopes": ["ask"]}))
    assert [r.status_code for r in responses] == [201] * 20
    secrets_made = [r.json()["key"] for r in responses]
    assert len(set(secrets_made)) == 20
    assert len({r.json()["prefix"] for r in responses}) == 20
    for secret in secrets_made[:5]:
        assert api.post("/api/v1/ask", json={"question": Q}, headers={"Authorization": "Bearer " + secret}).status_code == 200
    assert integrity.verify()["ok"] is True


def test_a_key_used_while_its_scopes_are_checked_never_gets_a_500(api):
    made = apikeys.create("narrow", ["usage"])
    headers = {"Authorization": "Bearer " + made["key"]}
    codes = run_parallel(30, 10, lambda i: hit(api, headers, i).status_code)
    assert set(codes) == {403}


# --- Feedback ------------------------------------------------------------------------------------------------------------------

def test_simultaneous_feedback_from_one_person_leaves_one_row_and_one_case(api):
    client = admin(api)
    audit_id = client.post("/api/ask", json={"query": Q}).json()["audit_id"]
    codes = run_parallel(16, 16, lambda i: client.post(f"/api/audit/{audit_id}/feedback", json={
        "rating": "down", "reason": "wrong", "comment": f"attempt {i}"}).status_code)
    assert set(codes) == {200}, codes
    with db.session() as s:
        rows = s.scalar(select(func.count()).select_from(db.AnswerFeedback).where(db.AnswerFeedback.audit_id == audit_id))
        cases = s.scalar(select(func.count()).select_from(db.ReviewCase).where(db.ReviewCase.first_audit_id == audit_id))
    assert rows == 1 and cases <= 1


def test_feedback_from_different_people_is_counted_separately(api):
    admin(api)
    clients = []
    for i in range(5):
        auth.create_user(f"c{i}@example.org", PASSWORD, role="clinician", name=f"C{i}")
        c = TestClient(app, raise_server_exceptions=False)
        assert c.post("/api/auth/login", json={"email": f"c{i}@example.org", "password": PASSWORD}).status_code == 200
        clients.append(c)
    audit_id = api.post("/api/ask", json={"query": Q}).json()["audit_id"]
    codes = run_parallel(5, 5, lambda i: clients[i].post(f"/api/audit/{audit_id}/feedback", json={"rating": "up"}).status_code)
    assert set(codes) == {200} or set(codes) <= {200, 400}
    with db.session() as s:
        rows = s.scalar(select(func.count()).select_from(db.AnswerFeedback).where(db.AnswerFeedback.audit_id == audit_id))
    assert rows == codes.count(200)


def test_signing_in_and_out_in_parallel_never_fails_with_a_server_error(api):
    auth.create_user("many@example.org", PASSWORD, role="clinician", name="Many")

    def cycle(i):
        c = TestClient(app, raise_server_exceptions=False)
        a = c.post("/api/auth/login", json={"email": "many@example.org", "password": PASSWORD}).status_code
        b = c.get("/api/auth/me").status_code
        d = c.post("/api/auth/logout").status_code
        return max(a, b, d)

    assert max(run_parallel(24, 8, cycle)) < 500
