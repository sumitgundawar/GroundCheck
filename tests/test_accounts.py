"""Accounts, sessions, two-factor authentication, roles and audit storage.

Runs against SQLite always, and against PostgreSQL too when TEST_POSTGRES_URL
is set, for example postgresql+psycopg://user@127.0.0.1:5432/groundcheck_test."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import pyotp
import pytest
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"

from fastapi.testclient import TestClient  # noqa: E402

from app import audit, auth, config, db, retrieval  # noqa: E402

PASSWORD = "correct horse battery staple"



# --- Users and passwords ------------------------------------------------------

def test_passwords_are_hashed_with_argon2id(database):
    user = auth.create_user("Ada@Example.org", PASSWORD, role="admin", name="Ada")
    assert user.email == "ada@example.org"
    with db.session() as s:
        stored = s.get(db.User, user.id).password_hash
    assert stored.startswith("$argon2id$")
    assert PASSWORD not in stored


@pytest.mark.parametrize("email, password, message", [
    ("not-an-email", PASSWORD, "valid email"),
    ("ada@example.org", "short", "at least"),
    ("adalovelace@example.org", "adalovelace-password-123", "email name"),
])
def test_invalid_accounts_are_refused(database, email, password, message):
    with pytest.raises(auth.AuthError, match=message):
        auth.create_user(email, password)


def test_duplicate_email_is_refused(database):
    auth.create_user("ada@example.org", PASSWORD)
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.create_user("ADA@example.org", PASSWORD)


def test_unknown_role_is_refused(database):
    with pytest.raises(auth.AuthError, match="Role must be"):
        auth.create_user("ada@example.org", PASSWORD, role="superuser")


# --- Sign-in and sessions -----------------------------------------------------

def test_sign_in_creates_a_session_stored_as_a_hash(database):
    auth.create_user("ada@example.org", PASSWORD)
    result = auth.sign_in("ada@example.org", PASSWORD)
    assert not result.mfa_required
    assert auth.session_principal(result.token).email == "ada@example.org"
    with db.session() as s:
        hashes = s.scalars(select(db.AuthSession.token_hash)).all()
    assert result.token not in hashes and len(hashes) == 1


def test_wrong_password_and_unknown_email_give_the_same_error(database):
    auth.create_user("ada@example.org", PASSWORD)
    with pytest.raises(auth.AuthError) as wrong_password:
        auth.sign_in("ada@example.org", "wrong password entirely")
    with pytest.raises(auth.AuthError) as unknown_email:
        auth.sign_in("nobody@example.org", PASSWORD)
    assert str(wrong_password.value) == str(unknown_email.value) == "Incorrect email or password."


def test_repeated_failures_lock_the_account(database, monkeypatch):
    monkeypatch.setattr(config, "LOGIN_MAX_FAILURES", 3)
    auth.create_user("ada@example.org", PASSWORD)
    for _ in range(3):
        with pytest.raises(auth.AuthError):
            auth.sign_in("ada@example.org", "wrong password entirely")
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.sign_in("ada@example.org", PASSWORD)


def test_expired_sessions_are_rejected_and_purged(database):
    auth.create_user("ada@example.org", PASSWORD)
    expired = auth.sign_in("ada@example.org", PASSWORD).token
    stale = auth.sign_in("ada@example.org", PASSWORD).token
    live = auth.sign_in("ada@example.org", PASSWORD).token
    with db.session() as s:
        for token in (expired, stale):
            record = s.scalar(select(db.AuthSession).where(db.AuthSession.token_hash == auth._hash_token(token)))
            record.expires_at = db.utcnow() - timedelta(minutes=1)

    assert auth.session_principal(expired) is None  # rejected, and deleted on sight
    assert auth.purge_expired_sessions() == 1        # the one nobody used
    assert auth.session_principal(live).email == "ada@example.org"


def test_sign_out_ends_the_session(database):
    auth.create_user("ada@example.org", PASSWORD)
    token = auth.sign_in("ada@example.org", PASSWORD).token
    auth.sign_out(token)
    assert auth.session_principal(token) is None


def test_deactivating_a_user_signs_them_out(database):
    admin = auth.create_user("admin@example.org", PASSWORD, role="admin")
    user = auth.create_user("ada@example.org", PASSWORD)
    token = auth.sign_in("ada@example.org", PASSWORD).token
    auth.update_user(user.id, is_active=False, acting_user_id=admin.id)
    assert auth.session_principal(token) is None
    with pytest.raises(auth.AuthError):
        auth.sign_in("ada@example.org", PASSWORD)


def test_changing_password_signs_out_every_session(database):
    user = auth.create_user("ada@example.org", PASSWORD)
    token = auth.sign_in("ada@example.org", PASSWORD).token
    with pytest.raises(auth.AuthError, match="current password"):
        auth.set_password(user.id, "a brand new long password", current_password="nope nope nope")
    auth.set_password(user.id, "a brand new long password", current_password=PASSWORD)
    assert auth.session_principal(token) is None
    assert auth.sign_in("ada@example.org", "a brand new long password").token


def test_the_last_admin_cannot_be_demoted_or_deactivated(database):
    admin = auth.create_user("admin@example.org", PASSWORD, role="admin")
    with pytest.raises(auth.AuthError, match="at least one active admin"):
        auth.update_user(admin.id, role="clinician")
    other = auth.create_user("second@example.org", PASSWORD, role="admin")
    auth.update_user(admin.id, role="reviewer", acting_user_id=other.id)
    with pytest.raises(auth.AuthError, match="at least one active admin"):
        auth.update_user(other.id, is_active=False)


# --- Two-factor authentication -----------------------------------------------

def test_two_factor_setup_requires_a_valid_code(database):
    user = auth.create_user("ada@example.org", PASSWORD)
    setup = auth.begin_mfa_setup(user.id)
    assert setup["otpauth_uri"].startswith("otpauth://totp/GroundCheck:ada%40example.org")
    with pytest.raises(auth.AuthError, match="didn't match"):
        auth.confirm_mfa_setup(user.id, "000000")
    auth.confirm_mfa_setup(user.id, pyotp.TOTP(setup["secret"]).now())


def test_sign_in_with_two_factor_is_pending_until_verified(database):
    user = auth.create_user("ada@example.org", PASSWORD)
    secret = auth.begin_mfa_setup(user.id)["secret"]
    auth.confirm_mfa_setup(user.id, pyotp.TOTP(secret).now())

    result = auth.sign_in("ada@example.org", PASSWORD)
    assert result.mfa_required
    assert auth.session_principal(result.token) is None
    with pytest.raises(auth.AuthError):
        auth.verify_mfa(result.token, "123456")
    auth.verify_mfa(result.token, pyotp.TOTP(secret).now())
    assert auth.session_principal(result.token).email == "ada@example.org"


def test_disabling_two_factor_requires_the_password(database):
    user = auth.create_user("ada@example.org", PASSWORD)
    secret = auth.begin_mfa_setup(user.id)["secret"]
    auth.confirm_mfa_setup(user.id, pyotp.TOTP(secret).now())
    with pytest.raises(auth.AuthError):
        auth.disable_mfa(user.id, "wrong password entirely")
    auth.disable_mfa(user.id, PASSWORD)
    assert not auth.sign_in("ada@example.org", PASSWORD).mfa_required


def test_roles_are_ordered():
    clinician = auth.Principal(1, "a@b.org", "", "clinician", False)
    reviewer = auth.Principal(2, "c@d.org", "", "reviewer", False)
    admin = auth.Principal(3, "e@f.org", "", "admin", False)
    assert clinician.can("clinician") and not clinician.can("reviewer")
    assert reviewer.can("reviewer") and not reviewer.can("admin")
    assert admin.can("clinician") and admin.can("admin")


# --- Audit storage -------------------------------------------------------------

def test_audit_records_are_stored_in_the_database_with_the_user(database):
    from app import pipeline

    retrieval.load_index()
    user = auth.create_user("ada@example.org", PASSWORD)
    store = audit.AuditStore(10, Path(tempfile.mkdtemp()) / "audit.jsonl", persist=True)
    original = audit.store
    audit.store = store
    try:
        r = pipeline.run("What is the standard dose of Caloradine? My email is ada@example.org",
                         user_id=user.id)
    finally:
        audit.store = original
    assert store.backend() == "database"
    with db.session() as s:
        row = s.scalar(select(db.AuditRecord).where(db.AuditRecord.audit_id == r.audit_id))
    assert row.user_id == user.id
    assert row.decision == r.decision
    assert "ada@example.org" not in row.query  # stored redacted
    assert "raw_query" not in row.record
    assert store.recent(5, user_id=user.id)[0]["audit_id"] == r.audit_id
    assert store.recent(5, user_id=user.id + 999) == []


# --- API -----------------------------------------------------------------------

@pytest.fixture()
def client(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "SESSION_COOKIE_SECURE", False)
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
    from app.main import app
    with TestClient(app) as c:
        yield c


def _sign_in(client, email, password=PASSWORD):
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r


def test_api_requires_sign_in_when_accounts_are_required(client):
    assert client.get("/api/health").status_code == 200
    for method, path in [("get", "/api/settings"), ("get", "/api/corpus"), ("get", "/api/audit"),
                         ("post", "/api/ask"), ("get", "/api/local-ai"), ("get", "/api/users")]:
        response = getattr(client, method)(path, **({"json": {"query": "x"}} if method == "post" else {}))
        assert response.status_code == 401, path


def test_first_admin_then_sign_in_and_ask(client):
    me = client.get("/api/auth/me").json()
    assert me["auth_required"] and me["needs_first_admin"] and me["user"] is None
    r = client.post("/api/auth/first-admin", json={"email": "admin@example.org", "password": PASSWORD})
    assert r.status_code == 200
    assert "gc_session" in client.cookies
    assert client.post("/api/auth/first-admin", json={"email": "x@example.org", "password": PASSWORD}).status_code == 409

    ask = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?"})
    assert ask.status_code == 200 and ask.json()["decision"] == "answer"


def test_first_admin_cannot_be_created_remotely(client):
    r = client.post("/api/auth/first-admin", json={"email": "admin@example.org", "password": PASSWORD},
                    headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 403


def test_roles_limit_what_each_user_can_do(client):
    auth.create_user("admin@example.org", PASSWORD, role="admin")
    auth.create_user("clin@example.org", PASSWORD, role="clinician")
    auth.create_user("rev@example.org", PASSWORD, role="reviewer")

    _sign_in(client, "clin@example.org")
    mine = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?"}).json()["audit_id"]
    assert client.get("/api/users").status_code == 403
    assert client.post("/api/local-ai/select", json={"model": None}).status_code == 403
    assert [r["audit_id"] for r in client.get("/api/audit").json()["recent"]] == [mine]
    client.post("/api/auth/logout")

    _sign_in(client, "rev@example.org")
    assert client.get(f"/api/audit/{mine}").status_code == 200
    assert client.get("/api/users").status_code == 403
    client.post("/api/auth/logout")

    _sign_in(client, "admin@example.org")
    users = client.get("/api/users").json()["users"]
    assert {u["email"] for u in users} == {"admin@example.org", "clin@example.org", "rev@example.org"}
    created = client.post("/api/users", json={"email": "new@example.org", "password": PASSWORD, "role": "reviewer"})
    assert created.status_code == 200
    new_id = created.json()["user"]["id"]
    assert client.patch(f"/api/users/{new_id}", json={"role": "admin"}).json()["user"]["role"] == "admin"


def test_clinicians_cannot_read_other_users_audit_records(client):
    auth.create_user("a@example.org", PASSWORD)
    auth.create_user("b@example.org", PASSWORD)
    _sign_in(client, "a@example.org")
    theirs = client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?"}).json()["audit_id"]
    client.post("/api/auth/logout")
    _sign_in(client, "b@example.org")
    assert client.get(f"/api/audit/{theirs}").status_code == 404


def test_two_factor_through_the_api(client):
    auth.create_user("ada@example.org", PASSWORD)
    _sign_in(client, "ada@example.org")
    setup = client.post("/api/auth/mfa/setup").json()
    assert setup["qr_svg_data_uri"].startswith("data:image/svg+xml")
    assert client.post("/api/auth/mfa/confirm", json={"code": pyotp.TOTP(setup["secret"]).now()}).status_code == 200
    client.post("/api/auth/logout")

    login = _sign_in(client, "ada@example.org").json()
    assert login["mfa_required"] and login["user"] is None
    assert client.get("/api/settings").status_code == 401  # pending session grants nothing
    assert client.get("/api/auth/me").json()["mfa_pending"] is True
    assert client.post("/api/auth/mfa", json={"code": pyotp.TOTP(setup["secret"]).now()}).status_code == 200
    assert client.get("/api/settings").status_code == 200


def test_cross_site_writes_are_refused(client):
    auth.create_user("ada@example.org", PASSWORD)
    r = client.post("/api/auth/login", json={"email": "ada@example.org", "password": PASSWORD},
                    headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_session_cookie_is_http_only_and_strict(client):
    auth.create_user("ada@example.org", PASSWORD)
    r = _sign_in(client, "ada@example.org")
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
