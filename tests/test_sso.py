"""Single sign-on with OpenID Connect, against a fake identity provider that
signs real RS256 tokens: the full sign-in, role mapping, account linking,
and every check that should refuse a sign-in."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, config, sso  # noqa: E402

ISSUER = "https://idp.example.org"
CLIENT_ID = "groundcheck-test"
PASSWORD = "correct horse battery staple"


class FakeProvider:
    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "key-1"
        self.codes: dict[str, dict] = {}
        self.claims: dict = {}
        self.token_overrides: dict = {}
        self.header_overrides: dict = {}
        self.signing_key = None

    def jwks(self):
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        return {"keys": [{**jwk, "kid": self.kid, "use": "sig", "alg": "RS256"}]}

    def approve(self, authorize_url: str, **claims) -> tuple[str, str]:
        """What the provider does after the person signs in: issue a code."""
        query = {k: v[0] for k, v in parse_qs(urlparse(authorize_url).query).items()}
        code = secrets.token_urlsafe(16)
        self.codes[code] = {"nonce": query["nonce"], "challenge": query["code_challenge"],
                            "redirect_uri": query["redirect_uri"], "claims": claims}
        return code, query["state"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize",
                "token_endpoint": f"{ISSUER}/token", "jwks_uri": f"{ISSUER}/jwks"})
        if path == "/jwks":
            return httpx.Response(200, json=self.jwks())
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            grant = self.codes.pop(form.get("code"), None)
            if grant is None:
                return httpx.Response(400, json={"error": "invalid_grant"})
            challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
            if challenge != grant["challenge"] or form["redirect_uri"] != grant["redirect_uri"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            now = int(time.time())
            claims = {"iss": ISSUER, "aud": CLIENT_ID, "sub": "user-123", "iat": now, "exp": now + 300,
                      "nonce": grant["nonce"], "email": "dr.ada@hospital.example", "email_verified": True,
                      "name": "Ada Lovelace", **grant["claims"], **self.token_overrides}
            headers = {"kid": self.kid, **self.header_overrides}
            algorithm = headers.pop("alg", "RS256")
            token = jwt.encode(claims, self.signing_key or self.key, algorithm=algorithm, headers=headers)
            return httpx.Response(200, json={"id_token": token, "access_token": "x", "token_type": "Bearer"})
        return httpx.Response(404)


@pytest.fixture()
def provider(database, monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "OIDC_ISSUER", ISSUER)
    monkeypatch.setattr(config, "OIDC_CLIENT_ID", CLIENT_ID)
    monkeypatch.setattr(config, "OIDC_CLIENT_SECRET", "s3cret")
    monkeypatch.setattr(config, "OIDC_PROVIDER_NAME", "Hospital ID")
    monkeypatch.setattr(config, "OIDC_ADMIN_VALUES", {"gc-admins"})
    monkeypatch.setattr(config, "OIDC_REVIEWER_VALUES", {"gc-reviewers"})
    monkeypatch.setattr(config, "OIDC_CLINICIAN_VALUES", set())
    monkeypatch.setattr(config, "OIDC_DEFAULT_ROLE", "clinician")
    monkeypatch.setattr(config, "OIDC_ALLOWED_DOMAINS", set())
    monkeypatch.setattr(config, "OIDC_REQUIRE_MFA", False)
    monkeypatch.setattr(config, "PASSWORD_SIGN_IN", True)
    monkeypatch.setattr(sso, "_client", lambda: httpx.Client(transport=httpx.MockTransport(fake.handler)))
    sso.forget_provider()
    yield fake
    sso.forget_provider()


@pytest.fixture()
def client(provider):
    from app.main import app

    with TestClient(app) as c:
        yield c


def sign_in(client, provider, next_path="/#/review", **claims) -> httpx.Response:
    start = client.get(f"/api/auth/sso/start?next={next_path}", follow_redirects=False)
    assert start.status_code == 302, start.text
    code, state = provider.approve(start.headers["location"], **claims)
    return client.get(f"/api/auth/sso/callback?code={code}&state={state}", follow_redirects=False)


def error_of(response: httpx.Response) -> str | None:
    marker = "sso_error="
    text = response.text
    if marker not in text:
        return None
    return unquote(text.split(marker, 1)[1].split('"', 1)[0])


def test_sign_in_creates_an_account_with_the_mapped_role(client, provider):
    me = client.get("/api/auth/me").json()
    assert me["sso"] == {"enabled": True, "provider": "Hospital ID"} and me["user"] is None

    start = client.get("/api/auth/sso/start?next=/%23/review", follow_redirects=False)
    params = {k: v[0] for k, v in parse_qs(urlparse(start.headers["location"]).query).items()}
    assert params["code_challenge_method"] == "S256" and params["client_id"] == CLIENT_ID
    assert params["redirect_uri"] == "http://testserver/api/auth/sso/callback"
    assert "gc_sso_state" in start.headers["set-cookie"] and "HttpOnly" in start.headers["set-cookie"]

    code, state = provider.approve(start.headers["location"], roles=["gc-reviewers"])
    done = client.get(f"/api/auth/sso/callback?code={code}&state={state}", follow_redirects=False)
    assert done.status_code == 200 and error_of(done) is None
    assert 'url=/#/review' in done.text and "gc_session=" in done.headers["set-cookie"]
    user = client.get("/api/auth/me").json()["user"]
    assert (user["email"], user["name"], user["role"]) == ("dr.ada@hospital.example", "Ada Lovelace", "reviewer")

    # The provider stays in charge of roles.
    client.post("/api/auth/logout")
    sign_in(client, provider, roles=["gc-admins"])
    assert client.get("/api/auth/me").json()["user"]["role"] == "admin"
    assert len(auth.list_users()) == 1


def test_the_state_must_come_back_to_the_same_browser_once(client, provider):
    start = client.get("/api/auth/sso/start", follow_redirects=False)
    code, state = provider.approve(start.headers["location"])
    other_browser = TestClient(client.app)
    refused = other_browser.get(f"/api/auth/sso/callback?code={code}&state={state}", follow_redirects=False)
    assert "didn't start in this browser" in error_of(refused)

    done = client.get(f"/api/auth/sso/callback?code={code}&state={state}", follow_redirects=False)
    assert error_of(done) is None
    client.cookies.set("gc_sso_state", state, path="/api/auth/sso")
    replay = client.get(f"/api/auth/sso/callback?code={code}&state={state}", follow_redirects=False)
    assert "didn't start in this browser" in error_of(replay)


@pytest.mark.parametrize("override, header, message", [
    ({"aud": "another-app"}, {}, "couldn't be verified"),
    ({"iss": "https://evil.example"}, {}, "couldn't be verified"),
    ({"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200}, {}, "couldn't be verified"),
    ({"nonce": "not-the-nonce"}, {}, "didn't match"),
    ({"aud": [CLIENT_ID, "other"], "azp": "other"}, {}, "another application"),
    ({}, {"kid": "unknown-key"}, "unknown key"),
])
def test_invalid_tokens_are_refused(client, provider, override, header, message):
    provider.token_overrides = override
    provider.header_overrides = header
    assert message in error_of(sign_in(client, provider))
    assert client.get("/api/auth/me").json()["user"] is None


def test_tokens_signed_with_a_shared_secret_or_another_key_are_refused(client, provider):
    provider.header_overrides = {"alg": "HS256"}
    provider.signing_key = "a-shared-secret-that-is-long-enough-for-hs256-use"
    assert "algorithm that isn't allowed" in error_of(sign_in(client, provider))

    provider.header_overrides = {}
    provider.signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert "couldn't be verified" in error_of(sign_in(client, provider))


def test_a_wrong_pkce_verifier_is_refused(client, provider, monkeypatch):
    start = client.get("/api/auth/sso/start", follow_redirects=False)
    code, state = provider.approve(start.headers["location"])
    provider.codes[code]["challenge"] = "something-else"
    done = client.get(f"/api/auth/sso/callback?code={code}&state={state}", follow_redirects=False)
    assert "didn't accept the sign-in" in error_of(done)


def test_provider_errors_are_shown(client):
    start = client.get("/api/auth/sso/start", follow_redirects=False)
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    done = client.get(f"/api/auth/sso/callback?error=access_denied&error_description=User+cancelled&state={state}",
                      follow_redirects=False)
    assert error_of(done) == "Hospital ID didn't sign you in: User cancelled"


def test_domains_roles_and_mfa_rules(client, provider, monkeypatch):
    monkeypatch.setattr(config, "OIDC_ALLOWED_DOMAINS", {"other.example"})
    assert "email domain" in error_of(sign_in(client, provider))
    monkeypatch.setattr(config, "OIDC_ALLOWED_DOMAINS", {"hospital.example"})

    monkeypatch.setattr(config, "OIDC_DEFAULT_ROLE", "none")
    assert "doesn't have a GroundCheckHealth role" in error_of(sign_in(client, provider, roles=["unrelated"]))

    monkeypatch.setattr(config, "OIDC_REQUIRE_MFA", True)
    assert "multi-factor" in error_of(sign_in(client, provider, roles=["gc-reviewers"], amr=["pwd"]))
    assert error_of(sign_in(client, provider, roles=["gc-reviewers"], amr=["pwd", "mfa"])) is None

    # Losing the role at the provider removes access at the next sign-in.
    client.post("/api/auth/logout")
    assert "no longer has a GroundCheckHealth role" in error_of(sign_in(client, provider, amr=["mfa"]))


def test_existing_accounts_link_only_by_verified_email(client, provider):
    auth.create_user("dr.ada@hospital.example", PASSWORD, role="admin", name="Ada")
    assert "didn't confirm the email" in error_of(sign_in(client, provider, email_verified=False))
    assert error_of(sign_in(client, provider, roles=["gc-admins"])) is None
    me = client.get("/api/auth/me").json()["user"]
    assert me["role"] == "admin" and len(auth.list_users()) == 1


def test_password_sign_in_can_be_turned_off(client, provider, monkeypatch):
    auth.create_user("local@hospital.example", PASSWORD, role="admin")
    monkeypatch.setattr(config, "PASSWORD_SIGN_IN", False)
    assert client.get("/api/auth/me").json()["password_sign_in"] is False
    refused = client.post("/api/auth/login", json={"email": "local@hospital.example", "password": PASSWORD})
    assert refused.status_code == 403 and "Hospital ID" in refused.json()["detail"]


def test_only_local_paths_are_followed_after_sign_in(client, provider):
    for bad in ("//evil.example/", "https://evil.example/", "/\\evil.example"):
        done = sign_in(client, provider, next_path=bad)
        assert "url=/\"" in done.text or 'url=/"' in done.text
        client.post("/api/auth/logout")


def test_a_misconfigured_provider_is_explained(client, provider, monkeypatch):
    monkeypatch.setattr(config, "OIDC_ISSUER", "https://idp.example.org/wrong")
    sso.forget_provider()
    start = client.get("/api/auth/sso/start", follow_redirects=False)
    assert "issuer doesn't match" in error_of(start)
