"""Random and hostile input at every v1 endpoint and every admin or clinician route: never a 500, and on /api/v1 always
the error envelope. Seeded, standard library only."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import apikeys, auth, config, guards_input, retrieval  # noqa: E402
from app.main import app  # noqa: E402

PASSWORD = "correct horse battery staple"
Q = "What is the standard dose of Caloradine?"
JSON = {"content-type": "application/json"}
SCALARS = [None, True, False, 0, -1, 1, 3.14, 1e308, -1e308, 2 ** 70, -(2 ** 70), "", " ", "a", "x" * 300, "é\u0000", "1",
           "true", "[]", "{}", "../../etc/passwd", "<script>alert(1)</script>", "' OR 1=1--", "😀", "NaN", "-0",
           "gck_x", "2026-13-45", "9" * 30, "‮", "\n\r\t", "%s%n", "{{7*7}}", "${jndi:ldap://x}", "\\", "\"", "\x7f",
           Q, "John Smith 12/03/1980 NHS 943 476 5919", "caloradine " * 400]
KEYS = ["id", "name", "email", "question", "text", "note", "status", "role", "site", "site_id", "patient", "sources", "answer",
        "limit", "offset", "from", "to", "scopes", "password", "comment", "title", "kind", "value", "age_years", "weight_kg",
        "egfr", "medicines", "allergies", "labs", "pregnant", "sex", "child_pugh", "expires_in_days", "rpm", "quota"]


def value(rng: random.Random, depth: int = 0):
    roll = rng.randint(0, 9)
    if depth > 2 or roll < 5:
        return rng.choice(SCALARS)
    if roll < 8:
        return [value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(KEYS): value(rng, depth + 1) for _ in range(rng.randint(0, 5))}


def raw_body(rng: random.Random) -> bytes:
    roll = rng.random()
    if roll < 0.8:
        return json.dumps(value(rng), allow_nan=False).encode()
    return rng.choice([b"{", b"[1,", b"\xff\xfe", b"null", b'{"a":NaN}', b"", b"{" * 5000, b"[" * 5000, b'{"question":"' + b"x" * 70000 + b'"}',
                       b'{"question":"\\ud800"}', b'{"question":"a","question":"b"}', b"\x00" * 10, b"1" * 5000, b"true"])


def query(rng: random.Random) -> dict:
    out = {}
    for key in rng.sample(["limit", "offset", "from", "to", "q", "group_by", "status", "days", "format", "site", "start", "end"],
                          rng.randint(0, 3)):
        v = rng.choice(SCALARS)
        if v is not None:
            out[key] = v if isinstance(v, (str, int, float)) else json.dumps(v)
    return out


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def world(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    guards_input.key_limiter._by_client.clear()
    auth.create_user("admin@example.org", PASSWORD, role="admin", name="Admin")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/auth/login", json={"email": "admin@example.org", "password": PASSWORD}).status_code == 200
    made = apikeys.create("fuzz", ["ask", "verify", "usage", "audit:read"])
    return client, {"Authorization": "Bearer " + made["key"]}, made


def envelope_ok(response) -> None:
    if response.status_code < 400:
        return
    body = response.json()
    assert set(body) <= {"error"} and set(body["error"]) <= {"code", "message", "details"}, body
    assert isinstance(body["error"]["code"], str) and isinstance(body["error"]["message"], str)


# --- /api/v1 ---------------------------------------------------------------------------------------------------

V1 = [("POST", "/api/v1/ask"), ("POST", "/api/v1/verify"), ("GET", "/api/v1/usage"), ("GET", "/api/v1/audit/{id}"),
      ("GET", "/api/v1/health")]


@pytest.mark.parametrize("seed", range(6))
def test_v1_never_fails_with_a_server_error_and_always_uses_the_envelope(world, seed):
    client, headers, _ = world
    rng = random.Random(seed)
    for _ in range(40):
        method, path = rng.choice(V1)
        path = path.format(id=rng.choice(["0123456789abcdef", "x" * 100, "%00", "..", "%2e%2e", "ä", "-1", " "]))
        h = dict(JSON)
        roll = rng.random()
        if roll < 0.7:
            h.update(headers)
        elif roll < 0.8:
            h["X-API-Key"] = headers["Authorization"][7:]
        elif roll < 0.9:
            h["Authorization"] = ("Bearer " + str(rng.choice(SCALARS + [None]))).encode("utf-8", "replace")
        r = client.request(method, path, params=query(rng), headers=h,
                           content=raw_body(rng) if method == "POST" else None)
        assert r.status_code < 500, (method, path, r.status_code, r.text[:200])
        envelope_ok(r)
        assert r.headers.get("x-request-id")


@pytest.mark.parametrize("body", [
    {"question": Q, "answer": "x", "sources": ["CALO-001"] * 500}, {"question": Q, "answer": "x" * 200000},
    {"question": Q, "answer": ""}, {"question": "", "answer": "x"}, 
    {"question": Q, "answer": "15 mg once daily", "patient": {"age_years": "x"}},
    {"question": Q, "answer": "15 mg once daily", "patient": {"labs": {"x": 1}}},
    {"question": Q, "answer": "15 mg once daily", "patient": {"allergies": ["a"] * 31}},
    {"question": Q, "answer": "15 mg once daily", "patient": {"weight_kg": 1e30}},
    {"question": Q, "answer": "15 mg once daily", "site": "nope"}, {"question": Q, "answer": "15 mg once daily", "extra": 1},
    {"question": ["a"], "answer": "x"}, {"question": Q, "answer": {"a": 1}}, [], "x", 1, None,
])
def test_v1_verify_rejects_bad_bodies_with_4xx(world, body):
    client, headers, _ = world
    r = client.post("/api/v1/verify", json=body, headers=headers)
    assert 400 <= r.status_code < 500, r.text
    envelope_ok(r)


def test_deeply_nested_json_is_a_4xx_not_a_crash(world):
    client, headers, _ = world
    for depth in (500, 100000):
        r = client.post("/api/v1/ask", content=b'{"question":' + b"[" * depth + b"]" * depth + b"}", headers={**headers, **JSON})
        assert 400 <= r.status_code < 500, depth
        envelope_ok(r)


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "", "application/json; charset=utf-16",
                                          "application/xml"])
def test_v1_other_content_types_are_4xx(world, content_type):
    client, headers, _ = world
    r = client.post("/api/v1/ask", content=json.dumps({"question": Q}).encode(), headers={**headers, "content-type": content_type})
    assert r.status_code < 500
    envelope_ok(r)


def test_v1_unknown_methods_and_paths_use_the_envelope(world):
    client, headers, _ = world
    for method, path in (("DELETE", "/api/v1/ask"), ("PUT", "/api/v1/verify"), ("GET", "/api/v1/ask"), ("POST", "/api/v1/usage"),
                         ("GET", "/api/v1/nothing"), ("GET", "/api/v1/"), ("PATCH", "/api/v1/audit/abc")):
        r = client.request(method, path, headers=headers)
        assert r.status_code in (404, 405), (method, path, r.status_code)
        envelope_ok(r)


# --- Admin and clinician routes ---------------------------------------------------------------------------------

SKIP = ("pull", "/sso", "/ehr/launch", "/ehr/callback", "pacs", "/training/runs", "monitoring/evaluate", "index/rebuild",
        "review-tests/run", "/local-ai/select", "retention/run", "/metrics", "logout", "/auth/password", "/auth/mfa",
        "imaging/upload", "/api/sources", "/api/training/browse", "/auth/first-admin", "/api/users", "/api/models/",
        "/api/imaging/series", "/api/imaging/reports", "/api/training/image", "/api/training/dataset", "/api/releases/rollback",
        "/api/releases/", "/api/audit/verify")


def admin_routes():
    out = []
    for r in app.routes:
        for m in getattr(r, "methods", None) or ():
            if m not in ("HEAD", "OPTIONS") and r.path.startswith(("/api", "/cds")) and not any(s in r.path for s in SKIP):
                out.append((m, r.path))
    return sorted(set(out))


@pytest.mark.parametrize("seed", range(3))
def test_admin_routes_never_fail_with_a_server_error(world, seed):
    client, _, _ = world
    rng = random.Random(100 + seed)
    failures = []
    for method, path in admin_routes():
        for _ in range(4):
            p = re.sub(r"\{[^}]+\}", lambda _m: rng.choice(["1", "0", "-1", "abc", "9" * 25, "%00", "x" * 200, "0123456789abcdef"]), path)
            params, content = query(rng), (None if method in ("GET", "DELETE") else raw_body(rng))
            r = client.request(method, p, params=params, content=content, headers=JSON if content is not None else None)
            if r.status_code >= 500:
                failures.append((method, p, r.status_code, params, (content or b"")[:200]))
    assert not failures, "\n".join(sorted(set(map(str, failures))))


def test_every_route_without_a_session_is_401_or_403_never_data(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    anonymous = TestClient(app, raise_server_exceptions=False)
    open_prefixes = ("/api/health", "/api/auth", "/api/ask", "/api/settings", "/api/examples", "/api/eval-summary",
                     "/api/v1", "/cds-services", "/healthz", "/metrics", "/api/ehr", "/api/corpus", "/api/data-protection",
                     "/api/formulary", "/api/embeddings", "/api/governance/safety-case", "/api/index", "/api/local-ai")
    leaks = []
    for method, path in admin_routes():
        if path.startswith(open_prefixes):
            continue
        p = re.sub(r"\{[^}]+\}", "1", path)
        r = anonymous.request(method, p, content=b"{}", headers=JSON)
        if r.status_code not in (401, 403):
            leaks.append((method, path, r.status_code))
    assert not leaks, leaks


# --- Routes skipped above because they need real objects: users, sources, models and imaging ---------------------

ODD_NAMES = ["", ".", "..", "../../etc/passwd", "a\x00b.md", "x" * 400 + ".md", "névé.md", "‮gnp.md", "CON", "a b\tc.md",
             "John Smith 12-03-1980.md", "<img src=x onerror=alert(1)>.md", "a;rm -rf.md", "%2e%2e%2f.md", "ä" * 120 + ".txt"]
ODD_DATA = [b"", b"\x00" * 100, b"\xff\xfe\xfd", b"# T\n\n## S\n\nbody\n", b"PK\x03\x04" + b"x" * 50, b"%PDF-1.4\n" + b"x" * 100,
            b"<html><script>1</script></html>", "é".encode("utf-16"), b"x" * 5_000_000]


@pytest.mark.parametrize("seed", range(3))
def test_uploads_with_hostile_names_and_bodies_never_crash(world, seed):
    client, _, _ = world
    rng = random.Random(500 + seed)
    for _ in range(25):
        name, data = rng.choice(ODD_NAMES), rng.choice(ODD_DATA)
        kind = rng.choice(["text/markdown", "application/pdf", "image/png", "", "application/octet-stream", "text/html"])
        for path in ("/api/sources", "/api/imaging/upload"):
            r = client.post(path, files={"file": (name, data, kind)},
                            data={"title": rng.choice(SCALARS).__str__(), "site_id": rng.choice(["", "1", "x", "-1", "9" * 25])})
            assert r.status_code < 500, (path, name, kind, len(data), r.status_code, r.text[:200])


@pytest.mark.parametrize("seed", range(3))
def test_user_and_model_routes_never_crash(world, seed):
    client, _, _ = world
    rng = random.Random(900 + seed)
    for _ in range(40):
        method, path = rng.choice([("POST", "/api/users"), ("PATCH", "/api/users/{i}"), ("DELETE", "/api/users/{i}"),
                                   ("GET", "/api/models/{i}"), ("DELETE", "/api/models/{i}"), ("POST", "/api/models/{i}/predict"),
                                   ("PUT", "/api/models/{i}/imaging"), ("GET", "/api/imaging/series/{i}"),
                                   ("GET", "/api/imaging/series/{i}/slices/{i}.png"), ("POST", "/api/imaging/series/{i}/analyses"),
                                   ("POST", "/api/imaging/series/{i}/reports"), ("GET", "/api/imaging/reports/{i}/sr.dcm"),
                                   ("GET", "/api/sources/{i}"), ("POST", "/api/sources/{i}/approve"), ("POST", "/api/sources/{i}/evaluate")])
        p = path.format(i=rng.choice(["1", "2", "0", "-1", "abc", "9" * 25, "x" * 100]))
        content = raw_body(rng) if method in ("POST", "PATCH", "PUT") else None
        r = client.request(method, p, params=query(rng), content=content, headers=JSON if content is not None else None)
        assert r.status_code < 500, (method, p, r.status_code, (content or b"")[:200])
