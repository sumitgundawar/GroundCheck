"""Round 2, tenants through API keys: a key of one site learns nothing about another site's documents, sites or answers."""

from __future__ import annotations

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

from app import apikeys  # noqa: E402
from test_tenant_documents import ask_text, docs, world  # noqa: E402,F401


@pytest.fixture()
def call(docs):
    from app.main import app

    made, client, ids = docs
    raw = TestClient(app)

    def go(who, path, body):
        return raw.post(f"/api/v1/{path}", json=body, headers={"Authorization": "Bearer " + made[who]["api_key"]["key"]})

    return go, made, ids


def blank(text: str, *names: str) -> str:
    for n in names:
        text = re.sub(n, "NAME", text, flags=re.I)
    return text


def test_a_refusal_about_another_sites_medicine_looks_like_a_refusal_about_an_unknown_one(call):
    go, made, ids = call
    other = go("alpha", "ask", {"question": ask_text("Betazol")}).json()
    unknown = go("alpha", "ask", {"question": ask_text("Zzyzxol")}).json()
    assert other["decision"] == unknown["decision"] == "refuse"
    for field in ("reason_code", "answer_text", "refused_reason"):
        assert blank(str(other.get(field)), "Betazol") == blank(str(unknown.get(field)), "Zzyzxol"), field
    assert all(not x["id"].startswith(f"DOC{ids['Betazol']}-") and "Betazol" not in x["title"] for x in other["sources"])


def test_a_near_miss_of_another_sites_medicine_does_not_suggest_it(call):
    go, made, ids = call
    for q in ("Betazoll", "Betazal", "betazol", "BETAZOL", "Betazol-X"):
        body = go("alpha", "ask", {"question": ask_text(q)}).json()
        assert "Betazol" not in str(body.get("answer_text")) and not body.get("claims"), (q, body)


def test_verify_does_not_say_whether_another_sites_source_or_site_exists(call):
    go, made, ids = call
    mine = {"question": ask_text("Betazol"), "answer": "Give the first dose of Betazol within 60 minutes of recognition."}
    theirs = go("alpha", "verify", {**mine, "sources": [f"DOC{ids['Betazol']}-001"]})
    nowhere = go("alpha", "verify", {**mine, "sources": ["DOC99999-001"]})
    assert theirs.status_code == nowhere.status_code == 422
    assert blank(theirs.text, f"DOC{ids['Betazol']}-001") == blank(nowhere.text, "DOC99999-001")
    beta_key = made["beta"]["site"]["key"]
    real = go("alpha", "ask", {"question": ask_text("Alphazol"), "site": beta_key})
    fake = go("alpha", "ask", {"question": ask_text("Alphazol"), "site": "no-such-site"})
    assert real.status_code == fake.status_code == 403, (real.text, fake.text)
    assert real.json()["error"]["code"] == fake.json()["error"]["code"]


def test_a_site_bound_key_cannot_read_another_sites_audit_record(call):
    go, made, ids = call
    from app.main import app

    raw = TestClient(app)
    audit_id = go("beta", "ask", {"question": ask_text("Betazol")}).json()["audit_id"]
    scoped = apikeys.create("audit", ["audit:read"], site_id=made["alpha"]["site"]["id"])
    mine = raw.get(f"/api/v1/audit/{audit_id}", headers={"Authorization": "Bearer " + scoped["key"]})
    gone = raw.get("/api/v1/audit/does-not-exist", headers={"Authorization": "Bearer " + scoped["key"]})
    assert mine.status_code == gone.status_code == 404
