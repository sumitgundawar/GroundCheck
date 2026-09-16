"""API-surface tests using FastAPI's TestClient. These run offline (extractive
mode) and cover the endpoints added beyond the core pipeline: health, examples,
settings, corpus map, audit list, and the ask route with tuning overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
# Deterministic: never draft with a cloud or a selected local model.
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import retrieval  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # Ensure the index exists before the app starts.
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corpus"] > 0
    assert isinstance(body["llm"], bool)


def test_examples(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    examples = r.json()
    assert len(examples) >= 4
    assert all("query" in e and "label" in e for e in examples)


def test_settings_defaults_and_bounds(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "defaults" in body and "bounds" in body
    d = body["defaults"]
    for key in ("retrieval_min_score", "grounding_min", "top_k",
                "enable_coverage_guard", "enable_grounding_guard",
                "enable_dosage_guard", "use_llm_judge"):
        assert key in d


def test_corpus_map(client):
    r = client.get("/api/corpus")
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["total"] == len(body["points"])
    # 3D projection: each point has x, y, z and a kind.
    p = body["points"][0]
    assert {"x", "y", "z", "kind", "id"} <= set(p)
    # Topic list is present and non-empty.
    assert body["stats"]["topic_list"]


def test_ask_answer_and_audit_roundtrip(client):
    r = client.post("/api/ask", json={"query": "What is the dose of Caloradine?"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "answer"
    assert body["sources"]
    # Source cards carry the richer metadata.
    assert {"rank", "kind", "section", "above_gate"} <= set(body["sources"][0])
    # Trace steps carry plain-English explanations.
    assert all("explain" in s for s in body["trace"])
    # The audit record is retrievable by id.
    audit_id = body["audit_id"]
    rec = client.get(f"/api/audit/{audit_id}")
    assert rec.status_code == 200
    assert rec.json()["audit_id"] == audit_id


def test_ask_refuses_unknown_drug(client):
    r = client.post("/api/ask", json={"query": "What is the dose of Zalortin?"})
    assert r.json()["decision"] == "refuse"


def test_ask_with_settings_override_changes_decision(client):
    q = {"query": "What is the dose of Zalortin?",
         "settings": {"enable_coverage_guard": False}}
    # With the coverage guard off, the unknown-drug trap is no longer caught
    # pre-generation, so the ungoverned pipeline returns an answer.
    assert client.post("/api/ask", json=q).json()["decision"] == "answer"


def test_audit_list_endpoint(client):
    client.post("/api/ask", json={"query": "What is the dose of Caloradine?"})
    r = client.get("/api/audit")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert isinstance(body["recent"], list)


def test_unknown_audit_id_is_404(client):
    assert client.get("/api/audit/deadbeef").status_code == 404


def test_query_validation_rejects_empty(client):
    # Empty query violates the request schema (min_length=1).
    assert client.post("/api/ask", json={"query": ""}).status_code == 422
