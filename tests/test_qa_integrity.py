"""Data integrity and consistency: the audit chain verifies after every kind of write, the schema migrates up and down, a
restored copy is the same database, and every response says the same thing in all of its fields."""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, inspect, select, text  # noqa: E402

from app import apikeys, audit, auth, config, db, guards_input, integrity, pipeline, retrieval  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import PatientContext  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "correct horse battery staple"
Q = "What is the standard dose of Caloradine?"
UNKNOWN = "What is the standard dose of Quelvarin?"


@pytest.fixture(scope="module", autouse=True)
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def world(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    pipeline.clear_answer_cache()
    guards_input.key_limiter._by_client.clear()
    auth.create_user("admin@example.org", PASSWORD, role="admin", name="Admin")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/api/auth/login", json={"email": "admin@example.org", "password": PASSWORD}).status_code == 200
    made = apikeys.create("integrity", ["ask", "verify", "usage", "audit:read"])
    return client, {"Authorization": "Bearer " + made["key"]}, made


def chain_ok(expect_growth_from: int | None = None) -> int:
    result = integrity.verify()
    assert result["ok"] is True and result["complete"] is True, result
    length = result["checked"]
    if expect_growth_from is not None:
        assert length > expect_growth_from
    return length


# --- The chain after every write path ------------------------------------------------------------------------------

def test_the_chain_verifies_after_every_kind_of_write(world):
    client, headers, made = world
    steps = [
        ("dashboard answer", lambda: client.post("/api/ask", json={"query": Q})),
        ("dashboard refusal routed to review", lambda: client.post("/api/ask", json={"query": UNKNOWN})),
        ("v1 ask", lambda: client.post("/api/v1/ask", json={"question": Q}, headers=headers)),
        ("v1 ask refused", lambda: client.post("/api/v1/ask", json={"question": UNKNOWN}, headers=headers)),
        ("v1 verify pass", lambda: client.post("/api/v1/verify", json={"question": Q, "answer": "Caloradine is 15 mg once daily."}, headers=headers)),
        ("v1 verify fail", lambda: client.post("/api/v1/verify", json={"question": Q, "answer": "Caloradine is 150 mg once daily."}, headers=headers)),
        ("v1 verify rejected", lambda: client.post("/api/v1/verify", json={"question": Q, "answer": ""}, headers=headers)),
        ("key created", lambda: client.post("/api/keys", json={"name": "second", "scopes": ["ask"]})),
        ("key revoked", lambda: client.post(f"/api/keys/{made['id']}/revoke")),
        ("patient question", lambda: client.post("/api/ask", json={"query": "What is the dose of Mendel solution?",
                                                                  "patient": {"age_years": 8, "weight_kg": 20}})),
    ]
    length = chain_ok()
    for name, step in steps:
        response = step()
        assert response.status_code < 500, (name, response.text)
        grown = chain_ok()
        assert grown >= length, name
        length = grown
    assert length >= 8
    audit_id = client.post("/api/ask", json={"query": Q}).json()["audit_id"]
    assert client.post(f"/api/audit/{audit_id}/flag", json={"note": "x"}).status_code < 500
    assert client.post(f"/api/audit/{audit_id}/feedback", json={"rating": "down", "reason": "wrong", "comment": "no"}).status_code < 500
    case = client.post("/api/ask", json={"query": UNKNOWN}).json()["review"]["case_id"]
    if case:
        client.post(f"/api/reviews/{case}/comment", json={"note": "looking"})
        client.post(f"/api/reviews/{case}/resolve", json={"outcome": "answered", "note": "done"})
    chain_ok(length - 1)
    assert client.post("/api/audit/verify").json()["ok"] is True


def test_a_changed_row_is_found_and_a_deleted_row_is_found(world):
    client, headers, _ = world
    for _ in range(4):
        client.post("/api/v1/ask", json={"question": Q}, headers=headers)
    chain_ok()
    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET query = 'edited' WHERE id = (SELECT min(id) + 1 FROM audit_records)"))
    assert integrity.verify()["ok"] is False
    with db.engine().begin() as conn:
        conn.execute(text("DELETE FROM audit_records WHERE id = (SELECT max(id) - 1 FROM audit_records)"))
    assert integrity.verify()["ok"] is False


def test_the_chain_holds_under_concurrent_writes(world):
    client, headers, _ = world

    def one(i: int) -> int:
        path = "/api/v1/ask" if i % 2 else "/api/v1/verify"
        body = {"question": Q} if i % 2 else {"question": Q, "answer": "Caloradine is 15 mg once daily."}
        return client.post(path, json=body, headers=headers).status_code

    with ThreadPoolExecutor(12) as pool:
        codes = list(pool.map(one, range(48)))
    assert codes.count(200) == 48
    result = integrity.verify()
    assert result["ok"] is True, result
    with db.session() as s:
        seqs = [r[0] for r in s.execute(text("SELECT seq FROM audit_records WHERE seq IS NOT NULL ORDER BY seq")).all()]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))


# --- Migrations and backup --------------------------------------------------------------------------------------------

def _schema(connection) -> dict:
    insp = inspect(connection)
    return {t: ({c["name"]: str(c["type"]) for c in insp.get_columns(t)},
                sorted((i["name"], tuple(i["column_names"]), bool(i["unique"])) for i in insp.get_indexes(t)),
                sorted(tuple(f["constrained_columns"]) + (f["referred_table"],) for f in insp.get_foreign_keys(t)))
            for t in sorted(insp.get_table_names()) if t != "alembic_version"}


def test_migrations_go_up_and_down_and_up_again_to_the_same_schema(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config

    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'm.db'}")
    db.reset_engine()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    try:
        with db.engine().begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
            first = _schema(conn)
            assert len(first) > 20
            command.downgrade(cfg, "base")
            assert _schema(conn) == {}
            command.upgrade(cfg, "head")
            assert _schema(conn) == first
    finally:
        db.reset_engine()


def test_every_migration_steps_down_one_at_a_time(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'step.db'}")
    db.reset_engine()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    revisions = [r.revision for r in ScriptDirectory.from_config(cfg).walk_revisions()]
    try:
        with db.engine().begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
            for revision in revisions:
                command.downgrade(cfg, "-1")
            assert _schema(conn) == {}
    finally:
        db.reset_engine()


def test_a_restored_backup_is_the_same_database_and_the_chain_verifies(world, tmp_path, monkeypatch):
    client, headers, _ = world
    for i in range(6):
        client.post("/api/v1/ask", json={"question": Q if i % 2 else UNKNOWN}, headers=headers)
    client.post("/api/v1/verify", json={"question": Q, "answer": "Caloradine is 15 mg once daily."}, headers=headers)
    before = chain_ok()
    head = integrity.head()
    source = config.DATABASE_URL.removeprefix("sqlite:///")
    backup = tmp_path / "backup.db"
    src, dst = sqlite3.connect(source), sqlite3.connect(backup)
    src.backup(dst)
    src.close(), dst.close()
    with db.session() as s:
        counts = {t: s.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in inspect(db.engine()).get_table_names()}
    restored = tmp_path / "restored.db"
    shutil.copy(backup, restored)
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{restored}")
    db.reset_engine()
    audit.store._backend = None
    with db.session() as s:
        again = {t: s.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in inspect(db.engine()).get_table_names()}
    assert again == counts
    assert chain_ok() == before
    assert integrity.head() == head
    assert apikeys.create("after-restore", ["ask"])["key"].startswith("gck_")


# --- Responses say the same thing in every field --------------------------------------------------------------------

REASON_STAGES = {
    "term_not_covered": {"source coverage"}, "population_not_covered": {"source coverage"},
    "combination_forbidden": {"source coverage"}, "limit_not_stated": {"source coverage"},
    "stated_value_unsupported": {"source coverage"}, "no_relevant_source": {"retrieval gate"},
    "input_blocked": {"scope check", "pii redaction"}, "rate_limited": {"rate limit"}, "patient_unsafe": {"patient checks"},
    "dose_unsupported": {"dosage guard"}, "claim_ungrounded": {"grounding check"}, "unsupported_language": {"scope check", "language check"},
}


def _cases():
    rng = random.Random(11)
    golden = json.loads((ROOT / "eval/golden.json").read_text())
    adversarial = json.loads((ROOT / "eval/adversarial.json").read_text())
    patients = json.loads((ROOT / "eval/patient_cases.json").read_text())
    cases = [(x["query"], None) for x in rng.sample(golden, 120)] + [(x["query"], None) for x in adversarial]
    cases += [(x["query"], PatientContext(**x["patient"])) for x in rng.sample(patients, 60)]
    return cases


def test_every_field_of_an_ask_response_agrees_with_the_others(world):
    for query, patient in _cases():
        r = pipeline.run(query, patient=patient, review=True)
        where = (query, r.decision, r.reason_code)
        assert r.decision in ("answer", "refuse") and r.outcome in ("answered", "refused", "not_evaluated"), where
        if r.decision == "answer":
            assert r.outcome == "answered" and r.reason_code is None and not r.refused_reason, where
            assert r.claims and all(c.grounded for c in r.claims), where
            assert not r.review.routed and r.review.case_id is None, where
        else:
            assert r.outcome in ("refused", "not_evaluated"), where
            assert r.reason_code is not None and r.refused_reason, where
            assert r.answer_text and not any(c.grounded and c.text in r.answer_text for c in r.claims), where
            failing = {t.name for t in r.trace if t.status == "fail"} - {"decision"}
            expected = REASON_STAGES.get(r.reason_code)
            if expected is not None and r.reason_code != "rate_limited":
                assert failing & expected, (where, failing)
        assert r.review.routed == (r.review.case_id is not None), where
        if r.review.case_id is not None:
            with db.session() as s:
                assert s.scalar(select(func.count()).select_from(db.ReviewCase).where(db.ReviewCase.id == r.review.case_id)) == 1, where
        assert all(0.0 <= c.relevance <= 1.0001 and 0.0 <= c.grounding_score <= 1.0001 for c in r.claims), where
        for c in r.claims:
            if c.support is not None:
                record = retrieval.record_for(c.support.source_id)
                assert record is not None, where
                assert record["text"][c.support.start:c.support.end] == c.support.sentence, (where, c.text)
                assert c.support.source_id in c.source_ids or not c.source_ids, (where, c.text)
            for source_id in c.source_ids:
                assert retrieval.record_for(source_id) is not None, (where, source_id)
        assert [s.id for s in r.sources] == list(dict.fromkeys(s.id for s in r.sources)), where
        stored = audit.store.get(r.audit_id)
        assert stored is not None and stored["response"]["decision"] == r.decision, where
        assert stored["response"]["reason_code"] == r.reason_code, where


def test_a_verify_response_agrees_with_itself(world):
    client, headers, _ = world
    answers = ["Caloradine is 15 mg once daily.", "Caloradine is 150 mg once daily.", "Caloradine is 15 mg twice daily.",
               "I cannot answer that.", "Mendel solution is 5 mL once daily for 7 days.", "Caloradine is used for Veltris syndrome.",
               "Caloradine can be combined with Orrin-blockers.", "The sky is blue. Caloradine is 15 mg once daily."]
    for answer in answers:
        r = client.post("/api/v1/verify", json={"question": Q, "answer": answer}, headers=headers).json()
        where = (answer, r["decision"], r["reason_code"])
        assert r["decision"] in ("pass", "fail", "review"), where
        assert (r["decision"] == "pass") == (r["reason_code"] == "verified"), where
        assert r["unsupported_claims"] == sorted(set(r["unsupported_claims"])), where
        assert all(0 <= n < len(r["claims"]) for n in r["unsupported_claims"]), where
        if r["decision"] == "pass":
            assert all(c["grounded"] for c in r["claims"]) and not r["unsupported_claims"] and not r["dose_issues"], where
            assert not r["review"]["routed"], where
        assert all(0 <= d["claim"] < len(r["claims"]) for d in r["dose_issues"]), where
        if r["dose_issues"]:
            assert r["decision"] == "fail" and r["reason_code"] in ("dose_unsupported", "patient_unsafe"), where
        for claim in r["claims"]:
            if claim.get("support"):
                record = retrieval.record_for(claim["support"]["source_id"])
                assert record["text"][claim["support"]["start"]:claim["support"]["end"]] == claim["support"]["sentence"], where
            if not claim["grounded"]:
                assert claim["issues"] or r["decision"] != "pass", where
        assert r["review"]["routed"] == (r["review"]["case_id"] is not None), where
        stored = client.get(f"/api/v1/audit/{r['audit_id']}", headers=headers).json()
        assert stored["decision"] == r["decision"] and stored["endpoint"] == "verify", where


def test_a_review_that_is_said_to_be_routed_is_in_the_queue_and_one_that_is_not_says_why(world):
    client, _, _ = world
    routed = client.post("/api/ask", json={"query": UNKNOWN}).json()
    assert routed["review"]["routed"] and routed["review"]["case_id"]
    cases = client.get("/api/reviews?status=all").json()
    assert any(c["id"] == routed["review"]["case_id"] for c in cases["cases"])
    test_run = pipeline.run(UNKNOWN, review=False)
    assert not test_run.review.routed and test_run.review.note
    answered = client.post("/api/ask", json={"query": Q}).json()
    assert not answered["review"]["routed"]
