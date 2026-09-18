"""Protecting stored data: encryption at rest and key rotation, the
tamper-evident audit chain, retention, and deleting a user. Runs on SQLite,
and on PostgreSQL when TEST_POSTGRES_URL is set."""

from __future__ import annotations

import os
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Text, select, text, type_coerce, update  # noqa: E402

from app import (  # noqa: E402
    audit, auth, config, db, encryption, governance, integrity, pipeline, rekey, retention, retrieval,
)
from app.schemas import AskResponse  # noqa: E402

PASSWORD = "correct horse battery staple"
REFUSED = "What is the standard dose of Zyntrafen?"
ANSWERED = "What is the standard dose of Caloradine?"
KEY_A = encryption.generate_key()
KEY_B = encryption.generate_key()


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", "")
    monkeypatch.setattr(config, "DATA_ENCRYPTION_RETIRED_KEYS", "")
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", "")
    encryption.reset()
    yield
    monkeypatch.undo()
    encryption.reset()


def use_keys(monkeypatch, keys: str, retired: str = ""):
    monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", keys)
    monkeypatch.setattr(config, "DATA_ENCRYPTION_RETIRED_KEYS", retired)
    encryption.reset()


def raw(table: str, column: str, where: str = "1=1") -> list:
    """Stored values as they are on disk, without decrypting."""
    t = db.Base.metadata.tables[table]
    with db.engine().connect() as conn:
        return [r[0] for r in conn.execute(select(type_coerce(t.c[column], Text())).where(text(where))
                                           .order_by(t.c.id))]


# --- Encryption ------------------------------------------------------------------

def test_keys_must_be_32_base64_bytes(monkeypatch):
    assert len(encryption._decode_key(encryption.generate_key())) == 32
    for bad in ("short", "not base64!!", encryption.generate_key()[:-8]):
        use_keys(monkeypatch, bad)
        with pytest.raises(encryption.EncryptionError, match="32 random bytes"):
            encryption.keyring()
    use_keys(monkeypatch, f"{KEY_A},{KEY_A}")
    with pytest.raises(encryption.EncryptionError, match="twice"):
        encryption.keyring()


def test_values_are_bound_to_their_column_and_key(monkeypatch):
    use_keys(monkeypatch, KEY_A)
    ring = encryption.keyring()
    token = ring.encrypt("patient on 15 mg", "audit_records.query")
    assert token.startswith("gcenc:v1:") and "15 mg" not in token
    assert token != ring.encrypt("patient on 15 mg", "audit_records.query")  # random nonce
    assert ring.decrypt(token, "audit_records.query") == "patient on 15 mg"
    with pytest.raises(encryption.EncryptionError, match="integrity check"):
        ring.decrypt(token, "review_cases.query")
    tampered = token[:-6] + ("A" if token[-6] != "A" else "B") + token[-5:]
    with pytest.raises(encryption.EncryptionError):
        ring.decrypt(tampered, "audit_records.query")
    assert ring.decrypt("written before encryption", "audit_records.query") == "written before encryption"

    use_keys(monkeypatch, KEY_B)
    with pytest.raises(encryption.EncryptionError, match="isn't in DATA_ENCRYPTION_KEYS"):
        encryption.keyring().decrypt(token, "audit_records.query")


def test_questions_answers_reviews_and_2fa_secrets_are_encrypted_on_disk(database, monkeypatch):
    use_keys(monkeypatch, KEY_A)
    answer = pipeline.run(ANSWERED)
    pipeline.run(REFUSED)
    user = auth.create_user("clin@example.org", PASSWORD)
    secret = auth.begin_mfa_setup(user.id)["secret"]
    case = governance.list_cases()["cases"][0]
    governance.comment(case["id"], "Checked the Zyntrafen formulary entry.", None)

    for table, column in (("audit_records", "query"), ("audit_records", "record"), ("review_cases", "query"),
                          ("review_cases", "reason"), ("review_events", "note"), ("users", "mfa_secret")):
        values = raw(table, column)
        assert values and all(v.lstrip('"').startswith("gcenc:v1:") for v in values), (table, column)
    stored = " ".join(raw("audit_records", "record") + raw("review_events", "note"))
    assert "Caloradine" not in stored and "Zyntrafen" not in stored and secret not in stored

    # Everything reads back normally.
    assert audit.store.get(answer.audit_id)["response"]["decision"] == "answer"
    assert governance.get_case(case["id"])["query"] == REFUSED
    assert governance.get_case(case["id"])["answer_text"]
    assert [r["query"] for r in audit.store.recent(5)] == [REFUSED, ANSWERED]


def test_rotating_keys_and_turning_encryption_off(database, monkeypatch):
    pipeline.run(REFUSED)  # written before encryption was on
    use_keys(monkeypatch, KEY_A)
    pipeline.run(ANSWERED)
    assert not raw("audit_records", "query")[0].startswith("gcenc:")

    use_keys(monkeypatch, KEY_B, retired=KEY_A)
    result = rekey.reencrypt()
    assert result["rewritten"]["audit_records"] == 2
    kid = encryption.key_id(encryption._decode_key(KEY_B))
    assert all(v.lstrip('"').startswith(f"gcenc:v1:{kid}:") for v in raw("audit_records", "record"))
    assert rekey.reencrypt()["rewritten"]["audit_records"] == 0  # nothing left to do

    use_keys(monkeypatch, KEY_A)  # without the new key, verification stops rather than guessing
    stopped = integrity.verify()
    assert not stopped["ok"] and not stopped["complete"] and stopped["problems"] == []
    assert "isn't in DATA_ENCRYPTION_KEYS" in stopped["error"]

    use_keys(monkeypatch, KEY_B)  # the old key is gone
    pipeline.run(REFUSED)  # still finds the open case through the re-keyed lookup hash
    assert governance.list_cases()["cases"][0]["occurrences"] == 2
    assert integrity.verify()["ok"]

    use_keys(monkeypatch, "", retired=KEY_B)
    rekey.reencrypt()
    assert raw("audit_records", "query") == [REFUSED, ANSWERED, REFUSED]
    assert integrity.verify()["ok"]


# --- Audit chain ---------------------------------------------------------------------

def _write(n: int, created_at=None) -> None:
    for i in range(n):
        response = AskResponse(decision="refuse", answer_text="", refused_reason="test", claims=[], sources=[],
                               trace=[], audit_id=audit.store.new_id(), total_ms=i, llm_used=False)
        with integrity.write_lock, db.session() as s:
            integrity.append(s, db.AuditRecord(audit_id=response.audit_id, decision="refuse", query=f"q{i}",
                                               total_ms=i, llm_used=False, record=response.model_dump(),
                                               created_at=created_at))


def test_an_untouched_chain_verifies(database):
    pipeline.run(ANSWERED)
    pipeline.run(REFUSED)
    _write(3)
    result = integrity.verify()
    assert result["ok"] and result["checked"] == 5 and result["head"]["seq"] == 5
    assert (result["first_seq"], result["last_seq"], result["signed"]) == (1, 5, False)
    with db.session() as s:
        rows = s.scalars(select(db.AuditRecord).order_by(db.AuditRecord.seq)).all()
    assert rows[0].prev_hash == integrity.GENESIS
    assert all(b.prev_hash == a.entry_hash for a, b in zip(rows, rows[1:]))


def test_changed_deleted_and_truncated_records_are_detected(database):
    _write(6)
    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET decision = 'answer' WHERE seq = 2"))
    result = integrity.verify()
    assert not result["ok"]
    assert result["problems"] == [{"seq": 2, "audit_id": result["problems"][0]["audit_id"],
                                   "problem": "Its content has changed since it was written."}]

    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET decision = 'refuse' WHERE seq = 2"))
        conn.execute(text("DELETE FROM audit_records WHERE seq = 4"))
    problems = [p["problem"] for p in integrity.verify()["problems"]]
    assert problems == ["1 record is missing before number 5.", "Doesn't follow the record before it."]

    _write(1)  # seq 7
    with db.engine().begin() as conn:
        conn.execute(text("DELETE FROM audit_records WHERE seq >= 6"))
    assert any("newest records" in p["problem"] for p in integrity.verify()["problems"])


def test_a_signed_chain_needs_its_key(database, monkeypatch):
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", KEY_A)
    _write(2)
    with db.session() as s:
        assert {r.chain_alg for r in s.scalars(select(db.AuditRecord))} == {
            f"hmac-sha256:{encryption.key_id(encryption._decode_key(KEY_A))}"}
    assert integrity.verify()["ok"] and integrity.verify()["signed"]

    # Recomputing a changed record's hash without the key doesn't hide it.
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", KEY_B)
    result = integrity.verify()
    assert not result["ok"] and result["unverifiable"] == 2

    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", f"{KEY_B},{KEY_A}")  # rotated: new records use B
    _write(1)
    assert integrity.verify()["ok"]


def test_concurrent_writers_keep_one_unbroken_chain(database):
    errors = []

    def worker():
        try:
            _write(10)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    result = integrity.verify()
    assert result["ok"] and result["checked"] == 40 and result["head"]["seq"] == 40


def test_existing_records_are_chained_by_the_migration(database):
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(config.ROOT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(config.ROOT_DIR / "migrations"))
    with db.engine().begin() as conn:
        cfg.attributes["connection"] = conn
        command.downgrade(cfg, "0003")
    now = db.utcnow().replace(tzinfo=None, microsecond=0)
    with db.engine().begin() as conn:
        for i in range(3):
            conn.execute(text("INSERT INTO audit_records (audit_id, created_at, decision, query, total_ms, llm_used, record) "
                              "VALUES (:a, :c, 'refuse', :q, 5, :f, :r)"),
                         {"a": f"old{i}", "c": (now + timedelta(seconds=i)).isoformat(sep=" "), "q": f"old question {i}", "f": False,
                          "r": '{"response": {"decision": "refuse", "score": 0.1234567890123}}'})
    with db.engine().begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
    result = integrity.verify()
    assert result["ok"] and result["checked"] == 3
    _write(1)
    assert integrity.verify()["ok"] and integrity.verify()["head"]["seq"] == 4


# --- Retention -------------------------------------------------------------------------

def test_retention_deletes_the_oldest_records_and_the_chain_still_verifies(database, monkeypatch):
    old = db.utcnow() - timedelta(days=40)
    _write(3, created_at=old)
    _write(2)
    monkeypatch.setattr(config, "AUDIT_RETENTION_DAYS", 30)
    plan = retention.plan()
    assert plan["audit_records_to_delete"] == 3 and plan["last_run"] is None

    run = retention.apply()
    assert run["audit_deleted"] == 3 and run["anchor_seq"] == 3
    assert audit.store.count() == 2
    result = integrity.verify()
    assert result["ok"] and result["first_seq"] == 4 and result["anchor_seq"] == 3
    assert retention.plan()["last_run"]["audit_deleted"] == 3
    assert retention.apply()["audit_deleted"] == 0


def test_retention_is_off_by_default(database):
    _write(2, created_at=db.utcnow() - timedelta(days=5000))
    assert retention.plan()["audit_records_to_delete"] == 0
    assert retention.apply()["audit_deleted"] == 0 and audit.store.count() == 2


def test_resolved_reviews_are_deleted_after_their_period_and_open_ones_never(database, monkeypatch):
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    pipeline.run(REFUSED)
    pipeline.run("What is the standard dose of Quelvarin?")
    closed, still_open = governance.list_cases()["cases"]
    governance.resolve(closed["id"], "add_test", "", None, "refuse")
    monkeypatch.setattr(config, "REVIEW_RETENTION_DAYS", 30)
    assert retention.plan(now=db.utcnow() + timedelta(days=31))["review_cases_to_delete"] == 1

    assert retention.apply(now=db.utcnow() + timedelta(days=31))["reviews_deleted"] == 1
    assert [c["id"] for c in governance.list_cases("all")["cases"]] == [still_open["id"]]
    assert governance.list_eval_cases()[0]["from_case_id"] is None  # the test stays
    with db.session() as s:
        assert s.scalars(select(db.ReviewEvent).where(db.ReviewEvent.case_id == closed["id"])).all() == []


# --- Deleting a user ---------------------------------------------------------------------

def test_deleting_a_user_removes_personal_details_but_keeps_their_records(database):
    admin = auth.create_user("admin@example.org", PASSWORD, role="admin")
    clin = auth.create_user("jane.doe@example.org", PASSWORD, name="Jane Doe")
    auth.begin_mfa_setup(clin.id)
    pipeline.run(ANSWERED, user_id=clin.id)
    with pytest.raises(auth.AuthError, match="own account"):
        auth.delete_user(admin.id, acting_user_id=admin.id)
    with pytest.raises(auth.AuthError, match="one active admin"):
        auth.delete_user(admin.id)

    auth.delete_user(clin.id, acting_user_id=admin.id)
    users = {u["id"]: u for u in auth.list_users()}
    assert users[clin.id]["email"] == f"deleted-user-{clin.id}@deleted.invalid"
    assert users[clin.id]["name"] == "" and not users[clin.id]["is_active"]
    with pytest.raises(auth.AuthError):
        auth.sign_in("jane.doe@example.org", PASSWORD)
    assert audit.store.recent(5, user_id=clin.id)[0]["decision"] == "answer"


# --- API ---------------------------------------------------------------------------------

def test_api_permissions(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    auth.create_user("admin@example.org", PASSWORD, role="admin")
    auth.create_user("rev@example.org", PASSWORD, role="reviewer")
    clin = auth.create_user("clin@example.org", PASSWORD)
    from app.main import app

    with TestClient(app) as client:
        def login(email):
            client.post("/api/auth/logout")
            assert client.post("/api/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200

        login("rev@example.org")
        assert client.post("/api/ask", json={"query": ANSWERED}).status_code == 200
        verified = client.post("/api/audit/verify")
        assert verified.status_code == 200 and verified.json()["ok"] and verified.json()["checked"] == 1
        assert client.get("/api/data-protection").status_code == 403
        assert client.post("/api/retention/run").status_code == 403
        assert client.delete(f"/api/users/{clin.id}").status_code == 403

        login("admin@example.org")
        status = client.get("/api/data-protection").json()
        assert status["encryption"]["enabled"] is False and status["audit_chain"]["seq"] == 1
        assert status["retention"]["audit_retention_days"] == 0
        assert client.post("/api/retention/run").json()["run"]["audit_deleted"] == 0
        assert client.delete(f"/api/users/{clin.id}").json()["user"]["deleted"] is True


def test_moving_a_record_to_another_site_breaks_the_chain(database):
    """Which site an answer belongs to decides who can see it, so a change of
    site has to be as visible as a change of the answer itself."""
    from app import sites

    one = sites.create("site-one", "Site One")
    two = sites.create("site-two", "Site Two")
    _write(3)
    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET site_id = :s"), {"s": one["id"]})
    # Re-chain the records now that they carry a site, then confirm a move is caught.
    with db.session() as s:
        rows = s.scalars(select(db.AuditRecord).order_by(db.AuditRecord.seq)).all()
        prev = integrity.GENESIS
        for row in rows:
            row.prev_hash = prev
            row.chain_alg, row.entry_hash = integrity.compute(prev, integrity._row_content(row))
            prev = row.entry_hash
        s.execute(update(db.AuditChain).where(db.AuditChain.id == 1).values(last_hash=prev))
    assert integrity.verify()["ok"]

    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET site_id = :s WHERE seq = 2"), {"s": two["id"]})
    result = integrity.verify()
    assert not result["ok"]
    assert result["problems"][0]["problem"] == "Its content has changed since it was written."

    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET site_id = :s WHERE seq = 2"), {"s": one["id"]})
    assert integrity.verify()["ok"]


def test_records_written_before_the_site_was_covered_still_verify(database):
    """The hashed content gained a field after the first release. Records
    written before that still verify, and are counted so an operator can see
    how many predate the change."""
    _write(2)
    with db.session() as s:
        rows = s.scalars(select(db.AuditRecord).order_by(db.AuditRecord.seq)).all()
        prev = integrity.GENESIS
        for row in rows:
            row.prev_hash = prev
            row.chain_alg, row.entry_hash = integrity.compute(prev, integrity._legacy_row_content(row))
            prev = row.entry_hash
        s.execute(update(db.AuditChain).where(db.AuditChain.id == 1).values(last_hash=prev))
    result = integrity.verify()
    assert result["ok"] and result["earlier_format"] == 2
