"""API keys: creation, lookup, revocation, expiry, scopes, limits and the migration."""

from __future__ import annotations

import os
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import apikeys, config, db, guards_input, metering  # noqa: E402


@pytest.fixture()
def keys(database):
    guards_input.key_limiter._by_client.clear()
    return database


def test_secret_shown_once_and_stored_hashed(keys):
    made = apikeys.create("ehr", ["ask"])
    assert made["key"].startswith("gck_") and "key" not in apikeys.list_keys()[0]
    with db.session() as s:
        row = s.get(db.ApiKey, made["id"])
        assert made["key"] not in (row.hash, row.salt, row.prefix) and len(row.hash) == 64
        assert made["key"][4:] not in row.hash
    principal = apikeys.authenticate(made["key"])
    assert principal.id == made["id"] and principal.has("ask") and not principal.has("verify")


@pytest.mark.parametrize("token,code", [
    (None, "missing_api_key"), ("", "missing_api_key"), ("gck_short", "invalid_api_key"),
    ("gck_" + "a" * 43, "invalid_api_key"), ("Basic abc", "invalid_api_key"), ("x" * 5000, "invalid_api_key"),
    ("gck_\u0000" + "a" * 40, "invalid_api_key"),
])
def test_bad_tokens_are_401(keys, token, code):
    with pytest.raises(apikeys.ApiKeyError) as e:
        apikeys.authenticate(token)
    assert e.value.status == 401 and e.value.code == code


def test_wrong_secret_with_right_prefix_is_refused(keys):
    made = apikeys.create("ehr", ["ask"])
    forged = made["key"][:12] + ("A" if made["key"][12] != "A" else "B") + made["key"][13:]
    with pytest.raises(apikeys.ApiKeyError) as e:
        apikeys.authenticate(forged)
    assert e.value.code == "invalid_api_key"


def test_revoked_and_expired(keys, monkeypatch):
    made = apikeys.create("a", ["ask"])
    apikeys.revoke(made["id"])
    with pytest.raises(apikeys.ApiKeyError) as e:
        apikeys.authenticate(made["key"])
    assert e.value.code == "revoked_api_key"
    short = apikeys.create("b", ["ask"], expires_in_days=1)
    apikeys.authenticate(short["key"])
    real = db.utcnow
    monkeypatch.setattr(db, "utcnow", lambda: real() + timedelta(days=2))
    with pytest.raises(apikeys.ApiKeyError) as e:
        apikeys.authenticate(short["key"])
    assert e.value.code == "expired_api_key"


def test_header_parsing():
    f = apikeys.token_from_headers
    assert f("Bearer gck_x", None) == "gck_x" and f(None, " gck_y ") == "gck_y"
    assert f("bearer gck_x", "gck_y") == "gck_x" and f(None, None) is None
    assert f("Basic zzz", None) == "Basic zzz" and f("Bearer  two", None) == "Bearer  two"


@pytest.mark.parametrize("kw", [{"name": ""}, {"name": "x" * 201}, {"scopes": []}, {"scopes": ["admin"]},
                                {"rpm_limit": 0}, {"monthly_quota": -1}, {"rpm_limit": True},
                                {"expires_in_days": 0}, {"site_id": 999}])
def test_create_validation(keys, kw):
    args = {"name": "k", "scopes": ["ask"], **kw}
    with pytest.raises(apikeys.ApiKeyError) as e:
        apikeys.create(**args)
    assert e.value.status == 422


def test_rate_limit_boundary_and_window(keys, monkeypatch):
    made = apikeys.create("k", ["ask"], rpm_limit=3)
    p = apikeys.authenticate(made["key"])
    clock = [1000.0]
    monkeypatch.setattr(apikeys, "_mono", lambda: clock[0])
    assert [apikeys.check_rate(p)[0] for _ in range(3)] == [2, 1, 0]
    with pytest.raises(apikeys.ApiKeyError) as e:
        apikeys.check_rate(p)
    assert e.value.status == 429 and int(e.value.headers["Retry-After"]) >= 1
    clock[0] += 61
    assert apikeys.check_rate(p)[0] == 2


def test_quota_exact_boundary_concurrency_and_month_reset(keys, monkeypatch):
    made = apikeys.create("k", ["ask"], monthly_quota=10)
    p = apikeys.authenticate(made["key"])
    taken, refused = [], []

    def go():
        try:
            apikeys.reserve_quota(p)
            taken.append(1)
        except apikeys.ApiKeyError as exc:
            refused.append(exc.code)

    threads = [threading.Thread(target=go) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(taken) == 10 and set(refused) == {"quota_exceeded"}
    assert apikeys.quota_status(p.id)["used"] == 10
    real = db.utcnow
    monkeypatch.setattr(db, "utcnow", lambda: real() + timedelta(days=32))
    apikeys.reserve_quota(p)
    assert apikeys.quota_status(p.id)["used"] == 1


def test_metering_counts_exactly(keys):
    for _ in range(7):
        metering.record("ask", "answer", None, 1)
    for _ in range(3):
        metering.record("ask", "refuse", None, 1)
    metering.record("verify", "error", None, 2)
    r = metering.usage("key")
    by = {row["key_id"]: row for row in r["rows"]}
    assert by[1]["requests"] == 10 and by[1]["decisions"] == {"answer": 7, "refuse": 3}
    assert by[2]["billable"] == 0 and r["total"] == 11 and r["billable"] == 10
    assert "key_id,name,requests" in metering.to_csv(r)
    with pytest.raises(metering.UsageError):
        metering.usage("nope")
    with pytest.raises(metering.UsageError):
        metering.usage("day", "2026-02-01", "2026-01-01")


def test_migration_up_and_down(tmp_path, monkeypatch):
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
            command.upgrade(cfg, "head")
            names = inspect(c).get_table_names()
            assert {"api_keys", "usage_daily"} <= set(names)
            assert "api_key_id" in {x["name"] for x in inspect(c).get_columns("audit_records")}
            command.downgrade(cfg, "0011")
            assert not {"api_keys", "usage_daily"} & set(inspect(c).get_table_names())
            command.upgrade(cfg, "head")
    finally:
        db.reset_engine()
