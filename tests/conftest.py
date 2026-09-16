"""Test-wide setup. Runs before any test module imports the app, so the app
never touches the real database, audit log or local AI selection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_tmp = Path(tempfile.mkdtemp(prefix="groundcheck-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp / 'test.db'}")
os.environ["AUDIT_LOG_PATH"] = str(_tmp / "audit.jsonl")
os.environ["LOCAL_AI_STATE_PATH"] = str(_tmp / "local_ai.json")
os.environ["SESSION_COOKIE_SECURE"] = "false"  # the test client talks plain HTTP

BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("TEST_POSTGRES_URL") else [])


def _forget_database() -> None:
    from app import audit, db

    db.reset_engine()
    db._ready = False
    audit.store._backend = None


@pytest.fixture(params=BACKENDS)
def database(request, tmp_path):
    """A fresh, migrated database for one test: SQLite always, and PostgreSQL
    too when TEST_POSTGRES_URL is set. Restores the default database after."""
    from sqlalchemy import text

    from app import config, db

    original_url = config.DATABASE_URL
    if request.param == "sqlite":
        config.DATABASE_URL = f"sqlite:///{tmp_path / 'test.db'}"
    else:
        config.DATABASE_URL = os.environ["TEST_POSTGRES_URL"]
    _forget_database()
    if request.param == "postgresql":
        with db.engine().begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    try:
        assert db.ready()
        yield request.param
    finally:
        config.DATABASE_URL = original_url
        _forget_database()
