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
# Every module used to set these itself before importing the app, so the order the files ran in decided
# whether the app saw them: the first file to import it fixes the settings for the whole run.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000000")
os.environ.setdefault("FORCE_EXTRACTIVE", "true")
os.environ["GROQ_API_KEY"] = ""
os.environ["SESSION_COOKIE_SECURE"] = "false"  # the test client talks plain HTTP
# Checking a release runs over a thousand questions; tests/test_releases.py
# turns it on with a small set.
os.environ.setdefault("RELEASE_CHECKS", "false")
os.environ.setdefault("ALERT_INTERVAL_SECONDS", "0")

@pytest.fixture(autouse=True)
def _no_shared_volume_cache():
    """Imaging volumes are cached by series id, and every test's database starts again at id 1."""
    from app.imaging import store

    store._cache.clear()
    yield
    store._cache.clear()


@pytest.fixture()
def config_restored():
    """Reloading the config module re-reads the whole environment, including variables other test modules
    export for the process (AUDIT_PERSIST, FORCE_EXTRACTIVE). Put back exactly what it held."""
    from app import config

    saved = dict(vars(config))
    yield
    for name in set(vars(config)) - set(saved):
        delattr(config, name)
    for name, value in saved.items():
        setattr(config, name, value)


BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("TEST_POSTGRES_URL") else [])


def _forget_database() -> None:
    from app import audit, db, knowledge, sites

    db.reset_engine()
    db._ready = False
    audit.store._backend = None
    sites.forget_user()
    knowledge._status = knowledge.IndexStatus()      # names the documents of the database it was built from


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

# A request from another machine. Tests used to simulate this with an
# X-Forwarded-For header, which the app no longer believes: "this came from the
# machine running it" is now decided by the socket the request arrived on, so a
# test has to arrive on a different one.
REMOTE_PEER = ("203.0.113.9", 44321)


def remote_headers() -> dict:
    """A header claiming to be local. It must not be believed."""
    return {"X-Forwarded-For": "127.0.0.1"}


# A request arriving from the loopback address, which is what a reverse proxy
# on the same host looks like, and what uvicorn --forwarded-allow-ips makes a
# remote request look like.
LOOPBACK_PEER = ("127.0.0.1", 50001)


def pytest_configure(config):
    config.addinivalue_line("markers", "req(id, ...): the requirement(s) in compliance/requirements.yaml this test verifies")


def _requirement_ids() -> set[str] | None:
    try:
        import yaml

        data = yaml.safe_load((Path(__file__).resolve().parent.parent / "compliance" / "requirements.yaml").read_text(encoding="utf-8"))
        return {str(r["id"]) for r in data["requirements"]}
    except Exception:  # noqa: BLE001 - no register (or no PyYAML): nothing to check against
        return None


def pytest_collection_modifyitems(config, items):
    """A test that names a requirement that doesn't exist is an error, not a silent gap in the trace matrix."""
    known = _requirement_ids()
    if known is None:
        return
    unknown = [f"{item.nodeid}: unknown requirement {a!r}" for item in items for m in item.iter_markers("req") for a in m.args
               if a not in known]
    if unknown:
        raise pytest.UsageError("Tests refer to requirements missing from compliance/requirements.yaml:\n  " + "\n  ".join(unknown))
