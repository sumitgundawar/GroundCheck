"""Opt-in telemetry: off by default, nothing leaves the machine when off, the
payload holds exactly the documented fields and nothing identifying, and a
failed post is silent."""

from __future__ import annotations

import conftest

import importlib
import json
import logging
import os
import re
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app import auth, config, db, pipeline, retrieval, sites, telemetry  # noqa: E402

PASSWORD = "correct horse battery staple"
URL = "https://telemetry.example.org/v1/ping"
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)      # a Wednesday; the week starts 2026-11-23
README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def no_network(monkeypatch):
    """Any attempt to reach the telemetry endpoint, or to post anything at all,
    fails the test loudly. (Other libraries may still fetch their own files.)"""
    attempts = []
    host = "telemetry.example.org"

    def forbidden(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("a telemetry post was attempted")

    real_getaddrinfo, real_send = socket.getaddrinfo, httpx.Client.send

    def guarded_getaddrinfo(name, *args, **kwargs):
        if name == host:
            return forbidden(name)
        return real_getaddrinfo(name, *args, **kwargs)

    def guarded_send(self, request, *args, **kwargs):
        if request.url.host == host:
            return forbidden(request)
        return real_send(self, request, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(httpx, "post", forbidden)
    monkeypatch.setattr(httpx.Client, "send", guarded_send)
    return attempts


@pytest.fixture()
def sent(monkeypatch):
    """Capture posts instead of making them."""
    posts = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return posts


def on(monkeypatch, url=URL):
    monkeypatch.setattr(config, "TELEMETRY", "on")
    monkeypatch.setattr(config, "TELEMETRY_URL", url)


def put(created, decision="answer", test_run=False, kind=None):
    import secrets

    audit_id = secrets.token_hex(8)
    with db.session() as s:
        s.add(db.AuditRecord(audit_id=audit_id, created_at=created, decision=decision, query="SECRET QUESTION TEXT",
                             record={"audit_id": audit_id, "test_run": test_run}, test_run=test_run))


# --- Off by default -------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [(None, "off"), ("", "off"), ("off", "off"), ("OFF", "off"), ("no", "off"),
                                            ("maybe", "off"), ("false", "off"), ("0", "off"), ("on", "on"),
                                            ("ON", "on"), ("true", "on"), ("1", "on"), ("yes", "on")])
def test_the_setting_defaults_to_off_and_only_clear_yes_turns_it_on(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TELEMETRY", raising=False)
    else:
        monkeypatch.setenv("TELEMETRY", value)
    try:
        importlib.reload(config)
        assert config.TELEMETRY == expected
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_shipped_default_is_off():
    assert config.TELEMETRY == "off" and not telemetry.enabled()
    assert config.TELEMETRY_URL == ""


def test_nothing_is_sent_when_off_even_with_an_endpoint_set(database, monkeypatch, no_network):
    monkeypatch.setattr(config, "TELEMETRY", "off")
    monkeypatch.setattr(config, "TELEMETRY_URL", URL)
    assert telemetry.send_if_due(NOW) == {"sent": False, "reason": "off"}
    pipeline.run("What is the standard dose of Caloradine?")
    assert telemetry.send_if_due(NOW)["reason"] == "off"
    assert no_network == []
    with db.session() as s:
        assert s.scalars(select(db.AppSetting)).all() == []      # not even an install id


def test_starting_the_app_with_telemetry_off_starts_no_reporting_task(database, monkeypatch, no_network):
    from app import main

    started = []

    async def fake_loop():
        started.append(1)

    monkeypatch.setattr(main, "_telemetry_loop", fake_loop)
    monkeypatch.setattr(config, "TELEMETRY", "off")
    with TestClient(main.app):
        pass
    assert started == [] and no_network == []


def test_starting_the_app_with_telemetry_on_starts_the_reporting_task(database, monkeypatch):
    from app import main

    started = []

    async def fake_loop():
        started.append(1)

    monkeypatch.setattr(main, "_telemetry_loop", fake_loop)
    monkeypatch.setattr(config, "TELEMETRY", "on")
    with TestClient(main.app):
        pass
    assert started == [1]


def test_on_without_an_endpoint_or_with_an_unsafe_one_sends_nothing(database, monkeypatch, no_network):
    for url in ("", "http://telemetry.example.org/ping", "ftp://x/y", "file:///etc/passwd", "telemetry.example.org",
                "http://localhost.evil.example.org/p"):
        on(monkeypatch, url)
        result = telemetry.send_if_due(NOW)
        assert result["sent"] is False and result["reason"] == "no endpoint", url
    assert no_network == []


def test_the_admin_preview_makes_no_network_call_and_stores_nothing(database, monkeypatch, no_network):
    on(monkeypatch)
    status = telemetry.status()
    assert status["enabled"] is True and status["sample"]["install_id"] == "0" * 32
    assert no_network == []
    with db.session() as s:
        assert s.scalars(select(db.AppSetting)).all() == []


# --- The payload ---------------------------------------------------------------------------------

def test_the_payload_has_exactly_the_documented_fields(database, monkeypatch, sent):
    on(monkeypatch)
    result = telemetry.send_if_due(NOW)
    assert result["sent"] is True and len(sent) == 1
    url, kwargs = sent[0]
    assert url == URL
    body = kwargs["json"]
    assert set(body) == set(telemetry.FIELDS)
    assert body == result["payload"]
    assert all(telemetry.FIELDS[k].strip() for k in body)
    assert body["schema"] == 1 and body["version"] == config.VERSION and body["db_kind"] == "sqlite"
    assert body["llm_provider_kind"] == "extractive" and body["week_start"] == "2026-11-23"
    assert re.fullmatch(r"[0-9a-f]{32}", body["install_id"])


def test_the_post_carries_nothing_but_the_json(database, monkeypatch, sent):
    on(monkeypatch)
    telemetry.send_if_due(NOW)
    _, kwargs = sent[0]
    assert set(kwargs) == {"json", "timeout", "follow_redirects"}
    assert kwargs["follow_redirects"] is False and kwargs["timeout"] <= 5


def test_nothing_identifying_is_in_the_payload(database, monkeypatch, sent):
    monkeypatch.setattr(config, "INSTANCE_NAME", "very-private-hostname")
    monkeypatch.setattr(config, "ORGANISATION_NAME", "Private General Hospital")
    site = sites.create("private-site", "Private Site Name")
    auth.create_user("someone.private@example.org", PASSWORD, role="clinician", name="Someone Private", site_id=site["id"])
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-private-key-123456")
    on(monkeypatch)
    pipeline.run("What is the standard dose of Caloradine? Patient John Smith NHS 943 476 5919")
    put(NOW - timedelta(hours=1))
    telemetry.send_if_due(NOW)
    text = json.dumps(sent[0][1]["json"])
    for needle in ("very-private-hostname", "Private General Hospital", "Private Site Name", "private-site",
                   "someone.private", "Someone Private", "sk-private-key", "John Smith", "943", "Caloradine",
                   "SECRET QUESTION TEXT", "sqlite:///", "groundcheck.db", socket.gethostname(), str(config.ROOT_DIR),
                   config.GEN_MODEL, "api.groq.com", "example.org"):
        assert needle not in text, needle
    for value in sent[0][1]["json"].values():
        assert isinstance(value, (str, int))


def test_the_install_id_is_random_stable_and_not_derived_from_the_machine(database, monkeypatch, sent):
    import hashlib

    on(monkeypatch)
    telemetry.send_if_due(NOW)
    telemetry.send_if_due(NOW + timedelta(days=8))
    ids = [kw["json"]["install_id"] for _, kw in sent]
    assert len(ids) == 2 and ids[0] == ids[1]
    host = socket.gethostname().encode()
    assert ids[0] not in {hashlib.md5(host).hexdigest(), hashlib.sha256(host).hexdigest()[:32]}
    with db.session() as s:
        assert s.get(db.AppSetting, telemetry.ID_KEY).value == ids[0]


def test_two_installs_get_different_ids(tmp_path, monkeypatch, sent):
    from conftest import _forget_database

    ids = []
    for n in range(2):
        monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / f'db{n}.db'}")
        _forget_database()
        assert db.ready()
        on(monkeypatch)
        telemetry.send_if_due(NOW)
        ids.append(sent[-1][1]["json"]["install_id"])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("n,label", [(0, "0"), (1, "1-9"), (9, "1-9"), (10, "10-99"), (99, "10-99"), (100, "100-999"),
                                     (999, "100-999"), (1000, "1000-9999"), (9999, "1000-9999"), (10000, "10000+"),
                                     (10 ** 9, "10000+")])
def test_bucket_boundaries(n, label):
    assert telemetry.bucket(n) == label


def test_counts_are_this_weeks_real_questions_only_and_bucketed(database, monkeypatch, sent):
    on(monkeypatch)
    week = datetime(2026, 11, 23, tzinfo=timezone.utc)
    for i in range(12):                                               # 12 answers this week
        put(week + timedelta(hours=i + 1))
    for i in range(3):                                                # 3 refusals
        put(week + timedelta(days=1, hours=i), "refuse")
    put(week - timedelta(seconds=1))                                  # the Sunday before: last week
    put(week + timedelta(hours=5), test_run=True)                     # an evaluation run
    put(week + timedelta(hours=6), "event")                           # an administrative event row
    telemetry.send_if_due(NOW)
    body = sent[0][1]["json"]
    assert (body["requests_bucket"], body["answers_bucket"], body["refusals_bucket"]) == ("10-99", "10-99", "1-9")


def test_the_week_boundary_is_monday_midnight_utc(database, monkeypatch):
    put(datetime(2026, 11, 22, 23, 59, 59, tzinfo=timezone.utc))
    put(datetime(2026, 11, 23, 0, 0, 0, tzinfo=timezone.utc))
    p = telemetry.payload("x" * 32, datetime(2026, 11, 23, 0, 0, 1, tzinfo=timezone.utc))
    assert (p["week_start"], p["requests_bucket"]) == ("2026-11-23", "1-9")
    sunday = telemetry.payload("x" * 32, datetime(2026, 11, 29, 23, 59, 59, tzinfo=timezone.utc))
    assert sunday["week_start"] == "2026-11-23"


def test_an_empty_week_reports_zero(database, monkeypatch, sent):
    on(monkeypatch)
    telemetry.send_if_due(NOW)
    body = sent[0][1]["json"]
    assert (body["requests_bucket"], body["answers_bucket"], body["refusals_bucket"]) == ("0", "0", "0")


def test_the_provider_kind_reports_kind_only(database, monkeypatch, sent):
    from app import llm

    monkeypatch.setattr(llm, "active_provider", lambda: {"kind": "cloud", "model": "secret-model", "provider": "anthropic"})
    on(monkeypatch)
    telemetry.send_if_due(NOW)
    body = sent[0][1]["json"]
    assert body["llm_provider_kind"] == "cloud" and "secret-model" not in json.dumps(body) and "anthropic" not in json.dumps(body)


# --- Cadence and failure ----------------------------------------------------------------------------

def test_it_sends_once_a_week(database, monkeypatch, sent):
    on(monkeypatch)
    assert telemetry.send_if_due(NOW)["sent"] is True
    assert telemetry.send_if_due(NOW + timedelta(days=6, hours=23))["reason"] == "not due"
    assert len(sent) == 1
    assert telemetry.send_if_due(NOW + timedelta(days=7))["sent"] is True and len(sent) == 2


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("dropped"),
    httpx.InvalidURL("bad"), ValueError("anything"), RuntimeError("odd"),
])
def test_a_failed_post_is_silent_and_tried_again(database, monkeypatch, caplog, failure):
    on(monkeypatch)
    calls = []

    def failing(url, **kwargs):
        calls.append(url)
        raise failure

    monkeypatch.setattr(httpx, "post", failing)
    with caplog.at_level(logging.DEBUG):
        result = telemetry.send_if_due(NOW)
    assert result == {"sent": False, "reason": "network"}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    with db.session() as s:
        assert s.get(db.AppSetting, telemetry.SENT_KEY) is None       # not marked sent
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(200, request=httpx.Request("POST", url)))
    assert telemetry.send_if_due(NOW + timedelta(minutes=1))["sent"] is True


@pytest.mark.parametrize("status", [301, 302, 400, 404, 500, 503])
def test_a_bad_status_counts_as_a_failure(database, monkeypatch, status):
    on(monkeypatch)
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(status, request=httpx.Request("POST", url)))
    result = telemetry.send_if_due(NOW)
    assert result["sent"] is False
    with db.session() as s:
        assert s.get(db.AppSetting, telemetry.SENT_KEY) is None


def test_a_database_failure_is_silent(database, monkeypatch, sent):
    on(monkeypatch)
    monkeypatch.setattr(db, "session", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    assert telemetry.send_if_due(NOW)["sent"] is False and sent == []


def test_no_database_means_no_send(monkeypatch, sent):
    on(monkeypatch)
    monkeypatch.setattr(db, "ready", lambda: False)
    assert telemetry.send_if_due(NOW) == {"sent": False, "reason": "no database"} and sent == []


def test_another_replica_holding_the_lock_means_no_send(database, monkeypatch, sent):
    from contextlib import contextmanager

    @contextmanager
    def busy(name, key):
        yield False

    monkeypatch.setattr(db, "try_advisory_lock", busy)
    on(monkeypatch)
    assert telemetry.send_if_due(NOW)["sent"] is False and sent == []


# --- Documentation and the admin page ------------------------------------------------------------------

def test_the_readme_documents_every_field_and_only_those():
    section = README.split("<!-- telemetry-fields -->")[1].split("<!-- /telemetry-fields -->")[0]
    documented = set(re.findall(r"^\| `(\w+)` \|", section, re.M))
    assert documented == set(telemetry.FIELDS)
    for name, meaning in telemetry.FIELDS.items():
        assert meaning.split(".")[0][:30] in section, name


def test_the_readme_says_it_is_off_by_default():
    assert "TELEMETRY=off" in README or "off by default" in README.lower()


def test_the_admin_endpoint_is_admin_only_and_lists_the_fields(database, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    auth.create_user("admin@example.org", PASSWORD, role="admin")
    auth.create_user("rev@example.org", PASSWORD, role="reviewer")
    from app.main import app

    with TestClient(app) as c:
        assert c.get("/api/telemetry").status_code == 401
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"email": "rev@example.org", "password": PASSWORD})
        assert c.get("/api/telemetry").status_code == 403
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"email": "admin@example.org", "password": PASSWORD})
        body = c.get("/api/telemetry").json()
    assert body["enabled"] is False and set(body["fields"]) == set(telemetry.FIELDS)
    assert set(body["sample"]) == set(telemetry.FIELDS) and "off by default" in body["how_to_enable"]
