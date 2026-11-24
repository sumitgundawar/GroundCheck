"""The Python SDK against the real service over a real socket: what the
service sends is what the client expects, errors map to the right exceptions,
and the rate-limit headers arrive."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sdk" / "python"))

import uvicorn  # noqa: E402

import groundcheck_client as gc  # noqa: E402
from app import apikeys, config, guards_input, metering, retrieval  # noqa: E402
from app.main import app  # noqa: E402

Q = "What is the standard dose of Caloradine?"
RIGHT = "The standard adult regimen is 15 mg once daily, taken in the morning with food, for an initial course of 14 days."


@pytest.fixture()
def server(database):
    guards_input.key_limiter._by_client.clear()
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


def client(server, scopes=("ask", "verify", "usage", "audit:read"), **kw):
    made = apikeys.create("sdk", list(scopes), **{k: v for k, v in kw.items() if k in ("rpm_limit", "monthly_quota", "site_id")})
    return gc.Client(made["key"], base_url=server, max_retries=0), made


def test_ask_verify_usage_and_health_round_trip(server):
    c, made = client(server)
    assert c.health().ok
    ask = c.ask(Q)
    assert ask.decision == "answer" and ask.claims and ask.request_id and ask.rate_limit.limit == config.API_KEY_RPM
    assert ask.rate_limit.remaining == config.API_KEY_RPM - 1
    refused = c.ask("What is the standard dose of Quelvarin?")
    assert refused.decision == "refuse" and refused.reason_code == "term_not_covered"
    v = c.verify(Q, RIGHT)
    assert v.passed and v.reason_code == "verified"
    bad = c.verify(Q, "Caloradine is 150 mg once daily.")
    assert not bad.passed and bad.decision == "fail" and bad.dose_issues
    report = c.usage("endpoint")
    rows = {r["endpoint"]: r for r in report.rows}
    assert rows["ask"]["decisions"] == {"answer": 1, "refuse": 1} and rows["verify"]["decisions"] == {"pass": 1, "fail": 1}
    assert report.total == 4 and report.billable == 4


def test_the_clients_errors_match_the_services(server):
    c, made = client(server, scopes=("ask",), rpm_limit=3)
    with pytest.raises(gc.PermissionDeniedError) as e:
        c.verify(Q, RIGHT)
    assert e.value.status == 403 and e.value.code == "insufficient_scope" and e.value.request_id
    with pytest.raises(gc.UnprocessableError) as e:
        c.ask(Q, site="nowhere")
    assert e.value.code == "unknown_site"
    with pytest.raises(gc.PayloadTooLargeError):
        c.ask("x" * (config.MAX_QUERY_CHARS + 1))
    with pytest.raises(gc.RateLimitError) as e:
        c.ask(Q)
    assert e.value.status == 429 and e.value.code == "rate_limited" and 1 <= e.value.retry_after <= 60
    with pytest.raises(gc.AuthenticationError) as e:
        gc.Client("gck_" + "z" * 43, base_url=server, max_retries=0).ask(Q)
    assert e.value.code == "invalid_api_key"
    apikeys.revoke(made["id"])
    with pytest.raises(gc.AuthenticationError) as e:
        c.ask(Q)
    assert e.value.code == "revoked_api_key"


def test_the_quota_error_carries_retry_after(server):
    c, made = client(server, monthly_quota=1)
    c.ask(Q)
    with pytest.raises(gc.RateLimitError) as e:
        c.ask(Q)
    assert e.value.code == "quota_exceeded" and e.value.retry_after >= 1


def test_a_patient_and_sources_travel_the_wire(server):
    c, _ = client(server)
    unsafe = c.ask(Q, patient=gc.Patient(age_years=60, weight_kg=70, egfr=10))
    assert unsafe.decision == "refuse" and [f.code for f in unsafe.patient_findings] == ["renal"]
    ids = [s.id for s in c.ask(Q).sources][:1]
    assert c.verify(Q, RIGHT, sources=ids).passed
    with pytest.raises(gc.UnprocessableError):
        c.verify(Q, RIGHT, sources=["NOPE-1"])
