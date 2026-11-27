"""The Python SDK against a mock HTTP server (never the real app)."""

from __future__ import annotations

import copy
import logging
import pickle
import socket
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import groundcheck_client as gc  # noqa: E402
from groundcheck_client import client as client_module  # noqa: E402
from gc_mock_server import ASK_OK, VERIFY_OK, MockServer, Step  # noqa: E402

KEY = "gck_s3cretvalueABCDEFGHIJKLMNOP"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "GROUNDCHECK_API_KEY",
                 "GROUNDCHECK_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.fixture()
def server():
    with MockServer() as s:
        yield s


class Sleeps(list):
    def __call__(self, seconds):
        self.append(seconds)


def make(server, **kw):
    sleeps = Sleeps()
    kw.setdefault("sleep", sleeps)
    kw.setdefault("jitter", lambda: 1.0)
    kw.setdefault("timeout", 5)
    c = gc.Client(KEY, base_url=server.url, **kw)
    c.sleeps = sleeps
    return c


def error_body(code="bad", message="Nope."):
    return {"error": {"code": code, "message": message}}


# --- Construction and secrets ----------------------------------------------------------

def test_an_api_key_is_required(monkeypatch):
    with pytest.raises(gc.InvalidArgumentError, match="API key"):
        gc.Client()
    with pytest.raises(gc.InvalidArgumentError):
        gc.Client("")
    monkeypatch.setenv("GROUNDCHECK_API_KEY", KEY)
    assert gc.Client(base_url="https://x.example")


def test_the_key_and_base_url_can_come_from_the_environment(server, monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_API_KEY", KEY)
    monkeypatch.setenv("GROUNDCHECK_BASE_URL", server.url + "/")
    server.default = Step(200, {"status": "ok"})
    assert gc.Client().health().status == "ok"
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("key", ["has space", "tab\tkey", "line\nbreak", "a\r\nX-Evil: 1", "ünïcode", "key\x00"])
def test_keys_that_could_inject_headers_are_rejected(key):
    with pytest.raises(gc.InvalidArgumentError):
        gc.Client(key, base_url="https://x.example")


@pytest.mark.parametrize("url", ["ftp://x.example", "x.example", "", "http://", "https://u:p@x.example",
                                 "https://x.example?a=1", "https://x.example#f", "file:///etc/passwd"])
def test_bad_base_urls_are_rejected(url):
    with pytest.raises(gc.InvalidArgumentError):
        gc.Client(KEY, base_url=url)


def test_plain_http_is_only_allowed_for_loopback_unless_opted_in():
    for ok in ("http://localhost:8000", "http://127.0.0.1:1", "http://[::1]:8000", "http://api.localhost"):
        assert gc.Client(KEY, base_url=ok)
    with pytest.raises(gc.InvalidArgumentError, match="plain http"):
        gc.Client(KEY, base_url="http://api.example.com")
    assert gc.Client(KEY, base_url="http://api.example.com", allow_insecure_http=True)
    assert gc.Client(KEY, base_url="https://api.example.com/")


def test_base_url_trailing_slash_and_path_prefix(server):
    c = gc.Client(KEY, base_url=server.url + "/gateway/")
    server.default = Step(200, {"status": "ok"})
    c.health()
    assert server.requests[0].path == "/gateway/api/v1/health"


@pytest.mark.parametrize("kw", [{"timeout": 0}, {"timeout": -1}, {"timeout": float("nan")}, {"timeout": True},
                                {"timeout": "5"}, {"max_retries": -1}, {"max_retries": 11}, {"max_retries": 1.5},
                                {"max_retries": True}])
def test_bad_options_are_rejected(kw):
    with pytest.raises(gc.InvalidArgumentError):
        gc.Client(KEY, base_url="https://x.example", **kw)


def test_the_key_never_appears_in_repr_str_pickle_or_copies(server):
    c = make(server)
    for text in (repr(c), str(c), repr(c._key), str(c._key), repr(vars(c))):
        assert KEY not in text and "s3cret" not in text
    with pytest.raises(TypeError):
        pickle.dumps(c)
    with pytest.raises(TypeError):
        pickle.dumps(c._key)
    with pytest.raises(TypeError):
        copy.copy(c)


def test_the_key_never_reaches_logs_or_exceptions(server, caplog):
    caplog.set_level(logging.DEBUG, logger="groundcheck_client")
    server.enqueue(Step(500, error_body("boom", "Server broke.")), Step(401, error_body("auth", "Bad key.")))
    c = make(server)
    with pytest.raises(gc.AuthenticationError) as info:
        c.ask("What is the dose?")
    text = caplog.text + str(info.value) + repr(info.value) + repr(info.value.__dict__)
    assert KEY not in text and "Bearer" not in text and caplog.text


def test_the_key_is_sent_as_a_bearer_token_and_only_to_the_configured_host(server):
    server.default = Step(200, ASK_OK)
    make(server).ask("What is the dose?")
    seen = server.requests[0]
    assert seen.headers["authorization"] == f"Bearer {KEY}"
    assert seen.headers["content-type"] == "application/json" and seen.headers["accept"] == "application/json"
    assert seen.headers["user-agent"].startswith("groundcheck-client-python/")


# --- Endpoints -------------------------------------------------------------------------

def test_ask_returns_a_typed_response(server):
    server.default = Step(200, ASK_OK, headers={"X-Request-Id": "req-1", "X-RateLimit-Limit": "60",
                                                "X-RateLimit-Remaining": "59", "X-RateLimit-Reset": "1700000000"})
    r = make(server).ask("What is the usual dose?", site="north", patient={"age_years": 50, "egfr": 80})
    assert server.requests[0].method == "POST" and server.requests[0].path == "/api/v1/ask"
    assert server.requests[0].json() == {"question": "What is the usual dose?", "site": "north",
                                         "patient": {"age_years": 50, "egfr": 80}}
    assert r.decision == "answer" and r.outcome == "answered" and r.answered and r.audit_id == "abc123"
    assert r.claims[0].support.sentence == "Take 10 mg once daily." and r.claims[0].grounded
    assert r.sources[0].id == "DOC1-001" and r.sources[0].above_gate and r.llm_used is False
    assert r.patient_findings[0].code == "renal_dose" and not r.patient_findings[0].blocks
    assert r.redaction.kinds == {"phone": 1} and r.trace[0]["name"] == "retrieval"
    assert r.request_id == "req-1" and (r.rate_limit.limit, r.rate_limit.remaining) == (60, 59)
    with pytest.raises(Exception):
        r.decision = "refuse"   # immutable


def test_a_refusal_is_not_answered(server):
    refusal = {**ASK_OK, "decision": "refuse", "outcome": "refused", "answer_text": "", "claims": [],
               "refused_reason": "No source covers this.", "reason_code": "no_relevant_source"}
    server.default = Step(200, refusal)
    r = make(server).ask("What is Quelvarin?")
    assert not r.answered and r.refused_reason and r.reason_code == "no_relevant_source"
    server.default = Step(200, {**ASK_OK, "outcome": "not_evaluated"})
    assert not make(server).ask("x").answered


def test_ask_accepts_a_patient_dataclass_and_drops_unset_fields(server):
    server.default = Step(200, ASK_OK)
    make(server).ask("q", patient=gc.Patient(age_years=70, egfr=40.5, allergies=("penicillin",), labs={"inr": 2.1}))
    assert server.requests[0].json()["patient"] == {"age_years": 70, "egfr": 40.5, "allergies": ["penicillin"],
                                                    "labs": {"inr": 2.1}}


def test_verify_sends_the_answer_and_source_ids(server):
    server.default = Step(200, VERIFY_OK, headers={"X-Request-Id": "v-1"})
    r = make(server).verify("What is the dose?", "Take 10 mg once daily.", sources=["DOC1-001"], site="s1",
                            patient=gc.Patient(age_years=30))
    assert server.requests[0].path == "/api/v1/verify"
    assert server.requests[0].json() == {"question": "What is the dose?", "answer": "Take 10 mg once daily.",
                                         "sources": ["DOC1-001"], "site": "s1", "patient": {"age_years": 30}}
    assert r.passed and r.decision == "pass" and r.reason_code == "verified" and r.audit_id == "v123"
    assert r.claims[0].source_id == "DOC1-001" and r.claims[0].issues == () and r.latency_ms == 12
    assert r.redaction.redacted_answer == "a" and r.request_id == "v-1" and r.sources[0].title == "Label"


def test_verify_failures_carry_dose_issues_and_are_never_passed(server):
    body = {**VERIFY_OK, "decision": "fail", "reason_code": "dose_unsupported", "unsupported_claims": [0],
            "dose_issues": [{"claim": 0, "value": "50 mg", "message": "Not in the source."}],
            "claims": [{"text": "Take 50 mg.", "grounded": False, "source_id": None, "support": None,
                        "relevance": 0.1, "issues": ["dose_unsupported"]}]}
    server.default = Step(200, body)
    r = make(server).verify("q", "Take 50 mg.")
    assert not r.passed and r.unsupported_claims == (0,) and r.dose_issues[0].value == "50 mg"
    assert r.claims[0].issues == ("dose_unsupported",) and r.claims[0].support is None
    assert "sources" not in server.requests[0].json()


@pytest.mark.parametrize("decision", ["review", "PASS", "passed", "", "unknown", "Pass "])
def test_only_an_exact_pass_counts_as_passed(server, decision):
    server.default = Step(200, {**VERIFY_OK, "decision": decision})
    assert make(server).verify("q", "a").passed is False


def test_usage_builds_the_query_and_parses_rows(server):
    server.default = Step(200, {"group_by": "day", "from": "2026-11-01", "to": "2026-11-24",
                                "rows": [{"day": "2026-11-24", "requests": 5, "billable": 4, "decisions": {"pass": 4}}],
                                "total": 5, "billable": 4})
    r = make(server).usage("day", start=date(2026, 11, 1), end=datetime(2026, 11, 24, 15, 0))
    assert server.requests[0].method == "GET"
    assert server.requests[0].path == "/api/v1/usage?group_by=day&from=2026-11-01&to=2026-11-24"
    assert server.requests[0].body == b"" and "content-type" not in server.requests[0].headers
    assert (r.group_by, r.start, r.end, r.total, r.billable) == ("day", "2026-11-01", "2026-11-24", 5, 4)
    assert r.rows[0]["decisions"] == {"pass": 4}
    make(server).usage()
    assert server.requests[1].path == "/api/v1/usage?group_by=key"
    make(server).usage("key", start="2026-01-02")
    assert server.requests[2].path == "/api/v1/usage?group_by=key&from=2026-01-02"


def test_health(server):
    server.default = Step(200, {"status": "ok", "version": "1"}, headers={"X-Request-Id": "h"})
    h = make(server).health()
    assert h.ok and h.status == "ok" and h.details["version"] == "1" and h.request_id == "h"
    server.default = Step(200, {"status": "degraded"})
    assert not make(server).health().ok


# --- Client-side validation ----------------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda c: c.ask(""), lambda c: c.ask("   "), lambda c: c.ask(None), lambda c: c.ask(5), lambda c: c.ask(b"x"),
    lambda c: c.ask("q", site=""), lambda c: c.ask("q", site=5), lambda c: c.ask("q", patient=5),
    lambda c: c.ask("q", patient={"egfr": float("nan")}), lambda c: c.ask("q", patient={"x": object()}),
    lambda c: c.ask("bad \ud800 surrogate"), lambda c: c.verify("q", ""), lambda c: c.verify("", "a"),
    lambda c: c.verify("q", "a", sources="DOC1"), lambda c: c.verify("q", "a", sources=[1, 2]),
    lambda c: c.verify("q", "a", sources=b"x"), lambda c: c.usage("week"), lambda c: c.usage(start="not a date"),
    lambda c: c.usage(start=20260101), lambda c: c.usage(end="2026-13-45"),
])
def test_bad_arguments_fail_before_anything_is_sent(server, call):
    with pytest.raises(gc.InvalidArgumentError):
        call(make(server))
    assert server.requests == []


def test_invalid_argument_is_also_a_value_error():
    assert issubclass(gc.InvalidArgumentError, ValueError)


# --- Error statuses ------------------------------------------------------------------------

@pytest.mark.parametrize("status,cls", [
    (400, gc.BadRequestError), (401, gc.AuthenticationError), (403, gc.PermissionDeniedError),
    (404, gc.NotFoundError), (413, gc.PayloadTooLargeError), (422, gc.UnprocessableError),
    (409, gc.UnexpectedStatusError), (418, gc.UnexpectedStatusError), (410, gc.UnexpectedStatusError),
])
def test_client_errors_map_to_their_class_and_are_never_retried(server, status, cls):
    server.default = Step(status, error_body("some_code", "Specific message."),
                          headers={"X-Request-Id": "rid-9", "Retry-After": "1"})
    c = make(server, max_retries=5)
    with pytest.raises(cls) as info:
        c.ask("q")
    e = info.value
    assert type(e) is cls and e.status == status and e.code == "some_code" and e.message == "Specific message."
    assert e.request_id == "rid-9" and "Specific message." in str(e) and "rid-9" in str(e)
    assert len(server.requests) == 1 and c.sleeps == []
    assert isinstance(e, gc.APIStatusError) and isinstance(e, gc.GroundCheckError)


def test_429_is_retried_then_raised_with_retry_after(server):
    server.default = Step(429, error_body("rate_limited", "Slow down."), headers={"Retry-After": "2"})
    c = make(server, max_retries=2)
    with pytest.raises(gc.RateLimitError) as info:
        c.verify("q", "a")
    assert info.value.retry_after == 2.0 and info.value.status == 429 and info.value.code == "rate_limited"
    assert len(server.requests) == 3 and c.sleeps == [2.0, 2.0]


def test_a_retry_after_header_is_honoured_over_backoff_and_then_the_call_succeeds(server):
    server.enqueue(Step(429, error_body("rate_limited", "x"), headers={"Retry-After": "3"}))
    server.default = Step(200, VERIFY_OK)
    c = make(server)
    assert c.verify("q", "a").passed
    assert c.sleeps == [3.0] and len(server.requests) == 2
    assert server.requests[0].body == server.requests[1].body


def test_retry_after_as_a_http_date_and_fractions(server):
    future = time.gmtime(time.time() + 10)
    date_header = time.strftime("%a, %d %b %Y %H:%M:%S GMT", future)
    server.enqueue(Step(429, error_body("r", "x"), headers={"Retry-After": date_header}),
                   Step(429, error_body("r", "x"), headers={"Retry-After": "0.25"}))
    server.default = Step(200, VERIFY_OK)
    c = make(server, max_retries=3)
    c.verify("q", "a")
    assert 8 <= c.sleeps[0] <= 11 and c.sleeps[1] == 0.25


@pytest.mark.parametrize("header", ["soon", "-5", "nan", "inf", "", "Wed, 99 Foo 2020", "1e999"])
def test_unusable_retry_after_falls_back_to_backoff(server, header):
    server.enqueue(Step(429, error_body("r", "x"), headers={"Retry-After": header}))
    server.default = Step(200, VERIFY_OK)
    c = make(server, backoff_base=0.5)
    c.verify("q", "a")
    assert len(c.sleeps) == 1
    assert c.sleeps[0] in (0.5, 0.0)


def test_a_retry_after_longer_than_the_cap_raises_immediately(server):
    server.default = Step(429, error_body("rate_limited", "Come back later."), headers={"Retry-After": "3600"})
    c = make(server, max_retry_wait=30)
    with pytest.raises(gc.RateLimitError) as info:
        c.ask("q")
    assert info.value.retry_after == 3600 and len(server.requests) == 1 and c.sleeps == []


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_5xx_is_retried_with_exponential_backoff_then_raised(server, status):
    server.default = Step(status, error_body("server", "Down."))
    c = make(server, max_retries=3, backoff_base=0.5, backoff_max=1.5)
    with pytest.raises(gc.ServerError) as info:
        c.ask("q")
    assert info.value.status == status and len(server.requests) == 4
    assert c.sleeps == [0.5, 1.0, 1.5]


def test_5xx_then_success(server):
    server.enqueue(Step(503, error_body("x", "y")), Step(502, b"<html>Bad gateway</html>", raw=True))
    server.default = Step(200, ASK_OK)
    c = make(server, max_retries=2)
    assert c.ask("q").answered and len(server.requests) == 3


def test_jitter_scales_the_backoff(server):
    server.default = Step(500, error_body("x", "y"))
    c = make(server, max_retries=2, jitter=lambda: 0.5)
    with pytest.raises(gc.ServerError):
        c.ask("q")
    assert c.sleeps == [0.25, 0.5]


def test_zero_retries_means_one_attempt(server):
    server.default = Step(503, error_body("x", "y"))
    with pytest.raises(gc.ServerError):
        make(server, max_retries=0).ask("q")
    assert len(server.requests) == 1


def test_retry_after_on_5xx_other_than_503_is_ignored_for_the_wait(server):
    server.enqueue(Step(500, error_body("x", "y"), headers={"Retry-After": "9"}))
    server.default = Step(200, ASK_OK)
    c = make(server)
    c.ask("q")
    assert c.sleeps == [0.5]


def test_html_and_odd_error_bodies_give_a_generic_message(server):
    for body, raw in [(b"<html><h1>502 Bad Gateway</h1></html>", True), (b"", True), ([1, 2], False),
                      ({"error": "just a string"}, False), ({"error": {"code": 5, "message": ""}}, False),
                      (b"\xff\xfe", True), ({"detail": "x"}, False)]:
        server.default = Step(400, body, raw=raw)
        with pytest.raises(gc.BadRequestError) as info:
            make(server).ask("q")
        assert info.value.message == "HTTP 400" and info.value.code is None


def test_very_long_error_messages_are_truncated(server):
    server.default = Step(400, error_body("c" * 500, "m" * 5000))
    with pytest.raises(gc.BadRequestError) as info:
        make(server).ask("q")
    assert len(info.value.message) == 1000 and len(info.value.code) == 100


def test_redirects_are_not_followed_and_the_key_is_not_resent(server):
    server.default = Step(302, b"", headers={"Location": "/elsewhere"}, raw=True)
    with pytest.raises(gc.UnexpectedStatusError) as info:
        make(server).ask("q")
    assert info.value.status == 302 and len(server.requests) == 1


# --- Network failures ------------------------------------------------------------------------

def test_connection_refused_is_a_connection_error():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    c = gc.Client(KEY, base_url=f"http://127.0.0.1:{port}", sleep=lambda x: None, max_retries=2)
    with pytest.raises(gc.APIConnectionError):
        c.ask("q")


def test_a_post_is_not_retried_after_the_connection_drops(server):
    server.enqueue(Step(drop=True))
    server.default = Step(200, ASK_OK)
    c = make(server)
    with pytest.raises(gc.APIConnectionError):
        c.ask("q")
    assert len(server.requests) == 1


def test_a_get_is_retried_after_the_connection_drops(server):
    server.enqueue(Step(drop=True), Step(drop=True))
    server.default = Step(200, {"status": "ok"})
    c = make(server, max_retries=2)
    assert c.health().ok and len(server.requests) == 3 and c.sleeps == [0.5, 1.0]


def test_a_get_gives_up_after_the_retries(server):
    server.default = Step(drop=True)
    with pytest.raises(gc.APIConnectionError):
        make(server, max_retries=1).health()
    assert len(server.requests) == 2


def test_a_slow_server_times_out(server):
    server.default = Step(200, ASK_OK, delay=1.5)
    c = make(server, timeout=0.3)
    start = time.time()
    with pytest.raises(gc.APITimeoutError):
        c.ask("q")
    assert time.time() - start < 1.2 and isinstance(gc.APITimeoutError("x"), gc.APIConnectionError)
    assert len(server.requests) == 1


def test_a_server_that_trickles_a_body_is_cut_off(server):
    server.default = Step(200, b"", trickle=True)
    c = make(server, timeout=0.4)
    start = time.time()
    with pytest.raises(gc.APITimeoutError):
        c.ask("q")
    assert time.time() - start < 2


def test_a_get_is_retried_after_a_timeout(server):
    server.enqueue(Step(200, {"status": "ok"}, delay=1.0))
    server.default = Step(200, {"status": "ok"})
    c = make(server, timeout=0.3)
    assert c.health().ok and len(server.requests) == 2


# --- Malformed responses ------------------------------------------------------------------------

@pytest.mark.parametrize("body", [b"", b"{", b"not json", b"<html></html>", b"\xff\xfe\x00", b"null", b"[]", b"42",
                                  b'"text"', b'{"decision": "answer"}', b"[" * 5000 + b"]" * 5000], ids=repr)
def test_malformed_200_responses_raise_invalid_response(server, body):
    server.default = Step(200, body, raw=True, headers={"X-Request-Id": "r"})
    with pytest.raises(gc.InvalidResponseError) as info:
        make(server).ask("q")
    assert info.value.status in (200, None)


@pytest.mark.parametrize("change", [
    {"decision": 5}, {"audit_id": None}, {"total_ms": "41"}, {"total_ms": True}, {"claims": "none"},
    {"claims": [1]}, {"claims": [{"text": 5}]}, {"sources": [{"id": "x"}]}, {"llm_used": "no"}, {"trace": [1]},
    {"patient_findings": [{"severity": "warn"}]}, {"redaction": {"kinds": {"a": "b"}}}, {"redaction": []},
    {"review": {"routed": "yes"}}, {"review": {"case_id": "1"}}, {"explanation": {"missing_terms": [1]}},
    {"claims": [{"text": "t", "source_ids": [1]}]}, {"claims": [{"text": "t", "support": {"start": 0}}]},
])
def test_wrongly_typed_fields_are_rejected_not_coerced(server, change):
    server.default = Step(200, {**ASK_OK, **change})
    with pytest.raises(gc.InvalidResponseError):
        make(server).ask("q")


def test_missing_required_fields_are_rejected(server):
    for field in ("decision", "answer_text", "audit_id", "total_ms"):
        body = {k: v for k, v in ASK_OK.items() if k != field}
        server.default = Step(200, body)
        with pytest.raises(gc.InvalidResponseError, match=field):
            make(server).ask("q")
    for field in ("decision", "reason_code", "audit_id", "latency_ms"):
        server.default = Step(200, {k: v for k, v in VERIFY_OK.items() if k != field})
        with pytest.raises(gc.InvalidResponseError, match=field):
            make(server).verify("q", "a")


def test_verify_rejects_bad_claim_lists(server):
    for change in ({"unsupported_claims": ["0"]}, {"dose_issues": [{"claim": "x"}]}, {"claims": [{"grounded": True}]},
                   {"unsupported_claims": [True]}):
        server.default = Step(200, {**VERIFY_OK, **change})
        with pytest.raises(gc.InvalidResponseError):
            make(server).verify("q", "a")


def test_unknown_extra_fields_are_ignored_for_forward_compatibility(server):
    server.default = Step(200, {**ASK_OK, "new_field": {"a": 1}, "claims": [{**ASK_OK["claims"][0], "extra": 1}]})
    assert make(server).ask("q").answered
    server.default = Step(200, {**VERIFY_OK, "new_field": [1]})
    assert make(server).verify("q", "a").passed


def test_a_utf8_bom_on_the_response_is_tolerated(server):
    import json as _json
    server.default = Step(200, b"\xef\xbb\xbf" + _json.dumps(VERIFY_OK).encode(), raw=True)
    assert make(server).verify("q", "a").passed


def test_missing_optional_fields_use_defaults(server):
    minimal = {"decision": "refuse", "answer_text": "", "audit_id": "a", "total_ms": 1}
    server.default = Step(200, minimal)
    r = make(server).ask("q")
    assert r.outcome == "refused" and r.claims == () and r.review.routed is False and r.redaction.kinds == {}
    assert r.rate_limit is None


def test_a_response_larger_than_the_limit_is_refused(server, monkeypatch):
    monkeypatch.setattr(client_module, "MAX_RESPONSE_BYTES", 1000)
    server.default = Step(200, {**ASK_OK, "answer_text": "x" * 5000})
    with pytest.raises(gc.InvalidResponseError, match="larger"):
        make(server).ask("q")


def test_rate_limit_headers_with_garbage_values_are_ignored(server):
    server.default = Step(200, ASK_OK, headers={"X-RateLimit-Limit": "abc", "X-RateLimit-Remaining": "5"})
    r = make(server).ask("q")
    assert r.rate_limit.limit is None and r.rate_limit.remaining == 5 and r.rate_limit.reset is None


# --- Unicode and size ----------------------------------------------------------------------------

UNICODE = ["Dosis für Müller: 5 mg ✓", "日本語の質問 — 用量は？", "שאלה בעברית مرحبا", "emoji 💊🧪 and combining é (e + ́)",
           "zero​width and ‮RTL override", "null \u0000 byte escape", "quote \" backslash \\ tab \t newline \n",
           "𝒳 astral plane"]


@pytest.mark.parametrize("text", UNICODE)
def test_unicode_survives_the_round_trip_exactly(server, text):
    server.default = Step(200, {**ASK_OK, "answer_text": text})
    r = make(server).ask(text)
    assert server.requests[0].json()["question"] == text and r.answer_text == text


def test_a_large_request_is_sent_intact(server):
    text = ("long question " * 400_000)[:5_000_000]
    server.default = Step(200, ASK_OK)
    make(server, timeout=30).ask(text)
    assert server.requests[0].json()["question"] == text


def test_a_large_response_is_parsed(server):
    big = "word " * 1_000_000
    server.default = Step(200, {**ASK_OK, "answer_text": big})
    assert len(make(server, timeout=30).ask("q").answer_text) == len(big)


def test_413_for_a_payload_the_server_rejects(server):
    server.default = Step(413, error_body("payload_too_large", "Too large."))
    with pytest.raises(gc.PayloadTooLargeError):
        make(server).verify("q", "a" * 100_000)
    assert len(server.requests) == 1


def test_one_client_can_be_used_from_many_threads(server):
    server.default = Step(200, ASK_OK)
    c = make(server)
    results, errs = [], []

    def work(i):
        try:
            results.append(c.ask(f"question {i}").audit_id)
        except Exception as exc:  # noqa: BLE001
            errs.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errs == [] and len(results) == 20 and len(server.requests) == 20
    assert sorted(s.json()["question"] for s in server.requests) == sorted(f"question {i}" for i in range(20))


def test_the_context_manager_works(server):
    server.default = Step(200, {"status": "ok"})
    with make(server) as c:
        assert c.health().ok


def test_all_public_names_are_exported():
    for name in ("Client", "Patient", "AskResponse", "VerifyResponse", "RateLimitError", "ServerError",
                 "APITimeoutError", "InvalidResponseError", "UsageReport", "Health"):
        assert hasattr(gc, name)


def test_the_readme_example_runs(server):
    import re
    text = (HERE.parent / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    code = code.replace("https://groundcheck.example.org", server.url)
    server.enqueue(Step(200, VERIFY_OK), Step(200, ASK_OK))
    namespace: dict = {}
    exec(compile(code, "README.md", "exec"), namespace)
    assert [r.path for r in server.requests] == ["/api/v1/verify", "/api/v1/ask"]
    assert server.requests[0].json()["patient"] == {"age_years": 50, "egfr": 80}


@pytest.mark.parametrize("message", [f"Invalid key {KEY}", f"Authorization: Bearer {KEY} was rejected",
                                     "bad token gck_s3cretvalueABCD", f"{'x' * 995}{KEY}"])
def test_a_server_that_echoes_the_api_key_does_not_put_it_in_the_exception(server, message):
    server.default = Step(401, {"error": {"code": f"bad_{KEY}", "message": message}}, headers={"x-request-id": KEY})
    with pytest.raises(gc.AuthenticationError) as info:
        make(server).ask("q")
    shown = " ".join([info.value.message, str(info.value), repr(info.value), info.value.code or "",
                      info.value.request_id or ""])
    assert KEY not in shown and "s3cretvalue" not in shown
