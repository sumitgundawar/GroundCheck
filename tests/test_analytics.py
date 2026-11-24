"""Adoption metrics and the pilot report, against fixtures whose every number
was worked out by hand (see the comments), including period and UTC-day
boundaries, empty periods, exclusions and site isolation."""

from __future__ import annotations

import conftest  # noqa: F401

import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from itertools import count
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import analytics, auth, config, db, governance, integrity, pipeline, retrieval, sites  # noqa: E402
from app.db import AnswerFeedback, AuditRecord, EvalCase, ReviewCase, Source  # noqa: E402

PASSWORD = "correct horse battery staple"
UTC = timezone.utc
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)          # a Tuesday
SNAPSHOT = Path(__file__).parent / "fixtures" / "pilot_report_snapshot.md"
_ids = count(1)


def at(day: str, clock: str = "12:00:00") -> datetime:
    return datetime.fromisoformat(f"2026-{day}T{clock}").replace(tzinfo=UTC)


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


def audit(created, decision="answer", user=None, site=None, code=None, terms=(), kinds=None, api_key=None,
          test_run=False, demo_seed=False, query="What is the dose?") -> str:
    audit_id = f"a{next(_ids):015x}"
    response = {"decision": decision, "reason_code": code, "refused_reason": "the sources do not cover it" if code else None,
                "explanation": {"missing_terms": list(terms)}, "redaction": {"kinds": kinds or {}, "redacted_query": query}}
    record = {"audit_id": audit_id, "response": response}
    if test_run:
        record["test_run"] = True
    if demo_seed:
        record["demo_seed"] = True
    with db.session() as s:
        s.add(AuditRecord(audit_id=audit_id, created_at=created, user_id=user, decision=decision, query=query,
                          record=record, test_run=test_run, site_id=site, api_key_id=api_key))
    return audit_id


def case(created, resolved=None, site=None, kind="refusal", status=None):
    with db.session() as s:
        c = ReviewCase(kind=kind, status=status or ("resolved" if resolved else "open"), query="q", query_key=f"k{next(_ids)}",
                       first_audit_id="x", last_audit_id="x", created_at=created, last_seen_at=created,
                       due_at=created + timedelta(hours=72), resolved_at=resolved, outcome_note="", site_id=site)
        s.add(c)
        s.flush()
        return c.id


def eval_case(created, from_case=None):
    with db.session() as s:
        s.add(EvalCase(query="q", expect="refuse", note="", from_case_id=from_case, created_at=created))


def feedback_row(created, user, rating, reason="", site=None, audit_id=None):
    with db.session() as s:
        s.add(AnswerFeedback(audit_id=audit_id or f"f{next(_ids):015x}", voter=f"u{user}", user_id=user, site_id=site,
                             rating=rating, reason=reason, comment="", created_at=created, updated_at=created))


def source(status, reviewed):
    with db.session() as s:
        s.add(Source(document_key=f"d{next(_ids)}", title="Doc", status=status, content_sha256="0" * 64,
                     reviewed_at=reviewed, uploaded_at=reviewed))


@pytest.fixture()
def fixture(database):
    """Two sites; users 1-3 at site A, 4 at site B, 5 only long ago.

    Site A questions (UTC):                      user  decision / reason
      11-17 23:59:59  (before a 7-day period)    1     answer
      11-18 00:00:00  (first instant of period)  1     answer
      11-18 10:00     1   refuse term_not_covered
      11-19 12:00     2   answer
      11-22 23:59:59  2   refuse stated_value_unsupported
      11-23 00:00:00  3   answer
      11-24 11:00     1   refuse term_not_covered
      11-24 11:30     --  answer through API key 7
      11-24 11:45     1   answer, test run         -- not counted
      11-24 11:50     1   answer, demo seed        -- not counted
      11-24 11:55     event row                    -- not counted
      11-01 09:00     5   answer (old)
    Site B: 11-20 user 4 answer; 11-21 user 4 refuse term_not_covered.
    """
    A, B = sites.create("site-a", "Site A")["id"], sites.create("site-b", "Site B")["id"]
    for n in range(1, 6):
        auth.create_user(f"u{n}@example.org", PASSWORD, role="clinician", site_id=B if n == 4 else A)
    ids = {u["email"]: u["id"] for u in auth.list_users()}
    u = {n: ids[f"u{n}@example.org"] for n in range(1, 6)}

    with db.session() as session:
        key = db.ApiKey(prefix="gck_test", salt="s", hash="h", name="k", site_id=A, scopes="ask")
        session.add(key)
        session.flush()
        key_id = key.id
    audit(at("11-17", "23:59:59"), user=u[1], site=A)
    audit(at("11-18", "00:00:00"), user=u[1], site=A)
    audit(at("11-18", "10:00:00"), "refuse", user=u[1], site=A, code="term_not_covered", terms=["zyntrafen"])
    audit(at("11-19"), user=u[2], site=A)
    audit(at("11-22", "23:59:59"), "refuse", user=u[2], site=A, code="stated_value_unsupported")
    audit(at("11-23", "00:00:00"), user=u[3], site=A)
    audit(at("11-24", "11:00:00"), "refuse", user=u[1], site=A, code="term_not_covered", terms=["zyntrafen", "velmorex"])
    audit(at("11-24", "11:30:00"), site=A, api_key=key_id)
    audit(at("11-24", "11:45:00"), user=u[1], site=A, test_run=True)
    audit(at("11-24", "11:50:00"), user=u[1], site=A, demo_seed=True)
    audit(at("11-24", "11:55:00"), "event", site=A, test_run=True)
    audit(at("11-01", "09:00:00"), user=u[5], site=A)
    audit(at("11-20"), user=u[4], site=B)
    audit(at("11-21"), "refuse", user=u[4], site=B, code="term_not_covered", terms=["quorimex"])

    # Feedback: in period for A: u1 up (11-20), u2 down wrong (11-21), u3 down incomplete (11-24 00:00:00);
    # not counted: 11-17 23:59:59, one on a test-run answer, and one at site B.
    feedback_row(at("11-20"), u[1], "up", site=A)
    feedback_row(at("11-21"), u[2], "down", "wrong", site=A)
    feedback_row(at("11-24", "00:00:00"), u[3], "down", "incomplete", site=A)
    feedback_row(at("11-17", "23:59:59"), u[1], "up", site=A)
    feedback_row(at("11-22"), u[1], "down", "too_slow", site=A, audit_id=audit(at("11-22"), user=u[1], site=A, test_run=True))
    feedback_row(at("11-20"), u[4], "down", "wrong", site=B)

    # Reviews. Site A cases: c1 12h, c2 48h, c3 240h to close (all closed inside the period), c4 still open.
    c1 = case(at("11-18", "08:00:00"), at("11-18", "20:00:00"), site=A)
    c2 = case(at("11-19", "00:00:00"), at("11-21", "00:00:00"), site=A, kind="flagged")
    case(at("11-10", "12:00:00"), at("11-20", "12:00:00"), site=A)
    case(at("11-23", "00:00:00"), site=A)
    c5 = case(at("11-20", "00:00:00"), at("11-20", "10:00:00"), site=B)      # 10h, site B

    eval_case(at("11-20"), c1)
    eval_case(at("11-21"), c2)
    eval_case(at("11-17", "23:59:59"), c1)       # outside
    eval_case(at("11-20"), c5)                   # site B
    eval_case(at("11-20"), None)                 # no case: belongs to no site

    source("approved", at("11-19"))
    source("approved", at("11-24", "23:59:59"))
    source("approved", at("11-10"))              # outside
    source("retired", at("11-20"))               # not an approval
    return A, B, u


def test_exact_numbers_for_a_site_over_seven_days(fixture):
    A, B, u = fixture
    m = analytics.adoption_metrics(7, A, now=NOW)
    assert (m["from"], m["to"], m["period_days"], m["timezone"]) == ("2026-11-18", "2026-11-24", 7, "UTC")
    # 7 counted questions: answers on 11-18, 11-19, 11-23, 11-24(api); refusals 11-18, 11-22, 11-24.
    q = m["questions"]
    assert (q["total"], q["answered"], q["refused"]) == (7, 4, 3)
    assert q["answer_rate"] == round(4 / 7, 4) == 0.5714 and q["refusal_rate"] == round(3 / 7, 4) == 0.4286
    assert q["refusal_by_reason"] == {"term not in sources": 2, "stated dose not in sources": 1}
    assert q["via_api"] == 1   # the keyed record at 11-24 11:30
    # Users: u1 asked 3 (11-18 x2, 11-24), u2 2, u3 1; the keyless 11-24 11:30 record has no user.
    c = m["clinicians"]
    assert c["active_in_period"] == 3 and c["weekly_active"] == 3
    assert c["questions_per_active_clinician"] == 2.0           # (3 + 2 + 1) / 3
    assert c["weekly"] == [{"week_start": "2026-11-16", "questions": 4, "active_clinicians": 2},   # 11-18 x2, 11-19, 11-22
                           {"week_start": "2026-11-23", "questions": 3, "active_clinicians": 2}]   # 11-23, 11-24 x2
    f = m["feedback"]
    assert (f["total"], f["up"], f["down"], f["positive_rate"]) == (3, 1, 2, 0.3333)
    assert f["by_reason"] == {"incomplete": 1, "wrong": 1}
    rv = m["reviews"]
    assert (rv["opened"], rv["resolved"], rv["open_now"], rv["median_hours_to_close"]) == (3, 3, 1, 48.0)  # 12, 48, 240
    assert m["tests_added"] == 2 and m["documents_approved"] == 2


def test_all_sites_adds_the_other_site_and_unattributed_rows(fixture):
    m = analytics.adoption_metrics(7, None, now=NOW)
    # Site B adds an answer (11-20) and a refusal (11-21).
    assert (m["questions"]["total"], m["questions"]["answered"], m["questions"]["refused"]) == (9, 5, 4)
    assert m["questions"]["refusal_by_reason"] == {"term not in sources": 3, "stated dose not in sources": 1}
    assert m["clinicians"]["active_in_period"] == 4          # + user 4
    assert m["clinicians"]["questions_per_active_clinician"] == round((3 + 2 + 1 + 2) / 4, 2) == 2.0
    assert m["feedback"]["total"] == 4 and m["feedback"]["down"] == 3
    # Site B adds a 10 hour case: median of 10, 12, 48, 240.
    assert m["reviews"]["median_hours_to_close"] == 30.0 and m["reviews"]["resolved"] == 4
    assert m["tests_added"] == 4 and m["documents_approved"] == 2


def test_site_b_sees_only_site_b_and_shared_documents(fixture):
    A, B, u = fixture
    m = analytics.adoption_metrics(7, B, now=NOW)
    assert (m["questions"]["total"], m["questions"]["answered"], m["questions"]["refused"]) == (2, 1, 1)
    assert m["clinicians"]["active_in_period"] == 1 and m["feedback"]["total"] == 1
    assert m["reviews"]["resolved"] == 1 and m["reviews"]["median_hours_to_close"] == 10.0
    assert m["tests_added"] == 1 and m["documents_approved"] == 2        # documents are shared
    serialised = str(m)
    assert "Site A" not in serialised


def test_a_site_without_activity_gets_zeros_not_other_sitess_numbers(fixture):
    C = sites.create("site-c", "Site C")["id"]
    m = analytics.adoption_metrics(30, C, now=NOW)
    assert m["questions"]["total"] == 0 and m["clinicians"]["active_in_period"] == 0
    assert m["feedback"]["total"] == 0 and m["reviews"]["opened"] == 0 and m["tests_added"] == 0


def test_the_thirty_day_period_counts_the_older_user_but_not_as_weekly_active(fixture):
    A, B, u = fixture
    m = analytics.adoption_metrics(30, A, now=NOW)
    assert m["from"] == "2026-10-26"
    assert m["questions"]["total"] == 9                      # + 11-17 23:59:59 and 11-01
    assert m["clinicians"]["active_in_period"] == 4 and m["clinicians"]["weekly_active"] == 3
    assert [w["week_start"] for w in m["clinicians"]["weekly"]] == ["2026-10-26", "2026-11-02", "2026-11-09", "2026-11-16", "2026-11-23"]
    assert [w["questions"] for w in m["clinicians"]["weekly"]] == [1, 0, 0, 5, 3]
    assert sum(w["questions"] for w in m["clinicians"]["weekly"]) == 9


def test_the_week_buckets_add_up_to_the_total(fixture):
    A, B, u = fixture
    for days in (7, 14, 30, 60):
        m = analytics.adoption_metrics(days, A, now=NOW)
        assert sum(w["questions"] for w in m["clinicians"]["weekly"]) == m["questions"]["total"]


# --- Boundaries --------------------------------------------------------------------

def test_utc_day_boundaries(database):
    audit(at("11-17", "23:59:59"))
    audit(at("11-18", "00:00:00"))
    audit(at("11-24", "23:59:59"))
    audit(at("11-25", "00:00:00"))
    m = analytics.adoption_metrics(7, None, now=NOW)
    assert m["questions"]["total"] == 2                       # 11-18 00:00:00 and 11-24 23:59:59
    m = analytics.adoption_metrics(None, None, start=date(2026, 11, 17), end=date(2026, 11, 17), now=NOW)
    assert (m["questions"]["total"], m["period_days"]) == (1, 1)
    m = analytics.adoption_metrics(None, None, start=date(2026, 11, 25), end=date(2026, 11, 25), now=NOW)
    assert m["questions"]["total"] == 1


def test_a_now_in_another_timezone_is_read_as_utc(database):
    audit(at("11-24", "20:00:00"))
    audit(at("11-25", "00:30:00"))
    plus_530 = timezone(timedelta(hours=5, minutes=30))
    now = datetime(2026, 11, 25, 2, 0, tzinfo=plus_530)       # 2026-11-24 20:30 UTC: still the 24th
    m = analytics.adoption_metrics(1, None, now=now)
    assert (m["from"], m["to"], m["questions"]["total"]) == ("2026-11-24", "2026-11-24", 1)
    minus_8 = timezone(timedelta(hours=-8))
    now = datetime(2026, 11, 24, 18, 0, tzinfo=minus_8)       # 2026-11-25 02:00 UTC: already the 25th
    m = analytics.adoption_metrics(1, None, now=now)
    assert (m["to"], m["questions"]["total"]) == ("2026-11-25", 1)


def test_weeks_start_on_monday_utc(database):
    audit(at("11-22", "23:59:59"))   # Sunday
    audit(at("11-23", "00:00:00"))   # Monday
    m = analytics.adoption_metrics(3, None, now=NOW)
    assert [(w["week_start"], w["questions"]) for w in m["clinicians"]["weekly"]] == [("2026-11-16", 1), ("2026-11-23", 1)]


@pytest.mark.parametrize("days", [0, -1, 3661])
def test_period_limits(database, days):
    with pytest.raises(governance.GovernanceError):
        analytics.adoption_metrics(days, None, now=NOW)


def test_period_given_as_dates_must_be_complete_and_ordered(database):
    with pytest.raises(governance.GovernanceError):
        analytics.adoption_metrics(None, None, start=date(2026, 11, 1), now=NOW)
    with pytest.raises(governance.GovernanceError):
        analytics.adoption_metrics(None, None, start=date(2026, 11, 2), end=date(2026, 11, 1), now=NOW)
    with pytest.raises(governance.GovernanceError):
        analytics.adoption_metrics(None, None, start=date(2000, 1, 1), end=date(2026, 1, 1), now=NOW)


def test_one_day_and_the_maximum_period_work(database):
    assert analytics.adoption_metrics(1, None, now=NOW)["period_days"] == 1
    assert analytics.adoption_metrics(3660, None, now=NOW)["period_days"] == 3660


# --- Empty periods -------------------------------------------------------------------

def test_an_empty_installation_gives_zeros_and_none_never_a_division_error(database):
    m = analytics.adoption_metrics(30, None, now=NOW)
    assert m["questions"] == {"total": 0, "answered": 0, "refused": 0, "answer_rate": None, "refusal_rate": None,
                              "via_api": 0, "refusal_by_reason": {}}
    assert m["clinicians"]["active_in_period"] == 0 and m["clinicians"]["questions_per_active_clinician"] is None
    assert m["feedback"] == {"total": 0, "up": 0, "down": 0, "positive_rate": None, "by_reason": {}}
    assert m["reviews"]["median_hours_to_close"] is None and m["tests_added"] == 0 and m["documents_approved"] == 0
    assert all(w["questions"] == 0 and w["active_clinicians"] == 0 for w in m["clinicians"]["weekly"])


def test_a_period_before_any_data_is_empty_even_when_data_exists(fixture):
    m = analytics.adoption_metrics(None, None, start=date(2025, 1, 1), end=date(2025, 1, 31), now=NOW)
    assert m["questions"]["total"] == 0 and m["feedback"]["total"] == 0 and m["reviews"]["resolved"] == 0


def test_only_event_rows_test_runs_and_demo_records_make_an_empty_period(database):
    audit(at("11-24", "10:00:00"), "event", test_run=True)
    audit(at("11-24", "10:01:00"), test_run=True)
    audit(at("11-24", "10:02:00"), demo_seed=True)
    assert analytics.adoption_metrics(7, None, now=NOW)["questions"]["total"] == 0


def test_every_answer_a_user_got_through_the_pipeline_is_counted_once_and_test_runs_are_not(database, monkeypatch):
    monkeypatch.setattr(config, "REVIEW_QUEUE", False)
    pipeline.run("What is the standard dose of Caloradine?")
    pipeline.run("What is the standard dose of Zyntrafen?")
    pipeline.run("What is the standard dose of Zyntrafen?", review=False)     # an evaluation run
    integrity.record_event("admin did something", {})
    m = analytics.adoption_metrics(1, None)
    assert (m["questions"]["total"], m["questions"]["answered"], m["questions"]["refused"]) == (2, 1, 1)


def test_feedback_on_an_evaluation_run_is_not_counted(fixture):
    A, B, u = fixture
    assert analytics.adoption_metrics(7, A, now=NOW)["feedback"]["total"] == 3     # the 4th was on a test run


# --- The API ------------------------------------------------------------------------

@pytest.fixture()
def client(fixture, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    A, B, u = fixture
    auth.create_user("admin@example.org", PASSWORD, role="admin")
    auth.create_user("a-admin@example.org", PASSWORD, role="admin", site_id=A)
    auth.create_user("rev@example.org", PASSWORD, role="reviewer", site_id=A)
    from app.main import app

    clients = {}

    def get(who):
        if who not in clients:
            c = TestClient(app)
            assert c.post("/api/auth/login", json={"email": f"{who}@example.org", "password": PASSWORD}).status_code == 200
            clients[who] = c
        return clients[who]
    return get


def test_the_endpoint_is_admin_only(client):
    from app.main import app

    with TestClient(app) as anonymous:
        assert anonymous.get("/api/metrics/adoption").status_code == 401
        assert anonymous.get("/api/governance/pilot-report").status_code == 401
    for who in ("rev", "u1"):
        assert client(who).get("/api/metrics/adoption").status_code == 403
        assert client(who).get("/api/governance/pilot-report").status_code == 403


def test_a_site_admin_sees_only_their_site_whatever_they_ask_for(client):
    body = client("a-admin").get("/api/metrics/adoption?days=7&site=site-b").json()
    assert body["questions"]["total"] > 0
    assert body["questions"]["total"] == analytics.adoption_metrics(7, sites.list_sites()[0]["id"])["questions"]["total"]
    assert "Site B" not in client("a-admin").get("/api/governance/pilot-report?site=site-b").text


def test_a_group_admin_chooses_a_site_or_sees_all(client):
    allsites = client("admin").get("/api/metrics/adoption?days=30").json()
    one = client("admin").get("/api/metrics/adoption?days=30&site=site-b").json()
    assert one["questions"]["total"] == 2 and allsites["questions"]["total"] == 11
    assert client("admin").get("/api/metrics/adoption?site=nowhere").status_code == 400


@pytest.mark.parametrize("query", ["days=0", "days=99999", "from=2026-11-01", "from=bad&to=2026-11-02",
                                   "from=2026-11-05&to=2026-11-01", "days=abc"])
def test_bad_periods_are_refused(client, query):
    assert client("admin").get(f"/api/metrics/adoption?{query}").status_code in (400, 422)


def test_dates_select_an_inclusive_period(client):
    body = client("admin").get("/api/metrics/adoption?from=2026-11-18&to=2026-11-18&site=site-a").json()
    assert (body["from"], body["to"], body["questions"]["total"]) == ("2026-11-18", "2026-11-18", 2)


# --- Pilot report -------------------------------------------------------------------------

def test_the_report_matches_a_committed_snapshot(fixture):
    A, B, u = fixture
    text = analytics.pilot_report_markdown(None, A, start=date(2026, 11, 18), end=date(2026, 11, 24),
                                           site_name="Site A", now=NOW)
    from app import __version__

    text = text.replace(f"Version {__version__}.", "Version X.")
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(text, encoding="utf-8")
    assert text == SNAPSHOT.read_text(encoding="utf-8")


def test_the_report_is_deterministic(fixture):
    A, B, u = fixture
    kwargs = dict(start=date(2026, 11, 18), end=date(2026, 11, 24), site_name="Site A", now=NOW)
    assert analytics.pilot_report_markdown(None, A, **kwargs) == analytics.pilot_report_markdown(None, A, **kwargs)


def test_the_report_states_what_it_is_not(fixture):
    text = analytics.pilot_report_markdown(7, None, now=NOW)
    assert "not evidence of clinical safety" in text and "not a certified medical device" in text
    assert "DEMONSTRATION DATA" in text           # the demo formulary is synthetic


def test_document_gaps_group_by_term_and_count_questions(fixture):
    A, B, u = fixture
    gaps = analytics.document_gaps(date(2026, 11, 18), date(2026, 11, 24), A, now=NOW)
    # 11-18 blocked "zyntrafen"; 11-24 blocked "zyntrafen" and "velmorex".
    assert gaps["terms"] == [{"term": "zyntrafen", "questions": 2}, {"term": "velmorex", "questions": 1}]
    assert gaps["distinct_terms"] == 2 and gaps["withheld"] == 0
    everyone = analytics.document_gaps(date(2026, 11, 18), date(2026, 11, 24), None, now=NOW)
    assert [t["term"] for t in everyone["terms"]] == ["zyntrafen", "velmorex", "quorimex"] or \
        {t["term"]: t["questions"] for t in everyone["terms"]} == {"zyntrafen": 2, "velmorex": 1, "quorimex": 1}


def test_identifying_terms_and_questions_with_identifiers_are_withheld(database):
    audit(at("11-24", "10:00:00"), "refuse", code="term_not_covered", terms=["zyntrafen"])
    audit(at("11-24", "10:01:00"), "refuse", code="term_not_covered", terms=["mr", "9434765919", "[NAME]", "o'brien", "jo smith"],
          kinds={"name": 1})
    audit(at("11-24", "10:02:00"), "refuse", code="term_not_covered", terms=["smythe", "07700900123", "a@b.org", "[PHONE]"])
    gaps = analytics.document_gaps(date(2026, 11, 24), date(2026, 11, 24), None, now=NOW)
    assert [t["term"] for t in gaps["terms"]] == ["smythe", "zyntrafen"]
    assert gaps["withheld"] == 5 + 3


def test_the_report_prints_no_question_text_or_identifier(database):
    secrets = ["943 476 5919", "9434765919", "John Smith", "jane@example.org", "07700 900123"]
    q = "Mr John Smith NHS 943 476 5919 jane@example.org 07700 900123 asked about Zyntrafen"
    audit(at("11-24", "10:00:00"), "refuse", code="term_not_covered", terms=["zyntrafen", "john smith"],
          kinds={"name": 1, "nhs_number": 1}, query=q)
    audit(at("11-24", "10:01:00"), "refuse", code="term_not_covered", terms=["velmorex"], query="What is velmorex?")
    text = analytics.pilot_report_markdown(1, None, now=NOW)
    for s in secrets:
        assert s not in text
    assert "What is velmorex" not in text and "asked about" not in text
    assert "- velmorex: 1 question" in text and "zyntrafen" not in text.split("## Document gaps")[1].split("withheld")[0]
    assert "withheld" in text
    assert not re.search(r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b", text)
    assert not re.search(r"[\w.]+@[\w.]+\.\w+", text)


def test_report_period_and_site_name_appear_and_sites_are_isolated(fixture):
    A, B, u = fixture
    text = analytics.pilot_report_markdown(None, B, start=date(2026, 11, 18), end=date(2026, 11, 24), site_name="Site B", now=NOW)
    assert "Site: Site B." in text and "Period: 2026-11-18 to 2026-11-24 (7 days, UTC)" in text
    assert "- Questions: 2 (1 answered, 1 refused)" in text and "quorimex" in text
    assert "velmorex" not in text and "zyntrafen" not in text


def test_report_counts_incidents_flagged_answers_and_harm(fixture):
    from app import incidents

    A, B, u = fixture
    incidents.report({"title": "Wrong formulation quoted", "category": "answer", "harm": "moderate",
                      "description": "Seen on the ward."}, None, "Dr Rev", site_id=A)
    text = analytics.pilot_report_markdown(1, A, now=datetime.now(UTC))
    assert "- Incidents reported: 1" in text and "harm moderate: 1" in text
    other = analytics.pilot_report_markdown(1, B, now=datetime.now(UTC))
    assert "- Incidents reported: 0" in other


def test_empty_report_renders(database):
    text = analytics.pilot_report_markdown(30, None, now=NOW)
    assert "No refusals in the period." in text and "No gaps recorded in the period." in text
    assert "Answer rate: n/a" in text and "Median hours to close: n/a" in text


def test_the_report_endpoint_serves_markdown_with_periods(client):
    r = client("admin").get("/api/governance/pilot-report?from=2026-11-18&to=2026-11-24&site=site-a")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert "attachment" in r.headers["content-disposition"] and "Site: Site A." in r.text
    assert client("admin").get("/api/governance/pilot-report?days=0").status_code == 400
