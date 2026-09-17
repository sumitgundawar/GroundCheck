"""Clinical review and governance: the review queue, flagged answers,
escalation, closing the loop into tests, the hazard log, reports, and the
API's permission rules. Runs on SQLite, and on PostgreSQL when
TEST_POSTGRES_URL is set."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, config, db, governance, pipeline, retrieval  # noqa: E402

PASSWORD = "correct horse battery staple"
REFUSED = "What is the standard dose of Zyntrafen?"
ANSWERED = "What is the standard dose of Caloradine?"


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def queue(database, monkeypatch):
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    yield


def _cases(status="open"):
    return governance.list_cases(status)["cases"]


# --- Opening cases -------------------------------------------------------------

def test_a_refusal_opens_one_case_and_repeats_add_to_it(queue):
    first = pipeline.run(REFUSED)
    assert first.decision == "refuse"
    pipeline.run(f"  {REFUSED.upper()} ")  # same question, different spacing and case
    cases = _cases()
    assert len(cases) == 1
    case = cases[0]
    assert case["occurrences"] == 2 and case["kind"] == "refusal" and case["priority"] == "normal"
    assert case["first_audit_id"] == first.audit_id and case["last_audit_id"] != first.audit_id
    assert case["reason_category"] == "term not in sources"
    assert not case["overdue"]


def test_answers_and_test_runs_open_no_cases(queue, monkeypatch):
    assert pipeline.run(ANSWERED).decision == "answer"
    assert pipeline.run(REFUSED, review=False).decision == "refuse"
    monkeypatch.setattr(config, "REVIEW_QUEUE", False)
    pipeline.run(REFUSED)
    assert _cases() == []


def test_a_resolved_question_asked_again_opens_a_new_case(queue):
    pipeline.run(REFUSED)
    case_id = _cases()[0]["id"]
    governance.resolve(case_id, "correct_refusal", "Not on our formulary.", None)
    pipeline.run(REFUSED)
    assert [c["id"] != case_id for c in _cases()] == [True]


def test_categories():
    assert governance.categorise("No sufficiently relevant source was found.") == "no relevant source"
    assert governance.categorise("Something new") == "other"
    assert governance.categorise("") == "other"


# --- Flagging answers ------------------------------------------------------------

def test_flagging_an_answer_opens_a_high_priority_case(queue):
    answer = pipeline.run(ANSWERED)
    with pytest.raises(governance.GovernanceError, match="Say briefly"):
        governance.flag_answer(answer.audit_id, "bad", None)
    case = governance.flag_answer(answer.audit_id, "The dose looks too high for renal patients.", None)
    assert case["kind"] == "flagged" and case["priority"] == "high"
    detail = governance.get_case(case["id"])
    assert detail["answer_text"] and detail["sources"]
    assert [e["action"] for e in detail["events"]] == ["flagged"]


def test_refusals_and_unknown_records_cannot_be_flagged(queue):
    refused = pipeline.run(REFUSED)
    with pytest.raises(governance.GovernanceError, match="Only answers"):
        governance.flag_answer(refused.audit_id, "Should have answered this.", None)
    with pytest.raises(governance.GovernanceError, match="isn't in the audit trail"):
        governance.flag_answer("no-such-id", "Should have answered this.", None)


def test_flagged_cases_come_first(queue):
    pipeline.run(REFUSED)
    governance.flag_answer(pipeline.run(ANSWERED).audit_id, "Please check this answer.", None)
    assert [c["kind"] for c in _cases()] == ["flagged", "refusal"]


# --- Working the queue -------------------------------------------------------------

def test_overdue_cases_escalate_once(queue):
    pipeline.run(REFUSED)
    case_id = _cases()[0]["id"]
    later = db.utcnow() + timedelta(hours=config.REVIEW_SLA_HOURS + 1)
    assert governance.escalate_overdue(later) == 1
    assert governance.escalate_overdue(later) == 0
    case = governance.get_case(case_id)
    assert case["escalated"] and case["priority"] == "high"
    assert [e["action"] for e in case["events"]] == ["opened", "escalated"]


def test_assign_comment_resolve_and_reopen(queue):
    clinician = auth.create_user("clin@example.org", PASSWORD, role="clinician")
    reviewer = auth.create_user("rev@example.org", PASSWORD, role="reviewer", name="Dr Rev")
    pipeline.run(REFUSED)
    case_id = _cases()[0]["id"]

    with pytest.raises(governance.GovernanceError, match="active reviewer"):
        governance.assign(case_id, clinician.id, reviewer.id)
    assert governance.assign(case_id, reviewer.id, reviewer.id)["assigned_to_name"] == "Dr Rev"
    assert governance.list_cases("open", assigned_to=reviewer.id)["cases"][0]["id"] == case_id
    with pytest.raises(governance.GovernanceError, match="Write a comment"):
        governance.comment(case_id, "  ", reviewer.id)
    governance.comment(case_id, "Checking the formulary.", reviewer.id)

    with pytest.raises(governance.GovernanceError, match="Choose an outcome"):
        governance.resolve(case_id, "shrug", "", reviewer.id)
    with pytest.raises(governance.GovernanceError, match="answer or to refuse"):
        governance.resolve(case_id, "add_test", "", reviewer.id)
    resolved = governance.resolve(case_id, "add_test", "Must always refuse.", reviewer.id, "refuse")
    assert resolved["status"] == "resolved" and resolved["resolved_by_name"] == "Dr Rev"
    with pytest.raises(governance.GovernanceError, match="already closed"):
        governance.resolve(case_id, "no_action", "", reviewer.id)
    with pytest.raises(governance.GovernanceError, match="Only open cases"):
        governance.assign(case_id, reviewer.id, reviewer.id)

    reopened = governance.reopen(case_id, "New guidance arrived.", reviewer.id)
    assert reopened["status"] == "open" and reopened["resolved_at"] is None
    assert [e["action"] for e in reopened["events"]] == [
        "opened", "assigned", "commented", "resolved", "reopened"]
    counts = governance.list_cases("all")["counts"]
    assert counts["open"] == 1 and counts["resolved"] == 0


def test_resolving_as_a_test_adds_a_permanent_evaluation_case(queue):
    pipeline.run(REFUSED)
    governance.flag_answer(pipeline.run(ANSWERED).audit_id, "Correct, keep answering this.", None)
    for case in _cases():
        expected = "answer" if case["kind"] == "flagged" else "refuse"
        governance.resolve(case["id"], "add_test", "", None, expected)
    tests = governance.list_eval_cases()
    assert {(t["query"], t["expect"]) for t in tests} == {(REFUSED, "refuse"), (ANSWERED, "answer")}

    result = governance.run_eval_cases()
    assert result == {**result, "total": 2, "passed": 2, "unsafe_answers": 0}
    assert _cases() == []  # running the tests opened no new cases
    assert governance.report(1)["questions"]["total"] == 2  # and isn't counted as use


# --- Hazard log --------------------------------------------------------------------

HAZARD = {"title": "Wrong dose cited", "cause": "Outdated protocol", "effect": "Overdose",
          "severity": 5, "likelihood": 3, "controls": "Expiry dates; dosage check",
          "residual_severity": 5, "residual_likelihood": 1, "owner": "Pharmacy"}


def test_risk_levels():
    assert governance.risk(1, 4) == {"score": 4, "level": "low"}
    assert governance.risk(3, 3)["level"] == "medium"
    assert governance.risk(4, 4)["level"] == "high"
    assert governance.risk(5, 5) == {"score": 25, "level": "very high"}


def test_hazards_are_validated_and_sorted_by_residual_risk(database):
    low = governance.save_hazard({**HAZARD, "title": "Slow answer", "severity": 2, "likelihood": 2,
                                  "residual_severity": 1, "residual_likelihood": 1}, None)
    high = governance.save_hazard(HAZARD, None)
    assert high["initial_risk"] == {"score": 15, "level": "high"}
    assert high["residual_risk"] == {"score": 5, "level": "medium"}
    assert [h["id"] for h in governance.list_hazards()] == [high["id"], low["id"]]

    updated = governance.save_hazard({**HAZARD, "status": "mitigated"}, None, high["id"])
    assert updated["status"] == "mitigated"

    for bad, message in (
        ({**HAZARD, "title": " "}, "title"),
        ({**HAZARD, "severity": 6}, "Severity"),
        ({**HAZARD, "likelihood": "3"}, "Likelihood"),
        ({**HAZARD, "status": "gone"}, "Status"),
        ({**HAZARD, "residual_likelihood": 4}, "higher than the initial"),
    ):
        with pytest.raises(governance.GovernanceError, match=message):
            governance.save_hazard(bad, None)
    with pytest.raises(governance.GovernanceError, match="No such hazard"):
        governance.save_hazard(HAZARD, None, 9999)


# --- Reports ---------------------------------------------------------------------------

def test_report_and_safety_case(queue):
    pipeline.run(ANSWERED)
    pipeline.run(REFUSED)
    case_id = _cases()[0]["id"]
    governance.resolve(case_id, "correct_refusal", "", None)
    governance.save_hazard(HAZARD, None)

    r = governance.report(30)
    assert r["questions"]["total"] == 2 and r["questions"]["refused"] == 1
    assert r["questions"]["refusal_rate"] == 0.5
    assert r["questions"]["refusal_reasons"] == {"term not in sources": 1}
    assert r["reviews"]["opened"] == 1 and r["reviews"]["resolved_on_time"] == 1
    assert r["reviews"]["outcomes"] == {"correct_refusal": 1}
    with pytest.raises(governance.GovernanceError, match="period"):
        governance.report(0)

    text = governance.safety_case_markdown(30)
    assert "# GroundCheck clinical safety case summary" in text
    assert "| H1 | Wrong dose cited | 15 (high)" in text
    assert "Clinical safety officer" in text


# --- API ---------------------------------------------------------------------------------

@pytest.fixture()
def client(queue, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    from app.main import app
    with TestClient(app) as c:
        yield c


def _login(client, email):
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200


def test_api_roles(client):
    auth.create_user("clin@example.org", PASSWORD, role="clinician")
    reviewer = auth.create_user("rev@example.org", PASSWORD, role="reviewer")
    auth.create_user("admin@example.org", PASSWORD, role="admin")

    _login(client, "clin@example.org")
    answer = client.post("/api/ask", json={"query": ANSWERED}).json()
    assert client.post("/api/ask", json={"query": REFUSED}).json()["decision"] == "refuse"
    flagged = client.post(f"/api/audit/{answer['audit_id']}/flag", json={"note": "Please double-check."})
    assert flagged.status_code == 200, flagged.text
    assert client.post(f"/api/audit/{answer['audit_id']}/flag", json={"note": "x"}).status_code == 400
    assert client.get("/api/reviews").status_code == 403
    assert client.get("/api/hazards").status_code == 403

    _login(client, "rev@example.org")
    body = client.get("/api/reviews").json()
    assert body["counts"]["open"] == 2 and body["me"] == reviewer.id
    assert {r["id"] for r in body["reviewers"]} >= {reviewer.id}
    case_id = body["cases"][0]["id"]
    assert client.post(f"/api/reviews/{case_id}/assign", json={"user_id": reviewer.id}).status_code == 200
    assert client.get("/api/reviews?mine=true").json()["cases"][0]["id"] == case_id
    assert client.get("/api/reviews?status=nope").status_code == 400
    r = client.post(f"/api/reviews/{case_id}/resolve", json={"outcome": "add_test", "expected_decision": "answer"})
    assert r.status_code == 200 and r.json()["case"]["status"] == "resolved"
    assert client.get("/api/reviews/999999").status_code == 400
    assert client.post("/api/hazards", json=HAZARD).status_code == 403
    assert client.post("/api/review-tests/run").json()["passed"] == 1
    assert client.get("/api/governance/report?days=7").json()["reviews"]["resolved"] == 1

    _login(client, "admin@example.org")
    created = client.post("/api/hazards", json=HAZARD)
    assert created.status_code == 200
    hazard_id = created.json()["hazard"]["id"]
    bad = client.put(f"/api/hazards/{hazard_id}", json={**HAZARD, "severity": 9})
    assert bad.status_code == 400 and "Severity" in bad.json()["detail"]
    safety = client.get("/api/governance/safety-case")
    assert safety.status_code == 200 and "attachment" in safety.headers["content-disposition"]
    assert "Wrong dose cited" in safety.text


def test_api_without_sign_in_is_local_only(queue, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    from app.main import app
    with TestClient(app) as c:
        assert c.get("/api/reviews").status_code == 200
        remote = {"X-Forwarded-For": "203.0.113.9"}
        assert c.get("/api/reviews", headers=remote).status_code == 403
        assert c.post("/api/hazards", json=HAZARD, headers=remote).status_code == 403


def test_usage_dashboard(queue):
    pipeline.run(ANSWERED)
    pipeline.run(REFUSED)
    pipeline.run("Mr John Smith: " + REFUSED)
    pipeline.run(REFUSED, review=False)  # a test run: not counted
    u = governance.usage(7)
    assert len(u["daily"]) == 7 and u["daily"][-1] == {"date": u["to"], "answered": 1, "refused": 2}
    t = u["totals"]
    assert (t["questions"], t["answered"], t["refused"], t["deidentified"]) == (3, 1, 2, 1)
    assert t["median_ms"] is not None and t["open_reviews"] == 2
    assert u["drafted_by"] == {"Extractive": 3, "Local model": 0, "Cloud model": 0}
    assert u["refusal_reasons"] == {"term not in sources": 2}
    assert u["top_sources"] and u["top_sources"][0]["citations"] >= 1
    with pytest.raises(governance.GovernanceError):
        governance.usage(0)


def test_embeddings_summary_endpoint():
    from app.main import app

    with TestClient(app) as c:
        body = c.get("/api/embeddings").json()
    assert body["dimensions"] == 384 and body["passages"] == len(retrieval.all_metadata())
    assert body["vector_store"] in ("local", "qdrant") and "state" in body["index"]


def test_surveillance_report_gathers_the_period(queue, monkeypatch):
    from app import incidents, monitoring

    monkeypatch.setattr(config, "ALERT_DISABLED_RULES", {"audit_chain"})
    pipeline.run(REFUSED)
    pipeline.run(ANSWERED)
    incidents.report({"title": "Wrong formulation quoted", "category": "answer", "harm": "moderate",
                      "description": "Seen on the ward."}, None, "Dr Rev")
    incidents.report({"title": "Export sent to the wrong address", "category": "data_protection",
                      "description": "Recalled."}, None, "Dr Rev")
    monitoring.evaluate()
    text = governance.surveillance_markdown(90)
    assert "# GroundCheck post-market surveillance report" in text
    assert "Questions: 2 (1 answered, 1 refused" in text
    assert "Moderate harm: 1" in text and "Data protection or security: 1" in text
    assert "Needing a decision on reporting to a regulator: 1" in text   # the data breach, not moderate harm
    assert "INC-" in text and "Quality or clinical safety lead:" in text
    assert "term not in sources" in text          # why the refusal happened


def test_surveillance_report_needs_a_reviewer(client):
    auth.create_user("clin@example.org", PASSWORD, role="clinician")
    auth.create_user("rev@example.org", PASSWORD, role="reviewer")
    _login(client, "clin@example.org")
    assert client.get("/api/governance/surveillance").status_code == 403
    _login(client, "rev@example.org")
    r = client.get("/api/governance/surveillance?days=30")
    assert r.status_code == 200 and r.text.startswith("# GroundCheck post-market surveillance report")
    assert r.headers["content-disposition"].endswith('filename="groundcheck-surveillance.md"')
