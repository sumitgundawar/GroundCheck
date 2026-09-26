"""Monitoring: Prometheus metrics, health probes, alert rules firing and
resolving from real database records, webhook notifications, and access."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, db, monitoring  # noqa: E402
from app.db import Alert, AuditRecord, ReviewCase, Source  # noqa: E402


@pytest.fixture()
def mon(database, monkeypatch):
    monitoring.reset()
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "")
    monkeypatch.setattr(config, "METRICS_TOKEN", "")
    from app.main import app

    return TestClient(app)


def add_questions(n: int, refused: int, minutes_ago: float, total_ms: int = 800, score: float | None = 0.6,
                  spread: float = 0.0, test_run: bool = False) -> None:
    """Audit rows written directly, spread over time, as the pipeline would."""
    now = db.utcnow()
    with db.session() as s:
        for i in range(n):
            s.add(AuditRecord(audit_id=os.urandom(4).hex(), created_at=now - timedelta(minutes=minutes_ago + i * spread),
                              decision="refuse" if i < refused else "answer", query="q", total_ms=total_ms,
                              llm_used=False, record={}, test_run=test_run,
                              top_score=None if score is None else score + (i % 10) * 0.01))


def test_metrics_count_requests_and_questions(mon, monkeypatch):
    client = mon
    assert client.post("/api/ask", json={"query": "What is the standard dose of Caloradine?"}).status_code == 200
    client.get("/api/health")
    text = client.get("/metrics").text
    assert 'groundcheck_http_requests_total{method="POST",route="/api/ask",status="200"} 1' in text
    assert 'groundcheck_questions_total{decision="answer",drafted_by="extractive"} 1' in text
    assert 'groundcheck_question_duration_seconds_bucket{decision="answer",le="+Inf"} 1' in text
    assert "groundcheck_review_cases_open 0" in text and 'groundcheck_alerts_firing{severity="critical"} 0' in text
    # Paths with IDs are labelled by their template.
    client.get("/api/imaging/series/12345")
    assert 'route="/api/imaging/series/{series_id}"' in client.get("/metrics").text

    monkeypatch.setattr(config, "METRICS_TOKEN", "s3cret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_health_probes(mon):
    assert mon.get("/healthz/live").json() == {"status": "ok"}
    ready = mon.get("/healthz/ready")
    assert ready.status_code == 200 and ready.json()["checks"] == {"database": True, "index": True}


def test_refusal_spike_fires_notifies_and_resolves(mon, monkeypatch):
    posted = []

    def hook(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200)

    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "https://hooks.example/alerts")
    monkeypatch.setattr(monitoring, "transport", httpx.MockTransport(hook))
    monkeypatch.setattr(config, "ALERT_DISABLED_RULES", {"audit_chain"})
    add_questions(300, refused=30, minutes_ago=120, spread=20)       # a week at 10% refused
    add_questions(40, refused=24, minutes_ago=1, spread=1)           # the last hour at 60%
    add_questions(40, refused=40, minutes_ago=1, test_run=True)      # test questions don't count

    changes = monitoring.evaluate()
    assert changes["fired"] == ["refusal_rate"] and not changes["errors"]
    [alert] = mon.get("/api/monitoring").json()["firing"]
    assert alert["severity"] == "critical" and "60% of 40 questions" in alert["detail"] and alert["notified"]
    assert posted[0]["alert"]["event"] == "fired" and posted[0]["text"].startswith("GroundCheckHealth Critical:")

    # Still firing: no second notification, and acknowledging records who.
    assert monitoring.evaluate()["fired"] == [] and len(posted) == 1
    acked = mon.post(f"/api/alerts/{alert['id']}/acknowledge").json()["alert"]
    assert acked["acknowledged_by"] == "Local user"

    with db.session() as s:
        for row in s.query(AuditRecord).filter(AuditRecord.decision == "refuse"):
            row.decision = "answer"
    assert monitoring.evaluate()["resolved"] == ["refusal_rate"]
    assert posted[-1]["alert"]["event"] == "resolved"
    overview = mon.get("/api/monitoring").json()
    assert overview["firing"] == [] and overview["resolved"][0]["rule"] == "refusal_rate"


def test_latency_drift_overdue_and_expiring_documents(mon, monkeypatch):
    monkeypatch.setattr(config, "ALERT_DISABLED_RULES", {"audit_chain"})
    now = db.utcnow()
    add_questions(400, refused=0, minutes_ago=60 * 26, spread=30, score=0.7)    # the month: scores near 0.7
    add_questions(60, refused=0, minutes_ago=5, spread=1, score=0.35, total_ms=40000)   # today: lower, slow
    with db.session() as s:
        s.add(ReviewCase(kind="refusal", query="q", query_key="k", first_audit_id="a", last_audit_id="a",
                         due_at=now - timedelta(hours=1)))
        s.add(Source(document_key="anticoag", title="Anticoagulation guideline", status="approved", owner="Pharmacy",
                     filename="a.pdf", content_sha256="h" * 64, expires_on=now + timedelta(days=3)))
    changes = monitoring.evaluate()
    assert set(changes["fired"]) == {"latency", "retrieval_drift", "review_overdue", "documents_expiring"}, changes
    firing = {a["rule"]: a for a in mon.get("/api/monitoring").json()["firing"]}
    assert "40.0 s" in firing["latency"]["detail"] and firing["latency"]["severity"] == "critical"
    assert "mostly lower" in firing["retrieval_drift"]["detail"]
    assert "Anticoagulation guideline" in firing["documents_expiring"]["detail"]
    assert firing["documents_expiring"]["severity"] == "warning"


def test_a_broken_audit_chain_is_critical(mon, monkeypatch):
    from app import audit
    from app.schemas import AskResponse

    monkeypatch.setattr(config, "ALERT_DISABLED_RULES", set())
    audit.store.save("abcd1234", AskResponse(decision="answer", answer_text="x", audit_id="abcd1234", total_ms=5,
                                             llm_used=False), {"redacted_query": "q"})
    assert monitoring.evaluate()["fired"] == []
    with db.session() as s:
        s.query(AuditRecord).filter(AuditRecord.audit_id == "abcd1234").one().total_ms = 1
    # Clear the cached result, not just its timestamp. The gate re-verifies
    # when time.monotonic() - at exceeds the check interval, and monotonic()
    # counts from boot, so setting at=0.0 only forces a re-check on a machine
    # that has been up longer than ALERT_CHAIN_CHECK_MINUTES (six hours by
    # default). On a freshly booted machine -- or a CI runner -- the stale
    # "ok" was reused and this test failed for a reason nothing to do with
    # the audit chain.
    monitoring._last_chain_check.update(at=0.0, result=None)
    assert "audit_chain" in monitoring.evaluate()["fired"]
    with db.session() as s:
        alert = s.query(Alert).filter(Alert.rule == "audit_chain").one()
        assert alert.severity == "critical"


def test_a_failing_rule_doesnt_stop_the_others(mon, monkeypatch):
    def broken(s, now):
        raise RuntimeError("boom")

    monkeypatch.setitem(monitoring.RULES, "latency", broken)
    monkeypatch.setattr(config, "ALERT_DISABLED_RULES", {"audit_chain"})
    changes = monitoring.evaluate()
    assert changes["errors"] == [{"rule": "latency", "error": "RuntimeError"}]


def test_monitoring_needs_a_reviewer(mon, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_ACCESS", "none")
    assert mon.get("/api/monitoring").status_code == 403
    assert mon.post("/api/monitoring/evaluate").status_code == 403


def test_the_overview_says_what_this_instance_runs_on(mon):
    instance = mon.get("/api/monitoring").json()["instance"]
    assert instance["cpus"] >= 0.5 and instance["memory_gb"] > 0
    assert instance["accelerator"] in ("cpu", "mps", "cuda")
    assert instance["embed_batch"] >= 32 and instance["web_workers"] >= 1
    assert instance["answer_cache_seconds"] == config.ANSWER_CACHE_SECONDS
    assert "passages_cached" in instance and instance["version"] == config.VERSION
