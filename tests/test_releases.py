"""Knowledge releases: a rebuilt index is checked before it goes live, a
release that answers a must-refuse question is blocked (and alerts), live
questions never see a candidate, promotion can wait for a person, and
rollback restores the previous index in one step."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, db, knowledge, monitoring, pipeline, releases, retrieval  # noqa: E402
from app.db import EvalCase  # noqa: E402

HTML = b"""<html><body><h1>Falls prevention</h1><h2>Assessment</h2>
<p>Complete a falls risk assessment within 6 hours of admission for every adult inpatient.</p></body></html>"""
QUESTION = "How soon should a falls risk assessment be completed after admission?"


@pytest.fixture()
def setup(database, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "RELEASES_DIR", "")
    monkeypatch.setattr(config, "INCLUDE_DEMO_CORPUS", False)
    monkeypatch.setattr(config, "RELEASE_CHECKS", True)
    monkeypatch.setattr(config, "RELEASE_AUTO_PROMOTE", True)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"query": "What is the standard dose of Zyntrafen?", "expect": "refuse"}]))
    monkeypatch.setattr(releases, "GOLDEN_PATH", golden)
    monkeypatch.setattr(releases, "PATIENT_PATH", tmp_path / "none.json")
    yield
    retrieval._active = None
    retrieval.load_index()


def _approve(html: bytes = HTML, name: str = "falls.html") -> dict:
    source = knowledge.import_document(name, html, uploaded_by=None)
    return knowledge.review(source["id"], "approve", None, "", allow_self_approval=True)


def _statuses() -> list[tuple[str, str]]:
    return [(r["name"], r["status"]) for r in releases.list_releases()["releases"]]


def test_a_passing_release_goes_live_and_rollback_restores_the_last_one(setup):
    knowledge.rebuild_index("Empty")
    assert pipeline.run(QUESTION).decision == "refuse"
    _approve()
    status = knowledge.rebuild_index("Approved falls guideline")
    release = status["release"]
    assert release["status"] == "live" and release["check"]["passed"] and release["documents"][0]["title"]
    assert pipeline.run(QUESTION).decision == "answer"
    assert _statuses() == [("R2", "live"), ("R1", "retired")]
    listing = releases.list_releases()["releases"]
    assert listing[1]["changes"] == {"added": [], "removed": ["Falls prevention"], "updated": []}

    # Rebuilding the same content makes no new release.
    assert knowledge.rebuild_index("Again")["release"]["unchanged"]

    back = releases.rollback(None, "Dr Admin")
    assert back["name"] == "R1" and back["rolled_back"]
    assert pipeline.run(QUESTION).decision == "refuse"
    assert _statuses() == [("R2", "rolled_back"), ("R1", "live")]
    forward = releases.promote(listing[0]["id"], None, "Dr Admin")
    assert forward["name"] == "R2" and pipeline.run(QUESTION).decision == "answer"
    with pytest.raises(releases.ReleaseError, match="already live"):
        releases.promote(listing[0]["id"], None, "Dr Admin")


def test_a_release_that_answers_a_must_refuse_question_never_goes_live(setup, monkeypatch):
    knowledge.rebuild_index("Empty")
    _approve()
    knowledge.rebuild_index("Falls guideline")
    with db.session() as s:
        # A review decided this must be refused; the next release answers it.
        s.add(EvalCase(query="How long does inpatient falls risk assessment take after admission?", expect="refuse"))
    _approve(HTML.replace(b"6 hours", b"4 hours"), name="falls.html")
    blocked = knowledge.rebuild_index("Falls guideline v2")["release"]
    assert blocked["status"] == "failed" and blocked["check"]["unsafe"] == 1
    assert blocked["check"]["groups"]["review_tests"]["unsafe"] == 1
    # Live answers still come from the previous release.
    live = [r for r in releases.list_releases()["releases"] if r["status"] == "live"][0]
    assert live["name"] == "R2"
    assert "6 hours" in pipeline.run(QUESTION).answer_text
    with pytest.raises(releases.ReleaseError, match="didn't pass"):
        releases.promote(blocked["id"], None, "x")
    monkeypatch.setattr(config, "ALERT_DISABLED_RULES", {"audit_chain"})
    assert "release_blocked" in monitoring.evaluate()["fired"]


def test_candidates_are_invisible_to_live_questions(setup):
    knowledge.rebuild_index("Empty")
    _approve()
    records = knowledge.approved_records()
    vectors = retrieval.embed([r["text"] for r in records])
    candidate = retrieval.state_from([{**r, "kind": "condition", "section": r.get("section", "")} for r in records], vectors)
    seen = {}

    def live_question():
        seen["live"] = pipeline.run(QUESTION, check_only=True).decision

    with retrieval.using(candidate):
        assert pipeline.run(QUESTION, check_only=True).decision == "answer"
        other = threading.Thread(target=live_question)
        other.start()
        other.join()
    assert seen["live"] == "refuse"


def test_promotion_can_wait_for_a_person_and_the_api_needs_an_admin(setup, monkeypatch):
    monkeypatch.setattr(config, "RELEASE_AUTO_PROMOTE", False)
    knowledge.rebuild_index("Empty")   # the first release always goes live
    _approve()
    waiting = knowledge.rebuild_index("Falls guideline")["release"]
    assert waiting["status"] == "ready" and pipeline.run(QUESTION).decision == "refuse"
    from app.main import app

    client = TestClient(app)
    assert client.get("/api/releases").json()["releases"][0]["status"] == "ready"
    promoted = client.post(f"/api/releases/{waiting['id']}/promote").json()["release"]
    assert promoted["status"] == "live" and promoted["live_by"] == "Local user"
    assert pipeline.run(QUESTION).decision == "answer"
    assert client.post("/api/releases/rollback").json()["release"]["name"] == "R1"
    monkeypatch.setattr(config, "ADMIN_ACCESS", "none")
    assert client.post("/api/releases/rollback").status_code == 403


def test_baseline_and_pruning(setup, monkeypatch):
    knowledge.rebuild_index("Empty")
    assert releases.baseline() is None     # releases already exist
    monkeypatch.setattr(config, "RELEASES_KEEP", 2)
    for hours in range(3, 7):
        _approve(HTML.replace(b"6 hours", f"{hours} hours".encode()))
        knowledge.rebuild_index(f"{hours} hours")
    names = [r["name"] for r in releases.list_releases()["releases"] if r["snapshot"]]
    assert names == ["R5", "R4"]


def test_an_index_that_does_not_match_the_live_release_is_put_back(setup):
    """Going live writes the index, then marks the row. A crash in between
    leaves this instance serving passages the database doesn't call live, so
    startup compares the two and the release's own snapshot wins."""
    _approve(HTML)
    knowledge.rebuild_index("A release to come back to")
    live = next(r for r in releases.list_releases()["releases"] if r["status"] == "live")

    # Serve something else, as a half-finished promotion would have left behind.
    state = retrieval._state()
    records, vectors = list(state.metadata), state.store.vectors()
    retrieval.write_index(records[:-1], vectors[:-1])
    retrieval.load_index()
    assert len(retrieval._state().metadata) != live["passages"]

    result = releases.reconcile()
    assert result == {"release": int(live["name"][1:]), "restored": True}
    assert len(retrieval._state().metadata) == live["passages"]
    assert releases.reconcile() is None     # nothing left to put right
