"""The control mapping matrix: every row complete, evidence that exists, a
validator that rejects what it should, the fixed no-certification banner in
every output, and the API and report script."""

from __future__ import annotations

import conftest

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, compliance_controls as cc, config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="module")
def data():
    return cc.load()


def good():
    return {"frameworks": {"f": {"name": "F", "controls": [
        {"id": "1", "title": "t", "status": "supported", "how": "h", "evidence": ["app/audit.py"]},
        {"id": "2", "title": "t", "status": "customer", "how": "h", "evidence": "none"}]}}}


def test_the_real_mapping_has_no_problems(data):
    assert cc.validate(data) == []
    assert cc.matrix()["problems"] == []


def test_four_frameworks_each_with_enough_rows(data):
    sizes = {k: len(v["controls"]) for k, v in data["frameworks"].items()}
    assert set(sizes) == {"hipaa", "gdpr", "soc2", "iso27001"}
    assert sizes["hipaa"] >= 25 and sizes["gdpr"] >= 15 and sizes["soc2"] >= 20 and sizes["iso27001"] >= 25


def test_every_row_has_a_status_and_an_evidence_pointer_or_an_explicit_none(data):
    for key, framework in data["frameworks"].items():
        for row in framework["controls"]:
            assert row["status"] in cc.STATUSES, (key, row["id"])
            assert row["evidence"] == "none" or (isinstance(row["evidence"], list) and row["evidence"]), (key, row["id"])
            assert row["title"].strip() and row["how"].strip()


def test_every_evidence_pointer_is_a_real_file(data):
    pointers = {p for f in data["frameworks"].values() for r in f["controls"]
                for p in (r["evidence"] if isinstance(r["evidence"], list) else [])}
    assert len(pointers) > 20
    for pointer in pointers:
        assert (ROOT / pointer).exists(), pointer


def test_a_supported_row_always_has_evidence(data):
    for framework in data["frameworks"].values():
        for row in framework["controls"]:
            if row["status"] == "supported":
                assert isinstance(row["evidence"], list) and row["evidence"], row["id"]


def test_ids_are_unique_within_a_framework(data):
    for framework in data["frameworks"].values():
        ids = [str(r["id"]) for r in framework["controls"]]
        assert len(ids) == len(set(ids))


def test_the_well_known_controls_are_present(data):
    ids = {k: {str(r["id"]) for r in v["controls"]} for k, v in data["frameworks"].items()}
    assert {"164.312(b)", "164.312(d)", "164.312(a)(2)(iv)", "164.308(a)(6)"} <= ids["hipaa"]
    assert {"Art. 32", "Art. 33", "Art. 25", "Art. 17"} <= ids["gdpr"]
    assert {"CC6.1", "CC7.2", "CC8.1", "C1.2"} <= ids["soc2"]
    assert {"A.8.15", "A.8.24", "A.5.15", "A.8.5"} <= ids["iso27001"]


def test_known_gaps_are_admitted(data):
    status = {(k, str(r["id"])): r["status"] for k, v in data["frameworks"].items() for r in v["controls"]}
    assert status[("gdpr", "Art. 17")] == "not_addressed" and status[("gdpr", "Art. 15")] == "not_addressed"
    assert status[("hipaa", "164.312(a)(2)(ii)")] == "not_addressed"
    assert status[("hipaa", "164.310(a)")] == "customer" and status[("gdpr", "Art. 6")] == "customer"
    assert any(s == "not_addressed" for s in status.values()) and any(s == "customer" for s in status.values())


def test_no_row_claims_certification_or_compliance(data):
    claim = re.compile(r"\b(certified|compliant|guarantee[sd]?|fully meets|satisfies)\b", re.I)
    for framework in data["frameworks"].values():
        for row in framework["controls"]:
            assert not claim.search(row["how"] + " " + row["title"]), row["id"]


# --- The validator -----------------------------------------------------------------------

def test_the_validator_accepts_a_good_mapping():
    assert cc.validate(good()) == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d["frameworks"]["f"]["controls"][0].pop("status"), "status must be"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(status="compliant"), "status must be"),
    (lambda d: d["frameworks"]["f"]["controls"][0].pop("evidence"), "evidence must be"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(evidence=[]), "evidence must be"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(evidence=""), "evidence must be"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(evidence=["app/nope.py"]), "does not exist"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(evidence=[1]), "evidence must be"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(evidence="none"), "supported needs evidence"),
    (lambda d: d["frameworks"]["f"]["controls"][1].update(id="1"), "duplicate id"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(how=""), "needs an id, title and how"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(title=""), "needs an id, title and how"),
    (lambda d: d["frameworks"]["f"]["controls"][0].update(evidence=["../etc/passwd"]), "does not exist"),
])
def test_the_validator_rejects(mutate, expect):
    d = good()
    mutate(d)
    problems = cc.validate(d)
    assert problems and any(expect in p for p in problems), problems


def test_the_validator_rejects_an_empty_mapping():
    assert cc.validate({}) == ["no frameworks"] and cc.validate({"frameworks": {}}) == ["no frameworks"]


# --- The banner ------------------------------------------------------------------------------

def test_the_banner_is_fixed_and_in_every_output():
    m = cc.matrix()
    assert "holds no certification" in m["banner"] and "not a statement that any requirement is met" in m["banner"]
    assert cc.BANNER in cc.to_markdown(m) and cc.BANNER in cc.to_csv(m)


def test_a_file_cannot_switch_the_banner_off(tmp_path, data):
    data = {**data, "banner": "", "show_banner": False}
    path = tmp_path / "controls.yaml"
    path.write_text(yaml.safe_dump(data))
    m = cc.matrix(path)
    assert m["banner"] == cc.BANNER and cc.BANNER in cc.to_markdown(m)


def test_counts_are_consistent():
    m = cc.matrix()
    assert sum(m["totals"].values()) == sum(len(f["controls"]) for f in m["frameworks"])
    for f in m["frameworks"]:
        assert sum(f["counts"].values()) == len(f["controls"])
        for status in cc.STATUSES:
            assert f["counts"][status] == sum(c["status"] == status for c in f["controls"])
    assert m["totals"]["supported"] > 30 and m["totals"]["customer"] > 30 and m["totals"]["not_addressed"] >= 4


def test_csv_round_trips_and_matches_the_matrix():
    m = cc.matrix()
    text = cc.to_csv(m)
    assert text.splitlines()[0] == "# " + cc.BANNER
    rows = list(csv.reader(io.StringIO(text.split("\n", 1)[1])))
    assert rows[0][:4] == ["Framework", "Control", "Title", "Status"]
    assert len(rows) - 1 == sum(len(f["controls"]) for f in m["frameworks"])
    assert all(r[5] for r in rows[1:])                      # evidence is a pointer or "none", never blank
    assert {r[3] for r in rows[1:]} <= set(cc.STATUSES.values())


def test_markdown_escapes_pipes_in_cells(tmp_path, data):
    data["frameworks"]["hipaa"]["controls"][0]["how"] = "a | b"
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    assert "a / b" in cc.to_markdown(cc.matrix(path))


# --- Script and API ---------------------------------------------------------------------------

def run_script(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "controls_report.py"), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_the_script_check_passes_and_prints_the_formats():
    r = run_script("--check")
    assert r.returncode == 0 and re.search(r"\d+ controls, 0 problem", r.stdout)
    assert cc.BANNER in run_script().stdout and cc.BANNER in run_script("--csv").stdout
    assert json.loads(run_script("--json").stdout)["totals"]["supported"] > 0


def test_the_script_check_fails_on_a_bad_file(tmp_path, monkeypatch):
    data = yaml.safe_load(cc.PATH.read_text())
    data["frameworks"]["hipaa"]["controls"][0]["evidence"] = ["app/does_not_exist.py"]
    bad = tmp_path / "controls.yaml"
    bad.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(cc, "PATH", bad)
    sys.path.insert(0, str(ROOT / "scripts"))
    import controls_report

    assert controls_report.main(["--check"]) == 1


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    from app.main import app

    return app


def test_the_api_needs_a_reviewer(database, client):
    for email, role in (("clin", "clinician"), ("rev", "reviewer")):
        auth.create_user(f"{email}@example.org", PASSWORD, role=role)
    with TestClient(client) as anonymous:
        assert anonymous.get("/api/governance/controls").status_code == 401
        assert anonymous.get("/api/governance/controls.csv").status_code == 401
    with TestClient(client) as c:
        c.post("/api/auth/login", json={"email": "clin@example.org", "password": PASSWORD})
        assert c.get("/api/governance/controls").status_code == 403
        assert c.get("/api/governance/controls.csv").status_code == 403
    with TestClient(client) as c:
        c.post("/api/auth/login", json={"email": "rev@example.org", "password": PASSWORD})
        body = c.get("/api/governance/controls").json()
        assert body["banner"] == cc.BANNER and body["problems"] == [] and len(body["frameworks"]) == 4
        r = c.get("/api/governance/controls.csv")
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        assert r.text.startswith("# " + cc.BANNER)
