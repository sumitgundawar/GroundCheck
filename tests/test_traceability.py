"""Requirements-to-test traceability: the register is well formed, every requirement has at least one test, a test
that names an unknown requirement fails the build, and the trace matrix is generated from the register and the
collected tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest  # noqa: E402
import trace_matrix as tm  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-047")


def write(tmp_path, text, name="requirements.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


GOOD = """
hazards:
  - {id: HZ-01, title: First hazard}
requirements:
  - id: GC-REQ-001
    area: Answering
    text: "Something shall hold."
    risk_control: "A guard."
    hazards: [HZ-01]
    tests: [tests/test_x.py::test_a]
  - id: GC-REQ-002
    area: Audit
    text: "Another thing."
    risk_control: "Another guard."
"""


# --- The register ---------------------------------------------------------------------------------

def test_the_real_register_loads_and_has_the_expected_shape():
    register = tm.load_register()
    ids = [r["id"] for r in register["requirements"]]
    assert len(ids) >= 40 and ids == sorted(ids) and len(set(ids)) == len(ids)
    assert all(tm.ID.match(i) for i in ids)
    assert register["hazards"] and all(tm.HAZARD.match(h) for h in register["hazards"])
    for r in register["requirements"]:
        assert set(r) <= tm.FIELDS and r["text"].strip().endswith(".") and r["area"] and r["risk_control"]
        assert r.get("hazards"), f"{r['id']} should say which hazard it controls"
        assert all(h in register["hazards"] for h in r["hazards"])


def test_every_hazard_is_controlled_by_at_least_one_requirement():
    register = tm.load_register()
    used = {h for r in register["requirements"] for h in r.get("hazards") or []}
    assert set(register["hazards"]) <= used, sorted(set(register["hazards"]) - used)


def test_a_good_register_loads(tmp_path):
    register = tm.load_register(write(tmp_path, GOOD))
    assert [r["id"] for r in register["requirements"]] == ["GC-REQ-001", "GC-REQ-002"] and register["hazards"] == {"HZ-01": "First hazard"}


@pytest.mark.parametrize("mutation,message", [
    (("GC-REQ-001", "GC-REQ-1"), "must look like"), (("GC-REQ-001", "REQ-001"), "must look like"),
    (("GC-REQ-002", "GC-REQ-001"), "listed twice"), (("HZ-01", "HZ-99"), "isn't in the hazards list"),
    (("    area: Answering\n", ""), "non-empty `area`"), (("    risk_control: \"A guard.\"\n", ""), "non-empty `risk_control`"),
    (("    text: \"Something shall hold.\"\n", "    text: \"  \"\n"), "non-empty `text`"),
    (("    tests: [tests/test_x.py::test_a]", "    tests: tests/test_x.py"), "list of node-id prefixes"),
    (("    tests: [tests/test_x.py::test_a]", "    tests: [1, 2]"), "list of node-id prefixes"),
    (("    risk_control: \"A guard.\"\n", "    risk_control: \"A guard.\"\n    owner: someone\n"), "unknown fields"),
    (("  - {id: HZ-01, title: First hazard}", "  - {id: HZ-01, title: First hazard}\n  - {id: HZ-01, title: Again}"), "twice"),
    (("  - {id: HZ-01, title: First hazard}", "  - {id: hazard-1, title: Bad id}"), "Bad hazard entry"),
    (("  - {id: HZ-01, title: First hazard}", "  - {id: HZ-01}"), "Bad hazard entry"),
])
def test_malformed_registers_are_refused_with_a_reason(tmp_path, mutation, message):
    old, new = mutation
    text = GOOD.replace(old, new, 1)
    assert text != GOOD
    with pytest.raises(tm.RegisterError, match=message):
        tm.load_register(write(tmp_path, text))


@pytest.mark.parametrize("content", ["", "just text", "[]", "requirements: x", "requirements: {a: 1}", "hazards: []", "a: [unclosed",
                                     "requirements:\n  - just a string\n"])
def test_a_register_that_is_not_a_register_is_refused(tmp_path, content):
    with pytest.raises(tm.RegisterError):
        tm.load_register(write(tmp_path, content))


def test_a_missing_register_is_refused(tmp_path):
    with pytest.raises(tm.RegisterError, match="Can't read"):
        tm.load_register(tmp_path / "nope.yaml")


# --- The matrix ---------------------------------------------------------------------------------------

REGISTER = {"hazards": {"HZ-01": "h"}, "requirements": [
    {"id": "GC-REQ-001", "area": "A", "text": "One.", "risk_control": "rc", "hazards": ["HZ-01"], "tests": ["tests/test_x.py::test_a"]},
    {"id": "GC-REQ-002", "area": "B", "text": "Two | pipe.", "risk_control": "rc", "hazards": []},
    {"id": "GC-REQ-003", "area": "C", "text": "Three.", "risk_control": "rc"}]}


def item(nodeid, markers=(), bad=False):
    return {"nodeid": nodeid, "markers": list(markers), "bad_markers": bad}


def test_markers_and_links_both_connect_tests_to_requirements():
    matrix = tm.build(REGISTER, [item("tests/test_x.py::test_a"), item("tests/test_x.py::test_a2"), item("tests/test_y.py::test_b", ["GC-REQ-002"]),
                                 item("tests/test_z.py::test_c", ["GC-REQ-002", "GC-REQ-001"]), item("tests/test_w.py::test_d")])
    rows = {r["id"]: r for r in matrix["rows"]}
    assert rows["GC-REQ-001"]["tests"] == ["tests/test_x.py::test_a", "tests/test_x.py::test_a2", "tests/test_z.py::test_c"]
    assert rows["GC-REQ-002"]["tests"] == ["tests/test_y.py::test_b", "tests/test_z.py::test_c"]
    assert rows["GC-REQ-003"]["tests"] == [] and matrix["uncovered"] == ["GC-REQ-003"] and matrix["problems"] == []
    assert matrix["tests_collected"] == 5 and matrix["tests_linked"] == 4


def test_a_test_naming_an_unknown_requirement_is_a_problem():
    matrix = tm.build(REGISTER, [item("tests/test_x.py::test_a", ["GC-REQ-999"]), item("tests/test_x.py::test_b", ["gc-req-001"])])
    assert any("tests/test_x.py::test_a" in p and "unknown requirement 'GC-REQ-999'" in p for p in matrix["problems"])
    assert any("'gc-req-001'" in p for p in matrix["problems"])


def test_a_marker_without_ids_or_with_non_strings_is_a_problem():
    matrix = tm.build(REGISTER, [item("tests/test_x.py::test_a", [], bad=True)])
    assert any("needs requirement ids as strings" in p for p in matrix["problems"])


def test_a_link_that_matches_no_test_is_a_problem_because_stale_links_hide_gaps():
    matrix = tm.build(REGISTER, [item("tests/test_other.py::test_a")])
    assert any("GC-REQ-001: link 'tests/test_x.py::test_a' matches no test" in p for p in matrix["problems"])
    assert matrix["uncovered"] == ["GC-REQ-001", "GC-REQ-002", "GC-REQ-003"]


def test_a_prefix_link_matches_a_family_of_tests_and_parametrised_ids():
    register = {"hazards": {}, "requirements": [{"id": "GC-REQ-001", "area": "A", "text": "x.", "risk_control": "r",
                                                 "tests": ["tests/test_x.py::test_fam"]}]}
    matrix = tm.build(register, [item("tests/test_x.py::test_family_one"), item("tests/test_x.py::test_fam[sqlite-a]"), item("tests/test_x.py::test_other")])
    assert matrix["rows"][0]["tests"] == ["tests/test_x.py::test_fam[sqlite-a]", "tests/test_x.py::test_family_one"]


def test_markdown_reports_coverage_problems_and_escapes_table_cells():
    matrix = tm.build(REGISTER, [item("tests/test_x.py::test_a", ["GC-REQ-777"])])
    text = tm.to_markdown(matrix)
    assert "1 of 3 requirements have at least one test" in text and "**NO TEST**" in text and "Two \\| pipe." in text
    assert "## Problems" in text and "GC-REQ-777" in text and "## Hazards" in text and "HZ-01" in text
    assert "tests/test_x.py (1)" in text


def test_csv_has_one_row_per_requirement():
    matrix = tm.build(REGISTER, [item("tests/test_x.py::test_a")])
    lines = tm.to_csv(matrix).strip().splitlines()
    assert len(lines) == 4 and lines[0].startswith("id,area,requirement") and lines[1].endswith(",1,yes") and lines[3].endswith(",0,NO")


# --- The command --------------------------------------------------------------------------------------------

@pytest.fixture()
def fake_suite(monkeypatch, tmp_path):
    items = [item("tests/test_x.py::test_a"), item("tests/test_y.py::test_b", ["GC-REQ-002"])]
    monkeypatch.setattr(tm, "collect", lambda *a, **k: items)
    return items, write(tmp_path, GOOD.replace("GC-REQ-002\n    area: Audit", "GC-REQ-002\n    area: Audit"))


def test_the_command_warns_for_uncovered_requirements_and_fails_with_strict(fake_suite, tmp_path, capsys):
    items, register = fake_suite
    items.pop(1)                                                             # GC-REQ-002 loses its test
    assert tm.main(["--register", str(register)]) == 0
    assert "warning: GC-REQ-002 has no test" in capsys.readouterr().err
    assert tm.main(["--register", str(register), "--strict"]) == 1


def test_the_command_fails_on_a_test_that_names_an_unknown_requirement(fake_suite, capsys):
    items, register = fake_suite
    items.append(item("tests/test_z.py::test_c", ["GC-REQ-404"]))
    assert tm.main(["--register", str(register)]) == 1
    assert "unknown requirement 'GC-REQ-404'" in capsys.readouterr().err


def test_the_command_writes_markdown_json_and_csv(fake_suite, tmp_path):
    _, register = fake_suite
    out = tmp_path / "build"
    assert tm.main(["--register", str(register), "--markdown", str(out / "m.md"), "--json", str(out / "m.json"), "--csv", str(out / "m.csv")]) == 0
    assert "# Requirements trace matrix" in (out / "m.md").read_text()
    data = json.loads((out / "m.json").read_text())
    assert [r["id"] for r in data["rows"]] == ["GC-REQ-001", "GC-REQ-002"] and data["problems"] == []
    assert (out / "m.csv").read_text().startswith("id,area")


def test_the_command_prints_the_matrix_when_no_file_is_asked_for(fake_suite, capsys):
    _, register = fake_suite
    assert tm.main(["--register", str(register)]) == 0
    assert "| **GC-REQ-001**" in capsys.readouterr().out


def test_a_bad_register_or_a_failed_collection_have_their_own_exit_codes(fake_suite, tmp_path, monkeypatch, capsys):
    assert tm.main(["--register", str(tmp_path / "missing.yaml")]) == 1
    assert "Register error" in capsys.readouterr().err

    def broken(*a, **k):
        raise RuntimeError("pytest couldn't collect")

    monkeypatch.setattr(tm, "collect", broken)
    _, register = fake_suite
    assert tm.main(["--register", str(register)]) == 2


# --- The build fails on unknown ids in the tests themselves -------------------------------------------------------

class FakeItem:
    def __init__(self, nodeid, *ids):
        self.nodeid = nodeid
        self._marks = [pytest.mark.req(i).mark for i in ids]

    def iter_markers(self, name):
        return iter(self._marks if name == "req" else [])


def test_collection_fails_for_a_test_that_names_an_unknown_requirement(monkeypatch):
    monkeypatch.setattr(conftest, "_requirement_ids", lambda: {"GC-REQ-001"})
    conftest.pytest_collection_modifyitems(None, [FakeItem("t::a", "GC-REQ-001"), FakeItem("t::b")])
    with pytest.raises(pytest.UsageError, match=r"t::c: unknown requirement 'GC-REQ-009'"):
        conftest.pytest_collection_modifyitems(None, [FakeItem("t::c", "GC-REQ-009", "GC-REQ-001")])


def test_collection_does_not_check_when_there_is_no_register(monkeypatch):
    monkeypatch.setattr(conftest, "_requirement_ids", lambda: None)
    conftest.pytest_collection_modifyitems(None, [FakeItem("t::a", "GC-REQ-009")])


def test_the_marker_is_registered(pytestconfig):
    assert any(line.startswith("req(") for line in pytestconfig.getini("markers"))


# --- The real suite ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("trace")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "trace_matrix.py"), "--strict", "--markdown", str(out / "m.md"),
                             "--json", str(out / "m.json"), "--csv", str(out / "m.csv")], cwd=ROOT, capture_output=True, text=True, timeout=600,
                            env={**os.environ, "GROQ_API_KEY": ""})
    return result, out


def test_every_requirement_has_at_least_one_test_in_the_real_suite(real_run):
    result, out = real_run
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads((out / "m.json").read_text())
    assert data["problems"] == [] and data["uncovered"] == []
    assert all(r["tests"] for r in data["rows"]) and len(data["rows"]) >= 40
    assert data["tests_collected"] > 1500 and data["tests_linked"] > 400


def test_the_real_matrix_names_this_requirements_own_tests(real_run):
    _, out = real_run
    data = {r["id"]: r for r in json.loads((out / "m.json").read_text())["rows"]}
    assert any("test_traceability.py" in t for t in data["GC-REQ-047"]["tests"])
    assert any("test_soup.py" in t for t in data["GC-REQ-048"]["tests"])
    assert any("test_tenant_isolation.py" in t for t in data["GC-REQ-036"]["tests"])
    assert any("test_burnedin.py" in t for t in data["GC-REQ-022"]["tests"])
    md = (out / "m.md").read_text()
    assert "requirements have at least one test" in md and "NO TEST" not in md
