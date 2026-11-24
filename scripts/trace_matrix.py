"""Requirements-to-test traceability, IEC 62304 style.

Reads compliance/requirements.yaml, collects the test suite (without running
it) and writes a trace matrix: each requirement, its hazards and risk control,
and the tests that verify it.

A test is linked to a requirement in either of two ways:

- a marker:   @pytest.mark.req("GC-REQ-012")     (or pytestmark = pytest.mark.req(...))
- a link in the register: `tests: [tests/test_guards.py::test_dosage_]` on the
  requirement, matched as a prefix of the test's node id, so a whole file, a
  family of tests or one test can be linked without editing the test.

Exit status:
  0  the register and links are consistent (requirements without a test are
     listed as warnings; with --strict they fail)
  1  a test refers to a requirement that doesn't exist, a link matches no test,
     the register is malformed, or (with --strict) a requirement has no test
  2  the tests could not be collected

    python scripts/trace_matrix.py                     # matrix to stdout
    python scripts/trace_matrix.py --strict --markdown build/trace.md --json build/trace.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "compliance" / "requirements.yaml"
ID = re.compile(r"^GC-REQ-\d{3}$")
HAZARD = re.compile(r"^HZ-\d{2}$")
FIELDS = {"id", "area", "text", "risk_control", "hazards", "tests"}


class RegisterError(Exception):
    pass


def load_register(path: Path = REGISTER) -> dict:
    """The register as {"hazards": {id: title}, "requirements": [ ... ]}, or RegisterError."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RegisterError("PyYAML is needed to read the register (pip install pyyaml).") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegisterError(f"Can't read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise RegisterError(f"{path} isn't valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("requirements"), list):
        raise RegisterError("The register needs a `requirements:` list.")
    hazards: dict[str, str] = {}
    for h in data.get("hazards") or []:
        if not isinstance(h, dict) or not HAZARD.match(str(h.get("id", ""))) or not str(h.get("title", "")).strip():
            raise RegisterError(f"Bad hazard entry: {h!r} (needs id HZ-nn and a title).")
        if h["id"] in hazards:
            raise RegisterError(f"Hazard {h['id']} is listed twice.")
        hazards[h["id"]] = h["title"]
    seen: set[str] = set()
    for r in data["requirements"]:
        if not isinstance(r, dict):
            raise RegisterError(f"A requirement must be a mapping, not {r!r}.")
        rid = str(r.get("id", ""))
        if not ID.match(rid):
            raise RegisterError(f"Requirement id {rid!r} must look like GC-REQ-012.")
        if rid in seen:
            raise RegisterError(f"{rid} is listed twice.")
        seen.add(rid)
        extra = set(r) - FIELDS
        if extra:
            raise RegisterError(f"{rid} has unknown fields: {', '.join(sorted(extra))}.")
        for field in ("area", "text", "risk_control"):
            if not isinstance(r.get(field), str) or not r[field].strip():
                raise RegisterError(f"{rid} needs a non-empty `{field}`.")
        for h in r.get("hazards") or []:
            if h not in hazards:
                raise RegisterError(f"{rid} refers to hazard {h}, which isn't in the hazards list.")
        links = r.get("tests") or []
        if not isinstance(links, list) or not all(isinstance(t, str) and t.strip() for t in links):
            raise RegisterError(f"{rid}: `tests` must be a list of node-id prefixes.")
    return {"hazards": hazards, "requirements": data["requirements"]}


class _Collector:
    """A pytest plugin that records every collected test and its req markers."""

    def __init__(self):
        self.items: list[dict] = []

    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            marks = list(item.iter_markers("req"))
            bad = any(not m.args or not all(isinstance(a, str) for a in m.args) for m in marks)
            self.items.append({"nodeid": item.nodeid, "markers": [a for m in marks for a in m.args if isinstance(a, str)],
                               "bad_markers": bad})


def collect(extra_args: list[str] | None = None) -> list[dict]:
    """Collect the suite. Raises RuntimeError when pytest can't."""
    import os

    import pytest

    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("GROQ_API_KEY", "")
    collector = _Collector()
    buffer = io.StringIO()
    real = sys.stdout
    sys.stdout = buffer
    try:
        code = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT), str(ROOT / "tests"),
                            *(extra_args or [])], plugins=[collector])
    finally:
        sys.stdout = real
    if code not in (0, 5):
        raise RuntimeError(f"pytest couldn't collect the tests (exit {code}):\n{buffer.getvalue()[-2000:]}")
    return collector.items


def build(register: dict, items: list[dict]) -> dict:
    """The matrix, and the problems found. Pure: takes the register and the collected tests."""
    requirements = {r["id"]: r for r in register["requirements"]}
    linked: dict[str, set[str]] = defaultdict(set)
    problems: list[str] = []
    for item in items:
        if item["bad_markers"]:
            problems.append(f"{item['nodeid']}: a req marker needs requirement ids as strings, e.g. req(\"GC-REQ-012\").")
        for rid in item["markers"]:
            if rid in requirements:
                linked[rid].add(item["nodeid"])
            else:
                problems.append(f"{item['nodeid']}: refers to unknown requirement {rid!r}.")
    for rid, r in requirements.items():
        for prefix in r.get("tests") or []:
            hits = [i["nodeid"] for i in items if i["nodeid"].startswith(prefix)]
            if not hits:
                problems.append(f"{rid}: link {prefix!r} matches no test.")
            linked[rid].update(hits)
    rows = []
    for rid in sorted(requirements):
        r = requirements[rid]
        tests = sorted(linked.get(rid, ()))
        rows.append({"id": rid, "area": r["area"], "text": r["text"], "risk_control": r["risk_control"],
                     "hazards": list(r.get("hazards") or []), "tests": tests, "covered": bool(tests)})
    uncovered = [r["id"] for r in rows if not r["covered"]]
    return {"rows": rows, "hazards": register["hazards"], "problems": problems, "uncovered": uncovered,
            "tests_collected": len(items),
            "tests_linked": len({t for r in rows for t in r["tests"]})}


def to_markdown(matrix: dict) -> str:
    rows = matrix["rows"]
    covered = sum(r["covered"] for r in rows)
    out = ["# Requirements trace matrix", "",
           f"{covered} of {len(rows)} requirements have at least one test. {matrix['tests_linked']} of "
           f"{matrix['tests_collected']} collected tests are linked to a requirement.", "",
           "| Requirement | Area | Hazards | Risk control | Tests | Status |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        families = _families(r["tests"])
        out.append(f"| **{r['id']}** {_cell(r['text'])} | {r['area']} | {', '.join(r['hazards']) or 'none'} | "
                   f"{_cell(r['risk_control'])} | {len(r['tests'])}: {_cell('; '.join(families))} | "
                   f"{'covered' if r['covered'] else '**NO TEST**'} |")
    if matrix["hazards"]:
        out += ["", "## Hazards", ""] + [f"- **{k}** {v}" for k, v in sorted(matrix["hazards"].items())]
    if matrix["problems"]:
        out += ["", "## Problems", ""] + [f"- {p}" for p in matrix["problems"]]
    return "\n".join(out) + "\n"


def _families(tests: list[str]) -> list[str]:
    """Test node ids grouped by file, for a readable cell."""
    files: dict[str, int] = defaultdict(int)
    for t in tests:
        files[t.split("::", 1)[0]] += 1
    return [f"{f} ({n})" for f, n in sorted(files.items())]


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def to_csv(matrix: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "area", "requirement", "hazards", "risk_control", "tests", "covered"])
    for r in matrix["rows"]:
        writer.writerow([r["id"], r["area"], r["text"], " ".join(r["hazards"]), r["risk_control"], len(r["tests"]),
                         "yes" if r["covered"] else "NO"])
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--register", type=Path, default=REGISTER)
    parser.add_argument("--markdown", type=Path, help="write the matrix as Markdown")
    parser.add_argument("--json", type=Path, help="write the matrix as JSON")
    parser.add_argument("--csv", type=Path, help="write the matrix as CSV")
    parser.add_argument("--strict", action="store_true", help="fail when a requirement has no test")
    args = parser.parse_args(argv)
    try:
        register = load_register(args.register)
    except RegisterError as exc:
        print(f"Register error: {exc}", file=sys.stderr)
        return 1
    try:
        items = collect()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    matrix = build(register, items)
    text = to_markdown(matrix)
    for path, content in ((args.markdown, text), (args.json, json.dumps(matrix, indent=1) + "\n"), (args.csv, to_csv(matrix))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if not (args.markdown or args.json or args.csv):
        print(text)
    for rid in matrix["uncovered"]:
        print(f"warning: {rid} has no test", file=sys.stderr)
    for problem in matrix["problems"]:
        print(f"error: {problem}", file=sys.stderr)
    if matrix["problems"] or (args.strict and matrix["uncovered"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
