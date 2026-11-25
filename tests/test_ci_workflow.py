"""The CI workflow runs the gates the platform promises."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
CI = yaml.safe_load(TEXT)


def commands(job: str) -> str:
    return "\n".join(str(step.get("run", "")) for step in CI["jobs"][job]["steps"])


def test_the_workflow_parses_and_has_its_jobs():
    assert {"test", "security", "restore-drill", "browser", "sdk-typescript"} <= set(CI["jobs"])
    triggers = CI.get("on", CI.get(True))
    assert "schedule" in triggers and "workflow_dispatch" in triggers and "pull_request" in triggers


def test_traceability_soup_and_controls_gates_are_in_ci():
    assert "python scripts/trace_matrix.py --strict --markdown build/trace.md" in commands("test")
    assert "python scripts/soup_list.py --sbom sbom.json --markdown soup.md" in commands("security")
    assert "python scripts/controls_report.py --check" in commands("test")


def test_the_soup_step_comes_after_the_sbom_it_reads():
    names = [s.get("name", "") for s in CI["jobs"]["security"]["steps"]]
    sbom = next(i for i, n in enumerate(names) if n.startswith("Bill of materials"))
    soup = next(i for i, n in enumerate(names) if n.startswith("Software of unknown provenance"))
    assert sbom < soup


def test_the_traceability_matrix_runs_after_the_tests_it_reads():
    names = [s.get("name", "") for s in CI["jobs"]["test"]["steps"]]
    tests = next(i for i, n in enumerate(names) if n.startswith("Run the test suite"))
    trace = next(i for i, n in enumerate(names) if n.startswith("Requirements traceability"))
    assert tests < trace


def test_the_restore_drill_runs_against_postgresql():
    job = CI["jobs"]["restore-drill"]
    assert job["services"]["postgres"]["image"].startswith("postgres:")
    assert job["env"]["DATABASE_URL"].startswith("postgresql+psycopg://")
    run = commands("restore-drill")
    assert "python scripts/db_restore_drill.py --tamper-check" in run
    assert "postgresql-client" in run
    assert "health-cmd" in job["services"]["postgres"]["options"]


def test_the_restore_drill_is_not_gated_to_a_branch():
    job = CI["jobs"]["restore-drill"]
    assert "if" not in job  # runs on every push and pull request, and on the weekly schedule


def test_the_tracing_tests_cannot_be_skipped_in_ci():
    test_job = CI["jobs"]["test"]
    assert "opentelemetry-sdk" in commands("test")
    pytest_step = next(s for s in test_job["steps"] if s.get("name", "").startswith("Run the test suite"))
    assert pytest_step["env"]["REQUIRE_OTEL"] == "1"


def test_language_and_string_checks_are_in_ci():
    run = commands("test")
    assert "python scripts/language_eval.py" in run
    assert "python scripts/i18n_extract.py --check" in run


def test_every_script_the_workflow_calls_exists():
    for path in set(re.findall(r"(?:python|bash)\s+(scripts/[\w./-]+)", TEXT)) | set(re.findall(r"run:\s+(scripts/[\w./-]+)", TEXT)):
        assert (ROOT / path).exists(), path


def test_generated_ci_outputs_are_ignored_by_git():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in ("build/", "sbom.json", "soup.md"):
        assert name in ignore.splitlines()
