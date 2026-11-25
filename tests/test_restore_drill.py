"""The backup and restore drill (scripts/db_restore_drill.py): it passes on a
healthy backup and fails, with a reason, on each way a restore can go wrong."""

from __future__ import annotations

import copy
import importlib.util
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("db_restore_drill", ROOT / "scripts" / "db_restore_drill.py")
drill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drill)

SNAPSHOT = {"tables": {"users": 2, "audit_records": 24, "review_cases": 4}, "head": {"seq": 24, "hash": "ab" * 32}}


APP_VARS = set(re.findall(r'_get(?:_float|_int)?\("([A-Z0-9_]+)"', (ROOT / "app" / "config.py").read_text(encoding="utf-8"))) | {
    "DATABASE_URL", "AUDIT_SIGNING_KEYS", "DATA_ENCRYPTION_KEYS", "AUDIT_LOG_PATH", "LOCAL_AI_STATE_PATH", "GROQ_API_KEY",
    "LLM_API_KEY"}


def clean_env(**extra: str) -> dict:
    """The environment without any setting an earlier test may have left behind."""
    return {**{k: v for k, v in os.environ.items() if k not in APP_VARS}, **extra}


@pytest.fixture(autouse=True)
def _no_leftover_settings(monkeypatch):
    for name in APP_VARS:
        monkeypatch.delenv(name, raising=False)


def run_drill(url: str | None, *args: str) -> subprocess.CompletedProcess:
    env = clean_env()
    if url:
        env["DATABASE_URL"] = url
    return subprocess.run([sys.executable, "scripts/db_restore_drill.py", *args], cwd=ROOT, env=env, text=True,
                          capture_output=True, timeout=300)


def test_a_healthy_backup_passes_on_sqlite_and_proves_tamper_detection():
    out = run_drill(None, "--tamper-check")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "counts match" in out.stdout and "edit detected" in out.stdout and "DRILL PASSED" in out.stdout


def test_matching_snapshots_have_no_problems():
    assert drill.compare(SNAPSHOT, copy.deepcopy(SNAPSHOT)) == []


def test_a_different_row_count_is_reported_by_table():
    after = copy.deepcopy(SNAPSHOT)
    after["tables"]["audit_records"] = 23
    assert drill.compare(SNAPSHOT, after) == ["audit_records: 24 before, 23 after"]


def test_a_missing_table_is_reported():
    after = copy.deepcopy(SNAPSHOT)
    del after["tables"]["review_cases"]
    assert drill.compare(SNAPSHOT, after) == ["review_cases: 4 before, None after"]


def test_a_different_chain_head_is_reported():
    after = copy.deepcopy(SNAPSHOT)
    after["head"]["hash"] = "cd" * 32
    problems = drill.compare(SNAPSHOT, after)
    assert len(problems) == 1 and "audit chain head" in problems[0]
    after = copy.deepcopy(SNAPSHOT)
    after["head"]["seq"] = 25
    assert "audit chain head" in drill.compare(SNAPSHOT, after)[0]


def test_an_extra_table_after_the_restore_is_reported():
    after = copy.deepcopy(SNAPSHOT)
    after["tables"]["sessions"] = 1
    assert drill.compare(SNAPSHOT, after) == ["sessions: None before, 1 after"]


def test_a_restore_from_a_truncated_backup_fails_the_drill(monkeypatch, tmp_path):
    """Corrupt the backup file between backup and restore."""
    real_backup = drill.backup

    def damaged(url, target):
        dump = real_backup(url, target)
        conn = sqlite3.connect(dump)
        conn.execute("DELETE FROM audit_records WHERE id > (SELECT min(id) + 5 FROM audit_records)")
        conn.commit()
        conn.close()
        return dump

    monkeypatch.setattr(drill, "backup", damaged)
    url = f"sqlite:///{tmp_path / 'drill.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    assert drill.drill(url, tamper_check=False) == 1


def test_a_restore_that_changes_one_audit_record_fails_the_drill(monkeypatch, tmp_path):
    real_backup = drill.backup

    def edited(url, target):
        dump = real_backup(url, target)
        conn = sqlite3.connect(dump)
        conn.execute("UPDATE audit_records SET query = query || ' x' WHERE id = (SELECT min(id) FROM audit_records)")
        conn.commit()
        conn.close()
        return dump

    monkeypatch.setattr(drill, "backup", edited)
    url = f"sqlite:///{tmp_path / 'drill.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    assert drill.drill(url, tamper_check=False) == 1


def test_the_urls_are_turned_into_what_pg_dump_takes():
    assert drill.libpq("postgresql+psycopg://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"
    assert drill.libpq("postgresql://u@h/db") == "postgresql://u@h/db"
    assert drill.is_postgres("postgresql+psycopg://u@h/db") and not drill.is_postgres("sqlite:///x.db")


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL") or not shutil.which("pg_dump"),
                    reason="needs TEST_POSTGRES_URL and the PostgreSQL client tools")
def test_the_drill_passes_on_postgresql():
    out = run_drill(os.environ["TEST_POSTGRES_URL"], "--tamper-check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_drill_refuses_a_database_that_has_data(tmp_path):
    url = f"sqlite:///{tmp_path / 'real.db'}"
    env = clean_env(DATABASE_URL=url, AUDIT_PERSIST="false")
    code = ("from app import auth, db; db.migrate(); "
            "auth.create_user('keep@example.org', 'Quartz-Anvil-Passw0rd-long', role='admin')")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True, capture_output=True)
    out = run_drill(url)
    assert out.returncode != 0 and "REFUSED" in (out.stdout + out.stderr)
    assert sqlite3.connect(tmp_path / "real.db").execute("SELECT count(*) FROM users").fetchone()[0] == 1


def test_the_drill_leaves_the_database_empty_afterwards(tmp_path):
    url = f"sqlite:///{tmp_path / 'drill.db'}"
    out = run_drill(url)
    assert out.returncode == 0, out.stdout + out.stderr
    assert not (tmp_path / "drill.db").exists()
    kept = run_drill(url, "--keep")
    assert kept.returncode == 0 and (tmp_path / "drill.db").exists()
