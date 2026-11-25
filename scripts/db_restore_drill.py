#!/usr/bin/env python3
"""Backup and restore drill for the database (PostgreSQL or SQLite).

Creates the schema, seeds users, audit records and review cases, takes a
backup, wipes the database, restores it, and fails unless:

  - every table has the same number of rows as before,
  - the audit chain head (sequence number and hash) is the same,
  - `python -m app.cli verify-audit` says the audit trail is intact,
  - and, with --tamper-check, editing one restored audit record is caught and
    restoring again repairs it.

    DATABASE_URL=postgresql://user:pw@host/db python scripts/db_restore_drill.py
    python scripts/db_restore_drill.py                  a throwaway SQLite database

PostgreSQL needs pg_dump and psql on the PATH. The Compose-level drill,
scripts/backup_restore_drill.sh, covers the data volume too.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = False
QUESTIONS = [
    "What is the dose of Caloradine?", "What is the dose of Zalortin?", "How is Veltris syndrome treated?",
    "How do I treat a broken arm at home?", "What is the dose of Caloradine for children?",
    "How often are quorl levels monitored during conservative management?", "What is the cure for cancer?",
    "Can I combine Caloradine with Orrin-blockers?",
]


def libpq(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")


def is_postgres(url: str) -> bool:
    return url.startswith("postgresql")


def app_env(url: str) -> dict:
    env = {**os.environ, "DATABASE_URL": url, "FORCE_EXTRACTIVE": "true", "GROQ_API_KEY": "",
           "RATE_LIMIT_PER_MINUTE": "1000000", "AUDIT_PERSIST": "true", "REVIEW_QUEUE": "true",
           "ALERT_INTERVAL_SECONDS": "0", "RELEASE_CHECKS": "false"}
    env.setdefault("AUDIT_SIGNING_KEYS", "")
    return env


def run(cmd: list[str], env: dict, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, **kw)


def must(cmd: list[str], env: dict, what: str) -> str:
    out = run(cmd, env)
    if out.returncode != 0:
        sys.exit(f"FAILED: {what}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return out.stdout


# --- Inside the app's environment ---------------------------------------------

def _seed() -> None:
    from app import auth, db, integrity, pipeline

    db.migrate()
    auth.create_user("drill-admin@example.org", "Drill-Passw0rd-long-enough", role="admin", name="Drill Admin")
    auth.create_user("drill-clinician@example.org", "Drill-Passw0rd-long-enough", role="clinician", name="Drill Clinician")
    for _ in range(3):
        for q in QUESTIONS:
            pipeline.clear_answer_cache()
            pipeline.run(q, review=True)
    print(json.dumps({"head": integrity.head()["seq"]}))


def _counts() -> None:
    from sqlalchemy import inspect, text

    from app import db, integrity

    engine = db.engine()
    out = {}
    with engine.connect() as conn:
        for table in sorted(inspect(engine).get_table_names()):
            out[table] = conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar()
    head = integrity.head() if "audit_chain" in out else {"seq": None, "hash": None}
    print(json.dumps({"tables": out, "head": {"seq": head["seq"], "hash": head["hash"]}}))


def _tamper() -> None:
    from sqlalchemy import text

    from app import db

    with db.engine().begin() as conn:
        conn.execute(text("UPDATE audit_records SET query = query || ' (edited)' WHERE id = (SELECT min(id) FROM audit_records)"))


# --- The drill ------------------------------------------------------------------

def snapshot(url: str) -> dict:
    out = must([sys.executable, __file__, "_counts"], app_env(url), "counting rows")
    return json.loads(out.strip().splitlines()[-1])


def backup(url: str, target: Path) -> Path:
    if is_postgres(url):
        dump = target / "groundcheck.sql"
        must(["pg_dump", "--clean", "--if-exists", "--no-owner", "--no-privileges", "-f", str(dump), libpq(url)],
             os.environ, "pg_dump")
        return dump
    path = Path(url.removeprefix("sqlite:///"))
    dump = target / "groundcheck.sqlite"
    src, dst = sqlite3.connect(path), sqlite3.connect(dump)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return dump


def wipe(url: str) -> None:
    if is_postgres(url):
        must(["psql", "-v", "ON_ERROR_STOP=1", libpq(url), "-c", "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"],
             os.environ, "wiping the schema")
    else:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(url.removeprefix("sqlite:///") + suffix).unlink(missing_ok=True)


def restore(url: str, dump: Path) -> None:
    if is_postgres(url):
        must(["psql", "-v", "ON_ERROR_STOP=1", "-q", "-f", str(dump), libpq(url)], os.environ, "restoring with psql")
    else:
        shutil.copyfile(dump, url.removeprefix("sqlite:///"))


def verify_audit(url: str) -> tuple[int, str]:
    out = run([sys.executable, "-m", "app.cli", "verify-audit"], app_env(url))
    return out.returncode, out.stdout + out.stderr


def compare(before: dict, after: dict) -> list[str]:
    """What differs between two snapshots, as readable lines; empty when they match."""
    problems = []
    for table in sorted(set(before["tables"]) | set(after["tables"])):
        if before["tables"].get(table) != after["tables"].get(table):
            problems.append(f"{table}: {before['tables'].get(table)} before, {after['tables'].get(table)} after")
    if before["head"] != after["head"]:
        problems.append(f"audit chain head {before['head']} before, {after['head']} after")
    return problems


def drill(url: str, tamper_check: bool) -> int:
    env = app_env(url)
    existing = snapshot(url)
    if any(n for t, n in existing["tables"].items() if t != "alembic_version"):
        # Checked before anything else: nothing below may touch a database that has data.
        sys.exit("REFUSED: this database already has data, and the drill wipes it. Point DATABASE_URL at an empty one.")
    work = Path(tempfile.mkdtemp(prefix="gc-drill-"))
    try:
        print("== Creating the database and seeding it")
        must([sys.executable, "-m", "app.cli", "migrate"], env, "migrate")
        must([sys.executable, __file__, "_seed"], env, "seeding")
        before = snapshot(url)
        print("   rows:", {k: v for k, v in before["tables"].items() if v}, "chain head", before["head"]["seq"])
        if not before["tables"].get("audit_records") or not before["tables"].get("users"):
            sys.exit("FAILED: the seed produced no audit records or users")
        code, text = verify_audit(url)
        if code != 0:
            sys.exit(f"FAILED: the audit trail was not intact before the backup\n{text}")

        print("== Backing up")
        dump = backup(url, work)
        print("   ", dump.name, dump.stat().st_size, "bytes")

        print("== Wiping the database")
        wipe(url)
        emptied = snapshot(url)
        if any(emptied["tables"].values()):
            sys.exit("FAILED: the wipe left rows behind")

        print("== Restoring")
        restore(url, dump)
        after = snapshot(url)
        problems = compare(before, after)
        code, text = verify_audit(url)
        if code != 0 or "The audit trail is intact." not in text:
            problems.append("verify-audit did not pass after the restore:\n" + text)
        users = run([sys.executable, "-m", "app.cli", "list-users"], app_env(url))
        if users.returncode != 0 or "drill-admin@example.org" not in users.stdout:
            problems.append("the seeded accounts are missing after the restore")
        if problems:
            print("FAILED:\n  " + "\n  ".join(problems))
            return 1
        print("   counts match, chain head", after["head"]["seq"], "and verify-audit passes")

        if tamper_check:
            print("== Proving a changed record is caught, and that restoring repairs it")
            must([sys.executable, __file__, "_tamper"], env, "editing a record")
            code, text = verify_audit(url)
            if code == 0:
                print("FAILED: verify-audit did not notice an edited audit record")
                return 1
            wipe(url)
            restore(url, dump)
            code, text = verify_audit(url)
            if code != 0:
                print("FAILED: verify-audit still fails after restoring the backup\n" + text)
                return 1
            print("   edit detected, restore repaired it")
        print("DRILL PASSED")
        return 0
    finally:
        if not KEEP:
            wipe(url)
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("_seed", "_counts", "_tamper"):
        sys.path.insert(0, str(ROOT))
        globals()[sys.argv[1]]()
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", action="store_true", help="leave the restored database in place afterwards")
    parser.add_argument("--tamper-check", action="store_true", help="also prove an edited record is detected")
    args = parser.parse_args()
    global KEEP
    KEEP = args.keep
    url = os.environ.get("DATABASE_URL")
    cleanup = None
    if not url:
        cleanup = tempfile.mkdtemp(prefix="gc-drill-db-")
        url = f"sqlite:///{cleanup}/drill.db"
    try:
        if is_postgres(url) and not (shutil.which("pg_dump") and shutil.which("psql")):
            sys.exit("pg_dump and psql are needed for PostgreSQL")
        return drill(url, args.tamper_check)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
