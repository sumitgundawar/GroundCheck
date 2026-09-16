"""Test-wide setup. Runs before any test module imports the app, so the app
never touches the real database, audit log or local AI selection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="groundcheck-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp / 'test.db'}")
os.environ["AUDIT_LOG_PATH"] = str(_tmp / "audit.jsonl")
os.environ["LOCAL_AI_STATE_PATH"] = str(_tmp / "local_ai.json")
os.environ["SESSION_COOKIE_SECURE"] = "false"  # the test client talks plain HTTP
