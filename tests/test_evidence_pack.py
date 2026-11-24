"""The evidence pack: contents, a manifest that matches every file, a digest
that detects tampering, deterministic output, no secrets and no personal data,
and admin-only access."""

from __future__ import annotations

import conftest

import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, config, db, encryption, evidence, governance, incidents, integrity, pipeline, retrieval, sites  # noqa: E402

PASSWORD = "correct horse battery staple"
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
HAZARD = {"title": "Wrong dose cited", "cause": "Guard gap", "effect": "Harm", "severity": 4, "likelihood": 2,
          "controls": "Dosage guard", "residual_severity": 4, "residual_likelihood": 1, "status": "open", "owner": "CSO"}
REQUIRED = {"safety-case.md", "hazard-log.csv", "audit-chain-verification.json", "release-history.json",
            "incident-register.csv", "config-snapshot.json", "manifest.json", "manifest.sig",
            "dependencies/requirements.txt", "eval/eval_summary.json"}
SENSITIVE_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|AUTHORIZATION", re.I)


@pytest.fixture(autouse=True, scope="module")
def _index():
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()


@pytest.fixture()
def populated(database, monkeypatch):
    monkeypatch.setattr(config, "REVIEW_QUEUE", True)
    governance.save_hazard(HAZARD, None)
    pipeline.run("What is the standard dose of Caloradine?")
    pipeline.run("What is the standard dose of Zyntrafen?")
    pipeline.run("Tell me about Quibblezap77 dosing please")
    incidents.report({"title": "Mr John Smith NHS 943 476 5919 was given the wrong dose", "category": "answer",
                      "harm": "moderate", "description": "DESCRIPTION-SECRET Seen on ward 4.",
                      "root_cause": "ROOTCAUSE-SECRET"}, None, "Reporter Name")
    return database


def unpack(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n: z.read(n) for n in z.namelist()}


# --- Contents and manifest ------------------------------------------------------------

def test_the_pack_has_every_required_file(populated):
    files = unpack(evidence.build(NOW))
    assert REQUIRED <= set(files)
    assert "Wrong dose cited" in files["hazard-log.csv"].decode()
    rows = list(csv.reader(io.StringIO(files["hazard-log.csv"].decode())))
    assert rows[0][0] == "ID" and rows[1][1] == "Wrong dose cited" and rows[1][6] == "8 (medium)"
    assert "# GroundCheckHealth clinical safety case summary" in files["safety-case.md"].decode()
    assert json.loads(files["eval/eval_summary.json"])["total"] > 0


def test_the_manifest_hashes_every_file_exactly(populated):
    files = unpack(evidence.build(NOW))
    manifest = json.loads(files["manifest.json"])
    listed = {f["path"]: f for f in manifest["files"]}
    assert set(listed) == set(files) - {"manifest.json", "manifest.sig"}
    for path, entry in listed.items():
        assert entry["sha256"] == hashlib.sha256(files[path]).hexdigest(), path
        assert entry["bytes"] == len(files[path])
    assert manifest["generated_at"] == "2026-11-24T12:00:00Z" and manifest["version"]
    assert [f["path"] for f in manifest["files"]] == sorted(listed)


def test_what_was_not_produced_is_said_so_not_silently_dropped(populated):
    manifest = json.loads(unpack(evidence.build(NOW))["manifest.json"])
    assert "sbom/sbom.json" in manifest["not_included"] and "CI" in manifest["not_included"]["sbom/sbom.json"]
    assert "dependencies/pip-audit.json" in manifest["not_included"]


def test_optional_files_are_included_when_they_exist(populated, tmp_path, monkeypatch):
    (tmp_path / "eval" / "security").mkdir(parents=True)
    (tmp_path / "sbom.json").write_text('{"bomFormat": "CycloneDX"}')
    (tmp_path / "eval" / "security" / "pip-audit.json").write_text('{"dependencies": []}')
    (tmp_path / "requirements.txt").write_text("fastapi==1\n")
    (tmp_path / "eval" / "guard_summary.json").write_text('{"x": 1}')
    (tmp_path / "eval" / "verify_summary.json").write_text("not json")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    files = unpack(evidence.build(NOW))
    manifest = json.loads(files["manifest.json"])
    assert files["sbom/sbom.json"] == b'{"bomFormat": "CycloneDX"}' and "dependencies/pip-audit.json" in files
    assert "eval/guard_summary.json" in files and "eval/eval_summary.json" in manifest["not_included"]
    assert manifest["not_included"]["eval/verify_summary.json"].startswith("Present but")
    assert "sbom/sbom.json" not in manifest["not_included"]


def test_the_audit_chain_result_and_release_history_are_included(populated):
    files = unpack(evidence.build(NOW))
    chain = json.loads(files["audit-chain-verification.json"])
    assert chain["ok"] is True and chain["checked"] >= 2 and "verified_at" not in chain
    releases = json.loads(files["release-history.json"])
    assert isinstance(releases, list)


def test_a_broken_audit_chain_shows_in_the_pack(populated):
    from sqlalchemy import update

    with db.session() as s:
        s.execute(update(db.AuditRecord).where(db.AuditRecord.seq == 1).values(decision="refuse" if True else "answer",
                                                                              total_ms=999999))
    chain = json.loads(unpack(evidence.build(NOW))["audit-chain-verification.json"])
    assert chain["ok"] is False and chain["problem_count"] >= 1


def test_the_incident_register_has_references_and_harm(populated):
    rows = list(csv.reader(io.StringIO(unpack(evidence.build(NOW))["incident-register.csv"].decode())))
    assert rows[0][:4] == ["Reference", "Title", "Concerns", "Harm"]
    assert len(rows) == 2 and rows[1][0].startswith("INC-") and "Moderate" in rows[1][3]


# --- Determinism ----------------------------------------------------------------------------

def test_the_same_data_and_time_give_the_same_bytes(populated):
    assert evidence.build(NOW) == evidence.build(NOW)


def test_zip_entries_carry_no_clock_or_path_information(populated):
    with zipfile.ZipFile(io.BytesIO(evidence.build(NOW))) as z:
        for info in z.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert not info.filename.startswith("/") and ".." not in info.filename
        assert z.namelist() == sorted(z.namelist())


# --- Signing and tamper detection ---------------------------------------------------------

def test_an_unsigned_installation_gets_a_plain_digest_and_it_verifies(populated, monkeypatch):
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", "")
    data = evidence.build(NOW)
    sig = json.loads(unpack(data)["manifest.sig"])
    assert sig["alg"] == "sha256" and len(sig["digest"]) == 64
    assert evidence.verify(data) == {"ok": True, "problems": []}


def test_a_signed_installation_gets_an_hmac_and_verification_needs_the_key(populated, monkeypatch):
    key = encryption.generate_key()
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", key)
    data = evidence.build(NOW)
    sig = json.loads(unpack(data)["manifest.sig"])
    assert sig["alg"].startswith("hmac-sha256:")
    assert evidence.verify(data)["ok"] is True
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", encryption.generate_key())
    result = evidence.verify(data)
    assert result["ok"] is False and "key" in result["problems"][0]


def _rewrite(data: bytes, change) -> bytes:
    files = unpack(data)
    change(files)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return out.getvalue()


@pytest.mark.parametrize("name,change", [
    ("a file edited", lambda f: f.__setitem__("hazard-log.csv", f["hazard-log.csv"] + b"extra\n")),
    ("a file removed", lambda f: f.pop("safety-case.md")),
    ("a file added", lambda f: f.__setitem__("extra.txt", b"x")),
    ("manifest edited", lambda f: f.__setitem__("manifest.json", f["manifest.json"].replace(b"2026", b"2025"))),
    ("signature edited", lambda f: f.__setitem__("manifest.sig", f["manifest.sig"].replace(b'"digest": "', b'"digest": "0'))),
])
def test_tampering_is_detected(populated, name, change):
    data = evidence.build(NOW)
    assert evidence.verify(data)["ok"]
    assert evidence.verify(_rewrite(data, change))["ok"] is False


def test_garbage_is_not_a_valid_pack():
    assert evidence.verify(b"not a zip")["ok"] is False
    assert evidence.verify(_rewrite_empty())["ok"] is False


def _rewrite_empty():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("hello.txt", "x")
    return out.getvalue()


# --- No secrets, no personal data --------------------------------------------------------------

SECRETS = {
    "LLM_API_KEY": "sk-llm-AAAAAAAAAAAAAAAA", "ANTHROPIC_API_KEY": "sk-ant-BBBBBBBBBBBBBBBB",
    "AZURE_OPENAI_API_KEY": "az-CCCCCCCCCCCCCCCC", "OIDC_CLIENT_SECRET": "oidc-DDDDDDDDDDDDDDDD",
    "SMART_CLIENT_SECRET": "smart-EEEEEEEEEEEEEEEE", "METRICS_TOKEN": "metrics-FFFFFFFFFFFFFFFF",
    "QDRANT_API_KEY": "qd-GGGGGGGGGGGGGGGG", "DICOMWEB_AUTHORIZATION": "Basic HHHHHHHHHHHHHHHH",
}


def test_no_secret_value_appears_anywhere_in_the_pack(populated, monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setattr(config, name, value)
    enc, sign = encryption.generate_key(), encryption.generate_key()
    monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", enc)
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", sign)
    for name, value in {"MY_API_KEY": "env-key-IIIIIIIIIIII", "SOME_SECRET": "env-secret-JJJJJJJJJJ",
                        "DEPLOY_TOKEN": "env-token-KKKKKKKKKKK", "DB_PASSWORD": "env-pass-LLLLLLLLLLL"}.items():
        monkeypatch.setenv(name, value)
    encryption.reset()
    try:
        data = evidence.build(NOW)
    finally:
        monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", "")
        monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", "")
        encryption.reset()
    values = list(SECRETS.values()) + [enc, sign] + [v for k, v in os.environ.items()
                                                    if SENSITIVE_NAME.search(k) and len(v) >= 6]
    for name, content in unpack(data).items():
        text = content.decode("utf-8", "replace")
        for value in values:
            assert value not in text, f"{name} contains a secret value"
    snapshot = json.loads(unpack(data)["config-snapshot.json"])
    assert all(isinstance(v, bool) for v in snapshot["credentials_set"].values())
    assert snapshot["credentials_set"]["LLM_API_KEY"] is True and snapshot["credentials_set"]["AUDIT_SIGNING_KEYS"] is True


def test_the_config_snapshot_is_an_allow_list(populated):
    snapshot = json.loads(unpack(evidence.build(NOW))["config-snapshot.json"])
    assert set(snapshot["settings"]) == set(evidence.SAFE_SETTINGS) | set(evidence.ALIASES)
    for name in snapshot["settings"]:
        assert not SENSITIVE_NAME.search(name), name
    for name in snapshot["credentials_set"]:
        assert name in evidence.CREDENTIALS
    # Names of everything in config that smells like a credential are only ever reported as set or not.
    sensitive_in_config = {n for n in dir(config) if n.isupper() and SENSITIVE_NAME.search(n)}
    assert not (sensitive_in_config & set(evidence.SAFE_SETTINGS))
    assert "INSTANCE_NAME" not in snapshot["settings"] and "DATABASE_URL" not in snapshot["settings"]


def test_no_personal_data_in_the_pack(populated):
    files = unpack(evidence.build(NOW))
    everything = "\n".join(c.decode("utf-8", "replace") for c in files.values())
    for needle in ("John Smith", "943 476 5919", "9434765919", "DESCRIPTION-SECRET", "ROOTCAUSE-SECRET",
                   "Reporter Name", "Seen on ward"):
        assert needle not in everything, needle
    register = files["incident-register.csv"].decode()
    assert "wrong dose" in register and "[" in register      # the title is kept, identifiers replaced
    # The evaluation summaries quote synthetic test questions, so they are left out of the e-mail scan.
    own = "\n".join(c.decode("utf-8", "replace") for n, c in files.items() if not n.startswith("eval/"))
    assert not re.search(r"[\w.+-]+@[\w-]+\.\w+", own)
    assert "Quibblezap77" not in everything and "dosing please" not in everything   # no question text


def test_the_pack_does_not_contain_the_database_url_or_host(populated, monkeypatch):
    monkeypatch.setattr(config, "INSTANCE_NAME", "secret-host-name-xyz")
    monkeypatch.setattr(config, "DATABASE_URL", config.DATABASE_URL)
    everything = "\n".join(c.decode("utf-8", "replace") for c in unpack(evidence.build(NOW)).values())
    assert "secret-host-name-xyz" not in everything and "sqlite:///" not in everything
    assert str(config.ROOT_DIR) not in everything


# --- Access ---------------------------------------------------------------------------------------

@pytest.fixture()
def client(populated, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", True)
    north = sites.create("north", "North")
    for email, role, site in (("clin", "clinician", None), ("rev", "reviewer", None), ("admin", "admin", None),
                              ("north-admin", "admin", north["id"])):
        auth.create_user(f"{email}@example.org", PASSWORD, role=role, site_id=site)
    from app.main import app

    clients = {}

    def get(who):
        if who not in clients:
            c = TestClient(app)
            assert c.post("/api/auth/login", json={"email": f"{who}@example.org", "password": PASSWORD}).status_code == 200
            clients[who] = c
        return clients[who]
    return get


def test_only_a_group_wide_admin_can_download_it(client):
    from app.main import app

    with TestClient(app) as anonymous:
        assert anonymous.get("/api/governance/evidence-pack").status_code == 401
    for who in ("clin", "rev"):
        assert client(who).get("/api/governance/evidence-pack").status_code == 403
    assert client("north-admin").get("/api/governance/evidence-pack").status_code == 403
    r = client("admin").get("/api/governance/evidence-pack")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    assert evidence.verify(r.content)["ok"] is True
    assert REQUIRED <= set(unpack(r.content))


def test_without_accounts_it_is_local_only(populated, monkeypatch):
    monkeypatch.setattr(config, "AUTH_REQUIRED", False)
    from app.main import app

    with TestClient(app) as c:
        assert c.get("/api/governance/evidence-pack").status_code == 200
    with TestClient(app, client=conftest.REMOTE_PEER) as c:
        assert c.get("/api/governance/evidence-pack").status_code == 403
