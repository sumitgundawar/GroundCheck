"""The evidence pack: one zip for a clinical safety officer.

It holds the safety case, the hazard log, the latest evaluation summaries, the
audit-chain verification result, the release history, the incident register, a
list of dependencies (and the SBOM and dependency audit when they have been
produced) and a snapshot of the configuration. Every file is listed in
manifest.json with its SHA-256, and manifest.sig holds the manifest's digest,
an HMAC under the audit signing key when one is configured.

Nothing secret goes in. The configuration snapshot is an allow-list of
settings that cannot hold a credential; for each credential it says only
whether one is set. Incident titles go through identifier removal and the free
text of incidents is left out. Building the pack twice from the same data and
the same time gives the same bytes."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import zipfile
from datetime import datetime, timezone

from . import __version__, config, db, governance, guards_input, incidents, integrity, releases
from .db import utcnow

ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SIG_CONTEXT = "groundcheck-evidence-pack"

# Settings safe to print. Anything not listed is not in the snapshot.
SAFE_SETTINGS = (
    "VERSION", "LLM_PROVIDER", "GEN_MODEL", "JUDGE_MODEL", "USE_LLM_JUDGE", "LLM_TIMEOUT_SECONDS",
    "FORCE_EXTRACTIVE", "AUTH_REQUIRED", "ADMIN_ACCESS", "ALLOW_GUARD_OVERRIDES", "LOGIN_MAX_FAILURES", "LOGIN_LOCKOUT_MINUTES", "SESSION_HOURS",
    "SESSION_COOKIE_SECURE", "EMBED_MODEL", "TOP_K", "VECTOR_STORE", "HYBRID_RETRIEVAL", "HYBRID_ALPHA",
    "RETRIEVAL_MIN_SCORE", "GROUNDING_MIN", "EXTRACT_MIN_SCORE", "MAX_QUERY_CHARS", "RATE_LIMIT_PER_MINUTE",
    "REVIEW_QUEUE", "REVIEW_SLA_HOURS", "FLAGGED_SLA_HOURS",
    "AUDIT_PERSIST", "AUDIT_RETENTION_DAYS", "REVIEW_RETENTION_DAYS", "RELEASE_CHECKS", "RELEASE_AUTO_PROMOTE",
    "RELEASES_KEEP", "INCLUDE_DEMO_CORPUS", "DEID_NER", "ANSWER_CACHE_SECONDS", "SMART_WRITE_NOTES",
    "CDS_HOOKS_ALLOW_UNSIGNED", "TELEMETRY",
)
# Reported under a name that does not look like a credential.
ALIASES = {"SIGN_IN_WITH_PASSPHRASE": "PASSWORD_SIGN_IN", "SIGN_IN_MIN_LENGTH": "PASSWORD_MIN_LENGTH"}
# Credentials: only whether each is set.
CREDENTIALS = ("LLM_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "DATA_ENCRYPTION_KEYS",
               "DATA_ENCRYPTION_RETIRED_KEYS", "AUDIT_SIGNING_KEYS", "OIDC_CLIENT_SECRET", "SMART_CLIENT_SECRET",
               "METRICS_TOKEN", "QDRANT_API_KEY", "DICOMWEB_AUTHORIZATION")
EVAL_FILES = ("eval_summary.json", "guard_summary.json", "verify_summary.json", "stress_summary.json")
OPTIONAL = {
    "sbom/sbom.json": ("sbom.json", "Produced in CI (cyclonedx-py); not present on this installation."),
    "dependencies/pip-audit.json": ("eval/security/pip-audit.json", "Produced in CI (pip-audit); not present on this installation."),
}


def config_snapshot() -> dict:
    values = {}
    for name in SAFE_SETTINGS:
        value = getattr(config, name, None)
        if name == "TELEMETRY":
            value = getattr(config, "TELEMETRY", "off")
        values[name] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    for label, attr in ALIASES.items():
        values[label] = getattr(config, attr)
    credentials = {name: bool(getattr(config, name, "")) for name in CREDENTIALS}
    return {"version": __version__, "settings": values, "credentials_set": credentials,
            "database": db.engine().dialect.name if db.ready() else None}


def _csv(rows: list[list]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def _json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _hazard_csv() -> str:
    rows = [["ID", "Hazard", "Cause", "Effect", "Severity", "Likelihood", "Initial risk", "Controls",
             "Residual severity", "Residual likelihood", "Residual risk", "Status", "Owner"]]
    for h in governance.list_hazards():
        rows.append([f"H{h['id']}", h["title"], h.get("cause", ""), h.get("effect", ""), h["severity"],
                     h["likelihood"], f"{h['initial_risk']['score']} ({h['initial_risk']['level']})", h["controls"],
                     h["residual_severity"], h["residual_likelihood"],
                     f"{h['residual_risk']['score']} ({h['residual_risk']['level']})", h["status"], h["owner"]])
    return _csv(rows)


def _incident_register() -> str:
    from sqlalchemy import select

    rows = [["Reference", "Title", "Concerns", "Harm", "Status", "Occurred", "Reported", "Closed", "Audit ID",
             "Review case", "Hazard"]]
    with db.session() as s:
        for i in s.scalars(select(db.Incident).order_by(db.Incident.reported_at, db.Incident.id)):
            rows.append([incidents.reference(i.id, i.reported_at), guards_input.deidentify(i.title).text,
                         incidents.CATEGORIES[i.category], incidents.HARM[i.harm], i.status,
                         i.occurred_at.isoformat() if i.occurred_at else "", i.reported_at.isoformat(),
                         i.closed_at.isoformat() if i.closed_at else "", i.audit_id or "",
                         i.review_case_id or "", i.hazard_id or ""])
    return _csv(rows)


def _release_history() -> str:
    out = []
    for r in releases.list_releases()["releases"]:
        out.append({k: r[k] for k in ("name", "status", "reason", "created_at", "created_by", "passages",
                                      "demo_passages", "content_sha256", "formulary_sha256", "live_at",
                                      "live_by") if k in r}
                   | {"documents": [{"key": d.get("key"), "title": d.get("title"), "version": d.get("version")}
                                    for d in r["documents"]],
                      "check": {k: v for k, v in (r.get("check") or {}).items() if k != "failures"}})
    return _json(out)


def build(now: datetime | None = None) -> bytes:
    """The pack, as zip bytes."""
    now = (now or utcnow()).astimezone(timezone.utc)
    files: dict[str, bytes] = {}
    missing: dict[str, str] = {}

    def add(path: str, text: str) -> None:
        files[path] = text.encode("utf-8")

    add("safety-case.md", governance.safety_case_markdown(30, None, now=now))
    add("hazard-log.csv", _hazard_csv())
    for name in EVAL_FILES:
        path = config.ROOT_DIR / "eval" / name
        if path.is_file():
            try:
                add(f"eval/{name}", _json(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                missing[f"eval/{name}"] = "Present but could not be read."
        else:
            missing[f"eval/{name}"] = "Not found on this installation."
    chain = integrity.verify()
    chain.pop("verified_at", None)
    add("audit-chain-verification.json", _json(chain))
    add("release-history.json", _release_history())
    add("incident-register.csv", _incident_register())
    requirements = config.ROOT_DIR / "requirements.txt"
    if requirements.is_file():
        add("dependencies/requirements.txt", requirements.read_text(encoding="utf-8"))
    for target, (source, why) in OPTIONAL.items():
        path = config.ROOT_DIR / source
        if path.is_file():
            files[target] = path.read_bytes()
        else:
            missing[target] = why
    add("config-snapshot.json", _json(config_snapshot()))

    manifest = {
        "title": "GroundCheckHealth evidence pack", "version": __version__,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": [{"path": p, "sha256": hashlib.sha256(files[p]).hexdigest(), "bytes": len(files[p])}
                  for p in sorted(files)],
        "not_included": dict(sorted(missing.items())),
        "note": "Supports, and does not replace, a clinical safety case signed by a qualified officer. "
                "No certification or clearance is held.",
    }
    manifest_bytes = _json(manifest).encode("utf-8")
    alg, digest = integrity.compute(SIG_CONTEXT, manifest_bytes)
    signature = _json({"alg": alg, "digest": digest, "covers": "manifest.json"}).encode("utf-8")
    files["manifest.json"], files["manifest.sig"] = manifest_bytes, signature

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(path, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[path])
    return buffer.getvalue()


def verify(data: bytes) -> dict:
    """Check a pack: every file against the manifest, and the manifest against
    its digest. Returns {"ok", "problems"}; an HMAC needs the signing key."""
    problems: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            manifest_bytes, signature = z.read("manifest.json"), json.loads(z.read("manifest.sig"))
            manifest = json.loads(manifest_bytes)
            listed = {f["path"]: f for f in manifest["files"]}
            for path, entry in listed.items():
                if path not in names:
                    problems.append(f"{path} is listed but missing")
                elif hashlib.sha256(z.read(path)).hexdigest() != entry["sha256"]:
                    problems.append(f"{path} has changed")
            for path in names - set(listed) - {"manifest.json", "manifest.sig"}:
                problems.append(f"{path} is not in the manifest")
    except (zipfile.BadZipFile, KeyError, ValueError):
        return {"ok": False, "problems": ["not a readable evidence pack"]}
    try:
        _, expected = integrity.compute(SIG_CONTEXT, manifest_bytes, signature.get("alg"))
        if not hmac.compare_digest(expected, signature.get("digest", "")):
            problems.append("the manifest does not match its digest")
    except KeyError:
        problems.append("the manifest was signed with a key this installation does not have")
    return {"ok": not problems, "problems": problems}
