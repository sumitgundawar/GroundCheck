"""Knowledge packs: a portable, versioned, signed bundle of documents, an
optional formulary and an optional evaluation set.

A pack is a zip archive:

    manifest.json    name, version, publisher, licence, jurisdiction, dates,
                     and the SHA-256 and size of every other file
    signature.json   optional Ed25519 signature over the manifest bytes
    documents/...    the documents (anything app/documents.py can read)
    formulary.json   optional, in the app/formulary.py format
    eval.json        optional list of {"query", "expect": "answer"|"refuse"}

Because the manifest lists every file's hash, signing the manifest covers the
whole pack. Importing never extracts to disk (so a hostile file name can't
write anywhere); it reads every member into memory with size limits.

Import puts the documents in the normal pending-approval queue: nothing from a
pack is cited until a reviewer approves it. A pack's formulary is validated and
staged beside the pack's record, never activated. Its eval set is kept for the
release gate (`release_gate_cases`).

Trust: PACK_TRUSTED_KEYS lists the publishers whose signatures are accepted,
`publisher=base64-public-key[@YYYY-MM-DD]` separated by commas, the date being
the day the key stops being accepted. An unsigned pack, one signed by an unknown
or expired key, or one with a bad signature is refused unless an administrator
overrides it with a reason; the override is written to the audit trail. A pack
whose files don't match its own manifest is corrupt and is never imported."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import stat
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import config, documents, integrity

FORMAT = 1
MANIFEST = "manifest.json"
SIGNATURE = "signature.json"
SIGN_DOMAIN = b"groundcheck-pack-v1\n"

MAX_PACK_BYTES = 200 * 1024 * 1024
MAX_FILE_BYTES = documents.MAX_FILE_BYTES
MAX_TOTAL_BYTES = 400 * 1024 * 1024
MAX_FILES = 2000
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PATH_CHARS = 120

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_VERSION = re.compile(r"^\d{1,6}\.\d{1,6}\.\d{1,6}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_DRIVE = re.compile(r"^[A-Za-z]:")


class PackError(ValueError):
    """The pack can't be built or imported. The message is safe to show."""


class PackVerificationError(PackError):
    """The pack is unsigned, untrusted, or doesn't match its manifest."""

    def __init__(self, message: str, verification: "Verification | None" = None):
        super().__init__(message)
        self.verification = verification


# --- Manifest --------------------------------------------------------------

class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str
    size: int = Field(gt=0)
    kind: Literal["document", "formulary", "eval"]
    title: str = Field(default="", max_length=300)

    @field_validator("sha256")
    @classmethod
    def _hash(cls, v: str) -> str:
        if not _SHA.match(v):
            raise ValueError("must be 64 lower-case hex characters")
        return v


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal[1]
    name: str
    version: str
    publisher: str = Field(min_length=1, max_length=120)
    licence: str = Field(min_length=1, max_length=200)
    jurisdiction: str = Field(min_length=2, max_length=32)
    description: str = Field(default="", max_length=2000)
    effective_from: date | None = None
    expires_on: date | None = None
    synthetic: bool = False
    files: list[FileEntry] = Field(max_length=MAX_FILES)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError("use 2 to 64 lower-case letters, digits, '.', '_' or '-', starting with a letter or digit")
        return v

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not _VERSION.match(v):
            raise ValueError("must look like 1.2.3")
        return v


def version_tuple(version: str) -> tuple[int, int, int]:
    if not _VERSION.match(version):
        raise PackError(f"{version!r} is not a version like 1.2.3.")
    a, b, c = (int(x) for x in version.split("."))
    return a, b, c


def safe_member_name(name: str) -> str:
    """Raise PackError unless `name` is a plain relative path inside the pack."""
    if not name or len(name) > 255:
        raise PackError("An archive member has an empty or overlong name.")
    if "\x00" in name or "\\" in name or name.startswith("/") or _DRIVE.match(name):
        raise PackError(f"Unsafe path in the archive: {name!r}.")
    if any(unicodedata.category(c).startswith("C") for c in name):
        raise PackError(f"Unsafe characters in the archive path {name!r}.")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise PackError(f"Unsafe path in the archive: {name!r}.")
    return name


def _check_entry(entry: FileEntry) -> None:
    safe_member_name(entry.path)
    if len(entry.path) > MAX_PATH_CHARS:
        raise PackError(f"The path {entry.path!r} is too long.")
    if entry.size > MAX_FILE_BYTES:
        raise PackError(f"{entry.path} is larger than the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.")
    if entry.kind == "document":
        suffix = PurePosixPath(entry.path).suffix.lower()
        if not entry.path.startswith("documents/") or suffix not in documents.MEDIA_TYPES:
            raise PackError(f"{entry.path} is not a readable document under documents/.")
    elif entry.kind == "formulary" and entry.path != "formulary.json":
        raise PackError("The formulary must be formulary.json.")
    elif entry.kind == "eval" and entry.path != "eval.json":
        raise PackError("The evaluation set must be eval.json.")


def parse_manifest(raw: bytes) -> Manifest:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise PackError("The manifest is too large.")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PackError(f"The manifest isn't valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PackError("The manifest must be a JSON object.")
    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise PackError(f"The manifest is invalid at {where}: {first['msg']}") from None
    seen: set[str] = set()
    for entry in manifest.files:
        _check_entry(entry)
        if entry.path in seen:
            raise PackError(f"The manifest lists {entry.path} twice.")
        seen.add(entry.path)
    if manifest.effective_from and manifest.expires_on and manifest.expires_on <= manifest.effective_from:
        raise PackError("The pack's expiry date must be after its effective date.")
    return manifest


# --- Keys and trust --------------------------------------------------------

def encode_key(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def decode_key(text: str) -> bytes:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise PackError("A key must be base64 text of 32 bytes.") from None
    if len(raw) != 32:
        raise PackError("A key must be base64 text of 32 bytes.")
    return raw


def generate_keypair() -> tuple[str, str]:
    """Returns (private, public) as base64 text."""
    private, public = integrity.ed25519_generate()
    return encode_key(private), encode_key(public)


@dataclass(frozen=True)
class TrustedKey:
    publisher: str
    public: bytes
    expires_on: date | None = None

    @property
    def key_id(self) -> str:
        return integrity.key_fingerprint(self.public)


def trusted_keys(spec: str | None = None) -> dict[str, TrustedKey]:
    """Keys from PACK_TRUSTED_KEYS (or `spec`), by key id. A malformed entry is
    an error rather than being skipped, so a typo can't quietly drop a key, or
    worse, leave an operator believing a publisher is trusted."""
    text = os.environ.get("PACK_TRUSTED_KEYS", "") if spec is None else spec
    keys: dict[str, TrustedKey] = {}
    for item in (t.strip() for t in text.split(",")):
        if not item:
            continue
        publisher, sep, rest = item.partition("=")
        if not sep or not publisher.strip() or not rest.strip():
            raise PackError("PACK_TRUSTED_KEYS entries look like publisher=base64key[@YYYY-MM-DD].")
        key_text, _, expiry = rest.partition("@")
        expires = None
        if expiry:
            try:
                expires = date.fromisoformat(expiry.strip())
            except ValueError:
                raise PackError(f"The expiry {expiry!r} for {publisher.strip()} isn't a date like 2027-01-31.") from None
        key = TrustedKey(publisher.strip(), decode_key(key_text), expires)
        keys[key.key_id] = key
    return keys


def _today(now: datetime | None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()


# --- Archive ---------------------------------------------------------------

@dataclass
class Archive:
    manifest_bytes: bytes
    signature_bytes: bytes | None
    files: dict[str, bytes]
    manifest: Manifest = None  # type: ignore[assignment]


def read_archive(source: bytes | Path | str) -> Archive:
    """Read a pack into memory, enforcing limits. Nothing is written to disk."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            if path.stat().st_size > MAX_PACK_BYTES:
                raise PackError("The pack file is too large.")
            data = path.read_bytes()
        except OSError as exc:
            raise PackError(f"Couldn't read the pack: {exc.strerror or exc}") from None
    else:
        data = source
    if len(data) > MAX_PACK_BYTES:
        raise PackError("The pack file is too large.")
    if not data:
        raise PackError("The pack file is empty.")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise PackError("This isn't a knowledge pack (not a readable zip archive).") from None

    infos = zf.infolist()
    if len(infos) > MAX_FILES + 2:
        raise PackError("The pack contains too many files.")
    seen: set[str] = set()
    members: dict[str, bytes] = {}
    total = 0
    with zf:
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                safe_member_name(name.rstrip("/"))
                continue
            safe_member_name(name)
            if name in seen:
                raise PackError(f"The archive contains {name!r} twice.")
            seen.add(name)
            if info.flag_bits & 0x1:
                raise PackError("Encrypted archives aren't supported.")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise PackError(f"The archive contains a link ({name!r}).")
            limit = MAX_MANIFEST_BYTES if name in (MANIFEST, SIGNATURE) else MAX_FILE_BYTES
            if info.file_size > limit:
                raise PackError(f"{name} is larger than allowed.")
            try:
                with zf.open(info) as fh:
                    body = fh.read(limit + 1)
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError, ValueError,
                    zlib.error) as exc:
                raise PackError(f"Couldn't read {name!r} from the archive: {exc}") from None
            if len(body) > limit:
                raise PackError(f"{name} is larger than allowed.")
            total += len(body)
            if total > MAX_TOTAL_BYTES:
                raise PackError("The pack expands to too much data.")
            members[name] = body
    if MANIFEST not in members:
        raise PackError("The pack has no manifest.json.")
    archive = Archive(members.pop(MANIFEST), members.pop(SIGNATURE, None), members)
    archive.manifest = parse_manifest(archive.manifest_bytes)
    return archive


# --- Verification ----------------------------------------------------------

SignatureStatus = Literal["verified", "unsigned", "untrusted", "expired_key", "invalid"]


@dataclass
class Verification:
    integrity: bool
    problems: list[str] = field(default_factory=list)
    signature: SignatureStatus = "unsigned"
    key_id: str | None = None
    signer: str | None = None
    detail: str = ""

    @property
    def trusted(self) -> bool:
        return self.integrity and self.signature == "verified"

    def as_dict(self) -> dict:
        return {"integrity": self.integrity, "problems": self.problems, "signature": self.signature,
                "key_id": self.key_id, "signer": self.signer, "detail": self.detail, "trusted": self.trusted}


def verify(archive: Archive, keys: dict[str, TrustedKey] | None = None, now: datetime | None = None) -> Verification:
    keys = trusted_keys() if keys is None else keys
    result = Verification(integrity=True)
    listed = {e.path: e for e in archive.manifest.files}
    for path in sorted(set(archive.files) - set(listed)):
        result.problems.append(f"{path} is in the archive but not in the manifest")
    for path, entry in listed.items():
        body = archive.files.get(path)
        if body is None:
            result.problems.append(f"{path} is listed in the manifest but missing")
        elif len(body) != entry.size or hashlib.sha256(body).hexdigest() != entry.sha256:
            result.problems.append(f"{path} doesn't match its hash in the manifest")
    result.integrity = not result.problems

    if archive.signature_bytes is None:
        result.detail = "The pack is not signed."
        return result
    try:
        sig = json.loads(archive.signature_bytes.decode("utf-8"))
        if not (isinstance(sig, dict) and sig.get("format") == FORMAT and sig.get("alg") == "ed25519"
                and isinstance(sig.get("key_id"), str) and isinstance(sig.get("signature"), str)):
            raise ValueError
        signature = base64.b64decode(sig["signature"], validate=True)
    except (UnicodeDecodeError, ValueError, binascii.Error, RecursionError):
        result.signature, result.detail = "invalid", "The signature file is malformed."
        return result
    result.key_id = sig["key_id"]
    key = keys.get(sig["key_id"])
    if key is None:
        result.signature = "untrusted"
        result.detail = f"The pack is signed with key {sig['key_id']}, which isn't in PACK_TRUSTED_KEYS."
        return result
    result.signer = key.publisher
    if not integrity.ed25519_verify(key.public, signature, SIGN_DOMAIN + archive.manifest_bytes):
        result.signature, result.detail = "invalid", "The signature doesn't match the manifest."
        return result
    if key.expires_on is not None and _today(now) >= key.expires_on:
        result.signature = "expired_key"
        result.detail = f"The signing key for {key.publisher} expired on {key.expires_on}."
        return result
    if key.publisher != archive.manifest.publisher:
        result.signature = "invalid"
        result.detail = (f"The key belongs to {key.publisher}, but the pack names "
                         f"{archive.manifest.publisher} as its publisher.")
        return result
    result.signature, result.detail = "verified", f"Signed by {key.publisher}."
    return result


# --- Build -----------------------------------------------------------------

_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, _ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name])
    return buf.getvalue()


def validate_eval(raw: bytes) -> list[dict]:
    try:
        cases = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackError(f"eval.json isn't valid JSON: {exc}") from None
    if not isinstance(cases, list) or not cases:
        raise PackError("eval.json must be a non-empty list of cases.")
    for i, case in enumerate(cases):
        if (not isinstance(case, dict) or not isinstance(case.get("query"), str) or not case["query"].strip()
                or case.get("expect") not in ("answer", "refuse") or set(case) - {"query", "expect", "patient"}):
            raise PackError(f"eval.json case {i + 1} needs a query and expect of 'answer' or 'refuse'.")
    return cases


def validate_formulary(raw: bytes):
    from . import formulary

    try:
        return formulary.Formulary.model_validate_json(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise PackError(f"formulary.json is invalid at {'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from None


def assemble(meta: dict, files: dict[str, bytes], titles: dict[str, str] | None = None,
             private_key: bytes | None = None) -> bytes:
    """Build a pack from its metadata and files (path to bytes), validating
    everything an importer would. Returns the archive bytes."""
    titles = titles or {}
    entries = []
    for path in sorted(files):
        body = files[path]
        safe_member_name(path)
        if not body:
            raise PackError(f"{path} is empty.")
        if path == "formulary.json":
            kind = "formulary"
        elif path == "eval.json":
            kind = "eval"
        elif path.startswith("documents/"):
            kind = "document"
        else:
            raise PackError(f"{path} isn't allowed: put documents under documents/.")
        entries.append({"path": path, "sha256": hashlib.sha256(body).hexdigest(), "size": len(body), "kind": kind,
                        **({"title": titles[path]} if path in titles else {})})
    manifest_data = {"format": FORMAT, **meta, "files": entries}
    manifest_bytes = json.dumps(manifest_data, indent=2, sort_keys=True, ensure_ascii=False,
                                default=str).encode("utf-8")
    manifest = parse_manifest(manifest_bytes)
    if not any(e.kind == "document" for e in manifest.files):
        raise PackError("A pack needs at least one document.")
    for entry in manifest.files:
        if entry.kind == "document":
            try:
                documents.parse(entry.path, files[entry.path])
            except documents.DocumentError as exc:
                raise PackError(f"{entry.path}: {exc}") from None
    if "formulary.json" in files:
        validate_formulary(files["formulary.json"])
    if "eval.json" in files:
        validate_eval(files["eval.json"])
    members = dict(files)
    members[MANIFEST] = manifest_bytes
    if private_key is not None:
        public = integrity.ed25519_public(private_key)
        members[SIGNATURE] = json.dumps({
            "format": FORMAT, "alg": "ed25519", "key_id": integrity.key_fingerprint(public),
            "signature": encode_key(integrity.ed25519_sign(private_key, SIGN_DOMAIN + manifest_bytes)),
        }, indent=2).encode("utf-8")
    return _zip_bytes(members)


META_FIELDS = ("name", "version", "publisher", "licence", "jurisdiction", "description", "effective_from",
               "expires_on", "synthetic")


def build_pack(src_dir: Path, out_path: Path, private_key: bytes | None = None) -> dict:
    """Build a pack from a folder holding pack.json (the metadata, plus an optional
    "titles" object mapping document paths to titles), documents/, and optionally
    formulary.json or formulary.csv, and eval.json."""
    src = Path(src_dir)
    try:
        meta_raw = json.loads((src / "pack.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PackError(f"Couldn't read pack.json in {src}: {exc}") from None
    if not isinstance(meta_raw, dict):
        raise PackError("pack.json must be a JSON object.")
    unknown = set(meta_raw) - set(META_FIELDS) - {"titles"}
    if unknown:
        raise PackError(f"pack.json has unknown fields: {', '.join(sorted(unknown))}.")
    titles = meta_raw.pop("titles", {}) or {}
    files: dict[str, bytes] = {}
    docs = src / "documents"
    if docs.is_dir():
        for p in sorted(docs.rglob("*")):
            if p.is_symlink():
                raise PackError(f"{p.relative_to(src)} is a link, which a pack can't contain.")
            if p.is_file():
                files[p.relative_to(src).as_posix()] = p.read_bytes()
    if (src / "formulary.json").exists() and (src / "formulary.csv").exists():
        raise PackError("Provide formulary.json or formulary.csv, not both.")
    if (src / "formulary.csv").exists():
        from . import formulary

        result = formulary.get_adapter("csv").load(
            src / "formulary.csv", formulary_name=str(meta_raw.get("name", "formulary")),
            version=str(meta_raw.get("version", "0.0.0")), synthetic=bool(meta_raw.get("synthetic", False)))
        if not result.ok:
            raise PackError("formulary.csv has errors: " + "; ".join(str(e) for e in result.errors[:5]))
        files["formulary.json"] = result.formulary.model_dump_json(indent=2).encode("utf-8")
    elif (src / "formulary.json").exists():
        files["formulary.json"] = (src / "formulary.json").read_bytes()
    if (src / "eval.json").exists():
        files["eval.json"] = (src / "eval.json").read_bytes()
    data = assemble(meta_raw, files, titles, private_key)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    manifest = parse_manifest(read_archive(data).manifest_bytes)
    return {"path": str(out), "name": manifest.name, "version": manifest.version, "files": len(manifest.files),
            "signed": private_key is not None, "sha256": hashlib.sha256(data).hexdigest()}


# --- Inspect ---------------------------------------------------------------

def inspect_pack(source: bytes | Path | str, keys: dict[str, TrustedKey] | None = None,
                 now: datetime | None = None) -> dict:
    archive = read_archive(source)
    verification = verify(archive, keys, now)
    m = archive.manifest
    return {
        "name": m.name, "version": m.version, "publisher": m.publisher, "licence": m.licence,
        "jurisdiction": m.jurisdiction, "description": m.description, "synthetic": m.synthetic,
        "effective_from": m.effective_from.isoformat() if m.effective_from else None,
        "expires_on": m.expires_on.isoformat() if m.expires_on else None,
        "files": [{"path": e.path, "kind": e.kind, "size": e.size, "sha256": e.sha256} for e in m.files],
        "verification": verification.as_dict(),
    }


# --- Import ----------------------------------------------------------------

def store_dir() -> Path:
    return Path(os.environ.get("PACK_STORE_DIR") or config.ROOT_DIR / "data" / "packs")


def installed(root: Path | None = None) -> list[dict]:
    """Imported packs, from the records written beside their staged files."""
    root = Path(root) if root else store_dir()
    out = []
    for record in sorted(root.glob("*/*/pack.json")) if root.is_dir() else []:
        try:
            out.append(json.loads(record.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda r: (r["name"], version_tuple(r["version"])))


def _staged_path(root: Path, name: str, version: str, filename: str) -> Path:
    return root / name / version / filename


def release_gate_cases(root: Path | None = None) -> list[dict]:
    """Every imported pack's evaluation cases, for the release gate."""
    cases = []
    root = Path(root) if root else store_dir()
    for record in installed(root):
        path = _staged_path(root, record["name"], record["version"], "eval.json")
        if path.exists():
            for case in validate_eval(path.read_bytes()):
                cases.append({**case, "pack": f"{record['name']}@{record['version']}"})
    return cases


def run_eval(cases: list[dict]) -> dict:
    """Run evaluation cases through the pipeline. A must-refuse case that is
    answered is unsafe; the run fails on any unsafe answer or wrong decision."""
    from . import pipeline
    from .schemas import PatientContext

    rows = []
    for case in cases:
        patient = PatientContext(**case["patient"]) if case.get("patient") else None
        response = pipeline.run(case["query"], patient=patient, review=False, client_id="pack-eval")
        rows.append({**case, "got": response.decision, "ok": response.decision == case["expect"]})
    unsafe = sum(r["expect"] == "refuse" and r["got"] == "answer" for r in rows)
    return {"total": len(rows), "passed": sum(r["ok"] for r in rows), "unsafe_answers": unsafe,
            "ok": all(r["ok"] for r in rows), "failures": [r for r in rows if not r["ok"]][:20]}


def _midnight(d: date | None) -> datetime | None:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc) if d else None


def import_pack(source: bytes | Path | str, *, keys: dict[str, TrustedKey] | None = None,
                now: datetime | None = None, override: bool = False, override_reason: str = "",
                allow_downgrade: bool = False, user_id: int | None = None, root: Path | None = None) -> dict:
    """Verify a pack and add its documents as pending versions."""
    from . import knowledge

    archive = read_archive(source)
    verification = verify(archive, keys, now)
    manifest = archive.manifest
    if not verification.integrity:
        raise PackVerificationError("The pack doesn't match its manifest and can't be imported: "
                                    + "; ".join(verification.problems[:5]), verification)
    needs_override = verification.signature != "verified"
    if needs_override:
        if not override:
            raise PackVerificationError(
                f"This pack can't be trusted ({verification.signature}): {verification.detail} "
                f"An administrator can override this with a reason.", verification)
        if len(override_reason.strip()) < 5:
            raise PackError("An override needs a reason of at least a few words.")
    if manifest.expires_on and manifest.expires_on <= _today(now):
        raise PackError(f"The pack expired on {manifest.expires_on}.")

    root = Path(root) if root else store_dir()
    record_dir = root / manifest.name / manifest.version
    manifest_sha = hashlib.sha256(archive.manifest_bytes).hexdigest()
    previous = installed(root)
    same = [r for r in previous if r["name"] == manifest.name]
    for r in same:
        if r["version"] == manifest.version:
            if r["manifest_sha256"] != manifest_sha:
                raise PackError(f"Version {manifest.version} of {manifest.name} was already imported with "
                                f"different content. Publish changes under a new version.")
            return {"status": "already_imported", "name": manifest.name, "version": manifest.version,
                    "verification": verification.as_dict(), "documents": [], "skipped": [],
                    "override_used": False, "audit_id": None}
    if same and version_tuple(manifest.version) < max(version_tuple(r["version"]) for r in same):
        newest = max(same, key=lambda r: version_tuple(r["version"]))["version"]
        if not allow_downgrade:
            raise PackError(f"Version {newest} of {manifest.name} is already imported; "
                            f"importing the older {manifest.version} needs allow_downgrade.")

    parsed: list[tuple[FileEntry, str]] = []
    keys_seen: dict[str, str] = {}
    for entry in manifest.files:
        if entry.kind != "document":
            continue
        try:
            doc = documents.parse(entry.path, archive.files[entry.path])
        except documents.DocumentError as exc:
            raise PackError(f"{entry.path}: {exc}") from None
        title = (entry.title or doc.title).strip()[:300] or doc.title
        key = knowledge.document_key(title)
        if key in keys_seen:
            raise PackError(f"{entry.path} and {keys_seen[key]} have the same title ({title!r}).")
        keys_seen[key] = entry.path
        parsed.append((entry, title))
    if "formulary.json" in archive.files:
        validate_formulary(archive.files["formulary.json"])
    if "eval.json" in archive.files:
        validate_eval(archive.files["eval.json"])

    audit_id = None
    if needs_override:
        audit_id = integrity.record_event("pack_import_override", {
            "pack": manifest.name, "version": manifest.version, "publisher": manifest.publisher,
            "manifest_sha256": manifest_sha, "signature": verification.signature,
            "detail": verification.detail, "reason": override_reason.strip()[:500]}, user_id)

    created: list[dict] = []
    skipped: list[str] = []
    effective, expires = _midnight(manifest.effective_from), _midnight(manifest.expires_on)
    try:
        for entry, title in parsed:
            filename = f"pack:{manifest.name}@{manifest.version}/{entry.path}"
            try:
                created.append(knowledge.import_document(
                    filename, archive.files[entry.path], title=title, owner=manifest.publisher,
                    effective_from=effective, expires_on=expires, uploaded_by=user_id))
            except knowledge.KnowledgeError as exc:
                if "already version" not in str(exc):
                    raise
                skipped.append(entry.path)
    except Exception as exc:
        for doc in created:
            try:
                knowledge.review(doc["id"], "reject", None, note="Pack import failed part way; rolled back.")
            except knowledge.KnowledgeError:
                pass
        raise PackError(f"Importing {manifest.name} {manifest.version} failed and was rolled back: {exc}") from exc

    record_dir.mkdir(parents=True, exist_ok=True)
    for name in ("formulary.json", "eval.json"):
        if name in archive.files:
            (record_dir / name).write_bytes(archive.files[name])
    record = {"name": manifest.name, "version": manifest.version, "publisher": manifest.publisher,
              "licence": manifest.licence, "jurisdiction": manifest.jurisdiction,
              "manifest_sha256": manifest_sha, "signature": verification.signature, "key_id": verification.key_id,
              "override": needs_override, "documents": [d["id"] for d in created], "skipped": skipped,
              "has_formulary": "formulary.json" in archive.files, "has_eval": "eval.json" in archive.files,
              "imported_at": datetime.now(timezone.utc).isoformat()}
    (record_dir / "pack.json.tmp").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (record_dir / "pack.json.tmp").replace(record_dir / "pack.json")
    if not needs_override:
        integrity.record_event("pack_import", {"pack": manifest.name, "version": manifest.version,
                                               "publisher": manifest.publisher, "manifest_sha256": manifest_sha,
                                               "signer": verification.signer}, user_id)
    return {"status": "imported", "name": manifest.name, "version": manifest.version,
            "verification": verification.as_dict(), "documents": created, "skipped": skipped,
            "formulary_path": str(record_dir / "formulary.json") if "formulary.json" in archive.files else None,
            "eval_path": str(record_dir / "eval.json") if "eval.json" in archive.files else None,
            "override_used": needs_override, "audit_id": audit_id}
