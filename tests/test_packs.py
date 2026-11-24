"""Knowledge packs: build, inspect, import, signing, tampering, hostile
archives, versioning, and that imported content flows through retrieval and the
guards like any other approved document."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import warnings
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import cli, config, db, integrity, knowledge, packs, pipeline, retrieval  # noqa: E402

FALLS = b"""<!doctype html><html><head><title>Falls Prevention Policy</title></head><body>
<h1>Falls Prevention Policy</h1><p>This fictional policy is for import testing.</p>
<h2>Risk assessment</h2><p>Complete a falls risk assessment within 6 hours of admission.</p>
<h2>After a fall</h2><p>Record the fall and review medication within 24 hours.</p></body></html>"""
TRANSFUSION = b"""# Transfusion Observation Policy

## Before starting

Confirm patient identity at the bedside with two staff members.

## During the transfusion

Record temperature and pulse every 15 minutes during the first hour.
"""
EVAL = json.dumps([
    {"query": "How soon should a falls risk assessment be completed after admission?", "expect": "answer"},
    {"query": "How soon should a Quelvarin risk assessment be completed after admission?", "expect": "refuse"},
]).encode()
META = {"name": "demo-policies", "version": "1.0.0", "publisher": "Acme Health", "licence": "CC0-1.0",
        "jurisdiction": "US", "description": "Synthetic policies for tests.", "synthetic": True}
FILES = {"documents/falls.html": FALLS, "documents/transfusion.md": TRANSFUSION, "eval.json": EVAL}
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def keypair():
    private, public = packs.generate_keypair()
    return packs.decode_key(private), public


def make(meta=None, files=None, key=None, titles=None) -> bytes:
    return packs.assemble({**META, **(meta or {})}, dict(files or FILES), titles, key)


def raw_pack(manifest: dict | bytes, files: dict[str, bytes] | None = None, signature: bytes | None = None,
             compress: int = zipfile.ZIP_STORED) -> bytes:
    """An archive written without any of the builder's checks."""
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(buf, "w", compress) as zf:
            zf.writestr("manifest.json", manifest if isinstance(manifest, bytes) else json.dumps(manifest))
            if signature is not None:
                zf.writestr("signature.json", signature)
            for name, body in (files or {}).items():
                zf.writestr(name, body)
    return buf.getvalue()


def manifest_for(files: dict[str, bytes], **over) -> dict:
    entries = []
    for path, body in files.items():
        kind = "eval" if path == "eval.json" else "formulary" if path == "formulary.json" else "document"
        entries.append({"path": path, "sha256": hashlib.sha256(body).hexdigest(), "size": len(body), "kind": kind})
    return {"format": 1, **META, "files": entries, **over}


def trusted(public: str, publisher="Acme Health", expires: str = "") -> dict:
    return packs.trusted_keys(f"{publisher}={public}" + (f"@{expires}" if expires else ""))


def verify_bytes(data: bytes, keys) -> packs.Verification:
    return packs.verify(packs.read_archive(data), keys, NOW)


# --- Build, inspect, determinism -------------------------------------------------------

def write_source(tmp_path: Path, **extra) -> Path:
    src = tmp_path / "src"
    (src / "documents").mkdir(parents=True)
    (src / "pack.json").write_text(json.dumps({**META, **extra}))
    (src / "documents" / "falls.html").write_bytes(FALLS)
    (src / "documents" / "transfusion.md").write_bytes(TRANSFUSION)
    (src / "eval.json").write_bytes(EVAL)
    return src


def test_build_is_deterministic_and_inspectable(tmp_path):
    src = write_source(tmp_path)
    private, public = keypair()
    a = packs.build_pack(src, tmp_path / "a.gcpack", private)
    b = packs.build_pack(src, tmp_path / "b.gcpack", private)
    assert a["sha256"] == b["sha256"] and a["files"] == 3 and a["signed"]
    info = packs.inspect_pack(tmp_path / "a.gcpack", trusted(public), NOW)
    assert (info["name"], info["version"], info["publisher"], info["licence"], info["jurisdiction"]) == (
        "demo-policies", "1.0.0", "Acme Health", "CC0-1.0", "US")
    assert info["verification"]["trusted"] and info["verification"]["signer"] == "Acme Health"
    assert [f["kind"] for f in info["files"]] == ["document", "document", "eval"]


def test_inspect_of_an_unsigned_pack_reports_it_without_raising(tmp_path):
    info = packs.inspect_pack(make(), trusted(keypair()[1]), NOW)
    assert info["verification"]["signature"] == "unsigned" and not info["verification"]["trusted"]


def test_build_from_a_csv_formulary(tmp_path):
    from app import formulary
    from app.formulary_adapters import csv_adapter

    src = write_source(tmp_path)
    (src / "formulary.csv").write_text(csv_adapter.to_csv(formulary.load()), encoding="utf-8")
    packs.build_pack(src, tmp_path / "f.gcpack")
    archive = packs.read_archive(tmp_path / "f.gcpack")
    built = packs.validate_formulary(archive.files["formulary.json"])
    assert built.model_dump()["medicines"] == formulary.load().model_dump()["medicines"]


def test_build_rejects_a_bad_csv_formulary(tmp_path):
    src = write_source(tmp_path)
    (src / "formulary.csv").write_text("record_type,medicine,adult_dose_amount\nmedicine,X,abc\n")
    with pytest.raises(packs.PackError, match="adult_dose_amount"):
        packs.build_pack(src, tmp_path / "f.gcpack")


def test_build_rejects_two_formularies(tmp_path):
    src = write_source(tmp_path)
    (src / "formulary.csv").write_text("record_type,medicine\nmedicine,X\n")
    (src / "formulary.json").write_text("{}")
    with pytest.raises(packs.PackError, match="not both"):
        packs.build_pack(src, tmp_path / "f.gcpack")


@pytest.mark.parametrize("meta,message", [
    ({"name": "Bad Name"}, "name"), ({"name": "x"}, "name"), ({"version": "1.0"}, "version"),
    ({"version": "v1.0.0"}, "version"), ({"version": "1.0.0-beta"}, "version"), ({"publisher": ""}, "publisher"),
    ({"licence": ""}, "licence"), ({"jurisdiction": "x"}, "jurisdiction"),
    ({"effective_from": "2030-01-01", "expires_on": "2029-01-01"}, "expiry"),
    ({"expires_on": "not a date"}, "expires_on"),
])
def test_build_validates_metadata(meta, message):
    with pytest.raises(packs.PackError, match=message):
        make(meta)


def test_build_requires_pack_json_and_documents(tmp_path):
    with pytest.raises(packs.PackError, match="pack.json"):
        packs.build_pack(tmp_path, tmp_path / "x.gcpack")
    src = write_source(tmp_path)
    (src / "pack.json").write_text(json.dumps({**META, "colour": "red"}))
    with pytest.raises(packs.PackError, match="unknown fields"):
        packs.build_pack(src, tmp_path / "x.gcpack")
    (src / "pack.json").write_text("[1]")
    with pytest.raises(packs.PackError, match="JSON object"):
        packs.build_pack(src, tmp_path / "x.gcpack")
    with pytest.raises(packs.PackError, match="at least one document"):
        make(files={"eval.json": EVAL})


def test_build_rejects_empty_unreadable_and_misplaced_files():
    with pytest.raises(packs.PackError, match="empty"):
        make(files={"documents/a.md": b""})
    with pytest.raises(packs.PackError):
        make(files={"documents/a.pdf": b"%PDF-1.4 not really"})
    with pytest.raises(packs.PackError, match="allowed"):
        make(files={"notes.md": b"# x\n\ntext"})
    with pytest.raises(packs.PackError):
        make(files={"documents/a.exe": b"MZ"})
    with pytest.raises(packs.PackError, match="Unsafe"):
        make(files={"documents/../a.md": b"# x\n\ntext"})


def test_build_rejects_bad_eval_and_formulary():
    for bad in (b"not json", b"[]", b'{"a":1}', b'[{"query":"q"}]', b'[{"query":"q","expect":"maybe"}]',
                b'[{"query":"","expect":"answer"}]', b'[{"query":"q","expect":"answer","extra":1}]', b"[1]"):
        with pytest.raises(packs.PackError):
            make(files={**FILES, "eval.json": bad})
    with pytest.raises(packs.PackError, match="formulary.json"):
        make(files={**FILES, "formulary.json": b'{"name": "x"}'})


def test_build_rejects_a_symlink_in_the_source(tmp_path):
    src = write_source(tmp_path)
    (src / "documents" / "link.md").symlink_to(tmp_path / "src" / "documents" / "falls.html")
    with pytest.raises(packs.PackError, match="link"):
        packs.build_pack(src, tmp_path / "x.gcpack")


def test_unicode_titles_and_large_documents_round_trip():
    big = ("# Large\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(400))).encode()
    data = make(files={"documents/ünï.md": "# Dosis über Nacht\n\nTägliche Dosis.".encode(), "documents/big.md": big},
                titles={"documents/ünï.md": "Übersicht ✓"})
    archive = packs.read_archive(data)
    assert archive.manifest.files[1].title == "Übersicht ✓" and len(archive.files["documents/big.md"]) == len(big)
    assert packs.verify(archive, {}, NOW).integrity


# --- Malformed manifests --------------------------------------------------------------------

@pytest.mark.parametrize("manifest,message", [
    (b"not json", "valid JSON"), (b"[]", "JSON object"), (b"\xff\xfe", "valid JSON"), (b"", "valid JSON"),
    (b"{" * 100000, "valid JSON"),
    ({}, "invalid"), ({"format": 2}, "invalid"),
])
def test_malformed_manifests_are_refused(manifest, message):
    with pytest.raises(packs.PackError, match=message):
        packs.read_archive(raw_pack(manifest))


def test_manifest_field_errors_name_the_field():
    good = manifest_for({"documents/a.md": b"# a\n\nb"})
    for change, where in [({"name": "BAD"}, "name"), ({"version": "x"}, "version"), ({"extra": 1}, "extra"),
                          ({"format": "1"}, "format"), ({"files": "no"}, "files"), ({"publisher": 5}, "publisher")]:
        with pytest.raises(packs.PackError, match=where):
            packs.read_archive(raw_pack({**good, **change}))


def test_manifest_entries_are_validated():
    body = b"# a\n\nb"
    base = manifest_for({"documents/a.md": body})
    entry = base["files"][0]
    cases = [
        {**entry, "path": "../a.md"}, {**entry, "path": "/abs/a.md"}, {**entry, "path": "documents\\a.md"},
        {**entry, "path": "documents/a.exe"}, {**entry, "path": "other/a.md"}, {**entry, "size": 0},
        {**entry, "size": -1}, {**entry, "size": 10 ** 12}, {**entry, "sha256": "ABC"}, {**entry, "kind": "script"},
        {**entry, "path": "x" * 300 + ".md"}, {**entry, "unexpected": True},
    ]
    for bad in cases:
        with pytest.raises(packs.PackError):
            packs.read_archive(raw_pack({**base, "files": [bad]}, {"documents/a.md": body}))
    with pytest.raises(packs.PackError, match="twice"):
        packs.read_archive(raw_pack({**base, "files": [entry, entry]}, {"documents/a.md": body}))
    with pytest.raises(packs.PackError, match="expiry"):
        packs.read_archive(raw_pack({**base, "effective_from": "2030-01-01", "expires_on": "2029-01-01"}))


def test_not_a_pack_at_all():
    for data in (b"", b"hello", b"PK\x03\x04garbage", b"%PDF-1.4"):
        with pytest.raises(packs.PackError):
            packs.read_archive(data)
    with pytest.raises(packs.PackError, match="no manifest"):
        packs.read_archive(raw_pack(b"{}", {}).replace(b"manifest.json", b"manifesx.json"))
    with pytest.raises(packs.PackError):
        packs.read_archive("/definitely/not/here.gcpack")


# --- Hostile archives ------------------------------------------------------------------------

HOSTILE = ["../evil.txt", "documents/../../evil.txt", "/etc/passwd", "documents//a.md", "./a.md",
           "C:/Windows/evil.txt", "documents\\..\\evil.txt", "a/./b.md", "documents/a.md/..", "documents/\x01a.md"]


@pytest.mark.parametrize("name", HOSTILE)
def test_zip_slip_member_names_are_refused_and_nothing_is_written(name, tmp_path):
    data = raw_pack(manifest_for({"documents/a.md": b"# a\n\nb"}), {"documents/a.md": b"# a\n\nb", name: b"evil"})
    before = set(tmp_path.parent.iterdir())
    with pytest.raises(packs.PackError):
        packs.read_archive(data)
    with pytest.raises(packs.PackError):
        packs.import_pack(data, keys={}, override=True, override_reason="testing", root=tmp_path / "store")
    assert set(tmp_path.parent.iterdir()) == before and not (tmp_path / "evil.txt").exists()


def test_a_manifest_naming_a_traversal_path_is_refused():
    body = b"# a\n\nb"
    manifest = manifest_for({"../escape.md": body})
    with pytest.raises(packs.PackError, match="Unsafe"):
        packs.read_archive(raw_pack(manifest, {"documents/a.md": body}))


def test_symlink_members_are_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest_for({})))
        info = zipfile.ZipInfo("documents/link.md")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(packs.PackError, match="link"):
        packs.read_archive(buf.getvalue())


def test_duplicate_member_names_are_refused():
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest_for({})))
            zf.writestr("documents/a.md", "# one\n\ntext")
            zf.writestr("documents/a.md", "# two\n\ntext")
    with pytest.raises(packs.PackError, match="twice"):
        packs.read_archive(buf.getvalue())


def test_encrypted_members_are_refused():
    data = bytearray(raw_pack(manifest_for({"documents/a.md": b"# a\n\nb"}), {"documents/a.md": b"# a\n\nb"}))
    for sig in (b"PK\x03\x04", b"PK\x01\x02"):
        i = 0
        while (i := data.find(sig, i)) != -1:
            off = i + (6 if sig == b"PK\x03\x04" else 8)
            data[off] |= 1
            i += 4
    with pytest.raises(packs.PackError):
        packs.read_archive(bytes(data))


def test_oversized_members_and_zip_bombs_are_refused(monkeypatch):
    monkeypatch.setattr(packs, "MAX_FILE_BYTES", 1024 * 1024)
    bomb = b"a" * (5 * 1024 * 1024)
    data = raw_pack(manifest_for({"documents/a.md": b"# a\n\nb"}), {"documents/a.md": b"# a\n\nb", "documents/b.md": bomb},
                    compress=zipfile.ZIP_DEFLATED)
    assert len(data) < 100_000
    with pytest.raises(packs.PackError, match="larger"):
        packs.read_archive(data)
    monkeypatch.setattr(packs, "MAX_TOTAL_BYTES", 10)
    with pytest.raises(packs.PackError, match="expands"):
        packs.read_archive(raw_pack(manifest_for({}), {"documents/a.md": b"x" * 100}))


def test_too_many_files_and_oversized_pack_and_manifest(monkeypatch):
    monkeypatch.setattr(packs, "MAX_FILES", 3)
    with pytest.raises(packs.PackError, match="too many"):
        packs.read_archive(raw_pack(manifest_for({}), {f"documents/{i}.md": b"x" for i in range(10)}))
    monkeypatch.setattr(packs, "MAX_PACK_BYTES", 100)
    with pytest.raises(packs.PackError, match="too large"):
        packs.read_archive(make())
    monkeypatch.setattr(packs, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(packs.PackError):
        packs.parse_manifest(b"{" + b" " * 100 + b"}")


def test_zero_length_file_listed_in_manifest_is_refused():
    manifest = manifest_for({"documents/a.md": b"x"})
    manifest["files"][0].update(size=0, sha256=hashlib.sha256(b"").hexdigest())
    with pytest.raises(packs.PackError):
        packs.read_archive(raw_pack(manifest, {"documents/a.md": b""}))


def test_huge_document_inside_limits_is_accepted_and_one_over_is_not(monkeypatch):
    monkeypatch.setattr(packs, "MAX_FILE_BYTES", 50_000)
    ok = b"# T\n\n" + b"a " * 20_000
    assert packs.verify(packs.read_archive(make(files={"documents/a.md": ok})), {}, NOW).integrity
    with pytest.raises(packs.PackError):
        make(files={"documents/a.md": ok + b"b" * 20_000})


# --- Signing and verification ---------------------------------------------------------------------

def test_signed_pack_verifies_with_the_trusted_key():
    private, public = keypair()
    v = verify_bytes(make(key=private), trusted(public))
    assert v.trusted and v.signature == "verified" and v.signer == "Acme Health" and v.key_id


def test_any_byte_flip_in_a_signed_pack_is_caught_or_harmless():
    """Flip every byte of the archive in turn. The result is an error, or the
    pack still verifies with exactly the original content (the byte was only
    zip bookkeeping, like a timestamp). It is never accepted with different
    content."""
    private, public = keypair()
    original = make(files={"documents/a.md": b"# A\n\nDose is 5 mg daily.", "eval.json": EVAL}, key=private)
    reference = packs.read_archive(original)
    keys = trusted(public)
    changed_but_accepted = 0
    for i in range(len(original)):
        mutated = bytearray(original)
        mutated[i] ^= 0x01
        try:
            archive = packs.read_archive(bytes(mutated))
            v = packs.verify(archive, keys, NOW)
        except packs.PackError:
            continue
        if v.trusted:
            assert archive.files == reference.files and archive.manifest_bytes == reference.manifest_bytes, i
            changed_but_accepted += 1
        else:
            assert not v.trusted
    assert changed_but_accepted < len(original)


def test_every_byte_of_every_file_is_covered_by_the_manifest():
    private, public = keypair()
    files = {"documents/a.md": b"# A\n\nDose is 5 mg daily.", "eval.json": EVAL}
    original = packs.read_archive(make(files=files, key=private))
    for path, body in files.items():
        for i in range(len(body)):
            tampered = dict(original.files)
            tampered[path] = body[:i] + bytes([body[i] ^ 0x20]) + body[i + 1:]
            archive = packs.Archive(original.manifest_bytes, original.signature_bytes, tampered, original.manifest)
            v = packs.verify(archive, trusted(public), NOW)
            assert not v.integrity and not v.trusted, (path, i)


def test_every_byte_of_the_manifest_is_covered_by_the_signature():
    private, public = keypair()
    original = packs.read_archive(make(key=private))
    for i in range(len(original.manifest_bytes)):
        mutated = bytearray(original.manifest_bytes)
        mutated[i] ^= 0x01
        archive = packs.Archive(bytes(mutated), original.signature_bytes, original.files, original.manifest)
        assert not packs.verify(archive, trusted(public), NOW).trusted, i


def test_a_swapped_file_is_caught():
    private, public = keypair()
    original = packs.read_archive(make(key=private))
    swapped = dict(original.files)
    swapped["documents/falls.html"], swapped["documents/transfusion.md"] = (
        swapped["documents/transfusion.md"], swapped["documents/falls.html"])
    v = packs.verify(packs.Archive(original.manifest_bytes, original.signature_bytes, swapped, original.manifest),
                     trusted(public), NOW)
    assert not v.integrity and len(v.problems) == 2


def test_a_swapped_file_with_a_rebuilt_manifest_fails_the_signature():
    private, public = keypair()
    signed = packs.read_archive(make(key=private))
    forged_files = {**FILES, "documents/falls.html": FALLS.replace(b"6 hours", b"60 hours")}
    forged = packs.read_archive(raw_pack(manifest_for(forged_files), forged_files, signed.signature_bytes))
    v = packs.verify(forged, trusted(public), NOW)
    assert v.integrity and v.signature == "invalid" and not v.trusted


def test_a_signature_from_another_pack_does_not_transfer():
    private, public = keypair()
    other = packs.read_archive(make(meta={"version": "9.9.9"}, key=private))
    mine = packs.read_archive(raw_pack(manifest_for(FILES), FILES, other.signature_bytes))
    assert packs.verify(mine, trusted(public), NOW).signature == "invalid"


def test_wrong_key_and_unknown_key():
    signer, _ = keypair()
    _, other_public = keypair()
    data = make(key=signer)
    v = verify_bytes(data, trusted(other_public))
    assert v.signature == "untrusted" and not v.trusted
    assert verify_bytes(data, {}).signature == "untrusted"


def test_key_trusted_under_another_publisher_name_is_not_enough():
    private, public = keypair()
    v = verify_bytes(make(key=private), trusted(public, publisher="Someone Else"))
    assert v.signature == "invalid" and "Someone Else" in v.detail


def test_same_key_id_but_forged_signature_bytes():
    private, public = keypair()
    archive = packs.read_archive(make(key=private))
    sig = json.loads(archive.signature_bytes)
    sig["signature"] = packs.encode_key(b"\x00" * 32) + packs.encode_key(b"\x00" * 32)[:-1]
    forged = packs.Archive(archive.manifest_bytes, json.dumps(sig).encode(), archive.files, archive.manifest)
    assert packs.verify(forged, trusted(public), NOW).signature == "invalid"


@pytest.mark.parametrize("signature", [
    b"not json", b"[]", b"{}", b'{"format":1}', b'{"format":2,"alg":"ed25519","key_id":"a","signature":"AA=="}',
    b'{"format":1,"alg":"rsa","key_id":"a","signature":"AA=="}', b'{"format":1,"alg":"ed25519","key_id":5,"signature":"AA=="}',
    b'{"format":1,"alg":"ed25519","key_id":"a","signature":"%%%"}', b"\xff\xfe",
])
def test_malformed_signature_files_are_invalid_not_crashes(signature):
    _, public = keypair()
    archive = packs.read_archive(raw_pack(manifest_for(FILES), FILES, signature))
    v = packs.verify(archive, trusted(public), NOW)
    assert v.signature == "invalid" and not v.trusted


def test_expired_key_is_refused_and_a_live_one_accepted():
    private, public = keypair()
    data = make(key=private)
    archive = packs.read_archive(data)
    keys = trusted(public, expires="2026-07-01")
    assert packs.verify(archive, keys, datetime(2026, 6, 30, tzinfo=timezone.utc)).trusted
    assert packs.verify(archive, keys, datetime(2026, 7, 1, tzinfo=timezone.utc)).signature == "expired_key"
    assert packs.verify(archive, keys, datetime(2030, 1, 1, tzinfo=timezone.utc)).signature == "expired_key"


def test_extra_and_missing_files_break_integrity():
    private, public = keypair()
    archive = packs.read_archive(make(key=private))
    extra = packs.Archive(archive.manifest_bytes, archive.signature_bytes,
                          {**archive.files, "documents/evil.md": b"# Evil\n\ntext"}, archive.manifest)
    assert "not in the manifest" in packs.verify(extra, trusted(public), NOW).problems[0]
    missing = dict(archive.files)
    missing.pop("eval.json")
    gone = packs.Archive(archive.manifest_bytes, archive.signature_bytes, missing, archive.manifest)
    assert "missing" in packs.verify(gone, trusted(public), NOW).problems[0]


def test_unsigned_pack_status():
    v = verify_bytes(make(), {})
    assert v.integrity and v.signature == "unsigned" and not v.trusted


def test_trusted_keys_parsing(monkeypatch):
    _, a = keypair()
    _, b = keypair()
    monkeypatch.setenv("PACK_TRUSTED_KEYS", f"Acme Health={a}, Other Co={b}@2030-12-31")
    keys = packs.trusted_keys()
    assert {k.publisher for k in keys.values()} == {"Acme Health", "Other Co"}
    assert [k.expires_on for k in keys.values() if k.publisher == "Other Co"] == [date(2030, 12, 31)]
    monkeypatch.delenv("PACK_TRUSTED_KEYS")
    assert packs.trusted_keys() == {}
    for bad in ("nokey", "=abc", "Acme=", f"Acme={a[:-4]}", "Acme=!!!!", f"Acme={a}@tomorrow",
                f"Acme={packs.encode_key(b'short')}"):
        with pytest.raises(packs.PackError):
            packs.trusted_keys(bad)


def test_keypair_helpers():
    private, public = keypair()
    assert len(private) == 32 and integrity.ed25519_public(private) == packs.decode_key(public)
    sig = integrity.ed25519_sign(private, b"msg")
    assert integrity.ed25519_verify(packs.decode_key(public), sig, b"msg")
    assert not integrity.ed25519_verify(packs.decode_key(public), sig, b"msh")
    assert not integrity.ed25519_verify(packs.decode_key(public), sig[:-1], b"msg")
    assert not integrity.ed25519_verify(b"short", sig, b"msg")


# --- Import: database, approval, idempotence, versions -------------------------------------------

@pytest.fixture()
def store(database, tmp_path, monkeypatch):
    monkeypatch.setenv("PACK_STORE_DIR", str(tmp_path / "store"))
    return tmp_path / "store"


def count_sources() -> int:
    with db.session() as s:
        return s.scalar(select(func.count(db.Source.id)))


def imp(data, **kw):
    kw.setdefault("now", NOW)
    return packs.import_pack(data, **kw)


def test_round_trip_import_leaves_documents_pending_not_live(store):
    private, public = keypair()
    result = imp(make(key=private), keys=trusted(public))
    assert result["status"] == "imported" and result["verification"]["trusted"] and not result["override_used"]
    assert sorted(d["title"] for d in result["documents"]) == ["Falls Prevention Policy", "Transfusion Observation Policy"]
    assert all(d["status"] == "pending" and d["owner"] == "Acme Health" for d in result["documents"])
    assert all(d["filename"].startswith("pack:demo-policies@1.0.0/documents/") for d in result["documents"])
    assert knowledge.approved_records() == []
    detail = knowledge.get_source(result["documents"][0]["id"])
    assert detail["sections"] and detail["status"] == "pending"
    assert (store / "demo-policies" / "1.0.0" / "eval.json").read_bytes() == EVAL
    assert [r["name"] for r in packs.installed(store)] == ["demo-policies"]


def test_untrusted_packs_import_nothing(store):
    with pytest.raises(packs.PackVerificationError, match="unsigned"):
        imp(make(), keys={})
    assert count_sources() == 0 and packs.installed(store) == []
    signer, _ = keypair()
    with pytest.raises(packs.PackVerificationError):
        imp(make(key=signer), keys={})
    assert count_sources() == 0


def test_tampered_pack_is_never_imported_even_with_override(store):
    private, public = keypair()
    original = packs.read_archive(make(key=private))
    files = dict(original.files)
    files["documents/falls.html"] = FALLS.replace(b"6 hours", b"60 hours")
    forged = raw_pack(original.manifest_bytes, files, original.signature_bytes)
    with pytest.raises(packs.PackVerificationError, match="doesn't match its manifest"):
        imp(forged, keys=trusted(public), override=True, override_reason="please just import it")
    assert count_sources() == 0


def test_override_requires_a_reason_and_is_audited(store):
    result = None
    with pytest.raises(packs.PackError, match="reason"):
        imp(make(), keys={}, override=True, override_reason="  ")
    assert count_sources() == 0
    result = imp(make(), keys={}, override=True, override_reason="Vendor pack, reviewed manually", user_id=None)
    assert result["status"] == "imported" and result["override_used"] and result["audit_id"]
    with db.session() as s:
        rows = s.scalars(select(db.AuditRecord).where(db.AuditRecord.audit_id == result["audit_id"])).all()
        assert len(rows) == 1
        record = rows[0].record
    assert record["event"] == "pack_import_override"
    assert record["detail"]["reason"] == "Vendor pack, reviewed manually"
    assert record["detail"]["signature"] == "unsigned" and record["detail"]["pack"] == "demo-policies"
    assert integrity.verify()["ok"]


def test_a_verified_import_is_audited_too(store):
    private, public = keypair()
    imp(make(key=private), keys=trusted(public))
    with db.session() as s:
        events = [r.record.get("event") for r in s.scalars(select(db.AuditRecord)) if r.decision == "event"]
    assert events == ["pack_import"]
    assert integrity.verify()["ok"]


def test_override_for_expired_key_and_invalid_signature(store):
    private, public = keypair()
    data = make(key=private)
    with pytest.raises(packs.PackVerificationError, match="expired"):
        imp(data, keys=trusted(public, expires="2026-01-01"))
    r = imp(data, keys=trusted(public, expires="2026-01-01"), override=True, override_reason="key rotation pending")
    assert r["override_used"] and r["verification"]["signature"] == "expired_key"


def test_importing_the_same_pack_twice_is_idempotent(store):
    private, public = keypair()
    data = make(key=private)
    first = imp(data, keys=trusted(public))
    n = count_sources()
    second = imp(data, keys=trusted(public))
    assert second["status"] == "already_imported" and second["documents"] == []
    assert count_sources() == n == len(first["documents"])
    # Re-signed with another key but identical content is the same manifest.
    other_private, other_public = keypair()
    third = imp(make(key=other_private), keys=trusted(other_public))
    assert third["status"] == "already_imported" and count_sources() == n


def test_same_version_with_different_content_is_refused(store):
    imp(make(), keys={}, override=True, override_reason="first import ok")
    changed = {**FILES, "documents/falls.html": FALLS.replace(b"6 hours", b"4 hours")}
    with pytest.raises(packs.PackError, match="different content"):
        imp(make(files=changed), keys={}, override=True, override_reason="second attempt")
    assert count_sources() == 2


def test_upgrade_adds_new_pending_versions_and_skips_unchanged_documents(store):
    def go(data):
        return imp(data, keys={}, override=True, override_reason="test pack")
    v1 = go(make())
    for d in v1["documents"]:
        knowledge.review(d["id"], "approve", None, allow_self_approval=True)
    changed = {**FILES, "documents/falls.html": FALLS.replace(b"6 hours", b"4 hours")}
    v2 = go(make(meta={"version": "1.1.0"}, files=changed))
    assert [d["version"] for d in v2["documents"]] == [2] and v2["skipped"] == ["documents/transfusion.md"]
    assert v2["documents"][0]["status"] == "pending"
    approved = knowledge.review(v2["documents"][0]["id"], "approve", None, allow_self_approval=True)
    assert approved["status"] == "approved"
    statuses = {(s["document_key"], s["version"]): s["status"] for s in knowledge.list_sources()}
    assert statuses[("falls-prevention-policy", 1)] == "retired"
    assert statuses[("falls-prevention-policy", 2)] == "approved"
    assert statuses[("transfusion-observation-policy", 1)] == "approved"
    assert [r["version"] for r in packs.installed(store)] == ["1.0.0", "1.1.0"]


def test_versions_compare_numerically_not_as_text(store):
    for v in ("1.9.0", "1.10.0"):
        imp(make(meta={"version": v}, files={"documents/a.md": f"# A\n\nversion {v}".encode()}), keys={},
            override=True, override_reason="ordering test")
    assert [r["version"] for r in packs.installed(store)] == ["1.9.0", "1.10.0"]
    with pytest.raises(packs.PackError, match="allow_downgrade"):
        imp(make(meta={"version": "1.9.5"}, files={"documents/a.md": b"# A\n\nversion 1.9.5"}), keys={},
            override=True, override_reason="ordering test")


def test_downgrade_is_refused_unless_allowed(store):
    imp(make(meta={"version": "2.0.0"}), keys={}, override=True, override_reason="current")
    old = make(meta={"version": "1.0.0"}, files={"documents/falls.html": FALLS.replace(b"6 hours", b"8 hours")})
    with pytest.raises(packs.PackError, match="allow_downgrade"):
        imp(old, keys={}, override=True, override_reason="rollback attempt")
    n = count_sources()
    result = imp(old, keys={}, override=True, override_reason="rollback attempt", allow_downgrade=True)
    assert result["status"] == "imported" and count_sources() == n + 1


def test_pack_names_are_independent(store):
    imp(make(meta={"name": "pack-one"}), keys={}, override=True, override_reason="first pack")
    other = make(meta={"name": "pack-two", "version": "0.1.0"}, files={"documents/b.md": b"# B\n\nOther text here."})
    assert imp(other, keys={}, override=True, override_reason="second pack")["status"] == "imported"
    assert {r["name"] for r in packs.installed(store)} == {"pack-one", "pack-two"}


def test_an_expired_pack_is_refused(store):
    with pytest.raises(packs.PackError, match="expired"):
        imp(make(meta={"expires_on": "2026-05-31"}), keys={}, override=True, override_reason="old pack")
    assert count_sources() == 0


def test_pack_dates_are_applied_to_every_document(store):
    r = imp(make(meta={"effective_from": "2026-06-15", "expires_on": "2027-01-01"}), keys={}, override=True,
            override_reason="dated pack")
    assert {d["effective_from"][:10] for d in r["documents"]} == {"2026-06-15"}
    assert {d["expires_on"][:10] for d in r["documents"]} == {"2027-01-01"}


def test_two_documents_with_the_same_title_are_refused_before_anything_is_stored(store):
    files = {"documents/a.md": b"# Same Title\n\nfirst", "documents/b.md": b"# Same Title\n\nsecond"}
    with pytest.raises(packs.PackError, match="same title"):
        imp(raw_pack(manifest_for(files), files), keys={}, override=True, override_reason="duplicate titles")
    assert count_sources() == 0 and packs.installed(store) == []


def test_an_unreadable_document_stops_the_whole_import(store):
    files = {"documents/a.md": b"# Fine\n\nbody text", "documents/b.pdf": b"%PDF-1.4 broken"}
    with pytest.raises(packs.PackError, match="b.pdf"):
        imp(raw_pack(manifest_for(files), files), keys={}, override=True, override_reason="bad document")
    assert count_sources() == 0


def test_a_failure_part_way_rolls_back_and_a_retry_succeeds(store, monkeypatch):
    real = knowledge.import_document
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real(*a, **kw)

    monkeypatch.setattr(knowledge, "import_document", flaky)
    with pytest.raises(packs.PackError, match="rolled back"):
        imp(make(), keys={}, override=True, override_reason="flaky import")
    assert packs.installed(store) == []
    assert [s["status"] for s in knowledge.list_sources()] == ["rejected"]
    monkeypatch.setattr(knowledge, "import_document", real)
    assert imp(make(), keys={}, override=True, override_reason="flaky import")["status"] == "imported"
    assert knowledge.approved_records() == []


def test_formulary_is_staged_but_never_activated(store, tmp_path):
    from app import formulary

    demo = formulary.load()
    data = make(files={**FILES, "formulary.json": demo.model_dump_json().encode()})
    r = imp(data, keys={}, override=True, override_reason="formulary pack")
    assert r["formulary_path"] and formulary.load(Path(r["formulary_path"])).model_dump() == demo.model_dump()
    assert formulary.load().name == demo.name and config.FORMULARY_PATH != Path(r["formulary_path"])


def test_the_pack_eval_set_is_kept_for_the_release_gate(store):
    imp(make(), keys={}, override=True, override_reason="eval pack")
    cases = packs.release_gate_cases(store)
    assert [c["expect"] for c in cases] == ["answer", "refuse"] and cases[0]["pack"] == "demo-policies@1.0.0"


# --- Imported content flows through retrieval and the guards ---------------------------------------

@pytest.fixture()
def live_index(store, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "INCLUDE_DEMO_CORPUS", False)
    monkeypatch.setattr(config, "VECTOR_STORE", "local")
    yield
    monkeypatch.undo()
    db.reset_engine()
    retrieval.load_index()


def test_imported_documents_are_answered_cited_and_guarded_after_approval(live_index):
    r = imp(make(), keys={}, override=True, override_reason="end to end")
    knowledge.rebuild_index()
    question = "How soon should a falls risk assessment be completed after admission?"
    assert pipeline.run(question, review=False).decision == "refuse"   # pending: not citable
    falls = next(d for d in r["documents"] if d["title"] == "Falls Prevention Policy")
    knowledge.review(falls["id"], "approve", None, allow_self_approval=True)
    knowledge.rebuild_index()
    answered = pipeline.run(question, review=False)
    assert answered.decision == "answer"
    assert any(sid.startswith(f"DOC{falls['id']}-") for c in answered.claims for sid in c.source_ids)
    assert "6 hours" in answered.answer_text
    assert pipeline.run("How soon should a Quelvarin risk assessment be completed after admission?",
                        review=False).decision == "refuse"
    gate = packs.run_eval(packs.release_gate_cases())
    assert gate["ok"] and gate["total"] == 2 and gate["unsafe_answers"] == 0


def test_a_failing_pack_eval_is_reported_and_unsafe_answers_are_counted(live_index):
    r = imp(make(), keys={}, override=True, override_reason="end to end")
    for d in r["documents"]:
        knowledge.review(d["id"], "approve", None, allow_self_approval=True)
    knowledge.rebuild_index()
    wrong = [{"query": "How soon should a falls risk assessment be completed after admission?", "expect": "refuse"},
             {"query": "How soon should a Quelvarin risk assessment be completed after admission?", "expect": "answer"}]
    result = packs.run_eval(wrong)
    assert not result["ok"] and result["passed"] == 0 and result["unsafe_answers"] == 1
    assert len(result["failures"]) == 2


# --- Command line -----------------------------------------------------------------------------------

def test_cli_keygen_build_inspect_import(store, tmp_path, capsys):
    key = tmp_path / "pub.key"
    assert cli.main(["pack", "keygen", "--out", str(key)]) == 0
    out = capsys.readouterr().out
    public = out.strip().splitlines()[-1]
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert cli.main(["pack", "keygen", "--out", str(key)]) == 1                      # never overwrite a key
    assert "refusing to overwrite" in capsys.readouterr().err

    src = write_source(tmp_path)
    out_file = tmp_path / "demo.gcpack"
    assert cli.main(["pack", "build", str(src), str(out_file), "--sign-key", str(key)]) == 0
    assert "signed" in capsys.readouterr().out

    os.environ["PACK_TRUSTED_KEYS"] = f"Acme Health={public}"
    try:
        assert cli.main(["pack", "inspect", str(out_file)]) == 0
        text = capsys.readouterr().out
        assert "demo-policies 1.0.0" in text and "Signature: verified" in text
        assert cli.main(["pack", "import", str(out_file)]) == 0
        assert "2 documents are pending approval" in capsys.readouterr().out
        assert cli.main(["pack", "import", str(out_file)]) == 0
        assert "already imported" in capsys.readouterr().out
    finally:
        del os.environ["PACK_TRUSTED_KEYS"]


def test_cli_refuses_untrusted_packs_and_accepts_an_override(store, tmp_path, capsys):
    out_file = tmp_path / "u.gcpack"
    out_file.write_bytes(make())
    assert cli.main(["pack", "inspect", str(out_file)]) == 2
    assert "unsigned" in capsys.readouterr().out
    assert cli.main(["pack", "import", str(out_file)]) == 1
    assert "can't be trusted" in capsys.readouterr().err
    assert count_sources() == 0
    assert cli.main(["pack", "import", str(out_file), "--override"]) == 1
    assert "reason" in capsys.readouterr().err
    assert cli.main(["pack", "import", str(out_file), "--override", "--reason", "reviewed by pharmacy lead"]) == 0
    assert "Override recorded" in capsys.readouterr().out


def test_cli_reports_bad_input_without_a_traceback(tmp_path, capsys):
    assert cli.main(["pack", "inspect", str(tmp_path / "missing.gcpack")]) == 1
    bad = tmp_path / "bad.gcpack"
    bad.write_bytes(b"not a zip")
    assert cli.main(["pack", "inspect", str(bad)]) == 1
    assert cli.main(["pack", "build", str(tmp_path), str(tmp_path / "o.gcpack")]) == 1
    assert cli.main(["pack", "build", str(tmp_path), str(tmp_path / "o.gcpack"), "--sign-key", str(tmp_path / "no")]) == 1
    err = capsys.readouterr().err
    assert err.count("Error:") == 4 and "Traceback" not in err
