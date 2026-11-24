# Knowledge packs

A knowledge pack is a portable, versioned bundle: documents, an optional formulary and an optional
evaluation set, with a manifest naming its publisher, licence and jurisdiction. Packs go through the same
approval process as any uploaded document, so nothing in a pack is cited until a reviewer approves it.

## Format

A pack (`.gcpack`) is a zip archive:

| File | Contents |
| --- | --- |
| `manifest.json` | `format` (1), `name`, `version`, `publisher`, `licence`, `jurisdiction`, optional `description`, `effective_from`, `expires_on`, `synthetic`, and `files`: the path, SHA-256, size and kind (`document`, `formulary`, `eval`) of every other file |
| `signature.json` | optional Ed25519 signature over the manifest bytes |
| `documents/...` | PDF, Word, HTML, Markdown or text documents |
| `formulary.json` | optional, in the format described in `docs/formulary.md` |
| `eval.json` | optional list of `{"query", "expect": "answer"\|"refuse"}` (and optionally `"patient"`) |

`name` is 2 to 64 lower-case letters, digits, `.`, `_`, `-`. `version` is `MAJOR.MINOR.PATCH`. Because the manifest
holds every file's hash, signing the manifest covers the whole pack.

## Building

Make a folder holding `pack.json` (the metadata above, plus an optional `titles` object mapping a document path
to a title), `documents/`, and optionally `formulary.json` or `formulary.csv` (converted with the CSV adapter) and
`eval.json`:

    python -m app.cli pack keygen --out publisher.key
    python -m app.cli pack build my-pack/ my-pack.gcpack --sign-key publisher.key

Building checks everything an importer would: names, readable documents, a valid formulary and eval set. The
output is reproducible: the same inputs give the same bytes.

`scripts/import_spl.py` builds a pack from drug labels in SPL XML; see the script's header.

## Importing

    python -m app.cli pack inspect my-pack.gcpack
    python -m app.cli pack import my-pack.gcpack

Import adds each document as a **pending** version. A reviewer approves it in the usual way, which retires the
previous approved version of a document with the same title. The pack's formulary is validated and staged
beside its record (under `PACK_STORE_DIR`, default `data/packs/<name>/<version>/`) but not activated; point
`FORMULARY_PATH` at it to use it. The eval set is kept for the release gate (`packs.release_gate_cases()`,
run with `packs.run_eval`).

Importing is safe to repeat: the same pack again changes nothing. A different pack with the same name and
version is refused (publish changes under a new version). A newer version adds new pending versions of the
documents that changed and skips the unchanged ones. An older version than one already imported is refused
unless `--allow-downgrade`. If an import fails part way, the documents it added are rejected and nothing is
recorded, so it can be retried.

## Trust and signing

`PACK_TRUSTED_KEYS` lists the publishers whose signatures are accepted, as
`publisher=base64-public-key[@YYYY-MM-DD]` separated by commas; the date is the day the key stops being accepted.
`pack keygen` prints the public key in this form. The publisher in a pack's manifest must match the publisher the
signing key is trusted for.

A pack is refused if it is unsigned, signed by an unknown or expired key, or its signature is invalid. An
administrator can import it anyway with `--override --reason "..."`; the override, the signature status and the
reason are written to the tamper-evident audit trail. A pack whose files don't match its own manifest (a changed
byte, a swapped, missing or extra file) is corrupt and is never imported, override or not. Every verified import is
also recorded in the audit trail.

Archives are read into memory, never extracted to disk. Names with `..`, absolute paths, drive letters, backslashes
or control characters, links, encrypted members, duplicate names, oversized members and over-large totals are refused.

## Not yet wired

The release gate itself (`app/releases.py`) does not yet read `packs.release_gate_cases()`; until it does, run
`packs.run_eval(packs.release_gate_cases())` after approving a pack and before relying on it.
