"""Command-line administration.

    python -m app.cli migrate
    python -m app.cli create-user --email you@example.org --role admin
    python -m app.cli set-password --email you@example.org
    python -m app.cli list-users
    python -m app.cli purge-sessions
    python -m app.cli escalate-reviews
    python -m app.cli delete-user --email someone@example.org
    python -m app.cli generate-key
    python -m app.cli reencrypt
    python -m app.cli verify-audit
    python -m app.cli audit-head
    python -m app.cli retention [--apply]
    python -m app.cli pack keygen --out publisher.key
    python -m app.cli pack build SRC_DIR OUT.gcpack [--sign-key publisher.key]
    python -m app.cli pack inspect FILE.gcpack
    python -m app.cli pack import FILE.gcpack [--override --reason TEXT] [--allow-downgrade]
    python -m app.cli tenant create --key acme --name "Acme Clinic" --admin-email admin@acme.org
                                    [--generate-password] [--scopes ask,verify] [--rpm N] [--monthly-quota N]

Passwords are always prompted for, never passed as arguments, so they don't
end up in shell history."""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from . import auth, db, encryption


def _prompt_password() -> str:
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        raise auth.AuthError("The passwords don't match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="GroundCheckHealth administration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="Upgrade the database to the latest schema")
    create = sub.add_parser("create-user", help="Create a user (prompts for the password)")
    create.add_argument("--email", required=True)
    create.add_argument("--role", default="clinician", choices=auth.ROLES)
    create.add_argument("--name", default="")
    reset = sub.add_parser("set-password", help="Set a user's password (prompts for it)")
    reset.add_argument("--email", required=True)
    sub.add_parser("list-users", help="List users")
    sub.add_parser("purge-sessions", help="Delete expired sign-in sessions")
    sub.add_parser("escalate-reviews", help="Escalate review cases past their due date")
    remove = sub.add_parser("delete-user", help="Delete a user's account and personal details")
    remove.add_argument("--email", required=True)
    sub.add_parser("generate-key", help="Print a new key for DATA_ENCRYPTION_KEYS or AUDIT_SIGNING_KEYS")
    sub.add_parser("reencrypt", help="Re-encrypt stored data with the current primary key")
    sub.add_parser("verify-audit", help="Check the audit trail hasn't been changed")
    sub.add_parser("audit-head", help="Print the audit chain head, to record somewhere else")
    keep = sub.add_parser("retention", help="Show, or with --apply delete, records past their retention period")
    keep.add_argument("--apply", action="store_true", help="Delete them")
    pack = sub.add_parser("pack", help="Build, inspect and import knowledge packs")
    pack_sub = pack.add_subparsers(dest="pack_command", required=True)
    keygen = pack_sub.add_parser("keygen", help="Create a signing key for a publisher")
    keygen.add_argument("--out", required=True, help="File to write the private key to")
    build = pack_sub.add_parser("build", help="Build a pack from a folder")
    build.add_argument("source")
    build.add_argument("output")
    build.add_argument("--sign-key", help="Private key file from `pack keygen`")
    look = pack_sub.add_parser("inspect", help="Show a pack's manifest and whether it can be trusted")
    look.add_argument("file")
    load = pack_sub.add_parser("import", help="Import a pack's documents as pending versions")
    load.add_argument("file")
    load.add_argument("--override", action="store_true", help="Import although the pack is unsigned or untrusted")
    load.add_argument("--reason", default="", help="Why the override is justified (recorded in the audit trail)")
    load.add_argument("--allow-downgrade", action="store_true", help="Import a version older than one already imported")
    tenant = sub.add_parser("tenant", help="Provision tenants (sites) for hosted use")
    tenant_sub = tenant.add_subparsers(dest="tenant_command", required=True)
    tcreate = tenant_sub.add_parser("create", help="Create a site, its admin, an API key and a quota in one step")
    tcreate.add_argument("--key", required=True, help="Short lower-case name for the site, such as st-marys")
    tcreate.add_argument("--name", required=True, help="The site's display name")
    tcreate.add_argument("--admin-email", required=True)
    tcreate.add_argument("--admin-name", default="")
    tcreate.add_argument("--generate-password", action="store_true",
                         help="Make a strong password for the admin and print it once, instead of prompting")
    tcreate.add_argument("--key-name", default="", help="Name for the API key")
    tcreate.add_argument("--scopes", default="ask,verify", help="Comma-separated API key scopes")
    tcreate.add_argument("--rpm", type=int, default=None, help="API key requests per minute")
    tcreate.add_argument("--monthly-quota", type=int, default=None, help="API key requests per month")
    tcreate.add_argument("--expires-days", type=float, default=None, help="API key lifetime in days")
    tcreate.add_argument("--json", action="store_true", help="Print the result as JSON (it includes the secrets, once)")
    args = parser.parse_args(argv)

    if args.command == "pack" and args.pack_command in ("keygen", "build", "inspect"):
        return _pack_offline(args)
    try:
        if args.command == "generate-key":
            print(encryption.generate_key())
            return 0
        # Check keys before touching data, so a mistyped key is caught at once.
        encryption.keyring()
        from . import integrity

        integrity.signing_summary()
        if args.command == "migrate":
            db.migrate()
            print("Database is up to date.")
            return 0
        db.migrate()
        if args.command == "create-user":
            user = auth.create_user(args.email, _prompt_password(), role=args.role, name=args.name)
            print(f"Created {user.role} {user.email}.")
        elif args.command == "set-password":
            email = auth.normalise_email(args.email)
            match = next((u for u in auth.list_users() if u["email"] == email), None)
            if match is None:
                raise auth.AuthError("No such user.")
            auth.set_password(match["id"], _prompt_password())
            print(f"Password updated for {email}. Their sessions were signed out.")
        elif args.command == "list-users":
            for u in auth.list_users():
                status = "active" if u["is_active"] else "inactive"
                mfa = "2FA on" if u["mfa_enabled"] else "2FA off"
                print(f"{u['email']:<40} {u['role']:<10} {status:<9} {mfa}")
        elif args.command == "purge-sessions":
            print(f"Deleted {auth.purge_expired_sessions()} expired sessions.")
        elif args.command == "escalate-reviews":
            from . import governance

            print(f"Escalated {governance.escalate_overdue()} overdue review cases.")
        elif args.command == "delete-user":
            email = auth.normalise_email(args.email)
            match = next((u for u in auth.list_users() if u["email"] == email), None)
            if match is None:
                raise auth.AuthError("No such user.")
            auth.delete_user(match["id"])
            print(f"Deleted {email}. Their audit records remain under account {match['id']}.")
        elif args.command == "reencrypt":
            from . import rekey

            result = rekey.reencrypt()
            state = result["encryption"]
            print("Encryption is " + (f"on, with key {state['primary_key_id']}." if state["enabled"] else "off."))
            for table, n in result["rewritten"].items():
                print(f"  {table:<16} {n} rows rewritten")
        elif args.command == "verify-audit":
            result = integrity.verify()
            kind = "signed" if result["signed"] else "unsigned"
            print(f"Checked {result['checked']} audit records ({kind} chain, head {result['head']['seq']}).")
            for p in result["problems"]:
                print(f"  #{p['seq']} {p['audit_id'] or ''}: {p['problem']}")
            if result["problem_count"] > len(result["problems"]):
                print(f"  ...and {result['problem_count'] - len(result['problems'])} more.")
            if not result["complete"]:
                print(f"Couldn't check the whole audit trail. {result['error']}")
                return 1
            print("The audit trail is intact." if result["ok"] else "The audit trail has been changed.")
            return 0 if result["ok"] else 2
        elif args.command == "audit-head":
            head = integrity.head()
            print(f"{head['seq']} {head['hash']} {head['updated_at']}")
        elif args.command == "pack":
            return _pack_import(args)
        elif args.command == "tenant":
            return _tenant_create(args)
        elif args.command == "retention":
            from . import retention

            if args.apply:
                run = retention.apply()
                print(f"Deleted {run['audit_deleted']} audit records, {run['reviews_deleted']} review cases "
                      f"and {run['sessions_deleted']} expired sessions.")
            else:
                p = retention.plan()
                if not (p["audit_retention_days"] or p["review_retention_days"]):
                    print("No retention period is set, so nothing would be deleted.")
                print(f"Would delete {p['audit_records_to_delete']} audit records and "
                      f"{p['review_cases_to_delete']} review cases. Run with --apply to delete them.")
    except encryption.EncryptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except auth.AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _tenant_create(args) -> int:
    import json
    import secrets

    from . import apikeys, sites

    generated = args.generate_password
    password = secrets.token_urlsafe(18) if generated else _prompt_password()
    scopes = [x.strip() for x in args.scopes.split(",") if x.strip()]
    try:
        made = sites.provision(args.key, args.name, args.admin_email, password, admin_name=args.admin_name,
                               key_name=args.key_name, scopes=scopes, rpm_limit=args.rpm, monthly_quota=args.monthly_quota,
                               expires_in_days=args.expires_days)
    except (sites.SiteError, apikeys.ApiKeyError) as exc:
        print(f"Error: {getattr(exc, 'message', None) or exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({**made, **({"admin_password": password} if generated else {})}, indent=2))
        return 0
    key = made["api_key"]
    print(f"Created site {made['site']['key']} (id {made['site']['id']}) with admin {made['admin']['email']}.")
    if generated:
        print(f"Admin password (shown once): {password}")
    print(f"API key {key['prefix']}... scopes {','.join(key['scopes'])}, "
          f"{key['rpm_limit'] or 'default'} requests/minute, {key['monthly_quota'] or 'no'} monthly quota.")
    print(f"API key secret (shown once, store it now): {key['key']}")
    return 0


def _pack_offline(args) -> int:
    """Pack commands that need neither the database nor the app's keys."""
    from pathlib import Path

    from . import packs

    try:
        if args.pack_command == "keygen":
            out = Path(args.out)
            if out.exists():
                raise packs.PackError(f"{out} already exists; refusing to overwrite a key.")
            private, public = packs.generate_keypair()
            fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(private + "\n")
            print(f"Private key written to {out}. Keep it secret.")
            print(f"Public key (for PACK_TRUSTED_KEYS as publisher={public}):\n{public}")
        elif args.pack_command == "build":
            key = None
            if args.sign_key:
                try:
                    key = packs.decode_key(Path(args.sign_key).read_text(encoding="utf-8"))
                except OSError as exc:
                    raise packs.PackError(f"Couldn't read the key file: {exc.strerror}") from None
            info = packs.build_pack(Path(args.source), Path(args.output), key)
            print(f"Built {info['name']} {info['version']}: {info['files']} files, "
                  f"{'signed' if info['signed'] else 'unsigned'}, sha256 {info['sha256']}")
            print(f"Wrote {info['path']}")
        else:
            info = packs.inspect_pack(Path(args.file))
            print(f"{info['name']} {info['version']} by {info['publisher']} ({info['licence']}, {info['jurisdiction']})")
            if info["description"]:
                print(f"  {info['description']}")
            print(f"  effective {info['effective_from'] or 'always'}, expires {info['expires_on'] or 'never'}"
                  f"{', synthetic' if info['synthetic'] else ''}")
            for f in info["files"]:
                print(f"  {f['kind']:<9} {f['path']}  {f['size']} bytes")
            v = info["verification"]
            for problem in v["problems"]:
                print(f"  PROBLEM: {problem}")
            print(f"Signature: {v['signature']}. {v['detail']}")
            return 0 if v["trusted"] else 2
    except packs.PackError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _pack_import(args) -> int:
    from pathlib import Path

    from . import packs

    try:
        result = packs.import_pack(Path(args.file), override=args.override, override_reason=args.reason,
                                   allow_downgrade=args.allow_downgrade)
    except packs.PackError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if result["status"] == "already_imported":
        print(f"{result['name']} {result['version']} was already imported; nothing changed.")
        return 0
    print(f"Imported {result['name']} {result['version']}: {len(result['documents'])} documents are pending approval"
          f"{', ' + str(len(result['skipped'])) + ' unchanged and skipped' if result['skipped'] else ''}.")
    if result["override_used"]:
        print(f"Override recorded in the audit trail ({result['audit_id']}).")
    if result["formulary_path"]:
        print(f"Formulary staged at {result['formulary_path']} (not active; point FORMULARY_PATH at it to use it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
