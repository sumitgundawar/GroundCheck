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

Passwords are always prompted for, never passed as arguments, so they don't
end up in shell history."""

from __future__ import annotations

import argparse
import getpass
import sys

from . import auth, db, encryption


def _prompt_password() -> str:
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        raise auth.AuthError("The passwords don't match.")
    return first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="GroundCheck administration")
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
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
