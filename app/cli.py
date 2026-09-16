"""Command-line administration.

    python -m app.cli migrate
    python -m app.cli create-user --email you@example.org --role admin
    python -m app.cli set-password --email you@example.org
    python -m app.cli list-users
    python -m app.cli purge-sessions

Passwords are always prompted for, never passed as arguments, so they don't
end up in shell history."""

from __future__ import annotations

import argparse
import getpass
import sys

from . import auth, db


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
    args = parser.parse_args(argv)

    try:
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
    except auth.AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
