"""Accounts: passwords, sign-in sessions, two-factor authentication and roles.

- Passwords are hashed with Argon2id (argon2-cffi defaults, per RFC 9106).
- A session is a random 256-bit token in an HttpOnly cookie. The database
  stores only its SHA-256 hash.
- Two-factor authentication uses time-based one-time codes (RFC 6238), the
  kind any authenticator app generates. A user with it enabled gets a session
  marked "pending" at sign-in, which grants nothing until a code is verified.
- Repeated wrong passwords lock the account for a while. Sign-in errors never
  say whether the email exists.

Roles, from least to most access:
  clinician  asks questions
  reviewer   also reads the audit trail
  admin      also manages users and local AI models
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from sqlalchemy import delete, func, select

from . import config, db
from .db import AuthSession, User, utcnow

ROLES = ("clinician", "reviewer", "admin")
_RANK = {role: i for i, role in enumerate(ROLES)}

_hasher = PasswordHasher()
# Verified against when an email doesn't exist, so a wrong email and a wrong
# password take the same time.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MFA_ISSUER = "GroundCheck"


class AuthError(Exception):
    """A sign-in or account operation was refused. The message is safe to show."""


@dataclass(frozen=True)
class Principal:
    """The signed-in user, as the API sees it."""
    id: int
    email: str
    name: str
    role: str
    mfa_enabled: bool

    def can(self, role: str) -> bool:
        return _RANK[self.role] >= _RANK[role]


def _principal(user: User) -> Principal:
    return Principal(id=user.id, email=user.email, name=user.name, role=user.role,
                     mfa_enabled=user.mfa_enabled)


# --- Validation ------------------------------------------------------------

def normalise_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL.match(email) or len(email) > 320:
        raise AuthError("Enter a valid email address.")
    return email


def check_password_strength(password: str, email: str = "") -> None:
    if len(password or "") < config.PASSWORD_MIN_LENGTH:
        raise AuthError(f"Use a password of at least {config.PASSWORD_MIN_LENGTH} characters.")
    local = email.split("@")[0].lower() if email else ""
    if len(local) >= 4 and local in password.lower():
        raise AuthError("Don't include your email name in your password.")


def check_role(role: str) -> str:
    if role not in ROLES:
        raise AuthError(f"Role must be one of: {', '.join(ROLES)}.")
    return role


# --- Users -----------------------------------------------------------------

def create_user(email: str, password: str, role: str = "clinician", name: str = "") -> Principal:
    email = normalise_email(email)
    check_password_strength(password, email)
    check_role(role)
    with db.session() as s:
        if s.scalar(select(User).where(User.email == email)):
            raise AuthError("A user with that email already exists.")
        user = User(email=email, name=name.strip()[:200], password_hash=_hasher.hash(password), role=role)
        s.add(user)
        s.flush()
        return _principal(user)


def list_users() -> list[dict]:
    with db.session() as s:
        users = s.scalars(select(User).order_by(User.created_at)).all()
        return [{
            "id": u.id, "email": u.email, "name": u.name, "role": u.role,
            "is_active": u.is_active, "mfa_enabled": u.mfa_enabled,
            "created_at": u.created_at.isoformat(),
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
        } for u in users]


def count_users() -> int:
    with db.session() as s:
        return s.scalar(select(func.count(User.id))) or 0


def _active_admins(s) -> int:
    return s.scalar(select(func.count(User.id)).where(User.role == "admin", User.is_active.is_(True))) or 0


def update_user(user_id: int, *, role: str | None = None, is_active: bool | None = None,
                name: str | None = None, acting_user_id: int | None = None) -> dict:
    with db.session() as s:
        user = s.get(User, user_id)
        if user is None:
            raise AuthError("No such user.")
        demoting = (role is not None and role != "admin") or is_active is False
        if user.role == "admin" and user.is_active and demoting and _active_admins(s) <= 1:
            raise AuthError("Keep at least one active admin.")
        if acting_user_id == user_id and is_active is False:
            raise AuthError("You can't deactivate your own account.")
        if role is not None:
            user.role = check_role(role)
        if name is not None:
            user.name = name.strip()[:200]
        if is_active is not None:
            user.is_active = is_active
            if not is_active:
                s.execute(delete(AuthSession).where(AuthSession.user_id == user_id))
        return {"id": user.id, "email": user.email, "role": user.role, "is_active": user.is_active}


def set_password(user_id: int, new_password: str, current_password: str | None = None) -> None:
    """Change a password. Users changing their own must give the current one.
    Every other session for the user is signed out."""
    with db.session() as s:
        user = s.get(User, user_id)
        if user is None:
            raise AuthError("No such user.")
        if current_password is not None and not _verify(user.password_hash, current_password):
            raise AuthError("Your current password is incorrect.")
        check_password_strength(new_password, user.email)
        user.password_hash = _hasher.hash(new_password)
        s.execute(delete(AuthSession).where(AuthSession.user_id == user_id))


def _verify(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# --- Sessions --------------------------------------------------------------

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SignIn:
    token: str
    principal: Principal
    mfa_required: bool


def sign_in(email: str, password: str, ip: str = "", user_agent: str = "") -> SignIn:
    """Check credentials and start a session. Raises AuthError with a message
    that never reveals whether the email exists."""
    generic = AuthError("Incorrect email or password.")
    try:
        email = normalise_email(email)
    except AuthError:
        _verify(_DUMMY_HASH, password or "")
        raise generic
    now = utcnow()
    with db.session() as s:
        user = s.scalar(select(User).where(User.email == email))
        if user is None:
            _verify(_DUMMY_HASH, password or "")
            raise generic
        if user.locked_until and user.locked_until > now:
            raise AuthError("Too many failed attempts. Try again later.")
        if not _verify(user.password_hash, password or ""):
            user.failed_logins += 1
            if user.failed_logins >= config.LOGIN_MAX_FAILURES:
                user.locked_until = now + timedelta(minutes=config.LOGIN_LOCKOUT_MINUTES)
                user.failed_logins = 0
            s.commit()
            raise generic
        if not user.is_active:
            raise generic
        if _hasher.check_needs_rehash(user.password_hash):
            user.password_hash = _hasher.hash(password)
        user.failed_logins = 0
        user.locked_until = None
        user.last_login_at = now
        token = secrets.token_urlsafe(32)
        s.add(AuthSession(
            token_hash=_hash_token(token),
            user_id=user.id,
            created_at=now,
            last_seen_at=now,
            expires_at=now + timedelta(hours=config.SESSION_HOURS),
            mfa_pending=user.mfa_enabled,
            ip_address=(ip or "")[:45],
            user_agent=(user_agent or "")[:300],
        ))
        return SignIn(token=token, principal=_principal(user), mfa_required=user.mfa_enabled)


def session_principal(token: str | None, allow_mfa_pending: bool = False) -> Principal | None:
    """The user for a session token, or None if it's missing, expired, pending
    two-factor verification, or the user is inactive."""
    if not token:
        return None
    now = utcnow()
    with db.session() as s:
        record = s.scalar(select(AuthSession).where(AuthSession.token_hash == _hash_token(token)))
        if record is None:
            return None
        if record.expires_at <= now:
            s.delete(record)
            return None
        if record.mfa_pending and not allow_mfa_pending:
            return None
        user = record.user
        if not user.is_active:
            return None
        # Update last-seen at most once a minute, to avoid a write per request.
        if (now - record.last_seen_at).total_seconds() > 60:
            record.last_seen_at = now
        return _principal(user)


def sign_out(token: str | None) -> None:
    if not token:
        return
    with db.session() as s:
        s.execute(delete(AuthSession).where(AuthSession.token_hash == _hash_token(token)))


def purge_expired_sessions() -> int:
    with db.session() as s:
        result = s.execute(delete(AuthSession).where(AuthSession.expires_at <= utcnow()))
        return result.rowcount or 0


# --- Two-factor authentication ---------------------------------------------

def begin_mfa_setup(user_id: int) -> dict:
    """Generate a new secret for the user to add to an authenticator app. It
    takes effect only after confirm_mfa_setup verifies a code from it."""
    secret = pyotp.random_base32()
    with db.session() as s:
        user = s.get(User, user_id)
        if user is None:
            raise AuthError("No such user.")
        if user.mfa_enabled:
            raise AuthError("Two-factor authentication is already on.")
        user.mfa_secret = secret
        uri = pyotp.TOTP(secret).provisioning_uri(name=user.email, issuer_name=MFA_ISSUER)
        return {"secret": secret, "otpauth_uri": uri}


def _code_ok(secret: str | None, code: str) -> bool:
    code = re.sub(r"\s", "", code or "")
    # Allow one 30-second step either side for clock drift.
    return bool(secret) and code.isdigit() and pyotp.TOTP(secret).verify(code, valid_window=1)


def confirm_mfa_setup(user_id: int, code: str) -> None:
    with db.session() as s:
        user = s.get(User, user_id)
        if user is None or not user.mfa_secret:
            raise AuthError("Start two-factor setup first.")
        if not _code_ok(user.mfa_secret, code):
            raise AuthError("That code didn't match. Check your authenticator app and try again.")
        user.mfa_enabled = True


def verify_mfa(token: str, code: str) -> Principal:
    """Complete sign-in for a session waiting on a two-factor code."""
    with db.session() as s:
        record = s.scalar(select(AuthSession).where(AuthSession.token_hash == _hash_token(token)))
        if record is None or record.expires_at <= utcnow() or not record.mfa_pending:
            raise AuthError("Sign in again.")
        user = record.user
        if not _code_ok(user.mfa_secret, code):
            raise AuthError("That code didn't match. Try the latest code from your app.")
        record.mfa_pending = False
        return _principal(user)


def disable_mfa(user_id: int, password: str) -> None:
    with db.session() as s:
        user = s.get(User, user_id)
        if user is None:
            raise AuthError("No such user.")
        if not _verify(user.password_hash, password):
            raise AuthError("Your password is incorrect.")
        user.mfa_enabled = False
        user.mfa_secret = None
