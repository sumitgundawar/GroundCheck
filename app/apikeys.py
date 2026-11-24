"""Service accounts: API keys for machines.

A key looks like `gck_<secret>`. The secret is 256 random bits, shown once when
the key is created. The database keeps only an eight-character prefix, to find
the row, and a salted SHA-256 hash, compared in constant time. A key belongs to
a site (or to none, meaning every site), carries scopes, and can expire or be
revoked; revoking takes effect on the next request.

Scopes:
  ask         POST /api/v1/ask
  verify      POST /api/v1/verify
  usage       GET  /api/v1/usage
  audit:read  GET  /api/v1/audit/{id}
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from . import config, db, guards_input
from .db import ApiKey

SCOPES = ("ask", "verify", "usage", "audit:read")
PREFIX = "gck_"
_TOKEN = re.compile(r"^gck_[A-Za-z0-9_-]{20,128}$")
_DUMMY_SALT = secrets.token_hex(8)
_DUMMY_HASH = hashlib.sha256((_DUMMY_SALT + "gck_" + secrets.token_urlsafe(32)).encode()).hexdigest()


class ApiKeyError(Exception):
    """A key was refused or an operation on keys failed. `code` is stable; the
    message is safe to show."""

    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


@dataclass(frozen=True)
class KeyPrincipal:
    id: int
    prefix: str
    name: str
    site_id: int | None
    scopes: frozenset[str]
    rpm_limit: int
    monthly_quota: int           # 0: no cap

    def has(self, scope: str) -> bool:
        return scope in self.scopes


def _now() -> datetime:
    return db.utcnow()


def _hash(salt: str, key: str) -> str:
    return hashlib.sha256((salt + key).encode("utf-8")).hexdigest()


def _out(k: ApiKey) -> dict:
    return {
        "id": k.id, "prefix": PREFIX + k.prefix, "name": k.name, "site_id": k.site_id,
        "scopes": [x for x in k.scopes.split(",") if x],
        "created_at": k.created_at.isoformat(), "created_by": k.created_by,
        "expires_at": k.expires_at.isoformat() if k.expires_at else None,
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        "revoked_at": k.revoked_at.isoformat() if k.revoked_at else None,
        "rpm_limit": k.rpm_limit, "monthly_quota": k.monthly_quota,
    }


def create(name: str, scopes: list[str], site_id: int | None = None, created_by: int | None = None,
           expires_in_days: float | None = None, rpm_limit: int | None = None,
           monthly_quota: int | None = None) -> dict:
    """Make a key. The returned dict carries the secret under "key"; nothing
    else can ever show it again."""
    from . import sites

    name = (name or "").strip()
    if not name or len(name) > 200 or any(ord(c) < 32 for c in name):
        raise ApiKeyError(422, "validation_error", "Give the key a name of up to 200 characters.")
    scopes = list(dict.fromkeys(scopes or []))
    bad = [x for x in scopes if x not in SCOPES]
    if not scopes or bad:
        raise ApiKeyError(422, "validation_error", f"Scopes must be some of: {', '.join(SCOPES)}.")
    for label, value in (("rpm_limit", rpm_limit), ("monthly_quota", monthly_quota)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1
                                  or value > 10**9):
            raise ApiKeyError(422, "validation_error", f"{label} must be a whole number of at least 1.")
    if expires_in_days is not None and not 0 < expires_in_days <= 3650:
        raise ApiKeyError(422, "validation_error", "expires_in_days must be between 0 and 3650.")
    now = _now()
    for _ in range(5):
        secret = secrets.token_urlsafe(32)
        key = PREFIX + secret
        salt = secrets.token_hex(8)
        try:
            with db.session() as s:
                try:
                    site_id = sites.check(s, site_id)
                except sites.SiteError as exc:
                    raise ApiKeyError(422, "validation_error", str(exc)) from exc
                row = ApiKey(prefix=secret[:8], salt=salt, hash=_hash(salt, key), name=name,
                             site_id=site_id, scopes=",".join(scopes), created_by=created_by,
                             created_at=now, rpm_limit=rpm_limit, monthly_quota=monthly_quota,
                             expires_at=now + timedelta(days=expires_in_days) if expires_in_days else None)
                s.add(row)
                s.flush()
                return {**_out(row), "key": key}
        except IntegrityError:
            continue
    raise ApiKeyError(500, "internal_error", "Couldn't make a key. Try again.")


def list_keys(site_scope: int | None = None) -> list[dict]:
    with db.session() as s:
        q = select(ApiKey).order_by(ApiKey.id)
        if site_scope is not None:
            q = q.where(ApiKey.site_id == site_scope)
        return [_out(k) for k in s.scalars(q)]


def revoke(key_id: int, site_scope: int | None = None) -> dict:
    with db.session() as s:
        row = s.get(ApiKey, key_id)
        if row is None or (site_scope is not None and row.site_id != site_scope):
            raise ApiKeyError(404, "not_found", "No such key.")
        if row.revoked_at is None:
            row.revoked_at = _now()
        return _out(row)


def token_from_headers(authorization: str | None, x_api_key: str | None) -> str | None:
    """The presented key, or None when the request carries no credentials. A
    credential that is present but not a Bearer key is returned as-is so it is
    refused rather than ignored."""
    if authorization is not None and authorization.strip():
        parts = authorization.split(" ")
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        if x_api_key is None:
            return authorization
    if x_api_key is not None and x_api_key.strip():
        return x_api_key.strip()
    return None


def authenticate(token: str | None) -> KeyPrincipal:
    """The key for a presented token, or raises ApiKeyError (401)."""
    if not token:
        raise ApiKeyError(401, "missing_api_key", "Send your API key as 'Authorization: Bearer gck_...'.",
                          {"WWW-Authenticate": "Bearer"})
    invalid = ApiKeyError(401, "invalid_api_key", "The API key is not valid.", {"WWW-Authenticate": "Bearer"})
    well_formed = bool(_TOKEN.match(token))
    row = None
    if well_formed:
        with db.session() as s:
            row = s.scalar(select(ApiKey).where(ApiKey.prefix == token[len(PREFIX):len(PREFIX) + 8]))
            if row is not None:
                s.expunge(row)
    # Hash against something even when there is no such key, so an unknown key
    # and a wrong one take the same time.
    salt, stored = (row.salt, row.hash) if row else (_DUMMY_SALT, _DUMMY_HASH)
    matches = hmac.compare_digest(_hash(salt, token if well_formed else PREFIX), stored)
    if row is None or not matches:
        raise invalid
    now = _now()
    if row.revoked_at is not None:
        raise ApiKeyError(401, "revoked_api_key", "This API key has been revoked.", {"WWW-Authenticate": "Bearer"})
    if row.expires_at is not None and row.expires_at <= now:
        raise ApiKeyError(401, "expired_api_key", "This API key has expired.", {"WWW-Authenticate": "Bearer"})
    if row.last_used_at is None or (now - row.last_used_at).total_seconds() > 60:
        with db.session() as s:
            s.execute(update(ApiKey).where(ApiKey.id == row.id).values(last_used_at=now))
    return KeyPrincipal(
        id=row.id, prefix=PREFIX + row.prefix, name=row.name, site_id=row.site_id,
        scopes=frozenset(x for x in row.scopes.split(",") if x),
        rpm_limit=row.rpm_limit or config.API_KEY_RPM,
        monthly_quota=row.monthly_quota if row.monthly_quota is not None else config.API_KEY_MONTHLY_QUOTA,
    )


# --- Limits ----------------------------------------------------------------

def _mono() -> float:
    return time.monotonic()


def check_rate(key: KeyPrincipal) -> tuple[int, int]:
    """Count a request against the key's per-minute limit. Returns (remaining,
    seconds until the window frees a slot); raises ApiKeyError (429) when over."""
    ok, remaining, wait = guards_input.key_limiter.hit(f"key:{key.id}", key.rpm_limit, _mono())
    if not ok:
        raise ApiKeyError(429, "rate_limited", "Too many requests. Slow down and retry.",
                          {"Retry-After": str(wait), "X-RateLimit-Limit": str(key.rpm_limit),
                           "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(wait)})
    return remaining, wait


def _period(now: datetime) -> str:
    return now.strftime("%Y-%m")


def _seconds_to_next_month(now: datetime) -> int:
    nxt = datetime(now.year + (now.month == 12), now.month % 12 + 1, 1, tzinfo=timezone.utc)
    return max(1, int((nxt - now).total_seconds()))


def reserve_quota(key: KeyPrincipal) -> None:
    """Take one billable request from the key's monthly quota, or raise
    ApiKeyError (429, quota_exceeded). One conditional UPDATE decides, so
    concurrent requests, in this process or another, cannot exceed the cap."""
    if key.monthly_quota <= 0:
        return
    now = _now()
    period = _period(now)
    with db.session() as s:
        s.execute(update(ApiKey).where(ApiKey.id == key.id, ApiKey.quota_period != period)
                  .values(quota_period=period, quota_used=0))
        taken = s.execute(update(ApiKey).where(
            ApiKey.id == key.id, ApiKey.quota_period == period, ApiKey.quota_used < key.monthly_quota,
        ).values(quota_used=ApiKey.quota_used + 1)).rowcount
    if taken != 1:
        wait = _seconds_to_next_month(now)
        raise ApiKeyError(429, "quota_exceeded", "This key has used its monthly quota.",
                          {"Retry-After": str(wait)})


def refund_quota(key: KeyPrincipal) -> None:
    if key.monthly_quota <= 0:
        return
    with db.session() as s:
        s.execute(update(ApiKey).where(ApiKey.id == key.id, ApiKey.quota_period == _period(_now()),
                                       ApiKey.quota_used > 0).values(quota_used=ApiKey.quota_used - 1))


def quota_status(key_id: int) -> dict:
    with db.session() as s:
        row = s.get(ApiKey, key_id)
        if row is None:
            raise ApiKeyError(404, "not_found", "No such key.")
        current = row.quota_used if row.quota_period == _period(_now()) else 0
        return {"period": _period(_now()), "used": current, "limit": row.monthly_quota}
