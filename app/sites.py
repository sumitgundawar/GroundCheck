"""Sites: several hospitals or clinics sharing one installation.

A person belongs to one site, or to none, which means the whole group. What
someone asks, and the review cases, incidents and imaging series that come
from it, belong to their site. People at a site see only their site's; people
with no site see every site's. Approved documents, the formulary, trained
models and releases are shared by the group.

Without any sites, nothing changes: everyone sees everything."""

from __future__ import annotations

import re
import threading
import time

from sqlalchemy import delete, select

from . import db
from .db import Site, User

_KEY = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


class SiteError(ValueError):
    """The request can't be carried out. The message is safe to show."""


def _out(site: Site) -> dict:
    return {"id": site.id, "key": site.key, "name": site.name}


def list_sites() -> list[dict]:
    with db.session() as s:
        return [_out(x) for x in s.scalars(select(Site).order_by(Site.name))]


def create(key: str, name: str) -> dict:
    key, name = (key or "").strip().lower(), (name or "").strip()
    if not _KEY.match(key):
        raise SiteError("The key is 2 to 63 lower-case letters, digits or hyphens, such as st-marys.")
    if not name:
        raise SiteError("Give the site a name.")
    with db.session() as s:
        if s.scalar(select(Site).where(Site.key == key)):
            raise SiteError("A site with that key already exists.")
        site = Site(key=key, name=name[:200])
        s.add(site)
        s.flush()
        return _out(site)


def rename(site_id: int, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise SiteError("Give the site a name.")
    with db.session() as s:
        site = s.get(Site, site_id)
        if site is None:
            raise LookupError(site_id)
        site.name = name[:200]
        return _out(site)


def by_key(s, key: str) -> Site | None:
    return s.scalar(select(Site).where(Site.key == key.strip().lower()))


def check(s, site_id: int | None) -> int | None:
    if site_id is None:
        return None
    if s.get(Site, int(site_id)) is None:
        raise SiteError("There's no such site.")
    return int(site_id)


# A person's site changes rarely and is read on every question, so it is
# remembered briefly rather than fetched each time. A move between sites takes
# effect within a minute.
_OF_USER_SECONDS = 60
_of_user_cache: dict[int, tuple[float, int | None]] = {}
_of_user_lock = threading.Lock()


def of_user(user_id: int | None) -> int | None:
    """The site a person belongs to, for tagging what they create."""
    if not user_id or not db.ready():
        return None
    now = time.monotonic()
    with _of_user_lock:
        found = _of_user_cache.get(user_id)
        if found and now - found[0] < _OF_USER_SECONDS:
            return found[1]
    with db.session() as s:
        site_id = s.scalar(select(User.site_id).where(User.id == user_id))
    with _of_user_lock:
        if len(_of_user_cache) > 10000:
            _of_user_cache.clear()
        _of_user_cache[user_id] = (now, site_id)
    return site_id


def forget_user(user_id: int | None = None) -> None:
    """Drop what was remembered, after a change of site."""
    with _of_user_lock:
        _of_user_cache.pop(user_id, None) if user_id else _of_user_cache.clear()


def visible(row_site_id: int | None, scope: int | None) -> bool:
    """Whether someone scoped to `scope` (None: every site) may see a row."""
    return scope is None or row_site_id == scope


# --- Provisioning a tenant ---------------------------------------------------------------------

def provision(key: str, name: str, admin_email: str, admin_password: str, *, admin_name: str = "", key_name: str = "",
              scopes: list[str] | None = None, rpm_limit: int | None = None, monthly_quota: int | None = None,
              expires_in_days: float | None = None, created_by: int | None = None) -> dict:
    """A site, its administrator, an API key and a quota, created together. Everything is checked before
    anything is written, and if a later step fails what was already created is removed, so a failed
    attempt leaves nothing behind. The audit trail records the creation (never the password or key).
    The key's secret is returned once, under ["api_key"]["key"]."""
    from . import apikeys, auth, integrity
    from .db import ApiKey

    key = (key or "").strip().lower()
    if not _KEY.match(key):
        raise SiteError("The key is 2 to 63 lower-case letters, digits or hyphens, such as st-marys.")
    if not (name or "").strip():
        raise SiteError("Give the site a name.")
    try:
        email = auth.normalise_email(admin_email)
        auth.check_password_strength(admin_password, email)
    except auth.AuthError as exc:
        raise SiteError(str(exc)) from exc
    with db.session() as s:
        if by_key(s, key) is not None:
            raise SiteError("A site with that key already exists.")
        if s.scalar(select(User).where(User.email == email)):
            raise SiteError("A user with that email already exists.")
    scopes = list(scopes) if scopes else ["ask", "verify"]
    site = admin = made = None
    try:
        site = create(key, name)
        admin = auth.create_user(email, admin_password, role="admin", name=admin_name, site_id=site["id"])
        made = apikeys.create(key_name or f"{key} service key", scopes, site_id=site["id"], created_by=admin.id,
                              expires_in_days=expires_in_days, rpm_limit=rpm_limit, monthly_quota=monthly_quota)
        integrity.record_event("tenant.created", {"site_id": site["id"], "site_key": key, "admin_id": admin.id,
                                                   "api_key_id": made["id"], "scopes": made["scopes"],
                                                   "rpm_limit": rpm_limit, "monthly_quota": monthly_quota}, created_by)
    except Exception as exc:
        with db.session() as s:
            if made is not None:
                s.execute(delete(ApiKey).where(ApiKey.id == made["id"]))
            if admin is not None:
                s.execute(delete(User).where(User.id == admin.id))
            if site is not None:
                s.execute(delete(Site).where(Site.id == site["id"]))
        if isinstance(exc, apikeys.ApiKeyError):
            raise SiteError(exc.message) from exc
        raise
    return {"site": site, "admin": {"id": admin.id, "email": admin.email, "role": admin.role}, "api_key": made}
