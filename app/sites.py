"""Sites: several hospitals or clinics sharing one installation.

A person belongs to one site, or to none, which means the whole group. What
someone asks, and the review cases, incidents and imaging series that come
from it, belong to their site. People at a site see only their site's; people
with no site see every site's. Approved documents, the formulary, trained
models and releases are shared by the group.

Without any sites, nothing changes: everyone sees everything."""

from __future__ import annotations

import re

from sqlalchemy import select

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


def of_user(user_id: int | None) -> int | None:
    """The site a person belongs to, for tagging what they create."""
    if not user_id or not db.ready():
        return None
    with db.session() as s:
        return s.scalar(select(User.site_id).where(User.id == user_id))


def visible(row_site_id: int | None, scope: int | None) -> bool:
    """Whether someone scoped to `scope` (None: every site) may see a row."""
    return scope is None or row_site_id == scope
