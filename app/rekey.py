"""Re-encrypting stored data after a key change.

`reencrypt()` rewrites every encrypted column that isn't already encrypted
with the primary key: values under an older key, values written before
encryption was turned on, and, when encryption has been turned off, values
that are still encrypted. Lookup hashes for repeated questions are
recomputed too. The audit chain covers plain text, so it still verifies.

Run it with `python -m app.cli reencrypt` after changing DATA_ENCRYPTION_KEYS,
and back up the database first."""

from __future__ import annotations

import json
import re

from sqlalchemy import Text, select, type_coerce
from sqlalchemy.orm.attributes import flag_modified

from . import db, encryption
from .db import EncryptedJSON, EncryptedText, ReviewCase


def _encrypted_models() -> list[tuple[type, list[str]]]:
    found = []
    for mapper in db.Base.registry.mappers:
        columns = [c.key for c in mapper.columns if isinstance(c.type, (EncryptedText, EncryptedJSON))]
        if columns:
            found.append((mapper.class_, columns))
    return sorted(found, key=lambda item: item[0].__tablename__)


def _needs_rewrite(ring: encryption.Keyring, value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.startswith('"'):
        # An encrypted JSON document is stored as a JSON string.
        try:
            value = json.loads(value)
        except ValueError:
            pass
    if ring.enabled:
        return not (isinstance(value, str) and ring.is_current(value))
    return encryption.is_encrypted(value)


def reencrypt(batch_size: int = 500) -> dict:
    encryption.reset()
    ring = encryption.keyring()
    counts: dict[str, int] = {}
    for model, columns in _encrypted_models():
        table = model.__table__
        rewritten, last_id = 0, 0
        while True:
            with db.engine().connect() as conn:
                # Read the stored values as they are, without decrypting.
                raw = conn.execute(
                    select(table.c.id, *[type_coerce(table.c[c], Text()).label(c) for c in columns])
                    .where(table.c.id > last_id).order_by(table.c.id).limit(batch_size)).all()
            if not raw:
                break
            last_id = raw[-1].id
            stale = [r.id for r in raw if any(_needs_rewrite(ring, getattr(r, c)) for c in columns)]
            if model is ReviewCase:
                stale = [r.id for r in raw]  # lookup hashes may use an old key
            if stale:
                with db.session() as s:
                    for obj in s.scalars(select(model).where(model.id.in_(stale))):
                        for c in columns:
                            flag_modified(obj, c)
                        if isinstance(obj, ReviewCase):
                            obj.query_key = ring.lookup_hash(re.sub(r"\s+", " ", obj.query.strip().lower()))
                rewritten += len(stale)
        counts[table.name] = rewritten
    from .imaging import store

    counts["imaging files"] = store.reencrypt_files()
    return {"encryption": ring.summary(), "rewritten": counts}
