"""Server-side lookup of the reason and stage wording in the dashboard's locale
files (web/i18n), for use by anything that must write them in a language other
than English. Pure: it reads two JSON files and nothing else."""

from __future__ import annotations

import json
from functools import lru_cache

from . import config

_DIR = config.ROOT_DIR / "web" / "i18n"


@lru_cache(maxsize=None)
def _table(locale: str) -> dict[str, str]:
    try:
        return json.loads((_DIR / f"{locale}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fill(text: str, values: dict[str, object] | None) -> str:
    for name, value in (values or {}).items():
        text = text.replace("{" + name + "}", str(value))
    return text


def message(key: str, locale: str = "en", **values: object) -> str | None:
    """The text under `key` in `locale`, else English, else None."""
    found = _table(locale).get(key)
    if found is None:
        found = _table("en").get(key)
    return None if found is None else _fill(found, values)


def reason(code: str | None, locale: str = "en") -> str:
    """The sentence for a reason code; the generic one for an unknown code."""
    return (message(f"reason.{code}", locale) if code else None) or message("reason.fallback", locale) or ""


def stage_explanation(stage: str, locale: str = "en") -> str | None:
    return message("srv.explain." + "_".join(stage.lower().replace("+", " ").split()), locale)
