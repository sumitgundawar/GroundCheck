"""Server-side lookup of the reason and stage wording in the dashboard's locale
files (web/i18n), for use by anything that must write them in a language other
than English. Pure: it reads two JSON files and nothing else."""

from __future__ import annotations

import json
import re
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


# --- Structured findings -----------------------------------------------------
# A patient finding is written from its code and params, with every number,
# unit, medicine and condition copied from the params. Anything this cannot
# put in the language whole (a formulary note or frequency it has no wording
# for) gives None and the caller keeps the English.

_NUM = r"\d+(?:\.\d+)?"
_NUMERIC_SLOTS = {"n", "a", "b"}


def _own(key: str, locale: str) -> str | None:
    return _table(locale).get(key)


def _slug(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _g(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, (int, float)):
        return format(float(value), "g")
    return str(value)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@lru_cache(maxsize=None)
def _patterns(prefix: str) -> tuple[tuple[str, re.Pattern, tuple[str, ...]], ...]:
    out = []
    for key, value in _table("en").items():
        if not key.startswith(prefix):
            continue
        names: list[str] = []
        body = ""
        for part in re.split(r"(\{\w+\})", value):
            m = re.fullmatch(r"\{(\w+)\}", part)
            if m:
                names.append(m.group(1))
                body += f"({_NUM})" if m.group(1) in _NUMERIC_SLOTS else "(.+?)"
            else:
                body += re.escape(part)
        out.append((key, re.compile(body), tuple(names)))
    return tuple(sorted(out, key=lambda e: -len(re.sub(r"\{\w+\}", "", _table("en")[e[0]]))))


def note(text: str, locale: str = "en") -> str | None:
    """A formulary rule's note in `locale`, when it has the wording for it."""
    if locale == "en":
        return text
    for key, pattern, names in _patterns("msg.note."):
        m = pattern.fullmatch(text)
        template = _own(key, locale)
        if m and template:
            return _fill(template, dict(zip(names, m.groups())))
    return None


def _fixed() -> list[tuple[str, str]]:
    return sorted(((v, k) for k, v in _table("en").items() if k.startswith("msg.pv.")), key=lambda e: -len(e[0]))


_PHRASES = (
    ("msg.pvp.range", re.compile(rf"({_NUM}) to ({_NUM}) (day|week|month|year)s?")),
    ("unit", re.compile(rf"({_NUM}) (day|week|month|year)s?")),
    ("msg.pvp.times_daily", re.compile(rf"({_NUM}) times daily")),
    ("msg.pvp.every_hours", re.compile(rf"every ({_NUM}) hours?")),
    ("msg.pvp.once_every_weeks", re.compile(rf"once every ({_NUM}) weeks")),
    ("msg.pvp.once_every_days", re.compile(rf"once every ({_NUM}) days")),
    ("quantity", re.compile(r"\d[\d,]*(?:\.\d+)? (?:mg|mcg|g|mL|L|units)\b")),
)


def _unit_word(unit: str, n: str, locale: str) -> str | None:
    return _own(f"srv.unit.{unit}.{'one' if n == '1' else 'other'}", locale)


def phrase(value: str, locale: str = "en") -> str | None:
    """A dose, frequency, duration, route or timing phrase in `locale`: every
    number and unit copied, every wording looked up. None when any part of it is
    not one the table knows."""
    if locale == "en":
        return value
    fixed = _fixed()
    out: list[str] = []
    pos = 0
    text = value.strip()
    while pos < len(text):
        if text[pos] == " ":
            out.append(" ")
            pos += 1
            continue
        for kind, pattern in _PHRASES:
            m = pattern.match(text, pos)
            if not m:
                continue
            if kind == "quantity":
                out.append(m.group(0))
            elif kind == "unit":
                word = _unit_word(m.group(2), m.group(1), locale)
                if word is None:
                    return None
                out.append(f"{m.group(1)} {word}")
            elif kind == "msg.pvp.range":
                word = _unit_word(m.group(3), m.group(2), locale)
                template = _own(kind, locale)
                if word is None or template is None:
                    return None
                out.append(_fill(template, {"a": m.group(1), "b": m.group(2), "unit": word}))
            else:
                template = _own(kind, locale)
                if template is None:
                    return None
                out.append(_fill(template, {"n": m.group(1)}))
            pos = m.end()
            break
        else:
            for english, key in fixed:
                if text.startswith(english, pos) and (pos + len(english) == len(text) or not text[pos + len(english)].isalnum()):
                    template = _own(key, locale)
                    if template is None:
                        return None
                    out.append(template)
                    pos += len(english)
                    break
            else:
                return None
    return "".join(out)


def _word(prefix: str, value: object, locale: str) -> str | None:
    if locale == "en":
        return str(value)
    return _own(f"{prefix}.{_slug(value)}", locale)


def _template(key: str, locale: str) -> str | None:
    return _own(key, locale)


def _subject(params: dict, locale: str) -> tuple[str, str] | None:
    found = _own(f"msg.subject.{params.get('subject') or 'answer'}", locale)
    return None if found is None else (found, _cap(found))


def _where(code: str, params: dict, locale: str) -> str | None:
    if code.startswith("hepatic"):
        template = _own("msg.where.liver", locale)
        return template and _fill(template, {"class": params.get("child_pugh")})
    template = _own("msg.where.egfr" if params.get("measure") == "eGFR" else "msg.where.crcl", locale)
    return template and _fill(template, {"value": _g(params.get("value"))})


def _note_of(params: dict, locale: str) -> str | None:
    return note(params.get("note") or "", locale)


def finding(code: str, params: dict, severity: str = "", locale: str = "en") -> str | None:
    """The sentence for a patient finding, from its structured fields."""
    return _finding(code, params, severity, locale) or None


def _finding(code: str, params: dict, severity: str, locale: str) -> str | None:
    p = dict(params or {})
    medicine = p.get("medicine", "")
    values = {"medicine": medicine}
    sub = _subject(p, locale)
    if sub:
        values.update(subject=sub[0], Subject=sub[1])
    if code.startswith("interaction_"):
        label = p.get("other", "")
        if p.get("note"):
            n = _note_of(p, locale)
            template = _template("msg.find.interaction", locale)
            return n and template and _fill(template, {**values, "other": label, "note": n})
        sev = _word("msg.sev", p.get("severity"), locale)
        template = _template("msg.find.interaction_sev", locale)
        return sev and template and _fill(template, {**values, "other": label, "severity": sev})
    if code == "duplicate_class":
        cls = _word("msg.class", p.get("class"), locale)
        template = _template("msg.find.duplicate_class", locale)
        return cls and template and _fill(template, {**values, "other": p.get("other"), "class": cls})
    if code.startswith("lab_"):
        n = _note_of(p, locale)
        sentence = _own(f"msg.lab.{p.get('lab')}", locale)
        template = _template("msg.find.lab", locale)
        return n and sentence and template and _fill(template, {
            "note": n, "sentence": _fill(sentence, {"value": _g(p.get("value"))})})
    if code in ("renal", "renal_caution", "hepatic", "hepatic_caution"):
        n = _note_of(p, locale)
        where = _where(code, p, locale)
        template = _template("msg.find.rule", locale)
        return n and where and template and _fill(template, {"note": n, "sentence": _cap(where) + "."})
    if code in ("renal_dose", "hepatic_dose"):
        n = _note_of(p, locale)
        where = _where(code, p, locale)
        dose = phrase(p.get("dose", ""), locale)
        template = _template("msg.find.rule_dose", locale)
        if not (n and where and dose and template and sub):
            return None
        high = ""
        if p.get("stated") is not None:
            high_t = _template("msg.find.high", locale)
            if not high_t:
                return None
            high = _fill(high_t, {**values, "stated": _g(p["stated"]), "unit": p.get("unit"), "where": where})
        return _fill(template, {"high": high, "note": n, "dose": dose})
    if code == "paediatric_dose":
        freq = phrase(p.get("frequency", ""), locale)
        cap_t = _template("msg.calc.child_cap", locale)
        calc_t = _template("msg.calc.child", locale)
        template = _template("msg.find.paed_block" if severity == "block" else "msg.find.paed_info", locale)
        if not (freq and cap_t and calc_t and template and sub):
            return None
        cap = _fill(cap_t, {"max_single": _g(p.get("max_single")), "unit": p.get("unit")}) if p.get("max_single") is not None else ""
        calc = _fill(calc_t, {"per_kg": _g(p.get("per_kg")), "unit": p.get("unit"), "weight": _g(p.get("weight")),
                              "uncapped": _g(p.get("uncapped")), "cap": cap})
        return _fill(template, {**values, "amount": _g(p.get("amount")), "unit": p.get("unit"), "frequency": freq, "calc": calc})
    if code in ("weight_dose", "dose_above_maximum"):
        if code == "dose_above_maximum" and "dose" in p:
            dose = phrase(p["dose"], locale)
            template = _template("msg.find.above_adult", locale)
            return dose and template and sub and _fill(template, {
                **values, "stated": _g(p.get("stated")), "unit": p.get("unit"), "dose": dose})
        freq = phrase(p.get("frequency", ""), locale)
        cap_t = _template("msg.calc.weight_cap", locale)
        calc_t = _template("msg.find.weight_dose", locale)
        if not (freq and cap_t and calc_t):
            return None
        cap = _fill(cap_t, {"max_single": _g(p.get("max_single")), "unit": p.get("unit")}) if p.get("max_single") is not None else ""
        calculated = _fill(calc_t, {"per_kg": _g(p.get("per_kg")), "unit": p.get("unit"), "weight": _g(p.get("weight")),
                                    "amount": _g(p.get("amount")), "frequency": freq, "cap": cap})
        if code == "weight_dose":
            return calculated
        template = _template("msg.find.above_weight", locale)
        return template and sub and _fill(template, {**values, "stated": _g(p.get("stated")), "unit": p.get("unit"),
                                                     "calculated": calculated})
    return None


_STATIC = {
    "allergy": ("srv.pat.f_allergy", ("allergy",)),
    "contraindicated_condition": ("srv.pat.f_contra", ("condition",)),
    "already_taking": ("srv.pat.f_already", ()),
    "high_alert": ("srv.pat.f_high_alert", ()),
    "pregnancy": ("srv.pat.f_pregnancy", ()),
    "pregnancy_no_data": ("srv.pat.f_pregnancy_nodata", ()),
    "breastfeeding": ("srv.pat.f_breastfeeding", ()),
    "breastfeeding_no_data": ("srv.pat.f_breastfeeding_nodata", ()),
}


def _static(code: str, params: dict, severity: str, message: str, locale: str) -> str | None:
    values = {"medicine": params.get("medicine", "")}
    if code in _STATIC:
        key, names = _STATIC[code]
        template = _own(key, locale)
        return template and _fill(template, {**values, **{n: params.get(n, "") for n in names}})
    if code == "paediatric_not_covered":
        template = _own("srv.pat.f_paed_none" if severity == "block" else "srv.pat.f_paed_adults", locale)
        return template and _fill(template, {**values, "years": _g(params.get("age"))})
    if code == "missing_data":
        table = _table("en")
        known = {table[k]: k for k in table if k.startswith("srv.item.")}
        words = []
        for item in params.get("missing") or []:
            hit = next((k for en_text, k in known.items() if en_text == item), None)
            found = hit and _own(hit, locale)
            if not found:
                return None
            words.append(found)
        if not words:
            return None
        if message.startswith("To calculate a dose"):
            template = _own("srv.pat.f_missing_weight", locale)
            return template and _fill(template, values)
        template = _own("srv.pat.f_missing", locale)
        return template and _fill(template, {**values, "items": ", ".join(words)})
    return None


def finding_text(f: object, locale: str = "en") -> str | None:
    """A PatientFinding (or its dict) in `locale`: written from its structured
    fields, or None when this has no wording for it (the caller keeps English)."""
    get = f.get if isinstance(f, dict) else lambda k, d=None: getattr(f, k, d)
    message = get("message") or ""
    if locale == "en":
        return message
    params = get("params") or {}
    return (finding(get("code"), params, get("severity") or "", locale)
            or _static(get("code"), params, get("severity") or "", message, locale) or None)


def pick_locale(*candidates: object) -> str:
    """The first language among the candidates (a tag, or an Accept-Language
    value) that has a locale file; English otherwise."""
    have = {p.stem for p in _DIR.glob("*.json")} - {"pseudo"}
    for value in candidates:
        if not isinstance(value, str):
            continue
        ranked = []
        for i, part in enumerate(value.split(",")):
            tag, _, rest = part.strip().partition(";")
            q = re.search(r"q=([0-9.]+)", rest)
            try:
                weight = float(q.group(1)) if q else 1.0
            except ValueError:
                weight = 0.0
            if tag and weight > 0:
                ranked.append((-weight, i, tag.lower().split("-")[0].split("_")[0]))
        for _, _, tag in sorted(ranked):
            if tag in have:
                return tag
    return "en"
