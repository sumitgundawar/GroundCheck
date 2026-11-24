"""CSV adapter: one flat file of medicines and their rules.

Each row has a `record_type`: `medicine` creates a medicine; `interaction`,
`renal`, `hepatic` and `lab` add a rule to the medicine named in the `medicine`
column (which must appear earlier in the file). The columns are documented in
docs/formulary.md. `to_csv` writes the same format from a Formulary, so a
formulary can be exported, edited in a spreadsheet and loaded back.

Nothing is dropped silently. Every bad cell is reported with its row and
column, and in strict mode (the default) any error means no formulary is
returned, because a formulary quietly missing some of its rules is unsafe."""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import formulary as fm
from ..formulary import AdapterResult, Formulary, Medicine, RowError

MEDICINE_COLUMNS = [
    "aliases", "classes", "high_alert", "source_ids", "allergy_groups", "contraindicated_conditions",
    "adult_dose_amount", "adult_dose_unit", "adult_dose_frequency", "adult_dose_max_daily", "adult_min_age",
    "paediatric_per_kg", "paediatric_unit", "paediatric_frequency", "paediatric_max_single", "paediatric_min_age",
    "weight_per_kg", "weight_unit", "weight_frequency", "weight_max_single",
    "pregnancy", "breastfeeding", "requires", "synthetic",
]
RULE_COLUMNS = [
    "rule_id", "rule_source", "note", "action", "egfr_below", "child_pugh", "with_medicine", "with_class",
    "severity", "lab", "above", "below", "dose_amount", "dose_unit", "dose_frequency", "dose_max_daily",
]
COLUMNS = ["record_type", "medicine", *MEDICINE_COLUMNS, *RULE_COLUMNS]

RECORD_TYPES = ("medicine", "interaction", "renal", "hepatic", "lab")

UNITS = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "mcg": "mcg", "ug": "mcg", "µg": "mcg", "μg": "mcg", "microgram": "mcg", "micrograms": "mcg",
    "g": "g", "gm": "g", "gram": "g", "grams": "g",
    "ml": "mL", "millilitre": "mL", "millilitres": "mL", "milliliter": "mL", "milliliters": "mL",
    "unit": "units", "units": "units",
}
_ID = re.compile(r"^[A-Z0-9][A-Z0-9-]{1,40}$")


def normalise_unit(text: str) -> str | None:
    return UNITS.get(text.strip().rstrip(".").lower())


class _Row:
    def __init__(self, number: int, cells: dict[str, str], errors: list[RowError]):
        self.number, self.cells, self.errors, self.failed = number, cells, errors, False

    def err(self, column: str, message: str) -> None:
        self.errors.append(RowError(self.number, column, message))
        self.failed = True

    def text(self, column: str) -> str:
        return (self.cells.get(column) or "").strip()

    def number_of(self, column: str, *, positive: bool = True) -> float | None:
        raw = self.text(column)
        if not raw:
            return None
        if "," in raw:
            self.err(column, f"{raw!r} contains a comma; write numbers with a decimal point and no thousands separator")
            return None
        try:
            value = float(raw)
        except ValueError:
            self.err(column, f"{raw!r} is not a number")
            return None
        if not math.isfinite(value):
            self.err(column, f"{raw!r} is not a finite number")
            return None
        if positive and value <= 0:
            self.err(column, f"{raw!r} must be greater than zero")
            return None
        return value

    def unit(self, column: str) -> str | None:
        raw = self.text(column)
        if not raw:
            return None
        unit = normalise_unit(raw)
        if unit is None:
            self.err(column, f"unknown unit {raw!r}; use one of mg, mcg, g, mL, units")
        return unit

    def choice(self, column: str, allowed: tuple[str, ...]) -> str | None:
        raw = self.text(column).lower()
        if not raw:
            return None
        if raw not in allowed:
            self.err(column, f"{self.text(column)!r} is not one of {', '.join(allowed)}")
            return None
        return raw

    def flag(self, column: str) -> bool | None:
        raw = self.text(column).lower()
        if not raw:
            return None
        if raw in ("true", "yes", "y", "1"):
            return True
        if raw in ("false", "no", "n", "0"):
            return False
        self.err(column, f"{self.text(column)!r} is not yes or no")
        return None

    def items(self, column: str) -> list[str]:
        return [p.strip() for p in self.text(column).split(";") if p.strip()]

    def dose(self, prefix: str, *, per_kg: str | None = None, max_column: str) -> dict | None:
        """A dose from `<prefix>_amount|unit|frequency`, all or none."""
        amount_col = per_kg or f"{prefix}_amount"
        unit_col, freq_col = f"{prefix}_unit", f"{prefix}_frequency"
        amount = self.number_of(amount_col)
        unit = self.unit(unit_col)
        frequency = self.text(freq_col)
        maximum = self.number_of(max_column)
        given = [bool(self.text(c)) for c in (amount_col, unit_col, freq_col)]
        if not any(given) and maximum is None:
            return None
        for ok, col in zip(given, (amount_col, unit_col, freq_col)):
            if not ok:
                self.err(col, "a dose needs an amount, a unit and a frequency; this one is missing")
        if self.failed:
            return None
        key = "per_kg" if per_kg else "amount"
        out: dict[str, Any] = {key: amount, "unit": unit, "frequency": frequency}
        if maximum is not None:
            out["max_single" if per_kg else "max_daily"] = maximum
        return out


def _cells(row: dict[str, str | None]) -> dict[str, str]:
    return {k: (v or "") for k, v in row.items() if k is not None}


def _header(name: str) -> str:
    return re.sub(r"\s+", "_", (name or "").strip().lower())


def _read_text(source: Any) -> str:
    data: bytes | str = Path(source).read_bytes() if isinstance(source, Path) else source
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig")
    return data.lstrip("﻿")


def _map_pydantic(row: _Row, exc: ValidationError, hints: dict[str, str]) -> None:
    for e in exc.errors():
        field = str(e["loc"][0]) if e["loc"] else ""
        row.err(hints.get(field, field), e["msg"])


MEDICINE_HINTS = {"adult_dose": "adult_dose_amount", "paediatric_dose": "paediatric_per_kg",
                  "weight_dose": "weight_per_kg", "name": "medicine", "requires": "requires"}


def parse_csv(source: Any, *, formulary_name: str, version: str, synthetic: bool = False,
              strict: bool = True) -> AdapterResult:
    errors: list[RowError] = []
    try:
        text = _read_text(source)
    except (OSError, UnicodeDecodeError) as exc:
        return AdapterResult(None, [RowError(0, "", f"couldn't read the file: {exc}")])
    if "\x00" in text:
        return AdapterResult(None, [RowError(0, "", "the file contains null bytes and is not a CSV")])

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header_raw = next(reader)
    except StopIteration:
        return AdapterResult(None, [RowError(0, "", "the file is empty")])
    except csv.Error as exc:
        return AdapterResult(None, [RowError(1, "", f"unreadable header: {exc}")])
    header = [_header(h) for h in header_raw]
    for name in ("record_type", "medicine"):
        if name not in header:
            errors.append(RowError(1, name, "required column is missing"))
    seen_cols: set[str] = set()
    for col in header:
        if col not in COLUMNS:
            errors.append(RowError(1, col, "unknown column; check the spelling against the template"))
        if col in seen_cols:
            errors.append(RowError(1, col, "column appears more than once"))
        seen_cols.add(col)
    if errors:
        return AdapterResult(None, errors)

    medicines: dict[str, dict] = {}   # normalised name -> medicine under construction
    order: list[str] = []
    rule_ids: dict[str, int] = {}
    rows_read = 0
    number = 1
    while True:
        number += 1
        try:
            raw = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            errors.append(RowError(number, "", f"unreadable row: {exc}"))
            continue
        if not any(c.strip() for c in raw):
            continue
        rows_read += 1
        if len(raw) > len(header):
            errors.append(RowError(number, "", f"has {len(raw)} cells but the header has {len(header)}; "
                                               f"an unquoted comma in a text field?"))
            continue
        row = _Row(number, dict(zip(header, raw + [""] * (len(header) - len(raw)))), errors)
        _read_row(row, medicines, order, rule_ids, synthetic)

    built: list[Medicine] = []
    for key in order:
        entry = medicines[key]
        probe = _Row(entry.pop("_row"), {}, errors)
        if entry["renal"] and "egfr" not in entry["requires"]:
            probe.err("requires", f"{entry['name']} has kidney rules but doesn't require egfr")
        if not probe.failed:
            try:
                built.append(Medicine.model_validate(entry))
            except ValidationError as exc:
                _map_pydantic(probe, exc, MEDICINE_HINTS)
    if not built and not errors:
        errors.append(RowError(0, "", "the file has no medicines"))
    try:
        formulary = Formulary(name=formulary_name, version=version, synthetic=synthetic, medicines=built) if built else None
    except ValidationError as exc:
        for e in exc.errors():
            errors.append(RowError(0, "", e["msg"].removeprefix("Value error, ")))
        formulary = None
    if errors and strict:
        formulary = None
    return AdapterResult(formulary, errors, rows_read)


def _read_row(row: _Row, medicines: dict, order: list[str], rule_ids: dict[str, int], synthetic: bool) -> None:
    kind = row.choice("record_type", RECORD_TYPES)
    if kind is None and not row.failed:
        row.err("record_type", "required")
    name = row.text("medicine")
    if not name:
        row.err("medicine", "required")
    if row.failed:
        return
    key = fm.normalise(name)
    if not key:
        row.err("medicine", f"{name!r} has no letters or digits")
        return

    if kind == "medicine":
        if key in medicines:
            row.err("medicine", f"{name!r} is already defined on row {medicines[key]['_row']}; "
                                f"each medicine appears once (put alternative names in `aliases`)")
            return
        entry = _medicine(row, name, synthetic)
        if entry is None:
            return
        for alias in entry["aliases"]:
            if any(fm.normalise(alias) == k for k in medicines):
                row.err("aliases", f"{alias!r} is already the name of another medicine")
                return
        entry["_row"] = row.number
        medicines[key] = entry
        order.append(key)
        return

    entry = medicines.get(key)
    if entry is None:
        entry = next((m for m in medicines.values() if key in {fm.normalise(a) for a in m["aliases"]}), None)
    if entry is None:
        row.err("medicine", f"{name!r} has not been defined; put its `medicine` row before its rules")
        return
    rule = _rule(row, kind)
    if rule is None:
        return
    if rule["id"] in rule_ids:
        row.err("rule_id", f"{rule['id']!r} is already used on row {rule_ids[rule['id']]}")
        return
    rule_ids[rule["id"]] = row.number
    entry[{"interaction": "interactions", "renal": "renal", "hepatic": "hepatic", "lab": "labs"}[kind]].append(rule)


def _medicine(row: _Row, name: str, synthetic: bool) -> dict | None:
    entry: dict[str, Any] = {
        "name": name, "aliases": row.items("aliases"), "classes": row.items("classes"),
        "source_ids": row.items("source_ids"), "allergy_groups": row.items("allergy_groups"),
        "contraindicated_conditions": row.items("contraindicated_conditions"),
        "interactions": [], "renal": [], "hepatic": [], "labs": [],
    }
    high_alert = row.flag("high_alert")
    if high_alert is not None:
        entry["high_alert"] = high_alert
    syn = row.flag("synthetic")
    entry["synthetic"] = synthetic if syn is None else syn
    adult = row.dose("adult_dose", max_column="adult_dose_max_daily")
    if adult:
        entry["adult_dose"] = adult
    paed = row.dose("paediatric", per_kg="paediatric_per_kg", max_column="paediatric_max_single")
    if paed:
        entry["paediatric_dose"] = paed
    weight = row.dose("weight", per_kg="weight_per_kg", max_column="weight_max_single")
    if weight:
        entry["weight_dose"] = weight
    for col, target in (("adult_min_age", "adult_min_age"), ("paediatric_min_age", "paediatric_min_age")):
        value = row.number_of(col, positive=False)
        if value is not None:
            if value < 0:
                row.err(col, "an age can't be negative")
            entry[target] = value
    for col in ("pregnancy", "breastfeeding"):
        value = row.choice(col, ("avoid", "no_data", "compatible"))
        if value:
            entry[col] = value
    requires = [r.lower() for r in row.items("requires")]
    bad = [r for r in requires if r not in ("age", "weight_kg", "egfr", "child_pugh", "pregnant")]
    if bad:
        row.err("requires", f"unknown measurement {bad[0]!r}; use age, weight_kg, egfr, child_pugh, pregnant")
    entry["requires"] = requires
    return None if row.failed else entry


def _rule(row: _Row, kind: str) -> dict | None:
    rule_id = row.text("rule_id").upper()
    if not rule_id:
        row.err("rule_id", "required for rules")
    elif not _ID.match(rule_id):
        row.err("rule_id", f"{rule_id!r} must be capital letters, digits and hyphens, 2 to 41 characters")
    rule: dict[str, Any] = {"id": rule_id, "source": row.text("rule_source")}
    note = row.text("note")

    def need(column: str, value: Any) -> Any:
        if value is None or value == "":
            row.err(column, f"required for {kind} rules")
        return value

    if kind == "interaction":
        with_medicine, with_class = row.text("with_medicine"), row.text("with_class")
        if bool(with_medicine) == bool(with_class):
            row.err("with_medicine", "name exactly one of with_medicine or with_class")
        rule.update(severity=need("severity", row.choice("severity", ("contraindicated", "major", "moderate"))),
                    note=note)
        if with_medicine:
            rule["with_medicine"] = with_medicine
        if with_class:
            rule["with_class"] = with_class
    elif kind == "renal":
        rule.update(egfr_below=need("egfr_below", row.number_of("egfr_below")),
                    action=need("action", row.choice("action", ("avoid", "reduce", "caution"))), note=note)
        dose = row.dose("dose", max_column="dose_max_daily")
        if dose:
            rule["dose"] = dose
    elif kind == "hepatic":
        child = row.text("child_pugh").upper()
        if child not in ("A", "B", "C"):
            row.err("child_pugh", "required for hepatic rules: A, B or C")
        rule.update(child_pugh=child, action=need("action", row.choice("action", ("avoid", "reduce", "caution"))),
                    note=note)
        dose = row.dose("dose", max_column="dose_max_daily")
        if dose:
            rule["dose"] = dose
    else:
        lab = need("lab", row.choice("lab", ("potassium", "sodium", "inr", "platelets", "alt", "qtc")))
        above, below = row.number_of("above", positive=False), row.number_of("below", positive=False)
        if above is None and below is None:
            row.err("above", "a lab rule needs `above`, `below` or both")
        rule.update(lab=lab, action=need("action", row.choice("action", ("avoid", "reduce", "caution"))), note=note)
        if above is not None:
            rule["above"] = above
        if below is not None:
            rule["below"] = below
    return None if row.failed else rule


# --- Export ----------------------------------------------------------------

def _num(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _list(values: list[str]) -> str:
    for v in values:
        if ";" in v:
            raise ValueError(f"{v!r} contains ';', which separates items in a CSV cell")
    return ";".join(values)


def to_csv(formulary: Formulary) -> str:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()

    def dose_cols(prefix: str, dose, per_kg: str | None = None, max_column: str = "") -> dict:
        if dose is None:
            return {}
        amount = dose.per_kg if per_kg else dose.amount
        maximum = dose.max_single if per_kg else dose.max_daily
        return {per_kg or f"{prefix}_amount": _num(amount), f"{prefix}_unit": dose.unit,
                f"{prefix}_frequency": dose.frequency, max_column: _num(maximum)}

    for m in formulary.medicines:
        writer.writerow({
            "record_type": "medicine", "medicine": m.name, "aliases": _list(m.aliases), "classes": _list(m.classes),
            "high_alert": "yes" if m.high_alert else "no", "source_ids": _list(m.source_ids),
            "allergy_groups": _list(m.allergy_groups),
            "contraindicated_conditions": _list(m.contraindicated_conditions),
            **dose_cols("adult_dose", m.adult_dose, max_column="adult_dose_max_daily"),
            "adult_min_age": _num(m.adult_min_age),
            **dose_cols("paediatric", m.paediatric_dose, "paediatric_per_kg", "paediatric_max_single"),
            "paediatric_min_age": _num(m.paediatric_min_age),
            **dose_cols("weight", m.weight_dose, "weight_per_kg", "weight_max_single"),
            "pregnancy": m.pregnancy, "breastfeeding": m.breastfeeding, "requires": _list(m.requires),
            "synthetic": "yes" if m.synthetic else "no",
        })
        base = {"medicine": m.name}
        for r in m.interactions:
            writer.writerow({**base, "record_type": "interaction", "rule_id": r.id, "rule_source": r.source,
                             "with_medicine": r.with_medicine or "", "with_class": r.with_class or "",
                             "severity": r.severity, "note": r.note})
        for r in m.renal:
            writer.writerow({**base, "record_type": "renal", "rule_id": r.id, "rule_source": r.source,
                             "egfr_below": _num(r.egfr_below), "action": r.action, "note": r.note,
                             **dose_cols("dose", r.dose, max_column="dose_max_daily")})
        for r in m.hepatic:
            writer.writerow({**base, "record_type": "hepatic", "rule_id": r.id, "rule_source": r.source,
                             "child_pugh": r.child_pugh, "action": r.action, "note": r.note,
                             **dose_cols("dose", r.dose, max_column="dose_max_daily")})
        for r in m.labs:
            writer.writerow({**base, "record_type": "lab", "rule_id": r.id, "rule_source": r.source,
                             "lab": r.lab, "above": _num(r.above), "below": _num(r.below),
                             "action": r.action, "note": r.note})
    return out.getvalue()


class CsvAdapter:
    name = "csv"

    def load(self, source: Any, *, formulary_name: str, version: str, synthetic: bool = False,
             strict: bool = True) -> AdapterResult:
        return parse_csv(source, formulary_name=formulary_name, version=version, synthetic=synthetic, strict=strict)


def register() -> None:
    fm.register_adapter(CsvAdapter())


register()
