"""Formulary adapters: the CSV template, its validation, and that a CSV export
of the demo formulary behaves exactly like the JSON original."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import formulary, pipeline, retrieval  # noqa: E402
from app.formulary_adapters import csv_adapter  # noqa: E402
from app.schemas import PatientContext  # noqa: E402

HEADER = ",".join(csv_adapter.COLUMNS)


def row(**cells) -> str:
    return ",".join(_q(cells.get(c, "")) for c in csv_adapter.COLUMNS)


def _q(v) -> str:
    v = str(v)
    return '"' + v.replace('"', '""') + '"' if any(ch in v for ch in ',"\n') else v


def load(*rows, strict=True, header=HEADER):
    text = "\n".join([header, *rows]) + "\n"
    return csv_adapter.parse_csv(text, formulary_name="T", version="1", synthetic=True, strict=strict)


def errors_of(result):
    return {(e.row, e.column) for e in result.errors}


BASE = row(record_type="medicine", medicine="Alphaquin", adult_dose_amount="10", adult_dose_unit="mg",
           adult_dose_frequency="once daily", adult_dose_max_daily="10")


# --- Round trip ---------------------------------------------------------------------

def test_demo_formulary_survives_csv_round_trip_exactly():
    demo = formulary.load()
    result = csv_adapter.parse_csv(csv_adapter.to_csv(demo), formulary_name=demo.name, version=demo.version,
                                   synthetic=demo.synthetic)
    assert result.ok and result.errors == []
    assert result.formulary.model_dump() == demo.model_dump()


def test_registry_returns_the_csv_adapter():
    assert formulary.get_adapter("csv").name == "csv"
    with pytest.raises(KeyError):
        formulary.get_adapter("nonexistent")


def test_a_csv_formulary_reproduces_every_patient_scenario(tmp_path, monkeypatch):
    """The 383 patient scenarios give the same results with the formulary loaded
    from CSV as with the JSON original: all correct, none unsafe."""
    try:
        retrieval.load_index()
    except FileNotFoundError:
        retrieval.build_index()
    demo = formulary.load()
    from_csv = csv_adapter.parse_csv(csv_adapter.to_csv(demo), formulary_name=demo.name, version=demo.version,
                                     synthetic=True).formulary
    monkeypatch.setattr(formulary, "_index", formulary.FormularyIndex(from_csv))
    cases = json.loads((Path(__file__).parent.parent / "eval" / "patient_cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 383
    wrong, unsafe = [], 0
    for case in cases:
        r = pipeline.run(case["query"], patient=PatientContext(**case["patient"]), review=False)
        codes = {f.code for f in r.patient_findings}
        if not (r.decision == case["expect"] and set(case["codes"]) <= codes):
            wrong.append(case["query"])
        unsafe += case["expect"] == "refuse" and r.decision == "answer"
    assert wrong == [] and unsafe == 0


# --- File-level handling ------------------------------------------------------------

def test_utf8_bom_is_ignored(tmp_path):
    path = tmp_path / "f.csv"
    path.write_bytes(b"\xef\xbb\xbf" + (HEADER + "\n" + BASE + "\n").encode())
    r = csv_adapter.parse_csv(path, formulary_name="T", version="1")
    assert r.ok and r.formulary.medicines[0].name == "Alphaquin"


def test_bom_in_a_string_is_ignored_too():
    r = csv_adapter.parse_csv("﻿" + HEADER + "\n" + BASE + "\n", formulary_name="T", version="1")
    assert r.ok


def test_bytes_are_accepted_and_crlf_line_endings():
    r = csv_adapter.parse_csv((HEADER + "\r\n" + BASE + "\r\n").encode(), formulary_name="T", version="1")
    assert r.ok


def test_empty_file_and_header_only_are_errors():
    assert not csv_adapter.parse_csv("", formulary_name="T", version="1").ok
    r = csv_adapter.parse_csv(HEADER + "\n", formulary_name="T", version="1")
    assert not r.ok and "no medicines" in str(r.errors[0])


def test_missing_file_is_reported_not_raised(tmp_path):
    r = csv_adapter.parse_csv(tmp_path / "nope.csv", formulary_name="T", version="1")
    assert r.formulary is None and r.errors[0].row == 0


def test_non_utf8_bytes_are_reported():
    r = csv_adapter.parse_csv(b"\xff\xfe\x00bad", formulary_name="T", version="1")
    assert r.formulary is None and r.errors


def test_null_bytes_rejected():
    r = csv_adapter.parse_csv(HEADER + "\n" + BASE + "\x00\n", formulary_name="T", version="1")
    assert r.formulary is None


def test_unknown_and_duplicate_and_missing_columns_named():
    r = csv_adapter.parse_csv("record_type,medicine,egfr_belw\nmedicine,X,1\n", formulary_name="T", version="1")
    assert (1, "egfr_belw") in errors_of(r)
    r = csv_adapter.parse_csv("record_type,record_type,medicine\n", formulary_name="T", version="1")
    assert (1, "record_type") in errors_of(r)
    r = csv_adapter.parse_csv("medicine\nX\n", formulary_name="T", version="1")
    assert (1, "record_type") in errors_of(r)


def test_header_case_and_spaces_are_tolerated():
    header = HEADER.replace("record_type", "Record Type").replace("adult_dose_amount", "ADULT DOSE AMOUNT")
    assert load(BASE, header=header).ok


# --- Quoting ------------------------------------------------------------------------

def test_quotes_commas_and_newlines_in_text_fields():
    note = 'Avoid, "strictly", when\neGFR is low'
    r = load(BASE, row(record_type="renal", medicine="Alphaquin", rule_id="R-1", egfr_below="30",
                       action="avoid", note=note))
    assert not r.ok  # no egfr in requires
    ok = row(record_type="medicine", medicine="Betadol", requires="egfr")
    r = load(ok, row(record_type="renal", medicine="Betadol", rule_id="R-1", egfr_below="30", action="avoid",
                     note=note))
    assert r.ok and r.formulary.medicines[0].renal[0].note == note


def test_unquoted_comma_in_a_text_field_is_reported_with_its_row():
    bad = BASE + ",extra,cells," * 30
    r = load(bad)
    assert r.formulary is None and r.errors[0].row == 2 and "comma" in r.errors[0].message


def test_unicode_names_and_notes():
    r = load(row(record_type="medicine", medicine="Zéphyrol", aliases="ゼフィロール;Zephyrol"))
    assert r.ok and "ゼフィロール" in r.formulary.medicines[0].aliases


# --- Bad rows are reported, never silently dropped --------------------------------------

def test_every_bad_row_is_reported_and_strict_mode_returns_nothing():
    r = load(BASE,
             row(record_type="medicine", medicine="Bad1", adult_dose_amount="abc", adult_dose_unit="mg",
                 adult_dose_frequency="daily"),
             row(record_type="medicine", medicine="Bad2", adult_dose_amount="5", adult_dose_unit="stone",
                 adult_dose_frequency="daily"),
             row(record_type="medicine", medicine="Bad3", pregnancy="maybe"),
             row(record_type="nonsense", medicine="Bad4"),
             row(record_type="medicine"))
    assert errors_of(r) == {(3, "adult_dose_amount"), (4, "adult_dose_unit"), (5, "pregnancy"),
                            (6, "record_type"), (7, "medicine")}
    assert r.formulary is None and r.rows_read == 6


def test_lenient_mode_keeps_good_rows_but_still_lists_errors():
    r = load(BASE, row(record_type="medicine", medicine="Bad", adult_dose_amount="-3", adult_dose_unit="mg",
                       adult_dose_frequency="daily"), strict=False)
    assert [m.name for m in r.formulary.medicines] == ["Alphaquin"]
    assert errors_of(r) == {(3, "adult_dose_amount")} and not r.ok


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1e999", "1,5", "1 000", ""])
def test_bad_numbers(value):
    r = load(row(record_type="medicine", medicine="X", adult_dose_amount=value or "x", adult_dose_unit="mg",
                 adult_dose_frequency="daily"))
    assert (2, "adult_dose_amount") in errors_of(r)


def test_partial_dose_names_the_missing_columns():
    r = load(row(record_type="medicine", medicine="X", adult_dose_amount="5"))
    assert {(2, "adult_dose_unit"), (2, "adult_dose_frequency")} <= errors_of(r)


def test_rule_before_its_medicine_or_for_unknown_medicine():
    r = load(row(record_type="interaction", medicine="Ghost", rule_id="I-1", with_class="x", severity="major"))
    assert (2, "medicine") in errors_of(r)


def test_rule_validation_names_column():
    med = row(record_type="medicine", medicine="X", requires="egfr")
    r = load(med,
             row(record_type="renal", medicine="X", rule_id="bad id", egfr_below="30", action="avoid"),
             row(record_type="renal", medicine="X", rule_id="R-2", egfr_below="30", action="delete"),
             row(record_type="renal", medicine="X", rule_id="R-3", action="avoid"),
             row(record_type="hepatic", medicine="X", rule_id="H-1", child_pugh="D", action="avoid"),
             row(record_type="lab", medicine="X", rule_id="L-1", lab="potassium", action="avoid"),
             row(record_type="lab", medicine="X", rule_id="L-2", lab="glucose", above="5", action="avoid"),
             row(record_type="interaction", medicine="X", rule_id="I-1", severity="major"),
             row(record_type="interaction", medicine="X", rule_id="I-2", severity="major", with_class="a",
                 with_medicine="b"))
    assert errors_of(r) == {(3, "rule_id"), (4, "action"), (5, "egfr_below"), (6, "child_pugh"), (7, "above"),
                            (8, "lab"), (9, "with_medicine"), (10, "with_medicine")}


def test_duplicate_rule_id_is_reported():
    med = row(record_type="medicine", medicine="X")
    rule = row(record_type="interaction", medicine="X", rule_id="I-1", severity="major", with_class="a")
    r = load(med, rule, rule)
    assert (4, "rule_id") in errors_of(r)


def test_kidney_rules_without_egfr_requirement_flagged_at_the_medicine_row():
    r = load(row(record_type="medicine", medicine="X"),
             row(record_type="renal", medicine="X", rule_id="R-1", egfr_below="30", action="avoid"))
    assert (2, "requires") in errors_of(r)


def test_unknown_requires_item():
    assert (2, "requires") in errors_of(load(row(record_type="medicine", medicine="X", requires="height")))


def test_bad_yes_no():
    assert (2, "high_alert") in errors_of(load(row(record_type="medicine", medicine="X", high_alert="perhaps")))


def test_negative_age():
    assert (2, "adult_min_age") in errors_of(load(row(record_type="medicine", medicine="X", adult_min_age="-1")))


# --- Duplicates ---------------------------------------------------------------------

def test_duplicate_drug_reported_not_merged():
    r = load(BASE, row(record_type="medicine", medicine="ALPHAQUIN"), row(record_type="medicine",
                                                                         medicine=" alphaquin! "))
    assert {(3, "medicine"), (4, "medicine")} <= errors_of(r)
    assert "row 2" in r.errors[0].message


def test_alias_clashing_with_another_medicine():
    r = load(BASE, row(record_type="medicine", medicine="Other", aliases="Alphaquin"))
    assert (3, "aliases") in errors_of(r)


def test_alias_clashing_between_aliases_is_a_file_error():
    r = load(row(record_type="medicine", medicine="A", aliases="Shared"),
             row(record_type="medicine", medicine="B", aliases="shared"))
    assert r.formulary is None and any(e.row == 0 and "more than one" in e.message for e in r.errors)


# --- Unit normalisation -----------------------------------------------------------------

@pytest.mark.parametrize("raw,unit", [
    ("mg", "mg"), ("MG", "mg"), ("Milligrams", "mg"), (" mg. ", "mg"),
    ("mcg", "mcg"), ("µg", "mcg"), ("μg", "mcg"), ("ug", "mcg"), ("Micrograms", "mcg"),
    ("g", "g"), ("Gram", "g"), ("ml", "mL"), ("ML", "mL"), ("millilitres", "mL"), ("Milliliter", "mL"),
    ("unit", "units"), ("Units", "units"),
])
def test_units_are_normalised(raw, unit):
    r = load(row(record_type="medicine", medicine="X", adult_dose_amount="5", adult_dose_unit=raw,
                 adult_dose_frequency="daily"))
    assert r.ok and r.formulary.medicines[0].adult_dose.unit == unit


@pytest.mark.parametrize("raw", ["kg", "IU", "mEq", "tablet", "%"])
def test_unsupported_units_are_rejected_not_guessed(raw):
    r = load(row(record_type="medicine", medicine="X", adult_dose_amount="5", adult_dose_unit=raw,
                 adult_dose_frequency="daily"))
    assert (2, "adult_dose_unit") in errors_of(r)


def test_paediatric_and_weight_doses_and_flags():
    r = load(row(record_type="medicine", medicine="X", paediatric_per_kg="0.5", paediatric_unit="MG",
                 paediatric_frequency="twice daily", paediatric_max_single="20", paediatric_min_age="1",
                 weight_per_kg="2", weight_unit="mcg", weight_frequency="daily", high_alert="Yes",
                 pregnancy="Avoid", requires="age; weight_kg"))
    m = r.formulary.medicines[0]
    assert (m.paediatric_dose.per_kg, m.paediatric_dose.max_single, m.paediatric_min_age) == (0.5, 20, 1)
    assert m.weight_dose.unit == "mcg" and m.high_alert and m.pregnancy == "avoid"
    assert m.requires == ["age", "weight_kg"]


def test_error_text_names_row_and_column():
    r = load(row(record_type="medicine", medicine="X", adult_dose_amount="x", adult_dose_unit="mg",
                 adult_dose_frequency="d"))
    assert str(r.errors[0]).startswith("row 2, column 'adult_dose_amount'")


def test_large_file_parses():
    rows = [row(record_type="medicine", medicine=f"Drug{i:05d}x", adult_dose_amount="1", adult_dose_unit="mg",
                adult_dose_frequency="daily") for i in range(3000)]
    r = load(*rows)
    assert r.ok and len(r.formulary.medicines) == 3000


def test_export_refuses_a_value_it_cannot_represent():
    f = formulary.Formulary(name="T", version="1", medicines=[formulary.Medicine(name="X", classes=["a;b"])])
    with pytest.raises(ValueError):
        csv_adapter.to_csv(f)
