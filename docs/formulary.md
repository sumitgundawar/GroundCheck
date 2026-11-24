# Loading a formulary from an external source

Patient-aware checks use a validated formulary (`app/data/formulary.json`, or the file named by
`FORMULARY_PATH`). Adapters turn an external source into that format. Every problem is reported with
the row and column it came from, and nothing is dropped silently.

## Adapter interface

An adapter is an object with a `name` and a `load` method (see `FormularyAdapter` in `app/formulary.py`):

```python
result = formulary.get_adapter("csv").load(path, formulary_name="Acme formulary", version="2025.1")
result.formulary   # a validated Formulary, or None
result.errors      # list of RowError(row, column, message)
result.ok          # True only when there is a formulary and no errors
```

`strict=True` (the default) returns no formulary if there is any error: a formulary that is quietly
missing some of its rules is unsafe. `strict=False` returns the rows that were valid and still lists
every error, which is useful for fixing a large file.

Register another adapter with `formulary.register_adapter(adapter)`.

## CSV template

One UTF-8 file (a byte-order mark is fine), comma separated, standard CSV quoting. The first row is
the header; column names are case-insensitive and spaces count as underscores. Unknown columns are an
error, so a misspelt column cannot silently disable a rule. Row 1 is the header, so the first
medicine is row 2.

`record_type` is one of `medicine`, `interaction`, `renal`, `hepatic`, `lab`. A `medicine` row defines a
medicine; the other rows add a rule to the medicine named in the `medicine` column, and must come after it.
Each medicine appears once. Put other names in `aliases`.

Lists (`aliases`, `classes`, `source_ids`, `allergy_groups`, `contraindicated_conditions`, `requires`) are
separated by `;`.

Medicine rows:

| Column | Meaning |
| --- | --- |
| `high_alert` | yes or no |
| `adult_dose_amount`, `adult_dose_unit`, `adult_dose_frequency`, `adult_dose_max_daily` | usual adult dose; amount, unit and frequency go together |
| `adult_min_age` | default 18 |
| `paediatric_per_kg`, `paediatric_unit`, `paediatric_frequency`, `paediatric_max_single`, `paediatric_min_age` | dose per kg for children; leave empty for no paediatric dosing, so children are refused |
| `weight_per_kg`, `weight_unit`, `weight_frequency`, `weight_max_single` | dosed by weight for adults too |
| `pregnancy`, `breastfeeding` | `avoid`, `no_data` or `compatible` |
| `requires` | measurements needed: `age`, `weight_kg`, `egfr`, `child_pugh`, `pregnant` |
| `synthetic` | yes or no; defaults to the loader's `synthetic` argument |

Rule rows share `rule_id` (capitals, digits, hyphens; unique in the file), `rule_source` (a citation) and `note`:

| record_type | Columns |
| --- | --- |
| `interaction` | `with_medicine` or `with_class` (exactly one), `severity` (`contraindicated`, `major`, `moderate`) |
| `renal` | `egfr_below`, `action` (`avoid`, `reduce`, `caution`), optional `dose_*`. The medicine must list `egfr` in `requires`. |
| `hepatic` | `child_pugh` (A, B, C), `action`, optional `dose_*` |
| `lab` | `lab` (`potassium`, `sodium`, `inr`, `platelets`, `alt`, `qtc`), `above` and/or `below`, `action` |

`dose_*` columns are `dose_amount`, `dose_unit`, `dose_frequency`, `dose_max_daily`.

## Numbers and units

Write numbers with a decimal point and no thousands separator (`1.5`, not `1,5` or `1,000`). Amounts must
be greater than zero. Units are normalised: `mg`, `milligram(s)` give `mg`; `mcg`, `ug`, `µg`, `microgram(s)`
give `mcg`; `g`, `gram(s)` give `g`; `ml`, `millilitre(s)`, `milliliter(s)` give `mL`; `unit(s)` gives `units`.
Any other unit (for example `IU`, `mEq`, `%`) is rejected rather than guessed, because a wrong unit changes a dose.

## Round trip

`csv_adapter.to_csv(formulary)` writes the same format from any formulary. The demo formulary exported
this way and loaded again is identical, and gives the same results on all 383 patient scenarios (all
correct, none unsafe). This is checked by `tests/test_formulary_adapter.py`.

Using the result: write `result.formulary.model_dump_json()` to a file and point `FORMULARY_PATH` at it.
