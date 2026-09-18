"""Medicine rules: the structured knowledge patient-aware checks use.

Documents say what a medicine is for and how it's usually given. Checking an
answer against a particular patient needs rules a program can apply: which
allergies and conditions rule a medicine out, which medicines it mustn't be
given with, and how its dose changes with age, weight, kidney and liver
function, pregnancy and breastfeeding.

The demo formulary (app/data/formulary.json) is generated from the synthetic
corpus by scripts/build_formulary.py, with extra illustrative rules for a few
medicines. Every medicine in it is invented. An organisation replaces it with
rules from its own licensed drug database (FORMULARY_PATH), in the same
format, validated when the app starts."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config

Action = Literal["avoid", "reduce", "caution"]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9-]{1,40}$")
    source: str = ""             # a citation: the document or formulary section the rule comes from


class Dose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: float = Field(gt=0)
    unit: Literal["mg", "mcg", "g", "mL", "units"]
    frequency: str
    max_daily: float | None = Field(default=None, gt=0)   # in the same unit


class WeightDose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    per_kg: float = Field(gt=0)
    unit: Literal["mg", "mcg", "g", "mL", "units"]
    frequency: str
    max_single: float | None = Field(default=None, gt=0)


class RenalRule(Rule):
    egfr_below: float = Field(gt=0, le=200)
    action: Action
    dose: Dose | None = None
    note: str = ""


class HepaticRule(Rule):
    child_pugh: Literal["A", "B", "C"]
    action: Action
    dose: Dose | None = None
    note: str = ""


class InteractionRule(Rule):
    with_medicine: str | None = None
    with_class: str | None = None
    severity: Literal["contraindicated", "major", "moderate"]
    note: str = ""

    @model_validator(mode="after")
    def one_target(self):
        if bool(self.with_medicine) == bool(self.with_class):
            raise ValueError("an interaction names exactly one of with_medicine or with_class")
        return self


class LabRule(Rule):
    lab: Literal["potassium", "sodium", "inr", "platelets", "alt", "qtc"]
    above: float | None = None
    below: float | None = None
    action: Action
    note: str = ""


class Medicine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    aliases: list[str] = []
    classes: list[str] = []
    high_alert: bool = False
    source_ids: list[str] = []
    allergy_groups: list[str] = []                 # an allergy to any of these rules the medicine out
    contraindicated_conditions: list[str] = []
    interactions: list[InteractionRule] = []
    adult_dose: Dose | None = None
    adult_min_age: float = 18
    paediatric_dose: WeightDose | None = None      # absent: no paediatric dosing, so refuse for children
    paediatric_min_age: float | None = None
    weight_dose: WeightDose | None = None          # dosed by weight for adults too
    renal: list[RenalRule] = []
    hepatic: list[HepaticRule] = []
    pregnancy: Literal["avoid", "no_data", "compatible"] = "no_data"
    breastfeeding: Literal["avoid", "no_data", "compatible"] = "no_data"
    labs: list[LabRule] = []
    requires: list[Literal["age", "weight_kg", "egfr", "child_pugh", "pregnant"]] = []
    synthetic: bool = True


class Formulary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    synthetic: bool = True
    medicines: list[Medicine]

    @model_validator(mode="after")
    def unique_names(self):
        seen: set[str] = set()
        for m in self.medicines:
            for n in [m.name, *m.aliases]:
                key = normalise(n)
                if key in seen:
                    raise ValueError(f"the name {n!r} is used by more than one medicine")
                seen.add(key)
        return self

    @model_validator(mode="after")
    def rules_can_be_reached(self):
        """A rule that depends on a measurement the medicine never asks for can
        never fire. Liver rules are the exception: an unknown Child-Pugh class
        is warned about rather than demanded, so those rules are still reached.
        Add "child_pugh" to a medicine's `requires` to make it mandatory."""
        for m in self.medicines:
            if m.renal and "egfr" not in m.requires:
                raise ValueError(f"{m.name} has kidney rules but doesn't require egfr in `requires`")
        return self


def normalise(text: str) -> str:
    """Lower case, with anything that isn't a letter or digit as a space.

    A strength written straight onto the name is separated too: electronic
    records hold "Tessorin10mg" as often as "Tessorin 10 mg", and without the
    split the name has no word boundary after it, so the medicine was not
    recognised at all and every rule about it was silently skipped.
    """
    text = re.sub(r"(?<=[a-zA-Z])(?=[0-9])|(?<=[0-9])(?=[a-zA-Z])", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class FormularyIndex:
    def __init__(self, formulary: Formulary):
        self.formulary = formulary
        self.by_name: dict[str, Medicine] = {}
        for m in formulary.medicines:
            for n in [m.name, *m.aliases]:
                self.by_name[normalise(n)] = m
        # Longest names first, so "Mendel solution" wins over a shorter overlap.
        names = sorted(self.by_name, key=len, reverse=True)
        self._pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b") if names else None

    def find(self, name: str) -> Medicine | None:
        return self.by_name.get(normalise(name))

    def mentioned(self, text: str) -> list[Medicine]:
        """Medicines named in free text, in order, without repeats."""
        if not self._pattern:
            return []
        found: list[Medicine] = []
        for m in self._pattern.finditer(normalise(text)):
            med = self.by_name[m.group(1)]
            if med not in found:
                found.append(med)
        return found


_index: FormularyIndex | None = None
_lock = threading.Lock()


def load(path: Path | None = None) -> Formulary:
    target = path or config.FORMULARY_PATH
    return Formulary.model_validate(json.loads(Path(target).read_text(encoding="utf-8")))


def index() -> FormularyIndex:
    global _index
    with _lock:
        if _index is None:
            _index = FormularyIndex(load())
        return _index


def reset() -> None:
    global _index
    with _lock:
        _index = None
