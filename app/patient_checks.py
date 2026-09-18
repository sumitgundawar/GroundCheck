"""Patient-aware checks: is this answer safe for this patient?

The rest of the pipeline checks an answer against the documents. These checks
apply the formulary's rules (app/formulary.py) to the patient the question
is about, for every formulary medicine named in the question or the answer:

- Allergies and contraindicated conditions
- Interactions with the patient's current medicines, and duplicate classes
- Children: refused where the formulary has no paediatric dosing, and a
  weight-based dose calculated where it does
- Dose adjustments for kidney function (eGFR, or creatinine clearance from
  serum creatinine) and liver function (Child-Pugh class)
- Pregnancy and breastfeeding, lab results, and high-alert medicines
- An answer that states a dose above the patient's maximum
- Required patient data that's missing

Each finding blocks the answer, warns alongside it, or is information. A
question that isn't about giving a medicine (what it's used for, say) turns
blocking findings into warnings, because the answer itself isn't unsafe."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import formulary
from .formulary import Dose, Medicine
from .schemas import Claim, PatientContext, PatientFinding

_PRESCRIBING = re.compile(
    r"\b(dose|doses|dosage|dosing|how much|regimen|mg|mcg|ml|units?|give|given|giving|start|started|starting|"
    r"prescribe|prescribing|administer|take|taking|use|using|treat|treatment|combine|combined|with|safe|increase|"
    r"maximum|max|continue)\b",
    re.IGNORECASE,
)
# Questions about a medicine rather than about giving it.
_INFORMATIONAL = re.compile(
    r"\b(used for|use of|what is \w[\w\s-]* for|what does|indicat\w*|side effects?|adverse|how does|mechanism)\b",
    re.IGNORECASE,
)
_DOSE_QUESTION = re.compile(
    r"\b(dose|doses|dosage|dosing|how much|how often|regimen|course|frequency|mg|mcg|ml|maximum|max)\b", re.IGNORECASE)
_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|mL|ml|g|units)\b")
_UNIT_FACTOR = {"mcg": 0.001, "mg": 1.0, "g": 1000.0}


@dataclass
class Review:
    findings: list[PatientFinding] = field(default_factory=list)
    medicines: list[str] = field(default_factory=list)
    derived: dict = field(default_factory=dict)

    @property
    def blocking(self) -> PatientFinding | None:
        return next((f for f in self.findings if f.severity == "block"), None)

    def add(self, severity, code, medicine, message, rule_id="", source=""):
        self.findings.append(PatientFinding(severity=severity, code=code, medicine=medicine.name,
                                            message=message, rule_id=rule_id, source=source))


def creatinine_clearance(patient: PatientContext) -> float | None:
    """Cockcroft-Gault, in mL/min, from serum creatinine in umol/L."""
    if None in (patient.creatinine_umol_l, patient.age_years, patient.weight_kg) or patient.age_years < 18:
        return None
    value = (140 - patient.age_years) * patient.weight_kg / (0.815 * patient.creatinine_umol_l)
    return round(value * (0.85 if patient.sex == "female" else 1.0), 1)


def _fmt(amount: float) -> str:
    return f"{amount:g}"


def _dose_text(dose: Dose) -> str:
    return f"{_fmt(dose.amount)} {dose.unit} {dose.frequency}"


def _stated_amounts(claims: list[Claim], medicine: Medicine, only_medicine: bool) -> list[tuple[float, str]]:
    """Dose values in the answer that belong to this medicine: every value when
    it's the only medicine involved (a sentence like "the course is 5 mL once
    daily" needn't repeat the name), otherwise values in sentences naming it."""
    names = [formulary.normalise(n) for n in [medicine.name, *medicine.aliases]]
    found = []
    for claim in claims:
        if only_medicine or any(re.search(rf"\b{re.escape(n)}\b", formulary.normalise(claim.text)) for n in names):
            found += [(float(v), u.replace("ml", "mL")) for v, u in _VALUE.findall(claim.text)]
    return found


def _exceeds(amounts: list[tuple[float, str]], limit: float, unit: str) -> float | None:
    for value, stated_unit in amounts:
        if stated_unit == unit and value > limit + 1e-9:
            return value
        if stated_unit in _UNIT_FACTOR and unit in _UNIT_FACTOR:
            if value * _UNIT_FACTOR[stated_unit] > limit * _UNIT_FACTOR[unit] + 1e-9:
                return value
    return None


def _matches(term: str, candidates: list[str]) -> bool:
    t = formulary.normalise(term)
    return any(t and (t == formulary.normalise(c) or re.search(rf"\b{re.escape(formulary.normalise(c))}\b", t)
                      or re.search(rf"\b{re.escape(t)}\b", formulary.normalise(c))) for c in candidates if c)


def _singular(text: str) -> str:
    return re.sub(r"s\b", "", formulary.normalise(text))


def review(query: str, claims: list[Claim], patient: PatientContext | None, subject: str = "answer") -> Review:
    """subject names what's being checked in messages: "answer", or "order" for CDS Hooks."""
    result = Review()
    if patient is None or patient.is_empty():
        return result
    idx = formulary.index()
    answer_text = " ".join(c.text for c in claims)
    medicines = idx.mentioned(f"{query} {answer_text}")
    result.medicines = [m.name for m in medicines]
    if not medicines:
        return result

    prescribing = bool(_PRESCRIBING.search(query)) and not (
        _INFORMATIONAL.search(query) and not _DOSE_QUESTION.search(query))
    dose_question = bool(_DOSE_QUESTION.search(query))
    prescribing = prescribing or dose_question  # asking for a dose is asking to give it
    crcl = creatinine_clearance(patient)
    kidney = patient.egfr if patient.egfr is not None else crcl
    if crcl is not None:
        result.derived["creatinine_clearance"] = crcl
    # EHRs write medicines with strength and form ("Tessorin 10 mg tablet"), so
    # a formulary name inside the text counts.
    current = [(text, idx.find(text) or next(iter(idx.mentioned(text)), None)) for text in patient.medicines]

    for med in medicines:
        stop = "block" if prescribing else "warn"
        amounts = _stated_amounts(claims, med, only_medicine=len(medicines) == 1)

        # Allergies and conditions
        for allergy in patient.allergies:
            if _matches(allergy, [med.name, *med.aliases, *med.classes, *med.allergy_groups]):
                result.add(stop, "allergy", med, f"The patient is allergic to {allergy}, and {med.name} is ruled out "
                           f"for that allergy.", source=", ".join(med.source_ids[:2]))
                break
        for condition in patient.conditions:
            if _matches(condition, med.contraindicated_conditions):
                result.add(stop, "contraindicated_condition", med,
                           f"{med.name} is contraindicated in {condition}.", source=", ".join(med.source_ids[:2]))
                break

        # Interactions and duplicates with current medicines
        for text, other in current:
            if other is med:
                result.add("info", "already_taking", med, f"The patient already takes {med.name}.")
                continue
            other_classes = " | ".join([_singular(c) for c in (other.classes if other else [])] + [_singular(text)])
            for rule in med.interactions:
                hit = (rule.with_medicine and other is not None and formulary.normalise(rule.with_medicine)
                       in {formulary.normalise(n) for n in [other.name, *other.aliases]}) or \
                      (rule.with_class and re.search(rf"\b{re.escape(_singular(rule.with_class))}\b", other_classes))
                if not hit:
                    continue
                label = other.name if other else text
                severity = {"contraindicated": stop, "major": "warn", "moderate": "info"}[rule.severity]
                result.add(severity, f"interaction_{rule.severity}", med,
                           f"{med.name} with {label}: {rule.note or rule.severity}", rule.id, rule.source)
            if other is not None and other is not med and set(map(_singular, other.classes)) & set(map(_singular, med.classes)):
                shared = sorted(set(other.classes) & set(med.classes)) or other.classes
                result.add("warn", "duplicate_class", med,
                           f"The patient already takes {other.name}, which is also one of the {shared[0]}.")

        if med.high_alert:
            result.add("warn", "high_alert", med, f"{med.name} is a high-alert medicine: use an independent double check.")

        # Pregnancy and breastfeeding
        if patient.pregnant:
            if med.pregnancy == "avoid":
                result.add(stop, "pregnancy", med, f"{med.name} should be avoided in pregnancy.", source="Formulary")
            elif med.pregnancy == "no_data":
                result.add(stop, "pregnancy_no_data", med, f"There's no pregnancy safety information for {med.name}.",
                           source="Formulary")
        if patient.breastfeeding:
            if med.breastfeeding == "avoid":
                result.add(stop, "breastfeeding", med, f"{med.name} should be avoided while breastfeeding.")
            elif med.breastfeeding == "no_data":
                result.add("warn", "breastfeeding_no_data", med,
                           f"There's no breastfeeding safety information for {med.name}.")

        # Lab results
        for rule in med.labs:
            value = patient.labs.get(rule.lab)
            if value is None:
                continue
            if (rule.above is not None and value > rule.above) or (rule.below is not None and value < rule.below):
                severity = stop if rule.action == "avoid" else "warn"
                result.add(severity, f"lab_{rule.lab}", med, f"{rule.note} The patient's {rule.lab} is {_fmt(value)}.",
                           rule.id, rule.source)

        if not dose_question and not prescribing:
            # Asking what a medicine is isn't asking to give it, so nothing here
            # blocks an answer. Say when the formulary wouldn't cover this
            # patient all the same, rather than leaving an adult answer looking
            # like it applies to the child or young person in front of them.
            if patient.age_years is not None and patient.age_years < med.adult_min_age and (
                    med.paediatric_dose is None or (med.paediatric_min_age is not None
                                                    and patient.age_years < med.paediatric_min_age)):
                result.add("warn", "paediatric_not_covered", med,
                           f"The formulary has no dosing for {med.name} in a patient aged "
                           f"{_fmt(patient.age_years)}: this answer is about adults.")
            continue

        # Missing data needed before a dose can be given
        needed = set(med.requires) | {"age"}
        if med.weight_dose:
            needed.add("weight_kg")
        missing = []
        if "age" in needed and patient.age_years is None:
            missing.append("age")
        if "weight_kg" in needed and patient.weight_kg is None:
            missing.append("weight")
        if "egfr" in needed and kidney is None:
            missing.append("kidney function (eGFR, or serum creatinine with age and weight)")
        if "child_pugh" in needed and patient.child_pugh is None:
            missing.append("liver function (Child-Pugh class)")
        if missing:
            result.add("block", "missing_data", med,
                       f"To check a dose of {med.name} for this patient, GroundCheck needs their {', '.join(missing)}.")
            continue

        # Children
        adult = patient.age_years >= med.adult_min_age
        if not adult:
            if med.paediatric_dose is None or (med.paediatric_min_age is not None
                                               and patient.age_years < med.paediatric_min_age):
                result.add("block", "paediatric_not_covered", med,
                           f"The formulary has no dosing for {med.name} in a patient aged {_fmt(patient.age_years)}.")
                continue
            if patient.weight_kg is None:
                result.add("block", "missing_data", med,
                           f"To calculate a dose of {med.name} for a child, GroundCheck needs their weight.")
                continue
            dose = med.paediatric_dose
            amount = dose.per_kg * patient.weight_kg
            capped = min(amount, dose.max_single) if dose.max_single else amount
            calc = (f"{_fmt(dose.per_kg)} {dose.unit}/kg × {_fmt(patient.weight_kg)} kg = {_fmt(round(amount, 2))} "
                    f"{dose.unit}{f', capped at {_fmt(dose.max_single)} {dose.unit}' if capped < amount else ''}")
            too_high = _exceeds(amounts, capped, dose.unit)
            if too_high is not None or amounts:
                result.add("block", "paediatric_dose", med,
                           f"The {subject} gives an adult dose. For this child the formulary gives {_fmt(round(capped, 2))} "
                           f"{dose.unit} {dose.frequency} ({calc}).", source="Formulary")
            else:
                result.add("info", "paediatric_dose", med,
                           f"For this child the formulary gives {_fmt(round(capped, 2))} {dose.unit} {dose.frequency} "
                           f"({calc}).", source="Formulary")
            continue

        limit = med.adult_dose
        # Kidney and liver function: the most restrictive rule that applies
        renal = sorted((r for r in med.renal if kidney is not None and kidney < r.egfr_below), key=lambda r: r.egfr_below)
        hepatic_order = {"C": 0, "B": 1, "A": 2}
        hepatic = sorted((r for r in med.hepatic if patient.child_pugh and
                          hepatic_order[patient.child_pugh] <= hepatic_order[r.child_pugh]),
                         key=lambda r: hepatic_order[r.child_pugh])
        measure = "eGFR" if patient.egfr is not None else "creatinine clearance"
        for rule, where in [*[(r, f"{measure} is {_fmt(kidney)}") for r in renal[:1]],
                            *[(r, f"liver function is Child-Pugh class {patient.child_pugh}") for r in hepatic[:1]]]:
            if rule.action == "avoid":
                result.add("block", "renal" if hasattr(rule, "egfr_below") else "hepatic", med,
                           f"{rule.note} The patient's {where}.", rule.id, rule.source)
            elif rule.action == "reduce" and rule.dose:
                limit = rule.dose
                too_high = _exceeds(amounts, rule.dose.amount, rule.dose.unit)
                result.add("block" if too_high is not None else "warn",
                           "renal_dose" if hasattr(rule, "egfr_below") else "hepatic_dose", med,
                           (f"The {subject}'s {_fmt(too_high)} {rule.dose.unit} is too high: the patient's {where}. "
                            if too_high is not None else "") +
                           f"{rule.note} The formulary gives {_dose_text(rule.dose)}.", rule.id, rule.source)
            else:
                result.add("warn", "renal_caution" if hasattr(rule, "egfr_below") else "hepatic_caution", med,
                           f"{rule.note} The patient's {where}.", rule.id, rule.source)

        # Weight-based adult dosing
        if med.weight_dose:
            dose = med.weight_dose
            amount = dose.per_kg * patient.weight_kg
            capped = min(amount, dose.max_single) if dose.max_single else amount
            result.add("info", "weight_dose", med,
                       f"By weight, {_fmt(dose.per_kg)} {dose.unit}/kg × {_fmt(patient.weight_kg)} kg gives "
                       f"{_fmt(round(capped, 2))} {dose.unit} {dose.frequency}"
                       f"{f' (capped at {_fmt(dose.max_single)} {dose.unit})' if capped < amount else ''}.",
                       source="Formulary")
        elif limit and limit.max_daily:
            daily_factor = 2 if limit.frequency in ("twice daily", "every 12 hours") else 1
            too_high = _exceeds(amounts, limit.max_daily / daily_factor, limit.unit)
            if too_high is not None and not any(f.code in ("renal_dose", "hepatic_dose") for f in result.findings
                                                if f.medicine == med.name):
                result.add("block", "dose_above_maximum", med,
                           f"The {subject}'s {_fmt(too_high)} {limit.unit} is above the maximum for {med.name} "
                           f"({_dose_text(limit)}).", source=", ".join(med.source_ids[:1]))

    return result
