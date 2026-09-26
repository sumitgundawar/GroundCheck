"""Build the demo formulary (app/data/formulary.json) from the synthetic corpus.

Each generated drug label gives a dose and frequency, a sensitivity that rules
the drug out, and a drug class it mustn't be combined with. Those become
structured rules. A few medicines also get illustrative rules for kidney and
liver function, weight-based dosing, children, pregnancy, lab results and
high-alert status, so every patient-aware check has something to test.

Everything here is invented, like the corpus. Run after regenerating the
corpus:  python scripts/build_formulary.py"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.formulary import Formulary  # noqa: E402

DOSE = re.compile(r"(\d+(?:\.\d+)?) (mg|mcg|mL) (once daily|twice daily|every 12 hours|once every morning)")
SENSITIVITY = re.compile(r"sensitiv(?:e|ity) to (\w+)|known (\w+) sensitivity")
CLASS = re.compile(r"\b([A-Z][a-z]+-blockers)\b")
DEMO_SOURCE = "Demo formulary (synthetic)"


def _daily(amount: float, frequency: str) -> float:
    return amount * (2 if frequency in ("twice daily", "every 12 hours") else 1)


def from_corpus(records: list[dict]) -> list[dict]:
    by_topic: dict[str, list[dict]] = {}
    for r in records:
        by_topic.setdefault(r["topic"], []).append(r)
    medicines = []
    for topic, group in by_topic.items():
        dosage = next((r for r in group if r["section"] == "Dosage and Administration"), None)
        if dosage is None or topic in ("caloradine", "mendel solution"):
            continue
        name = dosage["title"].split(":")[0].strip()
        dose = DOSE.search(dosage["text"])
        contra = next((r for r in group if r["section"] == "Contraindications"), None)
        interact = next((r for r in group if r["section"] == "Drug Interactions"), None)
        sensitivity = SENSITIVITY.search(contra["text"]) if contra else None
        blocker = CLASS.search(interact["text"]) if interact else None
        med = {
            "name": name,
            "source_ids": [r["id"] for r in (dosage, contra, interact) if r],
            "allergy_groups": [next(g for g in sensitivity.groups() if g)] if sensitivity else [],
            "interactions": [{"id": f"{dosage['id']}-INT", "with_class": blocker.group(1), "severity": "contraindicated",
                              "source": interact["id"], "note": f"{name} should not be combined with {blocker.group(1)}."}]
            if blocker else [],
        }
        if dose:
            amount, unit, frequency = float(dose.group(1)), dose.group(2), dose.group(3)
            med["adult_dose"] = {"amount": amount, "unit": unit, "frequency": frequency,
                                 "max_daily": _daily(amount, frequency)}
        medicines.append(med)
    return sorted(medicines, key=lambda m: m["name"].lower())


BASE = [
    {
        "name": "Caloradine", "classes": ["quorl modulators"], "source_ids": ["CALO-001", "VELT-002", "INTR-001"],
        "allergy_groups": ["quorl"], "contraindicated_conditions": ["quorl sensitivity"],
        "interactions": [
            {"id": "CALO-INT-1", "with_class": "Orrin-blockers", "severity": "contraindicated", "source": "INTR-001",
             "note": "Orrin-blockers must not be combined with Caloradine."},
            {"id": "CALO-INT-2", "with_medicine": "Mendel solution", "severity": "major", "source": DEMO_SOURCE,
             "note": "Use together only with specialist advice, and review for dizziness."},
        ],
        "adult_dose": {"amount": 15, "unit": "mg", "frequency": "once daily", "max_daily": 15},
        "renal": [
            {"id": "CALO-RENAL-1", "egfr_below": 30, "action": "avoid", "source": DEMO_SOURCE,
             "note": "Avoid when eGFR is below 30."},
            {"id": "CALO-RENAL-2", "egfr_below": 60, "action": "reduce", "source": DEMO_SOURCE,
             "dose": {"amount": 7.5, "unit": "mg", "frequency": "once daily", "max_daily": 7.5},
             "note": "Halve the dose when eGFR is 30 to 59."},
        ],
        "hepatic": [
            {"id": "CALO-HEP-1", "child_pugh": "C", "action": "avoid", "source": DEMO_SOURCE, "note": "Avoid in severe liver disease."},
            {"id": "CALO-HEP-2", "child_pugh": "B", "action": "caution", "source": DEMO_SOURCE, "note": "Monitor closely in moderate liver disease."},
        ],
        "pregnancy": "avoid", "breastfeeding": "no_data", "requires": ["egfr"],
    },
    {
        "name": "Mendel solution", "aliases": ["Mendel"], "classes": ["orrin supplements"], "source_ids": ["MEND-001", "INTR-001"],
        "interactions": [
            {"id": "MEND-INT-1", "with_class": "Orrin-blockers", "severity": "contraindicated", "source": "INTR-001",
             "note": "Orrin-blockers must not be combined with Mendel solution."},
        ],
        "adult_dose": {"amount": 5, "unit": "mL", "frequency": "once daily", "max_daily": 5},
        "paediatric_dose": {"per_kg": 0.1, "unit": "mL", "frequency": "once daily", "max_single": 5},
        "paediatric_min_age": 6,
        "pregnancy": "compatible", "breastfeeding": "compatible",
    },
    {
        "name": "Tessorin", "classes": ["Orrin-blockers"], "source_ids": ["INTR-001"],
        "adult_dose": {"amount": 10, "unit": "mg", "frequency": "twice daily", "max_daily": 20},
        "labs": [{"id": "TESS-LAB-1", "lab": "potassium", "above": 5.5, "action": "avoid", "source": DEMO_SOURCE,
                  "note": "Avoid when potassium is above 5.5 mmol/L."}],
        "pregnancy": "no_data",
    },
    {
        "name": "Vorantil", "classes": ["anticoagulants"], "high_alert": True, "source_ids": [],
        "weight_dose": {"per_kg": 1, "unit": "mg", "frequency": "every 12 hours", "max_single": 100},
        "renal": [{"id": "VORA-RENAL-1", "egfr_below": 30, "action": "avoid", "source": DEMO_SOURCE,
                   "note": "Avoid when eGFR is below 30."}],
        "labs": [{"id": "VORA-LAB-1", "lab": "platelets", "below": 100, "action": "avoid", "source": DEMO_SOURCE,
                  "note": "Avoid when platelets are below 100 x10^9/L."},
                 {"id": "VORA-LAB-2", "lab": "inr", "above": 1.5, "action": "caution", "source": DEMO_SOURCE,
                  "note": "Check with a haematologist when INR is above 1.5."}],
        "interactions": [{"id": "VORA-INT-1", "with_class": "anticoagulants", "severity": "contraindicated", "source": DEMO_SOURCE,
                          "note": "Don't combine two anticoagulants."}],
        "pregnancy": "avoid", "requires": ["weight_kg", "egfr"],
    },
]


def main() -> int:
    records = json.loads((ROOT / "app" / "data" / "corpus.json").read_text(encoding="utf-8"))
    formulary = {"name": "GroundCheckHealth demo formulary", "version": "2026.09", "synthetic": True,
                 "medicines": BASE + from_corpus(records)}
    Formulary.model_validate(formulary)  # fail loudly on any invalid rule
    out = ROOT / "app" / "data" / "formulary.json"
    out.write_text(json.dumps(formulary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}: {len(formulary['medicines'])} medicines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
