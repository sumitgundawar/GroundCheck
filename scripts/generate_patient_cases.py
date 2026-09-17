"""Generate the patient-aware evaluation set (eval/patient_cases.json).

For medicines with an answerable dose question in the golden set, six
scenarios each: a routine adult (answered), and an allergy, an interaction,
a child, pregnancy without safety data, and a patient with no age (all
refused). Hand-written cases cover kidney and liver rules, creatinine
clearance, weight-based and paediatric dosing, and questions that aren't
about giving a medicine. Deterministic:  python scripts/generate_patient_cases.py"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_MEDICINES = 60

ADULT = {"age_years": 45, "sex": "female", "weight_kg": 70, "egfr": 90}

HAND = [
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 72}, "answer", []),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 45}, "refuse", ["renal_dose"]),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 22}, "refuse", ["renal"]),
    ("What is the standard dose of Caloradine?", {"age_years": 82, "sex": "female", "weight_kg": 55,
                                                  "creatinine_umol_l": 160}, "refuse", ["renal"]),
    ("What is the standard dose of Caloradine?", {"age_years": 50, "sex": "male", "weight_kg": 90,
                                                  "creatinine_umol_l": 80}, "answer", []),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75}, "refuse", ["missing_data"]),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 90, "child_pugh": "C"},
     "refuse", ["hepatic"]),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 90, "child_pugh": "B"},
     "answer", []),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 90,
                                                  "allergies": ["Quorl sensitivity"]}, "refuse", ["allergy"]),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 90,
                                                  "conditions": ["known quorl sensitivity"]}, "refuse",
     ["contraindicated_condition"]),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 90, "medicines": ["Tessorin"]},
     "refuse", ["interaction_contraindicated"]),
    ("What is the standard dose of Caloradine?", {"age_years": 60, "weight_kg": 75, "egfr": 90,
                                                  "medicines": ["Mendel solution"]}, "answer", []),
    ("What is the standard dose of Caloradine?", {"age_years": 12, "weight_kg": 40, "egfr": 100}, "refuse",
     ["paediatric_not_covered"]),
    ("What is the standard dose of Caloradine?", {"age_years": 29, "sex": "female", "weight_kg": 60, "egfr": 110,
                                                  "pregnant": True}, "refuse", ["pregnancy"]),
    ("What is the standard dose of Caloradine?", {"age_years": 29, "sex": "female", "weight_kg": 60, "egfr": 110,
                                                  "breastfeeding": True}, "answer", []),
    ("What is Caloradine used for?", {"age_years": 60, "allergies": ["quorl"]}, "answer", []),
    # The answer describes the course without an adult dose, so it's safe, and
    # the child's weight-based dose is added as information.
    ("What is the standard course of Mendel solution?", {"age_years": 8, "weight_kg": 25}, "answer", ["paediatric_dose"]),
    # The answer includes Caloradine too, whose dose can't be checked for a child without kidney function.
    ("What is the dose of Mendel solution?", {"age_years": 8, "weight_kg": 25}, "refuse", ["missing_data"]),
    ("What is the standard course of Mendel solution?", {"age_years": 4, "weight_kg": 16}, "refuse",
     ["paediatric_not_covered"]),
    ("What is the standard course of Mendel solution?", {"age_years": 8}, "refuse", ["missing_data"]),
    ("What is the standard course of Mendel solution?", {"age_years": 34, "sex": "female", "weight_kg": 58,
                                                         "pregnant": True}, "answer", []),
    ("What is the standard course of Mendel solution?", {"age_years": 45, "weight_kg": 80, "medicines": ["an Orrin-blocker"]},
     "refuse", ["interaction_contraindicated"]),
    ("What is the standard course of Mendel solution?", {"age_years": 45, "weight_kg": 80}, "answer", []),
]


def main() -> int:
    golden = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    formulary = json.loads((ROOT / "app" / "data" / "formulary.json").read_text(encoding="utf-8"))
    base = {"caloradine", "mendel solution", "tessorin", "vorantil"}
    cases = [{"query": q, "patient": p, "expect": e, "codes": c} for q, p, e, c in HAND]
    chosen = 0
    for med in formulary["medicines"]:
        name = med["name"]
        if name.lower() in base or not med.get("allergy_groups") or not med.get("interactions") or not med.get("adult_dose"):
            continue
        question = next((c["query"] for c in golden if c["expect"] == "answer" and name.lower() in c["query"].lower()
                         and re.search(r"\b(dose|how much)\b", c["query"], re.I)), None)
        if question is None:
            continue
        blocker = med["interactions"][0]["with_class"]
        member = "a " + re.sub(r"s$", "", blocker)
        cases += [
            {"query": question, "patient": ADULT, "expect": "answer", "codes": []},
            {"query": question, "patient": {**ADULT, "allergies": [med["allergy_groups"][0]]}, "expect": "refuse",
             "codes": ["allergy"]},
            {"query": question, "patient": {**ADULT, "medicines": [member]}, "expect": "refuse",
             "codes": ["interaction_contraindicated"]},
            {"query": question, "patient": {"age_years": 9, "weight_kg": 30}, "expect": "refuse",
             "codes": ["paediatric_not_covered"]},
            {"query": question, "patient": {**ADULT, "age_years": 31, "pregnant": True}, "expect": "refuse",
             "codes": ["pregnancy_no_data"]},
            {"query": question, "patient": {"weight_kg": 70}, "expect": "refuse", "codes": ["missing_data"]},
        ]
        chosen += 1
        if chosen >= MAX_MEDICINES:
            break
    out = ROOT / "eval" / "patient_cases.json"
    out.write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}: {len(cases)} cases for {chosen} generated medicines and {len(HAND)} hand-written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
