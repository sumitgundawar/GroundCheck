#!/usr/bin/env python3
"""Add held-out mutation families to eval/verify_cases.json.

These cases are written from the corpus passages themselves, with their own
random seed, and are never used to tune the verifier: they are generated, run
and reported before any fix, and carry "split": "heldout". A correct answer is
a passage's own text; a wrong one changes one checkable thing in it.

    python scripts/build_verify_holdout.py

expect: pass      correct; must pass
        fail      wrong; must fail
        review    true but not what was asked; must go to a person
        not_pass  must not pass (a safe route is either fail or review)
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.update({"GROQ_API_KEY": "", "FORCE_EXTRACTIVE": "true", "AUDIT_PERSIST": "false", "REVIEW_QUEUE": "false"})
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import retrieval  # noqa: E402

OUT = ROOT / "eval" / "verify_cases.json"
PER_FAMILY = 70
SEED = 20261201
rng = random.Random(SEED)

ASKS = {
    "Dosage and Administration": "What is the standard dose of {n}?",
    "Adverse Reactions": "What are the side effects of {n}?",
    "Contraindications": "Who should not take {n}?",
    "Drug Interactions": "Can {n} be combined with other medicines?",
    "Indications and Usage": "What is {n} used for?",
}
SIDE = re.compile(r"side effect|adverse|drows|nausea|dizz|appetite|dry mouth|light-headed", re.I)


def medicines() -> list[dict]:
    by_topic: dict[str, dict] = {}
    for r in retrieval.all_metadata():
        if r.get("kind") != "drug":
            continue
        by_topic.setdefault(r["topic"], {})[r["section"]] = r["text"]
    out = [{"name": t[:1].upper() + t[1:], "passages": p} for t, p in sorted(by_topic.items())
           if all(s in p for s in ASKS)]
    rng.shuffle(out)
    return out


def swap_first(text: str, pattern: str, repl: str) -> str | None:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.IGNORECASE)
    return new if n else None


def main() -> None:
    global rng
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", help="write only the new cases here, leaving eval/verify_cases.json alone")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    retrieval.ensure_loaded()
    meds = medicines()
    cases = []

    def add(family, question, answer, expect, **kw):
        cases.append({"id": f"h{len(cases) + 1:04d}", "category": family, "split": "heldout",
                      "question": question, "answer": answer, "expect": expect, **kw})

    def pick(n=PER_FAMILY):
        pool = list(meds)
        rng.shuffle(pool)
        return pool[:n]

    def ask(section, med):
        return ASKS[section].format(n=med["name"])

    for section in ASKS:
        for m in pick(40):
            add("ho_correct_" + section.split()[0].lower(), ask(section, m), m["passages"][section], "pass")

    # --- the same dose written in another unit
    for m in pick(PER_FAMILY * 2):
        text = m["passages"]["Dosage and Administration"]
        mg = re.search(r"(\d+(?:\.\d+)?) mg\b", text)
        mcg = re.search(r"(\d+(?:\.\d+)?) mcg\b", text)
        if mg and float(mg.group(1)) >= 1:
            v = float(mg.group(1))
            as_g = f"{v / 1000:g} g"
            add("ho_unit_equivalent", ask("Dosage and Administration", m), text.replace(mg.group(0), as_g, 1), "pass")
            add("ho_unit_equivalent_wrong", ask("Dosage and Administration", m),
                text.replace(mg.group(0), f"{v / 100:g} g", 1), "fail")
            add("ho_unit_equivalent", ask("Dosage and Administration", m),
                text.replace(mg.group(0), f"{v * 1000:g} mcg", 1), "pass")
            add("ho_unit_equivalent_wrong", ask("Dosage and Administration", m),
                text.replace(mg.group(0), f"{v * 100:g} mcg", 1), "fail")
        if mcg and float(mcg.group(1)) >= 1:
            v = float(mcg.group(1))
            add("ho_unit_equivalent", ask("Dosage and Administration", m),
                text.replace(mcg.group(0), f"{v / 1000:g} mg", 1), "pass")
            add("ho_unit_equivalent_wrong", ask("Dosage and Administration", m),
                text.replace(mcg.group(0), f"{v / 100:g} mg", 1), "fail")
        ml = re.search(r"(\d+(?:\.\d+)?) mL\b", text)
        if ml:
            v = float(ml.group(1))
            add("ho_ml_without_concentration", ask("Dosage and Administration", m),
                text.replace(ml.group(0), f"{v:g} mg", 1), "fail")
            add("ho_ml_without_concentration", ask("Dosage and Administration", m),
                text.replace(ml.group(0), f"{v / 1000:g} L", 1), "pass")

    # --- true, about the right medicine, but not what was asked
    others = list(ASKS)
    for m in pick(PER_FAMILY * 2):
        asked = rng.choice(others)
        given = rng.choice([s for s in others if s != asked])
        text = m["passages"][given]
        if asked == "Adverse Reactions" and SIDE.search(text):
            continue
        if given == "Adverse Reactions" and asked != "Adverse Reactions":
            pass
        add("ho_aspect_mismatch", ask(asked, m), text, "review")

    # --- frequency
    for m in pick(PER_FAMILY):
        text = m["passages"]["Dosage and Administration"]
        q = ask("Dosage and Administration", m)
        for old, news in (("once daily", ["twice daily", "three times daily", "every 6 hours"]),
                          ("twice daily", ["once daily", "three times daily", "four times daily"]),
                          ("every 12 hours", ["every 8 hours", "every 6 hours", "every 24 hours", "once daily"])):
            if old in text:
                add("ho_frequency_change", q, text.replace(old, rng.choice(news), 1), "fail")
                break
        for old, abbrev, expect in (("twice daily", "tds", "fail"), ("twice daily", "qds", "fail"),
                                    ("once daily", "bd", "fail"), ("once daily", "tds", "fail"),
                                    ("twice daily", "bd", "not_pass")):
            if old in text and rng.random() < 0.5:
                add("ho_frequency_abbreviation", q, text.replace(old, abbrev, 1), expect)
                break

    # --- route
    for m in pick(PER_FAMILY):
        text = m["passages"]["Indications and Usage"]
        q = f"What is {m['name']} used for and how is it given?"
        route = re.search(r"\b(oral|subcutaneous|topical)\b", text, re.IGNORECASE)
        if route:
            old = route.group(1).lower()
            new = rng.choice([r for r in ("intravenous", "intramuscular", "oral", "topical", "inhaled",
                                          "subcutaneous", "rectal") if r != old])
            add("ho_route_swap", q, re.sub(old, new, text, count=1, flags=re.IGNORECASE), "fail")
        dose = m["passages"]["Dosage and Administration"]
        add("ho_route_swap", ask("Dosage and Administration", m),
            dose + " " + rng.choice(["It is given intravenously.", "It is given by IV infusion.",
                                     "Give it by intramuscular injection.", "It is applied topically."]), "fail")

    # --- population
    for m in pick(PER_FAMILY):
        text = m["passages"]["Dosage and Administration"]
        q = ask("Dosage and Administration", m)
        band = rng.choice(["paediatric", "child", "elderly", "neonatal", "pregnant"])
        swapped = swap_first(text, r"\badult\b", band)
        if swapped:
            add("ho_age_band_swap", q, swapped, "fail")
        else:
            add("ho_age_band_swap", q, text.replace(" is given as", f" is given to {rng.choice(['children', 'infants', 'older adults'])} as", 1)
                if " is given as" in text else text + " The same dose is used in children.", "fail")

    # --- a limit the source never states
    for m in pick(PER_FAMILY):
        text = m["passages"]["Dosage and Administration"]
        q = ask("Dosage and Administration", m)
        v = re.search(r"(\d+(?:\.\d+)?) (mg|mcg|mL)\b", text)
        if not v:
            continue
        top = float(v.group(1)) * rng.choice([3, 4, 6])
        add("ho_max_dose_exceeded", q, text + rng.choice([
            f" The maximum dose is {top:g} {v.group(2)} per day.",
            f" It can be increased up to {top:g} {v.group(2)} without specialist input.",
            " There is no maximum dose.", " The dose can be doubled if the response is poor."]), "fail")

    # --- negation
    flips = [("Contraindications", r"should not be used", "should be used"),
             ("Contraindications", r"\bAvoid (\w+)", r"Use \1"),
             ("Contraindications", r"is contraindicated", "is recommended"),
             ("Drug Interactions", r"should not be combined", "can be combined"),
             ("Drug Interactions", r"Do not give", "Give"),
             ("Drug Interactions", r"and is avoided", "and is encouraged"),
             ("Dosage and Administration", r"\bis not increased\b", "is increased"),
             ("Dosage and Administration", r"\bis given as\b", "is not given as")]
    for m in pick(PER_FAMILY * 2):
        section, pattern, repl = rng.choice(flips)
        swapped = swap_first(m["passages"][section], pattern, repl)
        if swapped:
            add("ho_negation", ask(section, m), swapped, "fail")
    for m in pick(PER_FAMILY):
        section = rng.choice(["Contraindications", "Drug Interactions", "Indications and Usage"])
        text = m["passages"][section]
        added = swap_first(text, r"\b(is|are) ", r"\1 not ")
        if added:
            add("ho_negation", ask(section, m), added, "fail")

    # --- combinations
    partners = sorted({p for m in meds for p in re.findall(r"\b[A-Z][a-z]+-blockers\b", m["passages"]["Drug Interactions"])})
    for m in pick(PER_FAMILY):
        text = m["passages"]["Drug Interactions"]
        q = f"Can {m['name']} be combined with other medicines?"
        found = re.findall(r"\b[A-Z][a-z]+-blockers\b", text)
        if not found:
            continue
        other = rng.choice([p for p in partners if p != found[0]])
        add("ho_combination", q, text.replace(found[0], other, 1), "fail")
        add("ho_combination", q, f"{m['name']} can safely be combined with {found[0]}.", "fail")
        add("ho_combination_ok", q, text.split(". ")[0].rstrip(".") + ".", "pass")

    if args.out:
        Path(args.out).write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
        print(len(cases), "cases written to", args.out)
        return
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data["cases"] = [c for c in data["cases"] if c.get("split") != "heldout"] + cases
    data["generated_by"] = "scripts/build_verify_cases.py, scripts/build_verify_holdout.py"
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    from collections import Counter
    print(len(cases), "held-out cases", dict(Counter(c["category"] for c in cases)))


if __name__ == "__main__":
    main()
