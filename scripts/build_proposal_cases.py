"""Build eval/proposal_cases.json: questions that propose a regimen, labelled
from the corpus's own generated records.

The labels come from the corpus template (a dose, a frequency, a timing and a
duration in fixed wording), read here with its own small patterns and rules,
not with app/proposal.py. Two families: `tuned` (phrasings and drugs the
parser was built against) and `heldout` (different drugs, different wording,
not looked at while the parser was written).

    python scripts/build_proposal_cases.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "proposal_cases.json"

RATE = {"once daily": 1, "twice daily": 2, "every 12 hours": 2, "once every morning": 1}
TIME = {"in the morning with food": {"morning", "with food"}, "after the evening meal": {"evening", "after food"},
        "twice daily with water": set(), "in the morning before food": {"morning", "before food"}}
WORDS = {2.5: "two point five", 5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 40: "forty",
         50: "fifty", 75: "seventy five", 100: "one hundred"}
DAYWORDS = {7: "seven", 10: "ten", 14: "fourteen", 21: "twenty one", 28: "twenty eight"}

FREQ_FORMS_TUNED = {
    1: ["once daily", "once a day", "daily", "od", "every 24 hours", "once every day"],
    2: ["twice daily", "twice a day", "two times a day", "bd", "bid", "every 12 hours", "q12h", "2x daily"],
    3: ["three times a day", "three times daily", "tds", "tid", "every 8 hours", "q8h", "3 times a day"],
    4: ["four times a day", "qid", "qds", "every 6 hours", "q6h", "4x daily"],
}
FREQ_FORMS_HELD = {
    1: ["one time a day", "every 24 hrs", "once per day", "qd", "1x/day"],
    2: ["2 times daily", "twice per day", "every twelve hours", "bds", "2x/day", "q.12.h"],
    3: ["thrice daily", "every eight hours", "three times per day", "3x a day", "t.d.s"],
    4: ["four times per day", "every six hours", "4 times a day", "q.i.d"],
}
ROUTE_FORMS = {"oral": ["orally", "by mouth", "as an oral dose"], "topical": ["topically", "on the skin"],
               "subcutaneous": ["subcutaneously", "under the skin", "by subcutaneous injection"],
               "intravenous": ["intravenously", "by IV infusion"]}
ROUTE_FORMS_HELD = {"oral": ["via the oral route", "po"], "topical": ["as a topical application"],
                    "subcutaneous": ["subcut", "sc"], "intravenous": ["iv", "by intravenous drip"],
                    "intramuscular": ["intramuscularly", "IM"]}

TEMPLATES_TUNED = [
    "Is {p} okay for {d}?", "Can I give {d} {p}?", "Would {p} be fine for {d}?", "Is it safe to give {d} {p}?",
    "{d} {p}, correct?", "Should {d} be given {p}?", "Is {p} acceptable for {d}?", "Is {d} {p} ok?",
    "Could I give {d} {p}?", "Is it alright to give {d} {p}?",
]
TEMPLATES_HELD = [
    "My colleague suggests {d} {p}. Is that right?", "Am I allowed to give {d} {p}?",
    "Quick question: is {d} {p} appropriate?", "Is giving {d} {p} recommended?", "What if I give {d} {p}?",
    "Do you agree that {d} should be {p}?", "{p} for {d}: is that fine?", "Would it be ok to prescribe {d} {p}?",
]
NEGATED = ["Isn't {p} fine for {d}?", "Is it not okay to give {d} {p}?", "Shouldn't {d} be given {p}?",
           "Can't I give {d} {p}?"]
NEGATED_HELD = ["Wouldn't {d} {p} be ok?", "Isn't it correct to give {d} {p}?"]
INVERTED = ["Is {p} too much for {d}?", "Is it dangerous to give {d} {p}?", "Is {p} of {d} an overdose?",
            "Would {p} be harmful for {d}?"]
INFO = ["What is the standard dose of {d}?", "How often should {d} be given?", "What is the usual course length of {d}?",
        "How is {d} given?", "Tell me about the dosage of {d}."]


def num(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)


def drugs() -> list[dict]:
    records = json.loads((ROOT / "index" / "metadata.json").read_text())
    by_topic: dict[str, list[dict]] = {}
    for r in records:
        if r.get("kind") == "drug":
            by_topic.setdefault(r["topic"], []).append(r)
    out = []
    for topic, recs in by_topic.items():
        text = " ".join(r["text"] for r in recs if r["section"] == "Dosage and Administration")
        m = re.search(r"(\d+(?:\.\d+)?) (mg|mcg|mL) (once daily|twice daily|every 12 hours|once every morning), "
                      r"(?:taken )?(.+?), for (?:an initial course of )?(\d+) days", text)
        route = re.search(r"\b(oral|topical|subcutaneous)\b", " ".join(r["text"] for r in recs if r["section"] == "Indications and Usage"))
        if not m or not route:
            continue
        dose, unit, freq, timing, days = float(m.group(1)), m.group(2), m.group(3), m.group(4), int(m.group(5))
        rate = RATE[freq]
        conflict = timing == "twice daily with water" and rate != 2
        tod = set(TIME[timing]) | ({"morning"} if freq == "once every morning" else set())
        out.append({"name": topic.title(), "topic": topic, "dose": dose, "unit": unit, "rate": None if conflict else rate,
                    "days": days, "route": route.group(1), "tod": tod, "has_tod": bool(tod), "conflict": conflict})
    return sorted(out, key=lambda d: d["topic"])


def dose_forms(d: dict, value: float, unit: str, held: bool) -> list[str]:
    forms = [f"{num(value)} {unit}", f"{num(value)}{unit}"]
    if value in WORDS and not held:
        forms.append(f"{WORDS[value]} {'mg' if unit == 'mg' else unit}")
    if unit == "mg":
        forms.append(f"{num(round(value / 1000, 6))} g")
        if held:
            forms.append(f"{num(round(value * 1000, 6))} mcg")
    if unit == "mcg":
        forms.append(f"{num(round(value / 1000, 6))} mg")
    if unit == "mL" and held:
        forms.append(f"{num(round(value / 1000, 6))} L")
    if held and value != int(value):
        forms.append(f"{num(value).replace('.', ',')} {unit}")
    return forms


def other_dose(d: dict, rng: random.Random) -> tuple[float, str]:
    choices = [d["dose"] * 2, d["dose"] / 2, d["dose"] + 5, d["dose"] * 4, d["dose"] + 1]
    value = rng.choice(choices)
    return round(value, 3), d["unit"]


def days_forms(days: int, held: bool) -> list[str]:
    forms = [f"{days} days", f"{days}-day course"]
    if days % 7 == 0:
        forms.append(f"{days // 7} weeks" if days > 7 else "a week")
        forms.append(f"{days // 7} week" if days == 7 else f"{days // 7} weeks")
    if not held and days in DAYWORDS:
        forms.append(f"{DAYWORDS[days]} days")
    return forms


def build(held: bool, drug_pool: list[dict], rng: random.Random, per_drug: int) -> list[dict]:
    templates = TEMPLATES_HELD if held else TEMPLATES_TUNED
    freqs = FREQ_FORMS_HELD if held else FREQ_FORMS_TUNED
    negated = NEGATED_HELD if held else NEGATED
    routes = ROUTE_FORMS_HELD if held else ROUTE_FORMS
    fam = "heldout" if held else "tuned"
    cases: list[dict] = []

    def add(kind, drug, parts, status, d, patient=None, dims=None, note=""):
        template = rng.choice(templates)
        text = " ".join(parts)
        query = template.format(p=text, d=drug)
        cases.append({"family": fam, "kind": kind, "query": query, "expect": status, "dims": dims or [], "patient": patient,
                      "drug": d["topic"], "note": note})

    for d in drug_pool:
        name = d["name"]
        for _ in range(per_drug):
            kind = rng.choice(["dose", "freq", "route", "duration", "max", "dose+freq", "dose+freq+duration", "timing",
                               "dose+route", "all"])
            dose_ok = rng.random() < 0.5
            freq_ok = rng.random() < 0.5
            dur_ok = rng.random() < 0.5
            route_ok = rng.random() < 0.5
            parts: list[str] = []
            statuses: list[str] = []
            dims: list[str] = []

            def dose_part():
                if dose_ok:
                    v, u = d["dose"], d["unit"]
                    form = rng.choice(dose_forms(d, v, u, held))
                else:
                    v, u = other_dose(d, rng)
                    form = f"{num(v)} {u}"
                parts.append(form)
                statuses.append("match" if dose_ok else "mismatch")
                dims.append("dose")

            def freq_part():
                if d["rate"] is None:
                    rate = rng.choice([1, 2, 3])
                    statuses.append("unverifiable")
                elif freq_ok:
                    rate = d["rate"]
                    statuses.append("match")
                else:
                    rate = rng.choice([r for r in (1, 2, 3, 4) if r != d["rate"]])
                    statuses.append("mismatch")
                parts.append(rng.choice(freqs[rate]))
                dims.append("frequency")

            def dur_part():
                if dur_ok:
                    days = d["days"]
                else:
                    days = rng.choice([x for x in (3, 5, 7, 10, 14, 21, 28, 30, 42) if x != d["days"]])
                form = rng.choice(days_forms(days, held)) if days % 7 == 0 or rng.random() < 0.8 else f"{days} days"
                parts.append("for " + form if "course" not in form else "as a " + form)
                statuses.append("match" if dur_ok else "mismatch")
                dims.append("duration")

            def route_part():
                if route_ok:
                    r = d["route"]
                else:
                    r = rng.choice([x for x in routes if x != d["route"]])
                parts.append(rng.choice(routes[r]))
                statuses.append("match" if route_ok else "mismatch")
                dims.append("route")

            if kind == "dose":
                dose_part()
            elif kind == "freq":
                freq_part()
            elif kind == "route":
                route_part()
            elif kind == "duration":
                dur_part()
            elif kind == "max":
                dims.append("max_daily")
                # No source states a maximum daily dose; the usual regimen is not one, so every maximum is unverifiable.
                status = "unverifiable"
                if d["rate"] is None:
                    total = d["dose"]
                elif dose_ok:
                    total = d["dose"] * d["rate"]
                else:
                    total = round(d["dose"] * d["rate"] * rng.choice([2, 0.5, 3]), 3)
                word = rng.choice(["a maximum of {x} a day", "no more than {x} per day", "up to {x} daily",
                                   "a daily maximum of {x}", "{x} a day at most", "a total of {x} per day at most"]
                                  if not held else ["a max of {x} per day", "not more than {x} a day",
                                                    "a maximum daily dose of {x}", "{x}/day maximum"])
                if word == "{x}/day maximum":
                    word = "{x} per day maximum"
                parts.append(word.format(x=f"{num(total)} {d['unit']}"))
                statuses.append(status)
            elif kind == "dose+freq":
                dose_part()
                freq_part()
            elif kind == "dose+freq+duration":
                dose_part()
                freq_part()
                dur_part()
            elif kind == "dose+route":
                dose_part()
                route_part()
            elif kind == "timing":
                dims.append("timing")
                options = [("in the morning", "morning"), ("in the evening", "evening"), ("with food", "with food"),
                           ("before food", "before food"), ("after the evening meal", "after food")]
                text, cls = rng.choice(options)
                times = {c for c in d["tod"] if c in ("morning", "evening")}
                if cls in ("morning", "evening"):
                    status = ("unverifiable" if len(times) != 1 else "match" if cls in times else "mismatch")
                else:
                    food = {c for c in d["tod"] if c in ("with food", "before food", "after food")}
                    status = "unverifiable" if not food else ("match" if cls in food else "mismatch")
                    if text == "after the evening meal":
                        status = ("unverifiable" if not food or len(times) != 1
                                  else "match" if ("after food" in d["tod"] and "evening" in d["tod"]) else "mismatch")
                parts.append(text)
                statuses.append(status)
            else:
                dose_part()
                freq_part()
                route_part()
                dur_part()
            status = ("mismatch" if "mismatch" in statuses else "unverifiable" if "unverifiable" in statuses else "match")
            add(kind, name, parts, status, d, dims=dims)
        # negated and inverted phrasing, informational controls
        parts = [f"{num(d['dose'])} {d['unit']}"] + ([rng.choice(freqs[d["rate"]])] if d["rate"] else [])
        wrong = [f"{num(d['dose'])} {d['unit']}", rng.choice(freqs[next(r for r in (1, 2, 3) if r != (d["rate"] or 0))])]
        pick = parts if rng.random() < 0.5 and d["rate"] else wrong
        status = "unverifiable"  # a negatively phrased question has no unambiguous yes or no
        cases.append({"family": fam, "kind": "negated", "query": rng.choice(negated).format(p=" ".join(pick), d=name),
                      "expect": status, "dims": ["dose", "frequency"], "patient": None, "drug": d["topic"], "note": ""})
        cases.append({"family": fam, "kind": "inverted", "query": rng.choice(INVERTED).format(p=" ".join(pick), d=name),
                      "expect": "unverifiable", "dims": [], "patient": None, "drug": d["topic"], "note": ""})
        cases.append({"family": fam, "kind": "info", "query": rng.choice(INFO).format(d=name), "expect": "none",
                      "dims": [], "patient": None, "drug": d["topic"], "note": ""})
    return cases


def specials(pool: list[dict], rng: random.Random, held: bool) -> list[dict]:
    fam = "heldout" if held else "tuned"
    cases = []
    two = [(pool[i], pool[i + 1]) for i in range(0, min(len(pool) - 1, 12), 2)]
    for a, b in two:
        cases.append({"family": fam, "kind": "two_drugs", "expect": "unverifiable", "dims": [], "patient": None,
                      "drug": a["topic"], "note": "",
                      "query": f"Is {num(a['dose'])} {a['unit']} of {a['name']} and {num(b['dose'])} {b['unit']} of {b['name']} okay?"})
    for d in pool[:12]:
        wrong = d["name"][:-1] if len(d["name"]) > 6 else d["name"] + "x"
        cases.append({"family": fam, "kind": "misspelt_drug", "expect": "refuse_any", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "",
                      "query": f"Is {num(d['dose'])} {d['unit']} of {wrong} okay once daily?"})
        cases.append({"family": fam, "kind": "unknown_drug", "expect": "refuse_any", "dims": [], "patient": None,
                      "drug": "", "note": "", "query": f"Is {num(d['dose'])} {d['unit']} of Zorbextine twice daily okay?"})
        cases.append({"family": fam, "kind": "no_unit", "expect": "unverifiable", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "",
                      "query": f"Is {num(d['dose'])} twice daily okay for {d['name']}?"})
        cases.append({"family": fam, "kind": "form_unit", "expect": "unverifiable", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "", "query": f"Can I give 2 tablets of {d['name']} once a day?"})
        cases.append({"family": fam, "kind": "per_kg", "expect": "unverifiable", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "", "query": f"Is 0.5 mg/kg of {d['name']} okay?"})
        cases.append({"family": fam, "kind": "thousands", "expect": "unverifiable", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "", "query": f"Is 1,000 mg of {d['name']} okay?"})
        cases.append({"family": fam, "kind": "relative", "expect": "unverifiable", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "", "query": f"Is it ok to double the dose of {d['name']}?"})
        cases.append({"family": fam, "kind": "misspelt_freq", "expect": "unverifiable_or_refuse", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "",
                      "query": f"Is {num(d['dose'])} {d['unit']} of {d['name']} twise a day okay?"})
        cases.append({"family": fam, "kind": "clock_time", "expect": "unverifiable_or_refuse", "dims": [], "patient": None,
                      "drug": d["topic"], "note": "",
                      "query": f"Is {num(d['dose'])} {d['unit']} of {d['name']} at 8am okay?"})
    # patient context
    for d in pool[:20]:
        if d["rate"] is None:
            continue
        right = f"{num(d['dose'])} {d['unit']} {FREQ_FORMS_TUNED[d['rate']][0]}"
        child = rng.choice([4, 8, 12, 15])
        cases.append({"family": fam, "kind": "patient_child", "expect": "refuse_patient",
                      "patient": {"age_years": child, "weight_kg": 30}, "dims": [], "drug": d["topic"], "note": "",
                      "query": f"Is {right} okay for {d['name']}?"})
        cases.append({"family": fam, "kind": "patient_adult_match", "expect": "match_or_patient",
                      "patient": {"age_years": 45, "weight_kg": 80, "egfr": 95}, "dims": [], "drug": d["topic"],
                      "note": "", "query": f"Is {right} okay for {d['name']}?"})
        other = next(r for r in (1, 2, 3, 4) if r != d["rate"])
        wrong = f"{num(d['dose'])} {d['unit']} {FREQ_FORMS_TUNED[other][0]}"
        cases.append({"family": fam, "kind": "patient_adult_mismatch", "expect": "mismatch_or_patient",
                      "patient": {"age_years": 45, "weight_kg": 80, "egfr": 95}, "dims": [], "drug": d["topic"],
                      "note": "", "query": f"Is {wrong} okay for {d['name']}?"})
    return cases


def main() -> None:
    pool = drugs()
    rng = random.Random(20260101)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    held_pool = shuffled[:50]
    tuned_pool = shuffled[50:]
    cases = build(False, tuned_pool[:70], rng, 3) + specials(tuned_pool[70:100], rng, False)
    cases += build(True, held_pool, random.Random(777), 3) + specials(held_pool, random.Random(778), True)
    for i, c in enumerate(cases):
        c["id"] = f"P{i:04d}"
    meta = {"cases": len(cases), "tuned": sum(c["family"] == "tuned" for c in cases),
            "heldout": sum(c["family"] == "heldout" for c in cases)}
    OUT.write_text(json.dumps({"meta": meta, "cases": cases}, indent=1, ensure_ascii=False))
    print(meta)


if __name__ == "__main__":
    main()
