"""Generate a large synthetic clinical corpus with REAL structure and FICTIONAL
content.

The document shapes mirror two real, public reference formats:
  - Disease entries follow the MedQuAD question-type taxonomy
    (Information, Causes, Symptoms, Susceptibility, Treatment, Prevention,
    Diagnosis, Prognosis, Monitoring). https://github.com/abachaa/MedQuAD
  - Drug entries follow the FDA Structured Product Labeling (SPL) sections used
    by DailyMed (Indications and Usage, Dosage and Administration,
    Contraindications, Drug Interactions, Adverse Reactions, Use in Specific
    Populations). https://dailymed.nlm.nih.gov

Every condition, medication, marker, and dosage below is INVENTED. No real drug,
condition, dose, or clinical guidance appears anywhere. The structure is real so
retrieval and grounding get a genuine workout; the content is fiction so the
demo stays safe and the "not medical advice" disclaimer stays true.

Deterministic: seeded, so the same corpus is produced every run.

Run from the repository root:
    python scripts/generate_corpus.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

SEED = 1729
N_CONDITIONS = 150
N_DRUGS = 150
N_MARKERS = 60        # fictional lab markers, 2 pages each
N_PROCEDURES = 50     # fictional diagnostic procedures, 2 pages each

# --- The 10 canonical demo records. Kept verbatim so the four demonstration
# queries and the golden evaluation behave exactly as specified. ------------
BASE_RECORDS = [
    {"id": "VELT-001", "title": "Veltris syndrome: overview", "topic": "veltris syndrome",
     "section": "Information",
     "text": "Veltris syndrome is a fictional metabolic condition characterised by intermittent fatigue and elevated serum quorl levels. First-line management is conservative: structured rest, hydration, and monitoring of quorl levels every two weeks. Pharmacological treatment is considered only when quorl levels remain elevated after four weeks of conservative management."},
    {"id": "VELT-002", "title": "Veltris syndrome: first-line medication", "topic": "veltris syndrome",
     "section": "Treatment",
     "text": "When pharmacological treatment of Veltris syndrome is indicated, the first-line agent is Caloradine. Caloradine is started at 15 mg once daily for 14 days, then reviewed. The dose is not increased without specialist input. Caloradine is contraindicated in patients with known quorl sensitivity."},
    {"id": "CALO-001", "title": "Caloradine: standard regimen", "topic": "caloradine",
     "section": "Dosage and Administration",
     "text": "Caloradine is a fictional oral agent used in the management of Veltris syndrome. The standard adult regimen is 15 mg once daily, taken in the morning with food, for an initial course of 14 days. Common side effects in this fictional model include mild drowsiness and dry mouth. Caloradine should not be combined with Orrin-blockers."},
    {"id": "ORRN-001", "title": "Orrin deficiency: overview", "topic": "orrin deficiency",
     "section": "Information",
     "text": "Orrin deficiency is a fictional condition in which the body produces insufficient orrin, leading to slowed recovery after exertion. It is diagnosed by a low orrin panel. Management focuses on graded activity and dietary support. Most fictional cases resolve within six weeks without medication."},
    {"id": "ORRN-002", "title": "Orrin deficiency: when to treat", "topic": "orrin deficiency",
     "section": "Treatment",
     "text": "Medication for orrin deficiency is reserved for fictional cases that do not improve after six weeks of graded activity. The first-line agent in this fictional model is Mendel solution, given as a short supervised course. Orrin-blockers must be avoided during treatment."},
    {"id": "MEND-001", "title": "Mendel solution: standard course", "topic": "mendel solution",
     "section": "Dosage and Administration",
     "text": "Mendel solution is a fictional preparation used for persistent orrin deficiency. The standard course in this fictional model is 5 mL once daily for 7 days, taken after the evening meal. It is supervised because of a fictional risk of transient dizziness. Mendel solution is not used in Veltris syndrome."},
    {"id": "QUOR-001", "title": "Quorl monitoring", "topic": "quorl levels",
     "section": "Monitoring",
     "text": "Quorl is a fictional serum marker used to track Veltris syndrome. Levels are reported on a scale of 0 to 100. A level above 60 is considered elevated. Monitoring is performed every two weeks during conservative management and weekly once medication is started."},
    {"id": "SAFE-001", "title": "General safety note", "topic": "safety",
     "section": "Information",
     "text": "All information in this reference is fictional and exists only to demonstrate a grounded retrieval system. No real condition, medication, or dosage is described. Nothing here is medical advice."},
    {"id": "INTR-001", "title": "Interactions: Orrin-blockers", "topic": "interactions",
     "section": "Drug Interactions",
     "text": "Orrin-blockers are a fictional drug class that must not be combined with Caloradine or Mendel solution. Combining them in this fictional model increases the risk of transient dizziness and is therefore avoided."},
    {"id": "VELT-003", "title": "Veltris syndrome: review schedule", "topic": "veltris syndrome",
     "section": "Prognosis",
     "text": "Patients with Veltris syndrome are reviewed at two weeks, four weeks, and then monthly. At each review, quorl levels and side effects are checked. If quorl levels normalise, medication is tapered under specialist guidance."},
]

# Names already in use, so generated names never collide with the demo set.
RESERVED_NAMES = {"veltris", "caloradine", "orrin", "mendel", "quorl"}

# Trap terms that must NEVER appear in generated content, so the demo refusals
# (an unknown drug "Zalortin", an out-of-scope "broken arm", "children",
# "cancer") keep working against the larger corpus. Matched at WORD START only
# (prefix-tolerant), so "arm" does not match inside "pharmacological".
BLOCKLIST = ["zalor", "child", "paediatr", "pediatr", "infant", "neonat",
             "broken", "fractur", "arm", "cancer", "tumor", "tumour",
             "oncolog", "cure", "pregn"]
_BLOCK_RE = re.compile(r"\b(?:" + "|".join(BLOCKLIST) + r")", re.IGNORECASE)

# --- Invented vocabulary pools --------------------------------------------
# A large pool of pronounceable consonant-vowel-consonant syllables. Built
# programmatically so there are hundreds of distinct first syllables: with this
# many, generated entity names rarely share a leading syllable, which keeps them
# distinct to the retriever and avoids one entity's documents being crowded out
# of the top-k by a similarly-named sibling.
def _build_prefixes() -> list[str]:
    c1 = list("bdfgklmnprstv")
    vowels = list("aeiou")
    c2 = ["l", "n", "r", "s", "t", "m", "d"]
    pool = []
    for a in c1:
        for v in vowels:
            for b in c2:
                syl = a + v + b
                if not _BLOCK_RE.search(syl):
                    pool.append(syl)
    return pool


PREFIXES = _build_prefixes()
COND_SUFFIX = ["ris", "on", "el", "ia", "eus", "ix", "ar", "oth", "und",
               "een", "ous", "ay", "ie", "as", "or", "yn"]
COND_TYPE = ["syndrome", "deficiency", "disorder", "dysregulation", "imbalance"]
DRUG_SUFFIX = ["adine", "oxine", "ulex", "ravil", "ostan", "ethol", "izane",
               "aprex", "ondel", "irin", "aleen", "orphil", "umide", "estal",
               "avine", "oltin", "endal", "ufen"]
DRUG_FORM = ["", "", "", " solution", " complex", " suspension"]
MARKERS = ["drovin", "kessel", "tarnil", "vexin", "orlin", "sennate",
           "brindyl", "morvex", "lunate", "praxel", "tovrin", "ulmate",
           "wendal", "yarnic", "glenor", "hossel", "irenic", "jadrin"]

SYMPTOMS = [
    "intermittent fatigue", "reduced stamina", "slowed recovery after exertion",
    "transient dizziness", "mild drowsiness", "low background energy",
    "delayed marker clearance", "episodic light-headedness",
    "reduced exercise tolerance", "occasional dry mouth",
]
CONSERVATIVE = [
    "structured rest", "hydration", "graded activity", "dietary support",
    "paced exertion", "regular sleep timing", "fluid balance monitoring",
]
SIDE_EFFECTS = [
    "mild drowsiness", "dry mouth", "transient dizziness", "mild nausea",
    "temporary loss of appetite", "light-headedness on standing",
]
TIMINGS = ["in the morning with food", "after the evening meal",
           "twice daily with water", "in the morning before food"]
FREQUENCIES = ["once daily", "twice daily", "every 12 hours", "once every morning"]
ROUTES = ["oral", "oral", "oral", "topical", "subcutaneous"]
DOSE_NUMS = ["2.5", "5", "10", "15", "20", "25", "40", "50", "75", "100"]
DOSE_UNITS = ["mg", "mg", "mg", "mcg", "mL"]
DURATIONS = ["7 days", "10 days", "14 days", "21 days", "28 days"]
WEEKS = ["four", "six", "eight"]
INTERVALS = ["two weeks", "three weeks", "four weeks"]
THRESHOLDS = ["55", "60", "65", "70"]
# Phrasing variety, so generated documents are not near-identical and the
# embedding map is less rigidly blocky.
MODEL_PHRASE = ["in this fictional model", "in this synthetic model",
                "in this reference", "for demonstration only",
                "in this invented dataset", "in this illustrative model"]
CONTEXT_SENTENCES = [
    "No real condition is described.",
    "The wording mirrors the shape of a real reference entry.",
    "This entry exists only to exercise the retrieval system.",
    "Everything here is invented for the demonstration.",
    "It carries no real-world clinical meaning.",
    "Details are fictional and intentionally synthetic.",
]
ONSET_PHRASES = [
    "Onset is usually gradual.", "It tends to build over days.",
    "Presentation varies between patients.", "Severity fluctuates over time.",
    "Most cases are mild.", "Course is typically self-limiting.",
]
MARKER_SUFFIX = ["in", "yl", "ate", "ix", "ol", "an", "ase", "een", "is", "or"]
PROC_SUFFIX = ["el", "ic", "an", "is", "or", "ux", "en", "al"]
PROC_FORM = [" panel", " assessment", " index", " screen", " profile"]
SAMPLE_TYPES = ["serum", "plasma", "whole blood", "salivary"]


def _has_blocked(text: str) -> bool:
    return bool(_BLOCK_RE.search(text))


# Short connective syllables used to build two-part stems. Two syllables give a
# far larger, more distinct name space, so generated entities rarely share a
# long prefix. That keeps retrieval recall high: a query naming one entity does
# not get crowded out of the top-k by near-identical siblings.
_INFIX = ["a", "e", "i", "o", "u", "an", "en", "or", "ar", "el", "in", "os",
          "ub", "ad", "em", "ol", "un", "ir"]


def make_unique_names(rng: random.Random, count: int, suffixes: list[str],
                      used: set[str], seen4: set[str],
                      forms: list[str] | None = None) -> list[str]:
    """Build distinct, drug-like names from two prefix syllables plus a suffix.
    A globally-shared set of first-four-character prefixes (seen4) guarantees no
    two entities, across any category, look near-identical to the retriever, so
    a query about one entity never gets crowded out by a similarly-named sibling."""
    names: list[str] = []
    attempts = 0
    while len(names) < count and attempts < count * 400:
        attempts += 1
        p1, p2 = rng.choice(PREFIXES), rng.choice(PREFIXES)
        if p1 == p2:
            continue
        stem = p1 + p2 + rng.choice(suffixes)
        form = rng.choice(forms) if forms else ""
        key = stem.lower()
        if key in used or _has_blocked(stem) or key[:4] in seen4:
            continue
        seen4.add(key[:4])
        used.add(key)
        names.append(stem.capitalize() + form)
    return names


def build_corpus() -> list[dict]:
    rng = random.Random(SEED)
    used = set(RESERVED_NAMES)
    # Shared guard so first-four-character prefixes are unique across every
    # category, keeping all entity names distinct to the retriever. Seed it with
    # the canonical demo names so generated names never shadow them.
    seen4 = {n[:4] for n in RESERVED_NAMES}

    condition_stems = make_unique_names(rng, N_CONDITIONS, COND_SUFFIX, used, seen4)
    drug_names = make_unique_names(rng, N_DRUGS, DRUG_SUFFIX, used, seen4, DRUG_FORM)

    records = list(BASE_RECORDS)

    # Pair each condition with a treating drug so cross-references are coherent.
    for i, stem in enumerate(condition_stems):
        cond_type = rng.choice(COND_TYPE)
        cond = f"{stem} {cond_type}"
        code = stem[:4].upper()
        marker = rng.choice(MARKERS)
        drug = drug_names[i % len(drug_names)]
        sx = rng.sample(SYMPTOMS, 3)
        cons = rng.sample(CONSERVATIVE, 2)
        weeks = rng.choice(WEEKS)
        interval = rng.choice(INTERVALS)
        threshold = rng.choice(THRESHOLDS)

        mp = rng.choice(MODEL_PHRASE)
        ctx = rng.choice(CONTEXT_SENTENCES)
        onset = rng.choice(ONSET_PHRASES)
        sections = [
            ("Information", f"{cond}: overview",
             rng.choice([
                f"{cond} is a fictional condition characterised by {sx[0]} and {sx[1]}. "
                f"It is tracked {mp} using the serum marker {marker}. {ctx}",
                f"Described {mp}, {cond} typically presents with {sx[0]}, alongside {sx[1]}. "
                f"Clinicians follow the marker {marker} to gauge its course. {ctx}",
                f"{cond} is an invented disorder in which {sx[1]} and {sx[0]} are seen. "
                f"Its activity is summarised by the {marker} marker. {ctx}",
             ])),
            ("Causes", f"{cond}: causes",
             rng.choice([
                f"{mp.capitalize()}, {cond} is associated with reduced {marker} "
                f"regulation. It is not contagious and has no real-world basis.",
                f"The driver of {cond} in this invented setting is poor {marker} "
                f"control. {onset} There is no real-world counterpart.",
                f"{cond} arises, {mp}, from disordered {marker} handling rather than "
                f"any external cause. It cannot be caught from another person.",
             ])),
            ("Symptoms", f"{cond}: symptoms",
             rng.choice([
                f"Common features of {cond} {mp} include {sx[0]}, {sx[1]}, and {sx[2]}.",
                f"People with {cond} often notice {sx[0]} and {sx[1]}; some also report {sx[2]}. {onset}",
                f"Typical complaints in {cond} are {sx[2]}, {sx[0]}, and {sx[1]}.",
             ])),
            ("Treatment", f"{cond}: first-line management",
             rng.choice([
                f"First-line management of {cond} is conservative: {cons[0]} and "
                f"{cons[1]}, with monitoring of {marker} every {interval}. "
                f"Pharmacological treatment is considered only when {marker} remains "
                f"elevated after {weeks} weeks, when the first-line agent is {drug}.",
                f"{cond} is managed first with {cons[0]} and {cons[1]}, while {marker} "
                f"is checked every {interval}. If {marker} stays high beyond {weeks} "
                f"weeks, the first-line agent {drug} is started.",
             ])),
            ("Monitoring", f"{marker.capitalize()} monitoring in {cond}",
             rng.choice([
                f"{marker.capitalize()} is a fictional serum marker for {cond}, "
                f"reported on a scale of 0 to 100. A level above {threshold} is "
                f"considered elevated. Monitoring is performed every {interval}.",
                f"To follow {cond}, the {marker} marker is measured on a 0 to 100 "
                f"scale every {interval}; readings over {threshold} count as raised.",
             ])),
            ("Prognosis", f"{cond}: review schedule",
             rng.choice([
                f"Patients with {cond} are reviewed at two weeks, {weeks} weeks, and "
                f"then monthly {mp}. If {marker} normalises, treatment is tapered "
                f"under specialist guidance.",
                f"Review for {cond} happens at two weeks, again at {weeks} weeks, then "
                f"monthly. Once {marker} settles, the dose is tapered with a specialist.",
             ])),
        ]
        for j, (section, title, text) in enumerate(sections, start=1):
            records.append({
                "id": f"{code}-{i:03d}{j}",
                "title": title,
                "topic": cond.lower(),
                "section": section,
                "text": text,
            })

    # Drug labels in DailyMed/SPL section shape.
    drug_class_names = [f"{rng.choice(PREFIXES).capitalize()}-blockers"
                        for _ in range(len(drug_names))]
    for i, drug in enumerate(drug_names):
        code = drug.split()[0][:4].upper()
        cond_stem = condition_stems[i % len(condition_stems)]
        cond_type = COND_TYPE[i % len(COND_TYPE)]
        cond = f"{cond_stem} {cond_type}"
        marker = MARKERS[i % len(MARKERS)]
        route = rng.choice(ROUTES)
        dose = f"{rng.choice(DOSE_NUMS)} {rng.choice(DOSE_UNITS)}"
        freq = rng.choice(FREQUENCIES)
        timing = rng.choice(TIMINGS)
        duration = rng.choice(DURATIONS)
        side = rng.sample(SIDE_EFFECTS, 2)
        drug_class = drug_class_names[(i + 3) % len(drug_class_names)]

        mp = rng.choice(MODEL_PHRASE)
        sections = [
            ("Indications and Usage", f"{drug}: indications and usage",
             rng.choice([
                f"{drug} is a fictional {route} agent indicated {mp} for the "
                f"management of {cond}. No real product is described.",
                f"Used {mp}, {drug} is a {route} treatment for {cond}. It is entirely invented.",
                f"{drug} is an invented {route} medicine given for {cond} {mp}.",
             ])),
            ("Dosage and Administration", f"{drug}: dosage and administration",
             rng.choice([
                f"The standard adult regimen of {drug} is {dose} {freq}, taken "
                f"{timing}, for an initial course of {duration}. The dose is not "
                f"increased without specialist input.",
                f"{drug} is given as {dose} {freq}, {timing}, for {duration} to start. "
                f"Any increase needs specialist input.",
             ])),
            ("Contraindications", f"{drug}: contraindications",
             rng.choice([
                f"{drug} is contraindicated in patients with known {marker} "
                f"sensitivity {mp}.",
                f"Avoid {drug} where there is known sensitivity to {marker}, {mp}.",
                f"{drug} should not be used if a patient is sensitive to {marker} ({mp}).",
             ])),
            ("Drug Interactions", f"{drug}: drug interactions",
             rng.choice([
                f"{drug} should not be combined with {drug_class}. Combining them "
                f"increases the fictional risk of {side[0]} and is avoided.",
                f"Do not give {drug} together with {drug_class}; the pairing raises "
                f"the invented risk of {side[0]}.",
             ])),
            ("Adverse Reactions", f"{drug}: adverse reactions",
             rng.choice([
                f"Common fictional side effects of {drug} include {side[0]} and "
                f"{side[1]}. These are described for demonstration only.",
                f"{drug} may cause {side[1]} or {side[0]} in this invented model; "
                f"both are illustrative.",
             ])),
        ]
        for j, (section, title, text) in enumerate(sections, start=1):
            records.append({
                "id": f"{code}-D{i:03d}{j}",
                "title": title,
                "topic": drug.split()[0].lower(),
                "section": section,
                "text": text,
            })

    # --- Fictional lab markers (their own category and colour) -------------
    marker_names = make_unique_names(rng, N_MARKERS, MARKER_SUFFIX, used, seen4)
    for i, marker in enumerate(marker_names):
        code = marker[:4].upper()
        cond = f"{condition_stems[i % len(condition_stems)]} {rng.choice(COND_TYPE)}"
        sample = rng.choice(SAMPLE_TYPES)
        threshold = rng.choice(THRESHOLDS)
        interval = rng.choice(INTERVALS)
        sections = [
            ("Laboratory marker", f"{marker}: laboratory marker",
             f"{marker} is a fictional {sample} marker used in this model to track "
             f"{cond}. Levels are reported on a scale of 0 to 100, and trends "
             f"matter more than any single reading."),
            ("Reference range", f"{marker}: reference range",
             f"In this fictional model a {marker} level above {threshold} is "
             f"considered elevated. Testing is repeated every {interval} while a "
             f"patient is monitored, and results are interpreted by a specialist."),
        ]
        for j, (section, title, text) in enumerate(sections, start=1):
            records.append({
                "id": f"{code}-K{i:03d}{j}",
                "title": title,
                "topic": marker.lower(),
                "section": section,
                "kind": "marker",
                "text": text,
            })

    # --- Fictional diagnostic procedures (their own category and colour) ---
    proc_names = make_unique_names(rng, N_PROCEDURES, PROC_SUFFIX, used, seen4, PROC_FORM)
    for i, proc in enumerate(proc_names):
        code = proc.split()[0][:4].upper()
        cond = f"{condition_stems[(i + 7) % len(condition_stems)]} {rng.choice(COND_TYPE)}"
        sections = [
            ("Diagnostic procedure", f"{proc}: overview",
             f"The {proc} is a fictional, non-invasive assessment used to support "
             f"the diagnosis of {cond} in this model. It can be repeated safely."),
            ("Diagnostic procedure", f"{proc}: interpretation",
             f"A raised result on the {proc} in this fictional model supports "
             f"closer monitoring of {cond}, and findings are reviewed with a "
             f"specialist before any treatment is started."),
        ]
        for j, (section, title, text) in enumerate(sections, start=1):
            records.append({
                "id": f"{code}-P{i:03d}{j}",
                "title": title,
                "topic": proc.split()[0].lower(),
                "section": section,
                "kind": "procedure",
                "text": text,
            })

    return records


def main() -> None:
    records = build_corpus()

    # Safety assertions: no trap term leaked into the generated corpus, and ids
    # are unique.
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate ids generated"
    leaked = [r["id"] for r in records
              if r["id"] not in {b["id"] for b in BASE_RECORDS}
              and _has_blocked(r["text"] + " " + r["title"])]
    assert not leaked, f"blocked trap term leaked into: {leaked[:5]}"

    with open(config.CORPUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {config.CORPUS_PATH}")
    print(f"  {len(BASE_RECORDS)} canonical demo records + "
          f"{len(records) - len(BASE_RECORDS)} generated")


if __name__ == "__main__":
    main()
