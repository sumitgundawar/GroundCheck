"""Generate a large golden evaluation set from the corpus.

Every case has a known expected decision, derived from how it relates to the
corpus, so the expectation is principled rather than hand-labelled:

  ANSWER cases  - questions built from entities that ARE in the corpus
                  (a real condition's treatment/symptoms/monitoring, or a real
                  drug's dose/indications). The trusted sources can ground them.

  REFUSE cases  - four families the system must decline:
                  1. unknown drug      a plausibly-named drug NOT in the corpus
                  2. unknown condition  a plausibly-named condition NOT present
                  3. out of scope      everyday questions the corpus never covers
                  4. blocked qualifier  paediatric / cancer / pregnancy style
                                        questions about real entities, where the
                                        specific guidance is absent

The eight original spec cases are included verbatim and first, so the headline
demo numbers are stable. Deterministic via a fixed seed.

Run from the repository root:
    python scripts/generate_golden.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

SEED = 4242
N_ANSWER = 1000
N_REFUSE = 1000

GOLDEN_PATH = config.ROOT_DIR / "eval" / "golden.json"

# The eight canonical spec cases, kept verbatim and first.
SPEC_CASES = [
    {"query": "What is the first-line management of Veltris syndrome?", "expect": "answer"},
    {"query": "What is the standard dose of Caloradine?", "expect": "answer"},
    {"query": "How often are quorl levels monitored during conservative management?", "expect": "answer"},
    {"query": "What is the recommended dose of Zalortin for Veltris syndrome?", "expect": "refuse"},
    {"query": "How do I treat a broken arm at home?", "expect": "refuse"},
    {"query": "What is the dose of Caloradine for children?", "expect": "refuse"},
    {"query": "Can I combine Caloradine with Orrin-blockers?", "expect": "answer"},
    {"query": "What is the cure for cancer?", "expect": "refuse"},
]

# Question templates keyed by the corpus section that answers them.
ANSWER_TEMPLATES = {
    "condition": [
        ("Treatment", "How is {name} treated?"),
        ("Treatment", "What is the treatment for {name}?"),
        ("Symptoms", "What are the symptoms of {name}?"),
        ("Information", "What is {name}?"),
        ("Monitoring", "How is {name} monitored?"),
        ("Prognosis", "How often is {name} reviewed?"),
    ],
    "drug": [
        ("Dosage and Administration", "What is the dose of {name}?"),
        ("Dosage and Administration", "How much {name} is given?"),
        ("Indications and Usage", "What is {name} used for?"),
        ("Contraindications", "When should {name} not be used?"),
        ("Adverse Reactions", "What are the side effects of {name}?"),
        ("Drug Interactions", "What should not be taken with {name}?"),
    ],
}

# Out-of-scope everyday questions the corpus never addresses. None contain a
# corpus entity, so retrieval should not find a relevant source.
# Everyday questions the corpus never addresses. Chosen to avoid vocabulary that
# overlaps the corpus (dosing, timing, monitoring, sleep, daily, hours), so a
# coincidental lexical match cannot let one slip past the retrieval gate. These
# must all refuse.
OUT_OF_SCOPE = [
    "How do I treat a sprained ankle?",
    "How do I remove a splinter?",
    "How do I treat a sunburn?",
    "What helps with hiccups?",
    "How do I stop a nosebleed?",
    "How do I get rid of dandruff?",
    "How do I treat a paper cut?",
    "What causes bad breath?",
    "How do I treat a bee sting?",
    "How do I soothe a mosquito bite?",
    "How do I treat athlete's foot?",
    "What helps with seasickness?",
    "How do I whiten my teeth?",
    "How do I treat a blister?",
    "How do I get rid of hiccups fast?",
    "How do I treat chapped lips?",
    "How do I remove a tick?",
    "What helps with a stubbed toe?",
    "How do I treat an ingrown toenail?",
    "How do I unclog a stuffy nose?",
]

# Qualifiers that name a real-but-absent population or context. Paired with real
# corpus entities, the specific guidance is not in any source, so refuse.
BLOCKED_QUALIFIERS = [
    "for children", "for infants", "during pregnancy", "for newborns",
    "in paediatric patients", "for breastfeeding mothers",
]


def load_corpus() -> list[dict]:
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _title_case_name(topic: str) -> str:
    return " ".join(w.capitalize() for w in topic.split())


def build_golden() -> list[dict]:
    rng = random.Random(SEED)
    corpus = load_corpus()

    # Group topics by whether they are a drug or a condition, via section types.
    drug_sections = {"Dosage and Administration", "Indications and Usage",
                     "Contraindications", "Drug Interactions", "Adverse Reactions"}
    cond_sections = {"Treatment", "Symptoms", "Information", "Monitoring",
                     "Prognosis", "Causes", "Susceptibility"}

    sections_by_topic: dict[str, set[str]] = {}
    kind_by_topic: dict[str, str] = {}
    for r in corpus:
        sections_by_topic.setdefault(r["topic"], set()).add(r["section"])
        if r["section"] in drug_sections:
            kind_by_topic[r["topic"]] = "drug"
        elif r["topic"] not in kind_by_topic and r["section"] in cond_sections:
            kind_by_topic[r["topic"]] = "condition"

    # Exclude the safety/interactions meta topics from question generation.
    topics = [t for t in sections_by_topic
              if t not in ("safety", "interactions")]
    drug_topics = [t for t in topics if kind_by_topic.get(t) == "drug"]
    cond_topics = [t for t in topics if kind_by_topic.get(t) == "condition"]

    answer_cases: list[dict] = []
    answer_seen: set[str] = set()
    # Build answerable cases, only when the answering section actually exists for
    # that topic, so every expected "answer" is genuinely grounded.
    pools = [("condition", cond_topics), ("drug", drug_topics)]
    attempts = 0
    while len(answer_cases) < N_ANSWER and attempts < N_ANSWER * 40:
        attempts += 1
        kind, pool = rng.choice(pools)
        if not pool:
            continue
        topic = rng.choice(pool)
        section, template = rng.choice(ANSWER_TEMPLATES[kind])
        if section not in sections_by_topic[topic]:
            continue
        query = template.format(name=_title_case_name(topic))
        if query in answer_seen:
            continue
        answer_seen.add(query)
        answer_cases.append({"query": query, "expect": "answer"})

    # Known corpus entity names, to avoid accidentally naming a real one in a
    # "refuse" case.
    real_names = {t.split()[0].lower() for t in topics}

    refuse_cases: list[dict] = []
    refuse_seen: set[str] = set()

    def add_refuse(query: str) -> bool:
        if query in refuse_seen:
            return False
        refuse_seen.add(query)
        refuse_cases.append({"query": query, "expect": "refuse"})
        return True

    # Family 1+2: unknown drugs and conditions (plausible names not in corpus).
    # Two-syllable stems give a large, plausible name space that does not collide
    # with the real corpus vocabulary.
    fake_p1 = ["zal", "xyr", "quom", "vrak", "blen", "drix", "phos", "trel",
               "wuld", "yest", "crov", "munt", "splen", "jorl", "klyn", "fros",
               "grav", "hesp", "ivor", "narb", "ocre", "plyx", "ryst", "thav"]
    fake_p2 = ["", "", "o", "a", "en", "ar", "il", "os", "un"]
    fake_drug_suffix = ["otin", "axine", "uvil", "endryl", "ostan", "iprex",
                        "afil", "oxen", "udine", "alom"]
    fake_cond_suffix = ["osis", "emia", "algia", "opathy", " syndrome",
                        " disorder", " deficiency", "itis"]
    unknown_q = [
        "What is the dose of {n}?",
        "How is {n} treated?",
        "What are the side effects of {n}?",
        "What is {n} used for?",
    ]
    n_unknown_target = int(N_REFUSE * 0.40)
    attempts = 0
    while len(refuse_cases) < n_unknown_target and attempts < n_unknown_target * 40:
        attempts += 1
        stem = rng.choice(fake_p1) + rng.choice(fake_p2)
        if rng.random() < 0.5:
            name = (stem + rng.choice(fake_drug_suffix)).capitalize()
        else:
            name = (stem + rng.choice(fake_cond_suffix)).capitalize()
        first = name.split()[0]
        # Skip names whose first token is too short to be a realistic entity
        # (a 3-letter drug name), and any that collide with a real corpus name.
        if len(first) < 5 or first.lower() in real_names:
            continue
        add_refuse(rng.choice(unknown_q).format(n=name))

    # Family 3: out of scope. Cycle the curated everyday list (all unique).
    for q in OUT_OF_SCOPE:
        add_refuse(q)

    # Family 4: blocked qualifier on a real entity. Large capacity:
    # topics x qualifiers x question forms, so it reliably fills to target.
    qual_forms = [
        "What is the dose of {n} {q}?",
        "How is {n} given {q}?",
        "Is {n} safe {q}?",
    ]
    attempts = 0
    while len(refuse_cases) < N_REFUSE and attempts < N_REFUSE * 60:
        attempts += 1
        topic = rng.choice(topics)
        name = _title_case_name(topic)
        qualifier = rng.choice(BLOCKED_QUALIFIERS)
        form = rng.choice(qual_forms)
        add_refuse(form.format(n=name, q=qualifier))

    # De-duplicate while preserving order, keeping the spec cases first.
    seen = set()
    out: list[dict] = []
    for case in SPEC_CASES + answer_cases + refuse_cases:
        key = case["query"]
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    return out


def main() -> None:
    cases = build_golden()
    with open(GOLDEN_PATH, "w", encoding="utf-8") as fh:
        json.dump(cases, fh, indent=2, ensure_ascii=False)
    n_ans = sum(1 for c in cases if c["expect"] == "answer")
    n_ref = sum(1 for c in cases if c["expect"] == "refuse")
    print(f"Wrote {len(cases)} golden cases to {GOLDEN_PATH}")
    print(f"  {n_ans} answerable, {n_ref} must-refuse")


if __name__ == "__main__":
    main()
