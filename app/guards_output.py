"""Output guards: source-coverage, grounding, and the dosage guard.

All three are deterministic so the demo is reproducible. The optional LLM judge
in llm.py is corroboration only; the checks here are authoritative."""

from __future__ import annotations

import re

import numpy as np

from . import config, retrieval

# --------------------------------------------------------------------------
# Source-coverage check
#
# Every salient term in the question must be represented in the retrieved
# sources. A question that names something the trusted sources never mention
# (an unknown drug like "Zalortin", or a qualifier like "children" that the
# sources do not address) cannot be grounded, so we refuse before generating.
# Matching is prefix-based so morphological variants line up, for example
# "monitored" against "monitoring".
# --------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Common words that carry no grounding signal. Kept deliberately broad so the
# check fires only on genuinely contentful, source-specific terms.
_STOPWORDS = {
    "what", "which", "when", "where", "whom", "whose", "that", "this", "these",
    "those", "with", "without", "from", "into", "about", "above", "below",
    "after", "before", "during", "between", "your", "yours", "have", "having",
    "does", "doing", "done", "should", "would", "could", "will", "shall",
    "must", "might", "they", "them", "their", "there", "here", "then", "than",
    "such", "some", "any", "many", "much", "more", "most", "other", "each",
    "every", "both", "either", "neither", "also", "very", "just", "only",
    "often", "ever", "never", "always", "still", "even", "like", "into",
    "over", "under", "again", "once", "how", "the", "and", "for", "are",
    "was", "were", "been", "being", "you", "can", "cannot", "may", "out",
    "off", "per", "via", "use", "used", "using", "tell", "give", "show",
    "explain", "describe", "want", "need", "please", "home", "good", "best",
    "right", "recommend", "recommended", "suggest", "suggested", "advise",
    "advised", "general", "overall", "thing", "things", "information",
    # Generic clinical-process and label-section words. The coverage guard's job
    # is to catch an unknown ENTITY or absent QUALIFIER in the question (an
    # invented drug, or "children"); it should not demand that framing verbs
    # like "monitored" or section words like "dosage" appear verbatim in the
    # retrieved passages, which causes false refusals at corpus scale.
    "treat", "treated", "treatment", "manage", "managed", "management",
    "monitor", "monitored", "monitoring", "schedule", "review", "reviewed",
    "regimen", "dose", "dosage", "doses", "dosing", "administration",
    "symptom", "symptoms", "feature", "features", "sign", "signs",
    "side", "effect", "effects", "adverse", "reaction", "reactions",
    "combine", "combined", "combination", "interact", "interaction",
    "interactions", "contraindicated", "contraindication", "contraindications",
    "indication", "indications", "indicated", "course", "agent", "level",
    "levels", "taken", "started", "standard", "adult", "daily", "common",
    "fictional", "model", "patient", "patients", "safe", "safety", "given",
    "cause", "causes", "diagnosis", "diagnosed", "prognosis", "prevention",
    "usage", "used", "managementof", "first", "second", "first-line",
    "line", "second-line", "oral", "topical", "supervised",
    # Generic English and clinical-process words. The coverage guard exists to
    # catch an unknown ENTITY (an invented drug) or an absent QUALIFIER
    # ("children", "pregnancy"), not to demand that ordinary words appear
    # verbatim in the corpus. Without these, naturally-phrased multi-step
    # questions ("what is the next step if conservative care fails?") refuse for
    # the wrong reason. Deliberately EXCLUDES entity/qualifier words.
    "next", "step", "steps", "follow", "following", "fail", "fails", "failed",
    "failing", "failure", "together", "maximum", "minimum", "max", "min",
    "needs", "needed", "take", "takes", "taking", "get", "getting", "make",
    "makes", "made", "work", "works", "working", "help", "helps", "start",
    "starts", "starting", "begin", "begins", "stop", "stops", "increase",
    "increased", "increasing", "decrease", "decreased", "reduce", "reduced",
    "change", "changed", "compare", "comparison", "versus", "difference",
    "different", "same", "instead", "alongside", "plus", "add", "added",
    "consider", "considered", "appropriate", "suitable", "option", "options",
    "available", "usual", "usually", "typical", "typically", "normal",
    "normally", "initial", "ongoing", "continue", "continued", "persist",
    "persists", "persistent", "remain", "remains", "raised", "result",
    "results", "finding", "findings", "current", "currently", "week", "weeks",
    "day", "days", "time", "times", "point", "stage", "stages", "phase",
    "manageme", "approach", "recommended", "next-step", "regarding", "about",
    "around", "still", "already", "yet", "soon", "later", "early", "long",
}


# Conversational words: greetings, hedges and politeness. They carry no
# clinical meaning and are never the name of a drug or condition, so a
# question phrased casually is not refused because of them. Kept explicit and
# short on purpose: a word that could name something clinical does not belong.
_CONVERSATIONAL = {
    "hello", "hey", "thanks", "thank", "cheers", "okay", "kindly",
    "honestly", "actually", "basically", "really", "simply", "exactly",
    "quick", "quickly", "question", "questions", "wondering", "wonder",
    "curious", "know", "think", "sure", "maybe", "perhaps", "wanted",
    "someone", "anyone", "somebody", "anything", "something", "whether",
    "correct", "true", "wrong", "confirm", "clarify", "understand",
    "according", "says", "said", "mention", "mentions",
}

# Contractions attach to words the check would otherwise treat as unknown
# ("what's", "doesn't"). The base word is what matters.
_CONTRACTION = re.compile(r"(?:'s|'re|'ve|'ll|'d|'m|n't)$")


# Short words that still change the question: another species, a route or a unit.
# Two-letter forms such as XR live in _FORM_TERMS below, which reads them only
# in capitals, so "Mr" and a hesitant "er" are still noise.
_SHORT_SALIENT = {"cat", "cats", "dog", "dogs", "pet", "pets", "cow", "pig", "rat", "rats", "horse", "iv", "im",
                  "gram", "kg", "mcg", "ml"}

# A form or a route changes the dose: 15 mg of a plain tablet is not 15 mg of a
# modified-release one, and an intravenous dose is rarely the oral dose. These
# words must be supported by a source about the medicine the question names,
# not by any passage that happens to mention them. "Solution" is deliberately
# absent: it is part of a medicine's name in the demo formulary.
_FORM_TERMS = {
    "xr", "sr", "er", "xl", "cr", "mr", "la", "iv", "im", "sc", "sl", "po", "pr",
    "topical", "topically", "oral", "orally", "intravenous", "intravenously",
    "intramuscular", "intramuscularly", "subcutaneous", "subcutaneously",
    "sublingual", "transdermal", "rectal", "rectally", "inhaled", "nebulised",
    "nebulized", "injection", "injectable", "infusion", "patch", "patches",
    "depot", "suppository", "suppositories", "syrup", "elixir", "lozenge",
    "tablet", "tablets", "capsule", "capsules", "cream", "ointment", "drops",
    "modified", "extended", "immediate", "prolonged", "sustained", "release",
}


def _salient_terms(query: str) -> list[str]:
    from .deid import PLACEHOLDER

    query = PLACEHOLDER.sub(" ", query)  # [NAME], [DATE]: removed identifiers, not topics
    terms = []
    for match in _WORD.finditer(query.replace("’", "'")):
        raw = _CONTRACTION.sub("", match.group(0))
        word = raw.lower()
        if word in _FORM_TERMS and (len(word) > 2 or raw.isupper()):
            # A form or a route is never noise, whatever the stopword list
            # says: "topical" and "oral" were both being dropped here, so a
            # question about a form the sources never mention was answered
            # with the plain product's dose. A two-letter abbreviation counts
            # only when written the way clinicians write it — "XR", "IV" — so
            # the honorific in "Mr Smith" is not read as modified release.
            terms.append(word)
            continue
        if (len(word) < 4 and word not in _SHORT_SALIENT) or word in _STOPWORDS or word in _CONVERSATIONAL:
            continue
        terms.append(word)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _source_vocabulary(sources_text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(sources_text)}


def _shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


# Word endings that don't change what a word is about: "monitored" and
# "monitoring", "treat" and "treatment". Only these may differ between a
# question word and a source word. A shared prefix alone isn't enough: it let
# one-letter misspellings of medicine names ("Rotmiitufen" for "Rotmitufen")
# count as covered.
_ENDINGS = ("", "s", "es", "ed", "d", "ing", "ly", "ment", "ments", "al", "ally", "ation", "ations", "ion", "ions",
            "er", "ers", "ive", "ity", "ies", "y", "ic", "ical", "ness", "ance", "ence", "ant", "ent")


def _stems(word: str) -> set[str]:
    out = {word}
    for ending in _ENDINGS:
        if ending and word.endswith(ending) and len(word) - len(ending) >= 3:
            stem = word[: -len(ending)]
            out.add(stem)
            if stem.endswith("i"):          # therapies -> therapy
                out.add(stem[:-1] + "y")
            # English doubles a final consonant before -ed and -ing (stop ->
            # stopped). Anywhere else, a doubled letter is a different word,
            # or a misspelling: "Sulsanaas" is not "Sulsanas".
            if ending in ("ed", "ing") and len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                out.add(stem[:-1])
            out.add(stem + "e")             # dosed -> dose, managing -> manage
    return out


# Words that ask for the same thing. A source that says "is given as 5 mL"
# answers "how is it dosed?", and one headed "Indications" answers "when is it
# prescribed?". Only ordinary clinical vocabulary is listed: never a name, so
# a misspelled medicine can't be covered by a synonym.
_SYNONYM_GROUPS = [
    {"dose", "dosed", "dosing", "dosage", "doses", "administer", "administered", "administration", "given", "give",
     "taken", "take"},
    {"prescribe", "prescribed", "prescription", "indication", "indications", "indicated", "used", "use", "uses",
     "usage", "treats", "treat"},
    {"monitor", "monitored", "monitoring", "review", "reviewed", "check", "checked", "follow-up", "followup"},
    {"side", "effects", "adverse", "reactions", "harms", "unwanted"},
    {"interaction", "interactions", "interacts", "combined", "combination", "together", "with"},
    {"contraindication", "contraindications", "contraindicated", "avoid", "avoided", "must", "not"},
    {"symptom", "symptoms", "signs", "presentation", "presents", "features"},
    {"cause", "causes", "caused", "aetiology", "etiology", "why"},
    {"manage", "managed", "management", "treatment", "treated", "therapy", "care"},
]
_SYNONYMS = {word: group for group in _SYNONYM_GROUPS for word in group}


def _covered(term: str, vocab: set[str]) -> bool:
    if term in vocab:
        return True
    term_stems = _stems(term)
    if any(term_stems & _stems(word) for word in vocab if abs(len(word) - len(term)) <= 6 and word[:3] == term[:3]):
        return True
    return bool(_SYNONYMS.get(term, frozenset()) & vocab)


# Qualifiers that change what a correct answer is. When one of these is
# missing from the sources, it is the most important reason to give, so it is
# reported ahead of other uncovered terms.
_QUALIFIER_TERMS = {
    "child", "children", "childs", "kid", "kids", "infant", "infants", "baby",
    "babies", "newborn", "newborns", "neonate", "neonates", "toddler",
    "toddlers", "adolescent", "adolescents", "teen", "teenager", "teenagers",
    "paediatric", "pediatric", "elderly", "geriatric", "pregnant", "pregnancy",
    "breastfeeding", "lactation", "lactating", "nursing", "alcohol", "kidney",
    "renal", "liver", "hepatic", "surgery", "dialysis", "weight", "overdose",
}

# Population words that show a source addresses an age group.
_CHILD_TERMS = {
    "child", "children", "paediatric", "pediatric", "infant", "infants",
    "adolescent", "adolescents", "neonate", "neonates", "newborn", "newborns",
}
_OLDER_TERMS = {"elderly", "older", "geriatric"}

def _source_vocab(sources: list[dict]) -> set[str]:
    vocab: set[str] = set()
    for record in sources:
        # Scan the body, the title, and the section label. A question word may
        # legitimately live in the title (for example "review schedule") rather
        # than the prose, and that still counts as covered.
        vocab |= _source_vocabulary(record.get("text", ""))
        vocab |= _source_vocabulary(record.get("title", ""))
        vocab |= _source_vocabulary(record.get("section", ""))
    return vocab


def _stated_age(query: str) -> tuple[float, str] | None:
    """The age a question is about, as (years, phrase), or None."""
    m = _AGE_RE.search(query)
    if not m:
        return None
    raw = m.group(1) or m.group(3)
    number = _word_to_number(raw)
    if number is None:
        return None
    unit = "year" if m.group(3) else re.sub(r"s$", "", (m.group(2) or "year").lower())
    unit = {"yr": "year"}.get(unit, unit)
    years = number if unit == "year" else 0.0
    return years, f"{_fmt_num(number)}-{unit}-old"


def _with_article(phrase: str) -> str:
    """ "a 6-year-old", but "an 8-year-old", "an 11-", "an 18-month-old"."""
    starts_with_vowel_sound = phrase.startswith("8") or re.match(r"1[18](?:\D|$)", phrase)
    return ("an " if starts_with_vowel_sound else "a ") + phrase


# Questions that ask for a dose limit. A source only answers them if it states
# a limit; a starting dose is not a maximum.
_LIMIT_QUESTION = re.compile(
    r"\b(?:(max(?:imum)?|highest|upper limit|most)|(min(?:imum)?|lowest|least))\b[^?.]*\b(?:dose|dosage|amount)\b"
    r"|\b(?:dose|dosage)\b[^?.]*\b(?:(max(?:imum)?|highest|upper limit)|(min(?:imum)?|lowest))\b",
    re.IGNORECASE,
)
_MAX_WORDS = ("maximum", "max", "highest", "exceed", "not more than", "no more than", "up to", "upper limit")
_MIN_WORDS = ("minimum", "lowest", "at least", "no less than")

# Questions that ask for a dose to use in combination: "combine X with Y at
# what dose", "the dose of X when taken together with Y".
_COMBINED_DOSE_QUESTION = re.compile(
    r"\bcombin\w*\b[^?.]*\bat what dos"
    r"|\bcombined dos"
    r"|\b(?:dose|dosage)\b[^?.,;]*\b(?:together with|in combination with|when combined with|if combined with)\b",
    re.IGNORECASE,
)
_DO_NOT_COMBINE = re.compile(
    r"\b(?:must|should)\s+not\s+be\s+combined\b|\bnot\s+be\s+combined\b|\bavoid\w*\s+combin",
    re.IGNORECASE,
)


def _unanswerable_request(query: str, sources: list[dict]) -> str | None:
    """A reason the sources can't answer what the question asks for, even
    though they mention its subject, or None."""
    text = " ".join(r.get("text", "") for r in sources).lower()

    m = _LIMIT_QUESTION.search(query)
    if m:
        wants_max = bool(m.group(1) or m.group(3))
        words = _MAX_WORDS if wants_max else _MIN_WORDS
        if not any(w in text for w in words):
            kind = "maximum" if wants_max else "minimum"
            return f"no trusted source states a {kind} dose"

    if _COMBINED_DOSE_QUESTION.search(query) and _DO_NOT_COMBINE.search(text):
        return "the trusted sources say these must not be combined, so no combined dose can be given"

    return None


def _prioritise(terms: list[str], query: str) -> list[str]:
    """Order uncovered terms by how much they matter: clinical qualifiers
    first, then names written with a capital letter, then everything else."""
    capitalised = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z'-]+", query)}
    def rank(term: str) -> int:
        if term in _QUALIFIER_TERMS:
            return 0
        if term in capitalised:
            return 1
        return 2
    return sorted(terms, key=rank)


def _sources_about(query: str, covered: list[str], sources: list[dict]) -> list[dict]:
    """The sources about the question's subject: those naming a capitalised
    name from the question, or, without one, its most specific covered term."""
    if not sources or not covered:
        return sources
    words = [(_source_vocabulary(f"{r.get('title', '')} {r.get('text', '')}")) for r in sources]
    names = [t for t in covered if t in {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z'-]+", query)}
             and t not in _STOPWORDS and sum(t in w for w in words) < len(sources)]
    if not names:
        # The rarest covered word across the whole index is the question's
        # subject: a medicine's name is in a few passages, "daily" in hundreds.
        from . import retrieval

        rarity = {t: retrieval.document_frequency(t) for t in covered}
        known = [t for t in covered if rarity[t] > 0]
        if not known:
            return sources
        names = [min(known, key=lambda t: (rarity[t], -len(t)))]
    about = [r for r, w in zip(sources, words) if any(_covered(n, w) for n in names)]
    return about or sources


def coverage_report(query: str, sources: list[dict]) -> dict:
    """Everything the coverage check looked at, for the trace: question terms
    found or missing in the retrieved sources, doses stated in the question,
    and any age the question is about."""
    vocab = _source_vocab(sources)
    terms = _salient_terms(query)
    covered = [t for t in terms if _covered(t, vocab)]
    uncovered = _prioritise([t for t in terms if not _covered(t, vocab)], query)

    # A form or route the question names has to be described by a source about
    # the medicine it names. Without this, "Caloradine XR" was answered with
    # the plain product's dose, because "xr" was too short to check and
    # "topical" appeared in some other medicine's passage.
    about = _sources_about(query, covered, sources)
    about_vocab = _source_vocab(about)
    unsupported_forms = [t for t in terms if t in _FORM_TERMS and not _covered(t, about_vocab)]

    # A dose the question states must come from a source about what the
    # question names, not from another medicine's passage that happened to be
    # retrieved alongside it ("50 mcg of Rulpuraprex" is not supported by
    # "Lembitulex is given as 50 mcg").
    source_values: set[str] = set()
    for record in about:
        source_values |= _canonical_pairs(record.get("text", ""))
    question_values = sorted(_canonical_pairs(query))
    unsupported_values = [v for v in question_values if v not in source_values]

    age = _stated_age(query)
    age_report = None
    if age is not None:
        years, phrase = age
        if years < 18:
            group, needed = "children", _CHILD_TERMS
        elif years >= 65:
            group, needed = "older adults", _OLDER_TERMS
        else:
            group, needed = "adults", set()
        age_report = {
            "phrase": phrase,
            "years": years,
            "group": group,
            "covered": not needed or bool(needed & vocab),
        }

    return {
        "checked_terms": terms,
        "covered": covered,
        "uncovered": uncovered,
        "question_values": question_values,
        "unsupported_values": unsupported_values,
        "unsupported_forms": unsupported_forms,
        "age": age_report,
        "unanswerable_request": _unanswerable_request(query, sources),
    }


def coverage_check(query: str, sources: list[dict]) -> tuple[bool, str]:
    """Return (covered, detail). The detail names the most important gap:
    a dose the question states that no source supports, then an age group no
    source addresses, then the most significant missing term, then a request
    the sources can't answer (a dose limit or a combined dose they don't give)."""
    report = coverage_report(query, sources)
    if report["unsupported_values"]:
        value = report["unsupported_values"][0]
        return False, f"the value '{value}' in the question does not appear in any trusted source"
    if report["unsupported_forms"]:
        form = report["unsupported_forms"][0]
        return False, (f"no trusted source about this describes a '{form}' form or route, "
                       f"and the dose of one form is not the dose of another")
    age = report["age"]
    if age and not age["covered"]:
        return False, (f"the question is about {_with_article(age['phrase'])}, and no "
                       f"trusted source covers {age['group']}")
    if report["uncovered"]:
        return False, f"the term '{report['uncovered'][0]}' does not appear in any trusted source"
    if report["unanswerable_request"]:
        return False, report["unanswerable_request"]
    return True, "all question terms are represented in the retrieved sources"


# --------------------------------------------------------------------------
# Grounding check (deterministic, embedding-based)
# --------------------------------------------------------------------------

def _lexical_overlap(claim_text: str, source_text: str) -> float:
    """Fraction of the claim's content words that appear in the source. A claim
    lifted verbatim from a source scores ~1.0; this is the right signal for
    extractive claims, where a single sentence can have low embedding cosine
    against its multi-topic parent passage."""
    claim_tokens = {w for w in (m.group(0).lower() for m in _WORD.finditer(claim_text))
                    if len(w) >= 3}
    if not claim_tokens:
        return 0.0
    source_tokens = {m.group(0).lower() for m in _WORD.finditer(source_text)}
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def grounding_score(claim_text: str, source_text: str) -> float:
    """Take the stronger of two signals: embedding cosine (catches faithful
    paraphrase from the LLM) and lexical containment (catches verbatim extracted
    text). Either path alone is sufficient evidence of grounding."""
    vectors = retrieval.embed([claim_text, source_text])
    cosine = float(vectors[0] @ vectors[1])
    lexical = _lexical_overlap(claim_text, source_text)
    return max(cosine, lexical)


def check_claim_grounded(claim_text: str, source_ids: list[str],
                         grounding_min: float | None = None) -> tuple[bool, float]:
    """A claim is grounded only if it cites at least one valid source and its
    similarity to the best cited source meets the threshold."""
    threshold = config.GROUNDING_MIN if grounding_min is None else grounding_min
    valid_texts = [t for t in (retrieval.corpus_text_for(sid) for sid in source_ids) if t]
    if not valid_texts:
        return False, 0.0
    best = max(grounding_score(claim_text, text) for text in valid_texts)
    return best >= threshold, best


# --------------------------------------------------------------------------
# Dosage guard (deterministic, never calls a model)
#
# The cheapest check that catches the most dangerous mistake: every value with a
# clinical unit in the answer must be supported by a retrieved source.
#
# Matching is done on a CANONICAL form, not raw substrings, so the guard is not
# fooled by surface variation. "fifteen milligrams", "15 mg", and "15mg" all
# canonicalise to (15.0, "mg"); a value in the answer is supported only if its
# canonical (number, unit) pair also appears in a source. This catches
# written-out numbers, unit spellings (microgram/micrograms/mcg), and spacing
# differences that a verbatim substring check would miss.
# --------------------------------------------------------------------------

# Canonical unit for every spelling we recognise.
_UNIT_CANON = {
    "mg/day": "mg/day", "mg/kg": "mg/kg",
    "mg": "mg", "milligram": "mg", "milligrams": "mg", "milligramme": "mg",
    "milligrammes": "mg",
    "mcg": "mcg", "ug": "mcg", "µg": "mcg", "microgram": "mcg",
    "micrograms": "mcg", "microgramme": "mcg", "microgrammes": "mcg",
    "g": "g", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "millilitres": "ml", "cc": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "units": "units", "unit": "units", "iu": "iu",
    "%": "%", "percent": "%",
}
# Ordered alternation: longer / multi-word spellings first so they win.
_UNIT_ALT = "|".join(
    re.escape(u) for u in sorted(_UNIT_CANON, key=len, reverse=True)
)

# Number words for written-out quantities (0 to 99 plus a few fractions).
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_FRACTIONS = {"half": 0.5, "quarter": 0.25}
_NUM_WORDS = {**_ONES, **_TENS, **_FRACTIONS}

# A quantity is either digits (15, 2.5) or one-to-two number words
# (fifteen, twenty five, twenty-five), optionally part of a range.
_NUMWORD_ALT = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
_QTY = rf"(?:\d+(?:\.\d+)?|(?:{_NUMWORD_ALT})(?:[\s-]+(?:{_NUMWORD_ALT}))?)"
_RANGE_SEP = r"(?:\s*(?:to|-|–|—|or)\s*)"
_VALUE_RE = re.compile(
    rf"\b({_QTY})(?:{_RANGE_SEP}({_QTY}))?\s*({_UNIT_ALT})\b",
    re.IGNORECASE,
)


# Ages a question can state: "6 year old", "six-year-old", "18 months old",
# "aged 70", "age 4". Group 1 is the number and group 2 the unit for the "old"
# forms; group 3 is the number for the "aged" forms.
_AGE_QTY = rf"(?:\d{{1,3}}|(?:{_NUMWORD_ALT})(?:[\s-]+(?:{_NUMWORD_ALT}))?)"
_AGE_RE = re.compile(
    rf"\b({_AGE_QTY})[\s-]*(years?|yrs?|months?|weeks?|days?)[\s-]*old\b"
    rf"|\baged?\s+({_AGE_QTY})\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _word_to_number(token: str) -> float | None:
    """Parse a digit string or one-to-two number words into a number."""
    token = token.strip().lower()
    try:
        return float(token)
    except ValueError:
        pass
    parts = re.split(r"[\s-]+", token)
    total = 0.0
    matched = False
    for p in parts:
        if p in _NUM_WORDS:
            total += _NUM_WORDS[p]
            matched = True
        else:
            return None
    return total if matched else None


def _fmt_num(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def _canonical_pairs(text: str) -> set[str]:
    """Every (number, unit) pair in the text, as canonical strings like
    "15 mg". A range yields a pair for each endpoint."""
    pairs: set[str] = set()
    for m in _VALUE_RE.finditer(text):
        unit = _UNIT_CANON.get(m.group(3).lower())
        if not unit:
            continue
        for raw in (m.group(1), m.group(2)):
            if raw is None:
                continue
            num = _word_to_number(raw)
            if num is not None:
                pairs.add(f"{_fmt_num(num)} {unit}")
    return pairs


def extract_values(text: str) -> list[str]:
    """Surface strings of detected value-with-unit spans, for display."""
    return [re.sub(r"\s+", " ", m.group(0)).strip() for m in _VALUE_RE.finditer(text)]


def dosage_guard(answer_text: str, sources: list[dict]) -> tuple[bool, str, list[str]]:
    """Return (ok, detail, checked_values).
    ok is False if any value-with-unit in the answer has no canonical match in a
    source. Canonicalisation means written-out numbers and unit spellings are
    matched against symbol forms in the sources."""
    answer_pairs = _canonical_pairs(answer_text)
    surface = extract_values(answer_text)
    if not answer_pairs:
        return True, "no dosage values to verify", []

    source_pairs: set[str] = set()
    for r in sources:
        source_pairs |= _canonical_pairs(r.get("text", ""))

    checked = sorted(answer_pairs)
    for pair in checked:
        if pair not in source_pairs:
            return (
                False,
                f"the value '{pair}' is not supported by any source",
                checked,
            )
    cited = ", ".join(checked)
    return True, f"{cited} verified against sources", checked
