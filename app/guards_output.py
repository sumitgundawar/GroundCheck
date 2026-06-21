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
}


def _salient_terms(query: str) -> list[str]:
    terms = []
    for match in _WORD.finditer(query.lower()):
        word = match.group(0)
        if len(word) < 4 or word in _STOPWORDS:
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


def _covered(term: str, vocab: set[str]) -> bool:
    for word in vocab:
        # Tolerate plural/verb-form variation via a shared prefix.
        threshold = min(5, len(term), len(word))
        if _shared_prefix_len(term, word) >= threshold and threshold >= 4:
            return True
    return False


def coverage_check(query: str, sources: list[dict]) -> tuple[bool, str]:
    """Return (covered, detail). If a salient term is uncovered, the second
    element names it for the refusal reason."""
    vocab: set[str] = set()
    for record in sources:
        # Scan the body, the title, and the section label. A question word may
        # legitimately live in the title (for example "review schedule") rather
        # than the prose, and that still counts as covered.
        vocab |= _source_vocabulary(record.get("text", ""))
        vocab |= _source_vocabulary(record.get("title", ""))
        vocab |= _source_vocabulary(record.get("section", ""))
    uncovered = [t for t in _salient_terms(query) if not _covered(t, vocab)]
    if uncovered:
        offending = uncovered[0]
        return False, f"the term '{offending}' does not appear in any trusted source"
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
