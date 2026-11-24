"""Verify an answer that something else wrote.

The caller sends a question, an answer from any model, and optionally the ids
of the sources the answer must rest on. The answer is split into claims and
each claim is looked for in the sources: the sentence (or two adjacent
sentences) that states it. The check is deliberately strict, in one direction:

- every number, unit and name in the claim must be in that passage;
- a negation the passage has may not be dropped, nor one it lacks added;
- a word that changes a dose, a population, a route, a frequency or a quantity
  may not appear unless the passage has it;
- a word the passage lacks that names something (an invented medicine, a
  different side effect) fails the claim, and any other new wording makes the
  claim "review", never "pass".

An answer passes only when every claim is supported, its doses are stated by a
source about the same medicine, it answers the question that was asked (about
the medicine or condition named in it), and, when a patient is given, the
patient checks find nothing blocking. Nothing here calls a model.
"""

from __future__ import annotations

import re
import time
from collections import Counter
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from . import config, guards_input, guards_output, pipeline, retrieval
from .schemas import (Claim, ClaimSupport, DoseIssue, Explanation, LLMClaim, PatientContext, Settings,
                      VerifiedClaim, VerifyRedaction, VerifyResponse, VerifySource)


class SourceNotFound(LookupError):
    def __init__(self, ids: list[str]):
        super().__init__(", ".join(ids))
        self.ids = ids


# --- Text ----------------------------------------------------------------------

_INVISIBLE = re.compile("[​-‏⁠﻿­‪-‮⁦-⁩]")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", text))
    return text.replace("\r\n", "\n").replace("\r", "\n")


_CITATION = re.compile(r"\[([^\[\]]{1,80})\]")
_ID_LIKE = re.compile(r"^[A-Za-z]{2,}[-_:][A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?![\[a-z])")
_TRAILING_CITE = re.compile(r"([.!?])\s*((?:\[[^\[\]]{1,80}\]\s*)+)(?=\S|$)")
_BULLET = re.compile(r"^\s*(?:[-*•●]|\d{1,2}[.)])\s+")


def split_claims(text: str) -> list[str]:
    """The answer's sentences, one per claim. A citation stays with the
    sentence before it."""
    out = []
    for line in text.split("\n"):
        line = _BULLET.sub("", line.replace("**", "").replace("__", "")).strip()
        line = _TRAILING_CITE.sub(lambda m: " " + m.group(2).strip() + m.group(1) + " ", line)
        if not line:
            continue
        out.extend(part.strip() for part in _SPLIT.split(line) if part.strip())
    return out


# --- Words ---------------------------------------------------------------------

_HEDGES = re.compile(r"\b(?:" + "|".join([
    r"as far as i (?:know|can tell|remember)", r"to (?:the best of )?my knowledge", r"in my (?:opinion|view)",
    r"i (?:think|believe|guess|suppose|recall|understand|assume|would say)(?: that)?",
    r"i(?:'m| am) (?:not )?(?:entirely |completely |totally |100% )?(?:sure|certain)(?: that| but| though)?",
    r"i (?:do not|don't) (?:know|remember)(?: for (?:certain|sure)| exactly)?",
    r"it (?:seems|appears)(?: that)?", r"according to (?:the |these |my |our |this )?"
    r"(?:sources?|guidance|guidelines?|label|labelling|documentation|formulary|information|documents?|text|passage)",
    r"(?:please )?note(?: that)?", r"basically", r"generally(?: speaking)?", r"maybe", r"perhaps", r"probably", r"possibly", r"presumably",
    r"in general", r"overall",
    r"in short", r"in summary", r"sure", r"certainly", r"of course",
]) + r")\b", re.IGNORECASE)

_REFUSALS = re.compile("|".join([
    r"(?:i|we)(?:'m| am)? (?:sorry|afraid)?,? ?(?:but )?(?:i |we )?(?:cannot|can't|can not|am unable to|'m unable to|"
    r"won't|will not|am not able to|'m not able to) (?:answer|help|provide|say|tell|advise|recommend|confirm|"
    r"determine|comment)(?: (?:this|that|it|the question|your question|with that|safely|reliably|confidently|"
    r"from the sources|from the available sources|without more information))*",
    r"(?:i )?(?:do not|don't) (?:know|have enough information)(?: the answer| that| this)?",
    r"(?:there is|there's) not enough information(?: to answer(?: this| safely)*)?",
    r"(?:the|these) sources? (?:do|does) not (?:contain|cover|say|state|mention)(?: enough information| this| that| it)?",
    r"(?:please )?(?:consult|ask|speak to|talk to|contact|see) (?:a|an|your|the) (?:doctor|pharmacist|clinician|"
    r"physician|nurse|prescriber|specialist|healthcare professional|medical professional|health professional)"
    r"(?: (?:before|about|for|if|who)[a-z ]{0,40})?",
    r"(?:this|that) is not (?:medical )?advice", r"(?:this is )?for information(?:al)? (?:purposes )?only",
    r"(?:i )?hope (?:this|that) helps", r"thank(?:s| you)(?: for (?:asking|your question))?", r"hello", r"hi",
    r"(?:the )?answer is (?:unknown|not known|unclear)", r"i (?:have )?no idea",
]), re.IGNORECASE)

_AFFIRMATION = {"yes", "yeah", "yep", "no", "nope", "correct", "right", "true", "false", "definitely", "absolutely"}

_FUNCTION = frozenset("""
a an the and or but if then than so of in on at to for from by with as into onto per via this that these those it its
they them their he she his her we our you your i me my is are was were be been being am do does did has have had
there here which who whom whose what when where how also very just about around
over under again still even such some other others same one's itself themselves been more_than whereas while although
though there's it's that's
can't won't don't doesn't isn't aren't wasn't weren't shouldn't couldn't wouldn't mustn't
""".split())

_CRITICAL = frozenset("""
once twice thrice daily nightly weekly monthly hourly hour hours day days week weeks month months year years minute minutes
morning evening night bedtime every each per
adult adults child children kid kids paediatric pediatric pediatrics infant infants neonate neonates newborn elderly older
geriatric pregnant pregnancy breastfeeding lactating lactation men women male female
oral orally topical topically intravenous intravenously iv im sc subcutaneous intramuscular sublingual rectal inhaled
injection infusion tablet tablets capsule capsules solution suspension complex syrup cream ointment drops patch depot
all any both either neither none no never always only must required mandatory need needs needed
more less most least higher lower greater fewer larger smaller increase increased increases increasing decrease
decreased decreases decreasing reduce reduced reduces reducing double doubled triple half quarter
before after until during while
above below exceed exceeds exceeding maximum minimum max min limit up down
first second third last initial then next
start started starts starting stop stopped stops stopping discontinue discontinued continue continued continues
avoid avoided avoids avoiding unless except without
food meal meals empty stomach
whenever regardless irrespective anytime freely unrestricted
safe safely safer harmless recommended recommend recommends advisable permitted allowed ok okay fine effective
effectively cure cures cured curing prevent prevents prevented eliminate eliminates
""".split())

_NEGATIONS = frozenset("""
not no never cannot without avoid avoided avoids avoiding contraindicated contraindication prohibited forbidden unsafe
neither nor none nothing unless except
""".split())

_CONTRACTIONS = [(re.compile(p, re.IGNORECASE), r) for p, r in (
    (r"\bcan['’]t\b", "cannot"), (r"\bwon['’]t\b", "will not"), (r"\bshan['’]t\b", "shall not"),
    (r"\b(\w+)n['’]t\b", r"\1 not"))]

_NUM_RE = re.compile(rf"\b{guards_output._QTY}(?![A-Za-z])", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> tuple[list[str], set[str], set[str]]:
    """(words, numbers, number-unit pairs) of a text, lower-cased. Number
    words become numbers and unit spellings are made canonical, so "fifteen
    milligrams" and "15 mg" are the same fact."""
    text = text.lower().replace("’", "'")
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    pairs: set[str] = set()
    numbers: set[str] = set()

    def strength(match: re.Match) -> str:
        pairs.add(f"{guards_output._shown(guards_output.concentrations(match.group(0))[0])} mg/ml")
        return " "

    text = guards_output._CONC_RE.sub(lambda m: strength(m) if guards_output.concentrations(m.group(0)) else " ", text)

    def value(match: re.Match) -> str:
        unit = guards_output._UNIT_CANON.get(match.group(3).lower())
        for raw in (match.group(1), match.group(2)):
            if raw is None:
                continue
            number = guards_output._word_to_number(raw)
            if number is not None:
                if unit:
                    pairs.add(f"{guards_output._fmt_num(number)} {unit}")
                else:
                    numbers.add(guards_output._fmt_num(number))
        return " "

    text = guards_output._VALUE_RE.sub(value, text)

    def plain(match: re.Match) -> str:
        number = guards_output._word_to_number(match.group(0))
        if number is not None:
            numbers.add(guards_output._fmt_num(number))
        return " "

    text = _NUM_RE.sub(plain, text)
    return _WORD.findall(text), numbers, pairs


_PLAIN_NEGATION = frozenset({"no", "not", "never", "cannot"})


def _plain(token: str) -> str:
    """The everyday negations count as one: "never" for "not" is a choice of
    word. Any other negation ("without", "avoid") stands for itself."""
    return "not" if token in _PLAIN_NEGATION else token


def _negated(tokens) -> set[str]:
    """The word each negation governs: the first word of substance after it
    ("not be *used*", "without *specialist*")."""
    found = set()
    for i, token in enumerate(tokens):
        if token in _NEGATIONS:
            for later in tokens[i + 1: i + 6]:
                if later not in _FUNCTION and later not in _NEGATIONS and len(later) > 1:
                    found.add(later)
                    break
    return found


def _critical_sequence(words) -> list[str]:
    return [w for w in words if w in _CRITICAL or w in _NEGATIONS]


def _same(a: str, b: str) -> bool:
    return a == b or (not _is_name(a) and bool(guards_output._stems(a) & guards_output._stems(b)))


def _in_order(claim: list[str], window: list[str]) -> str | None:
    """None if `claim` is a subsequence of `window` (each word matched once, in
    order); otherwise the first word that could not be placed."""
    pos = 0
    for word in claim:
        while pos < len(window) and not _same(word, window[pos]):
            pos += 1
        if pos >= len(window):
            return word
        pos += 1
    return None


def _is_name(word: str) -> bool:
    """A word that names something specific: not everyday English and not a
    clinical term. It must match exactly: "Caloradin" is not "Caloradine"."""
    return not guards_output._is_common(word) and not guards_output._is_clinical(word)


def _meaningful(words: list[str]) -> list[str]:
    return [w for w in words if w not in _FUNCTION and len(w) > 1]


@dataclass(frozen=True)
class _Window:
    source_id: str
    start: int
    end: int
    text: str
    sentences: int
    words: frozenset[str]
    stems: frozenset[str]
    numbers: frozenset[str]
    pairs: frozenset[str]
    pairs_norm: frozenset[str]
    negations: tuple[str, ...]
    tokens: tuple[str, ...]
    numbers_seen: tuple[str, ...] = ()


@lru_cache(maxsize=4096)
def _windows(source_id: str, text: str) -> tuple[_Window, ...]:
    spans = retrieval.sentence_spans(text)
    out = []
    for i in range(len(spans)):
        for n in (1, 2):
            if i + n > len(spans):
                continue
            start, end = spans[i][0], spans[i + n - 1][1]
            piece = text[start:end]
            words, numbers, pairs = _tokens(piece)
            stems: set[str] = set()
            for w in words:
                stems |= guards_output._stems(w)
            out.append(_Window(source_id, start, end, piece, n, frozenset(words), frozenset(stems),
                               frozenset(numbers), frozenset(pairs),
                               frozenset(guards_output.normalise_pair(p) for p in pairs),
                               tuple(w for w in words if w in _NEGATIONS), tuple(words)))
    return tuple(out)


def _has(word: str, window: _Window) -> bool:
    if word in window.words:
        return True
    if _is_name(word):
        return False
    return bool(guards_output._stems(word) & window.stems)


@dataclass
class Assessment:
    text: str
    status: str = "supported"          # supported | review | unsupported
    source_id: str | None = None
    support: ClaimSupport | None = None
    issues: list[str] = field(default_factory=list)
    dose_values: list[str] = field(default_factory=list)
    code: str = "claim_ungrounded"
    coverage: float = 0.0
    clean: str = ""


def _topic(record: dict) -> str:
    return (record.get("topic") or "").lower()


def canonical_text(text: str) -> str:
    """The text with quantities written the way sources write them: "fifteen
    milligrams" as "15 mg"."""
    def value(m: re.Match) -> str:
        unit = guards_output._UNIT_CANON.get(m.group(3).lower())
        a = guards_output._word_to_number(m.group(1))
        b = guards_output._word_to_number(m.group(2)) if m.group(2) else None
        if unit is None or a is None or (m.group(2) and b is None):
            return m.group(0)
        shown = guards_output._fmt_num(a) + (f"-{guards_output._fmt_num(b)}" if b is not None else "")
        return f"{shown} {unit}"

    return guards_output._VALUE_RE.sub(value, text)


def _align_units(text: str, window: _Window) -> str:
    """The claim with each dose written the way the source writes it, when it
    is the same quantity in another unit ("0.015 g" as "15 mg"), so the
    checks that compare wording see the same thing."""
    def swap(m: re.Match) -> str:
        unit = guards_output._UNIT_CANON.get(m.group(3).lower())
        a = guards_output._word_to_number(m.group(1))
        if unit is None or a is None or m.group(2):
            return m.group(0)
        pair = f"{guards_output._fmt_num(a)} {unit}"
        found = guards_output.matching_pair(pair, set(window.pairs), [window.text])
        return found if found and found != pair else m.group(0)

    return guards_output._VALUE_RE.sub(swap, text)


def _cited(text: str) -> tuple[str, list[str]]:
    ids = []

    def take(match: re.Match) -> str:
        inner = match.group(1).strip()
        if _ID_LIKE.match(inner):
            ids.append(inner)
            return " "
        if re.fullmatch(r"[\d,;\s–-]+", inner):      # [1], [2, 3]: not ours
            return " "
        return match.group(0)

    return _CITATION.sub(take, text), list(dict.fromkeys(ids))


_PII_FILLER = frozenset("""born dob birth date nhs number mrn name named called phone telephone mobile email address
postcode record id ssn contact mr mrs ms miss dr prof lives living patient""".split())


def assess(text: str, candidates: list[dict], allowed: set[str] | None,
           prefer: frozenset[str] = frozenset()) -> Assessment | None:
    """How well the sources support one claim, or None when the sentence is
    not a claim (a greeting, a refusal, a pointer to a clinician)."""
    body, cited = _cited(text)
    from .deid import PLACEHOLDER

    has_pii = bool(PLACEHOLDER.search(body))
    body = PLACEHOLDER.sub(" ", body)
    flat = " ".join(_HEDGES.sub(" ", body).split()).strip(" ,;:-")
    plain = re.sub(r"[^\w\s']", " ", flat.lower())
    plain = " ".join(plain.split())
    if body.strip().endswith("?"):
        return None
    if _REFUSALS.fullmatch(plain) or _REFUSALS.fullmatch(" ".join(re.sub(r"[^\w\s']", " ", body.lower()).split())):
        return None
    words, numbers, pairs = _tokens(flat)
    if has_pii:
        words = [w for w in words if w not in _PII_FILLER]
    if not _meaningful(words) and not numbers:
        return None
    result = Assessment(text=text, clean=" ".join(flat.split()))
    unknown = [c for c in cited if retrieval.record_for(c) is None]
    outside = [c for c in cited if c not in unknown and allowed is not None and c not in allowed]
    for c in unknown:
        result.issues.append(f"cites '{c}', which is not a source")
    for c in outside:
        result.issues.append(f"cites '{c}', which is not one of the sources given")
    if unknown or outside:
        result.status, result.code = "unsupported", "citation_invalid"
        return result
    pool = [r for r in candidates if not cited or r["id"] in cited]
    if not pool:
        result.status = "unsupported"
        result.issues.append("no source was available to check it against")
        return result

    content = list(dict.fromkeys(_meaningful(words)))
    pairs_norm = {guards_output.normalise_pair(p) for p in pairs}
    needed = len(content) + len(numbers) + len(pairs_norm)
    best, best_key = None, None
    for record in pool:
        for window in _windows(record["id"], record["text"]):
            hit = (sum(1 for w in content if _has(w, window)) + len(numbers & window.numbers)
                   + len(pairs_norm & window.pairs_norm))
            key = (hit / needed if needed else 0.0, _topic(record) in prefer, -window.sentences)
            if best_key is None or key > best_key:
                best, best_key = window, key
    assert best is not None
    result.coverage = best_key[0]
    result.clean = _align_units(canonical_text(result.clean), best)
    if has_pii:
        result.clean = " ".join(w for w in result.clean.split() if re.sub(r"\W", "", w.lower()) not in _PII_FILLER)
    result.source_id = best.source_id
    result.support = ClaimSupport(source_id=best.source_id, start=best.start, end=best.end, sentence=best.text)
    if best_key[0] < 0.5:
        result.status = "unsupported"
        result.issues.append("no source states this")
        return result

    fail: list[str] = []
    review: list[str] = []
    for n in sorted(numbers - best.numbers, key=float):
        fail.append(f"the number {n} is not stated by the source")
    for pair in sorted(pairs):
        if not guards_output.value_supported(pair, set(best.pairs), [best.text]):
            fail.append(f"'{pair}' is not stated by the source")
            result.dose_values.append(pair)
    novel = [w for w in content if not _has(w, best)]
    for w in novel:
        if w in _CRITICAL or w in _NEGATIONS:
            fail.append(f"'{w}' changes the meaning and is not in the source")
        elif _is_name(w) or guards_output._is_clinical(w):
            fail.append(f"'{w}' is not in the source")
        else:
            review.append(f"'{w}' is not in the source")
    if not fail:
        stray = _in_order(_critical_sequence(words), _critical_sequence(best.tokens))
        if stray is not None:
            fail.append(f"'{stray}' is not in the source, or not where the claim puts it")
    if not fail and not review:
        stray = _in_order(content, [t for t in best.tokens if t not in _FUNCTION])
        if stray is not None:
            review.append(f"'{stray}' is not in the order the source gives it")
    claim_neg = [w for w in words if w in _NEGATIONS]
    win_neg = list(best.negations)
    if claim_neg and not win_neg:
        fail.append(f"it says '{claim_neg[0]}', which the source does not")
    elif claim_neg and not set(claim_neg) & set(win_neg) and not set(claim_neg) <= {"no", "not", "never", "cannot"} | set(win_neg):
        fail.append(f"its negation '{claim_neg[0]}' differs from the source's")
    missing = Counter(_plain(t) for t in win_neg) - Counter(_plain(t) for t in claim_neg)
    if missing:
        claim_set = set(content)
        for i, tok in enumerate(best.tokens):
            if _plain(tok) in missing:
                near = {t for t in best.tokens[max(0, i - 3): i + 5] if t not in _FUNCTION and t not in _NEGATIONS}
                if any(t in claim_set or any(guards_output._stems(t) & guards_output._stems(c) for c in claim_set)
                       for t in near):
                    fail.append(f"the source says '{tok}' here, and the claim leaves it out")
                    break
    if not fail and claim_neg and win_neg:
        theirs, ours = _negated(best.tokens), _negated(list(words))
        if ours and theirs and not any(_same(a, b) for a in ours for b in theirs):
            review.append("its negation applies to a different word than the source's")
    bare = re.match(r"(?:" + "|".join(sorted(_AFFIRMATION)) + r")\b\s*(?:[,.:;!\u2014-]|$)", flat.lower())
    if bare and not fail:
        review.append("a bare yes/no cannot be checked against a source")
    if fail:
        result.status, result.issues = "unsupported", fail + review
    elif review:
        result.status, result.issues, result.code = "review", review, "claim_unverified"
    return result


# --- The answer ------------------------------------------------------------------

_SOURCE_CAP = 12
# The best claim's similarity to the question. Correct answers in the evaluation
# set never fall below 0.45; an answer to some other question rarely reaches it.
MIN_RELEVANCE = 0.40


def _record(source_id: str) -> dict:
    return retrieval.record_for(source_id)


def _candidates(question: str, claims: list[str], given: list[str] | None, top_k: int) -> list[dict]:
    if given is not None:
        return [_record(i) for i in given]
    seen: dict[str, dict] = {}
    for record, _ in retrieval.search(question, top_k):
        seen.setdefault(record["id"], record)
    for text in claims:
        body, _ = _cited(text)
        if len(_meaningful(_tokens(body)[0])) < 1:
            continue
        for record, _ in retrieval.search(body, 3):
            seen.setdefault(record["id"], record)
    return list(seen.values())[:_SOURCE_CAP * 3]


def _merge_kinds(*redactions) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for found in redactions:
        for k, v in found.items():
            kinds[k] = kinds.get(k, 0) + v
    return kinds


_EXPLAIN = {
    "verified": "Every claim is stated by a trusted source and every dose matches.",
    "input_blocked": "The question was stopped by an input guard, so nothing was checked.",
    "claim_ungrounded": "At least one claim is not supported by the sources, or contradicts them.",
    "dose_unsupported": "A dose in the answer is not stated by a source about that medicine.",
    "patient_unsafe": "The patient's details make this answer unsafe to give.",
    "citation_invalid": "The answer cites a source that does not exist or is not among the sources given.",
    "answer_off_topic": "The claims may be true, but they are not about what the question asked.",
    "claim_unverified": "The claims are close to the sources but use wording that could not be matched; "
                        "a person should check them.",
    "no_checkable_claims": "The answer makes no claim that can be checked, so nothing was verified.",
    "no_relevant_source": "No trusted source is relevant to this question, so the answer cannot be verified.",
}


def _aspects(question: str) -> list[tuple[str, set[str], re.Pattern]]:
    """What the question asks for (a dose, side effects, an interaction...),
    from the same classifier that picks the sentences of an answer."""
    from . import llm

    found = [(name, sections, cue) for name, pattern, sections, cue in llm._INTENTS if pattern.search(question)]
    if re.search(r"\bnot (?:take|taken|use|used|given|be given)\b", question, re.IGNORECASE):
        found = [f for f in found if f[0] != "dose"]      # "who should not take" asks about contraindications
    return found


@lru_cache(maxsize=4096)
def _mixed_text(text: str) -> bool:
    from . import llm

    return sum(bool(cue.search(text)) for *_, cue in llm._INTENTS) >= 3


def _mixed(record: dict) -> bool:
    """A passage that covers several aspects at once (dose, side effects,
    contraindications...), so the section it is filed under says little about
    which one a sentence from it is about."""
    return _mixed_text(record.get("text", ""))


def _addresses(question: str, claims: list[VerifiedClaim]) -> bool | None:
    """Whether any claim is about what the question asks: it states a source
    in a section that answers it, or it reads like such an answer and comes
    from a passage that is not about some other aspect. None when the question
    does not say what it asks for. Needs only one claim, so an answer that
    also says more is not penalised."""
    from . import llm

    aspects = _aspects(question)
    if not aspects:
        return None
    specific = set().union(*(sections for _, _, sections, _ in llm._INTENTS))
    for claim in claims:
        record = _record(claim.source_id) if claim.source_id else None
        section = (record or {}).get("section", "")
        body = _cited(claim.text)[0]
        mixed = record is not None and _mixed(record)
        for _, sections, cue in aspects:
            if section in sections or ((section not in specific or mixed) and cue.search(body)):
                return True
    return False


def verify(question: str, answer: str, sources: list[str] | None = None, patient: PatientContext | None = None,
           ) -> tuple[VerifyResponse, dict]:
    """Check `answer` to `question`. Returns the response (without an audit
    id) and the extras to record with it. Raises SourceNotFound for ids that are
    not in the index."""
    started = time.perf_counter()
    retrieval.ensure_loaded()
    if sources is not None:
        missing = [i for i in sources if retrieval.record_for(i) is None]
        if missing:
            raise SourceNotFound(missing)
    cfg = Settings(use_llm_judge=False, force_extractive=True)
    q = guards_input.deidentify(clean(question))
    a = guards_input.deidentify(clean(answer))
    redaction = VerifyRedaction(kinds=_merge_kinds(dict(q.found), dict(a.found)), redacted_query=q.text,
                                redacted_answer=a.text)
    extras: dict = {"endpoint": "verify", "redacted_query": q.text, "test_run": True,
                    "redaction": redaction.model_dump(), "source_ids_requested": sources}

    def done(decision: str, code: str, claims=None, unsupported=None, dose=None, used=None, findings=None,
             missing_terms=None) -> tuple[VerifyResponse, dict]:
        response = VerifyResponse(
            decision=decision, reason_code=code, claims=claims or [], unsupported_claims=unsupported or [],
            dose_issues=dose or [], explanation=Explanation(summary=_EXPLAIN[code], missing_terms=missing_terms or []),
            redaction=redaction, audit_id="", latency_ms=int(round((time.perf_counter() - started) * 1000)),
            sources=[VerifySource(id=r["id"], title=r["title"], topic=r.get("topic", "")) for r in used or []],
            patient_findings=findings or [],
        )
        return response, extras

    scope = guards_input.check_scope_and_injection(q.text, True)
    if not scope.ok:
        return done("fail", "input_blocked")

    sentences = split_claims(a.text)[:config.API_MAX_CLAIMS * 2]
    candidates = _candidates(q.text, sentences, sources, cfg.top_k)
    allowed = set(sources) if sources is not None else None
    named = retrieval.topics_mentioned(q.text)
    prefer = frozenset(named.values())
    assessed = [x for x in (assess(s, candidates, allowed, prefer) for s in sentences) if x is not None]
    assessed = assessed[:config.API_MAX_CLAIMS]
    if not assessed:
        return done("review", "no_checkable_claims", used=candidates[:_SOURCE_CAP])
    if not candidates:
        return done("review", "no_relevant_source")

    # The standard guards, applied to the same claims.
    llm_claims = []
    for x in assessed:
        body, cited = _cited(x.text)
        ids = [i for i in cited if retrieval.record_for(i)] or ([x.source_id] if x.source_id else [])
        llm_claims.append(LLMClaim(text=x.clean or " ".join(body.split()), source_ids=ids))
    stage = pipeline.verify_claims(q.text.strip(), llm_claims, candidates, patient, cfg)

    dose_issues: list[DoseIssue] = []
    out: list[VerifiedClaim] = []
    unsupported: list[int] = []
    for n, (x, lc) in enumerate(zip(assessed, llm_claims)):
        g = stage.claims[n]
        issues = list(x.issues)
        dose_ok, detail, _ = guards_output.dosage_guard(lc.text, candidates, q.text.strip())
        if not dose_ok:
            issues.append(detail)
            value = re.search(r"'([^']+)'", detail)
            dose_issues.append(DoseIssue(claim=n, value=value.group(1) if value else "", message=detail))
        for v in x.dose_values:
            if not any(d.claim == n and d.value == v for d in dose_issues):
                dose_issues.append(DoseIssue(claim=n, value=v, message=f"'{v}' is not stated by the source"))
        if not g.grounded and x.status == "supported":
            issues.append("it is not close enough to the source it cites")
        grounded = x.status == "supported" and g.grounded and dose_ok
        if not grounded:
            unsupported.append(n)
        out.append(VerifiedClaim(text=x.text, grounded=grounded, source_id=x.source_id, support=x.support,
                                 relevance=g.relevance, issues=issues))
    used = [r for r in candidates if r["id"] in {c.source_id for c in out if c.source_id}] or candidates[:4]
    used = used[:_SOURCE_CAP]
    extras["retrieved"] = [{"id": r["id"], "score": 0.0} for r in candidates]

    def resolve(decision: str, code: str, **kw):
        return done(decision, code, claims=out, unsupported=unsupported, dose=dose_issues, used=used,
                    findings=stage.findings, **kw)

    if any(x.code == "citation_invalid" and x.status == "unsupported" for x in assessed):
        return resolve("fail", "citation_invalid")
    if dose_issues:
        return resolve("fail", "dose_unsupported")
    if any(x.status == "unsupported" for x in assessed) or stage.reason_code == "claim_ungrounded":
        return resolve("fail", "claim_ungrounded")
    if stage.reason_code == "patient_unsafe":
        return resolve("fail", "patient_unsafe")

    if named:
        wanted = set(named.values())
        about = {_topic(_record(c.source_id) or {}) for c in out if c.source_id}
        about |= {t for c in out for t in retrieval.topics_mentioned(c.text).values()}
        if not wanted & about:
            return resolve("fail", "answer_off_topic", missing_terms=sorted(named))
    elif sources is None:
        asked = {r["id"] for r, _ in retrieval.search(q.text, cfg.top_k)}
        if not asked & {c.source_id for c in out if c.source_id}:
            return resolve("review", "no_relevant_source")
    addressed = _addresses(q.text, out)
    if addressed is False or (addressed is None and max(c.relevance for c in out) < MIN_RELEVANCE):
        return resolve("review", "answer_off_topic")
    if any(x.status == "review" for x in assessed):
        return resolve("review", "claim_unverified")
    return resolve("pass", "verified")
