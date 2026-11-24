"""De-identification of questions before they're searched, sent to a model,
logged or stored.

Each detector proposes candidate spans for one kind of identifier, each with a
base score. Validators (checksums, real calendar dates) and nearby cue words
(within four words either side) raise or lower the score, candidates that
overlap are resolved by score and then length, and what survives is replaced
with a placeholder such as [NAME] or [DATE] in a single pass, so the question
still reads naturally and nothing identifying leaves this step:

- Names: after a title (Mr, Mrs, Dr... in any case), a relation or role cue
  (husband, nurse, "seen by"...), or a cue like "patient" and "name:"; a given
  name from a bundled list followed by a capitalised surname; "SURNAME, Given";
  and a lone given name only beside age, pronoun or patient cues. Words the
  document index knows, medicine names, months and weekdays are never names.
- Dates: numeric and written forms that are real calendar dates, and a month
  with a day or year. A year on its own is kept.
- Ages over 89 become 90, as HIPAA Safe Harbor requires. Younger ages are
  kept: dosing depends on them.
- NHS numbers (mod-11), US Social Security numbers, UK National Insurance
  numbers, provider numbers (NPI, Luhn with the 80840 prefix), payment cards
  (Luhn, 13 to 19 digits), IBANs (mod-97) and labelled record, insurance and
  similar ids.
- Phone numbers, email addresses, web addresses and IP addresses.
- UK postcodes, US ZIP codes after a state, and street addresses.
- Any remaining run of seven or more digits.

If spaCy or Presidio is installed, DEID_NER=auto also lets it propose people,
places and dates. It never overrides a known term and its absence changes
nothing.

This is mostly rules. It catches identifiers written in common forms, not
every way a person can be identified: a rare condition or a unique event can
still identify someone. Treat it as a safety net, not permission to type
patient details into questions."""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Callable

from . import config

PLACEHOLDER = re.compile(
    r"\[(?:NAME|DATE|NHS NUMBER|SSN|ID|PHONE|EMAIL|URL|IP ADDRESS|POSTCODE|ZIP|ADDRESS|NUMBER|CARD)\]")

_PLACEHOLDERS = {
    "name": "[NAME]", "date": "[DATE]", "NHS number": "[NHS NUMBER]", "Social Security number": "[SSN]",
    "record number": "[ID]", "provider number": "[ID]", "bank account": "[ID]", "national insurance number": "[ID]",
    "phone number": "[PHONE]", "email address": "[EMAIL]", "web address": "[URL]", "IP address": "[IP ADDRESS]",
    "postcode": "[POSTCODE]", "ZIP code": "[ZIP]", "street address": "[ADDRESS]", "place": "[ADDRESS]",
    "long number": "[NUMBER]", "card number": "[CARD]",
}

THRESHOLD = 0.5

_MONTH = (r"(?:(?i:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)|May)")
_DAY = r"(?:[12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?"
_UNIT_AFTER = (r"(?!\s*(?:mg|mcg|µg|g|kg|ml|mL|l|units?|iu|IU|mmol|%|tablets?|doses?|hours?|hrs?|days?|weeks?|"
               r"months?|years?)\b)")

_TITLE = r"(?i:mr|mrs|ms|miss|mx|dr|prof|professor)"
_NAME_WORD = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"


@dataclass
class Entity:
    """One thing removed, by position in the original text. Never its value."""

    kind: str
    start: int
    end: int
    score: float
    detector: str


@dataclass
class Result:
    text: str
    found: Counter = field(default_factory=Counter)
    entities: list[Entity] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.found)

    def summary(self) -> str:
        """For the trace: what kinds were removed, never the values."""
        if not self.found:
            return "no identifiers found"
        parts = [f"{n} {kind}{'' if n == 1 else 's'}" for kind, n in sorted(self.found.items())]
        return "removed " + ", ".join(parts)


@dataclass
class _Candidate:
    start: int
    end: int
    kind: str
    score: float
    detector: str
    repl: str | None = None


# --- Validators -------------------------------------------------------------

def _nhs_checksum_ok(digits: str) -> bool:
    if len(digits) != 10 or len(set(digits)) == 1:
        return False
    total = sum(int(d) * w for d, w in zip(digits[:9], range(10, 1, -1)))
    check = 11 - total % 11
    check = 0 if check == 11 else check
    return check != 10 and check == int(digits[9])


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _npi_ok(digits: str) -> bool:
    return len(digits) == 10 and _luhn_ok("80840" + digits)


def _ssn_ok(digits: str) -> bool:
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"


def _iban_ok(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).upper()
    if not 15 <= len(compact) <= 34:
        return False
    moved = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def _real_date(a: int, b: int, year: int) -> bool:
    for day, month in ((a, b), (b, a)):
        try:
            date(year, month, day)
            return True
        except ValueError:
            continue
    return False


def _plausible_numeric_date(text: str) -> bool:
    parts = [int(p) for p in re.split(r"[/.-]", text)]
    if len(text.split("-")[0]) == 4 and text[:4].isdigit():
        year, month, day = parts
        try:
            date(year, month, day)
            return True
        except ValueError:
            return False
    a, b, year = parts
    if year < 100:
        year += 2000 if year < 30 else 1900
    return _real_date(a, b, year)


# --- Word lists -------------------------------------------------------------

@lru_cache(maxsize=None)
def _wordlist(name: str) -> frozenset[str]:
    try:
        return frozenset(w for w in (config.DATA_DIR / name).read_text(encoding="utf-8").split() if w)
    except OSError:
        return frozenset()


@lru_cache(maxsize=1)
def _formulary_words() -> frozenset[str]:
    try:
        from . import formulary

        return frozenset(w for name in formulary.index().by_name for w in name.split())
    except Exception:  # noqa: BLE001 - a missing formulary only means fewer exclusions
        return frozenset()


_MONTH_DAY_WORDS = frozenset(
    "january february march april may june july august september october november december jan feb mar apr jun jul "
    "aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur "
    "thurs fri sat sun".split())

# Words that follow a cue or a first name without being a surname.
_FUNCTION = frozenset("""
with who whose whom has had have is was were are be been being on in at for after before taking the a an and or of to
from what how when which where why should can could would will shall may might must if aged age male female man woman
boy girl child adult elderly pregnant i my our their his her its your this that these those it he she they we you
presenting admitted discharged not no yes ok okay hello hi hey dear please thanks thank sorry sir madam however also
then so but as by into over under about than there here some any all each every both either neither just only very
presents complains complained reports reported states stated says said asks asked needs needed wants wanted takes
took gets got goes went comes came seems seemed looks looked feels felt does did done doing now today tomorrow
yesterday tonight morning evening afternoon last next previous prior current new old young other another same
practitioner specialist manager consultant doctor nurse carer physician surgeon pharmacist therapist midwife
registrar mr mrs ms miss mx dr prof professor clinic ward hospital surgery unit department team service centre center
""".split())

# Given names that are also everyday words: on their own they are not names,
# and before a capitalised word they are only if that word is a surname.
_AMBIGUOUS = frozenset("""
will may bill pat sue art mark rose hope joy dawn grace faith charity patience honor liberty bliss pearl ruby amber
hazel ivy iris lily daisy poppy holly heather ginger candy cherry coral crystal diamond sage summer autumn april june
august jack king major rich frank grant sandy penny chip bob rob don ray rod jay ken ron sam tim tom ben dan max guy
miles drew dean earl glen chase case cash gray bird buck bunny baby infant judge justice prince princess queen lord
baron bride butler cannon carry else list part page root real revised plastic space sergeant journey lucky happy
fancy noble precious sincere sanity serenity glad glory harmony heaven hardy honey jewel miracle rogue royal shell
skip shadow rich brook brooks blaze angel bella belle destiny eden essence fern flora forest genesis haven june
lane mason ward wells wade chance mercy merry sunny
""".split())

_SURNAMES = frozenset("""
smith jones taylor brown williams wilson johnson davies robinson wright thompson evans walker white roberts green hall
wood jackson clarke clark turner hill scott cooper morris ward moore king watson baker harrison morgan patel young
allen mitchell james anderson phillips lee bell parker davis miller martin thomas hughes edwards lewis harris cook
bailey kelly murphy khan ali ahmed shah singh kaur begum hussain ahmad akhtar rahman chowdhury nguyen garcia
rodriguez martinez hernandez lopez gonzalez perez sanchez ramirez torres flores rivera gomez diaz reyes cruz
morales ortiz gutierrez chavez ruiz mendoza vasquez castillo jimenez moreno romero herrera medina aguilar
oneil o'neil o'brien o'connor
""".split())

_RELATION_CUE = (r"(?i:husband|wife|son|daughter|mother|father|mum|mom|mummy|dad|daddy|brother|sister|partner|spouse|"
                 r"grandson|granddaughter|grandmother|grandfather|nurse|carer|caregiver|friend|neighbour|neighbor|"
                 r"seen\s+by|referred\s+by|seen\s+with|admitted\s+by|reviewed\s+by|signed\s+by|assessed\s+by|"
                 r"examined\s+by|spoke\s+(?:to|with)|contact(?:ed)?)")
_PATIENT_CUE = r"(?i:patient|pt|named|called|(?:first |sur|fore|last |full )?name(?: is|:)?)"

_PATIENT_WORDS = frozenset("""
patient pt he she his her him himself herself aged age years year yo old male female man woman boy girl pregnant
dob born name named called mr mrs ms miss mx son daughter husband wife mother father nurse carer admitted
presents presented""".split())
_PHONE_WORDS = frozenset("phone call tel telephone mobile mob cell contact ring text fax whatsapp number".split())
_DATE_WORDS = frozenset("dob born birth admitted discharged since on date seen visit appointment review".split())
_CARD_WORDS = frozenset("card visa mastercard amex credit debit payment cvv expiry".split())
_SSN_WORDS = frozenset("ssn ss social security".split())
_NPI_WORDS = frozenset("npi provider prescriber".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# --- Structured patterns ----------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])")
_LABELLED_ID = re.compile(
    r"\b(?:MRN|medical record (?:number|no\.?)|hospital (?:number|no\.?)|patient (?:id|number|no\.?)|"
    r"record (?:number|no\.?)|case (?:number|no\.?)|CHI(?: number)?|account (?:number|no\.?)|NHS (?:number|no\.?)|"
    r"insurance (?:id|number|no\.?|policy(?: number)?)|(?:member|policy|group|subscriber|claim|passport|licen[cs]e|"
    r"driving licen[cs]e|reference|visa|employee|staff|national insurance|NI|social security) (?:id|number|no\.?|#)|"
    r"passport|(?-i:ID)|identification)"
    r"\s*(?:is|:|#|=)?\s*((?:[A-Za-z]{1,4}[ -]?)?\d[A-Za-z0-9-]*(?: \d{2,}[A-Za-z0-9-]*)*)",
    re.IGNORECASE,
)
_NI = re.compile(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")
_NHS = re.compile(r"(?<![\d-])(\d{3})[ -]?(\d{3})[ -]?(\d{4})(?![\d-])")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_SSN_DASHED = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
_SSN_PLAIN = re.compile(r"(?<![\d-])\d{9}(?![\d-])")
_NPI = re.compile(r"(?<![\d-])\d{10}(?![\d-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_PHONE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}"          # international
    r"|\(?0\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}"                     # UK
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"                            # North American
    r")(?!\w)"
)
_DATE_NUMERIC = re.compile(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2}))\b")
_DATE_WRITTEN = re.compile(
    rf"\b(?:{_DAY}\s+(?:of\s+)?{_MONTH}\.?,?(?:\s+\d{{4}})?"
    rf"|{_MONTH}\.?\s+{_DAY}{_UNIT_AFTER}(?:,?\s+\d{{4}})?"
    rf"|{_MONTH}\.?\s+\d{{4}})\b"
)
_AGE_OVER_89 = re.compile(
    r"\b(?:(?:aged?|age)\s+(9\d|1[0-4]\d)\b|(9\d|1[0-4]\d)(?=[\s-]*(?:years?|yrs?|y\.?o\.?)\b))",
    re.IGNORECASE,
)
_UK_POSTCODE = re.compile(r"\b(?:GIR ?0AA|[A-PR-UWYZ][A-HK-Y]?\d[A-Z\d]? ?\d[ABD-HJLNP-UW-Z]{2})\b")
_US_ZIP = re.compile(r"(,\s*[A-Z]{2}\s+)(\d{5}(?:-\d{4})?)\b")
_ADDRESS = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Close|Way|Court|Ct|Place|Pl|Crescent|Terrace|"
    r"Boulevard|Blvd|Gardens|Grove|Square|Row|Mews|Hill|Park)\b\.?"
)
_LONG_DIGITS = re.compile(rf"(?<!\w)\+?\d(?:[\s().-]?\d){{6,}}(?!\w){_UNIT_AFTER}")
_UNIT_FOLLOWS = re.compile(r"\s*(?:mg|mcg|µg|g|kg|ml|mL|l|units?|iu|IU|mmol|%|tablets?|doses?|hours?|hrs?|days?|weeks?|"
                           r"months?|years?)\b")

_TITLED = re.compile(rf"\b{_TITLE}\.?\s+(?=\S)")
_CUED = re.compile(rf"\b(?:{_RELATION_CUE}|{_PATIENT_CUE})\b[:\s,-]*(?=\S)")
_SURNAME_COMMA = re.compile(r"\b([A-Z][A-Za-z'’-]+),\s+([A-Z][a-z]+)\b")
_AGE_COMMA = re.compile(r"^,\s*\d{1,3}\s*(?:,|yo\b|y/o\b|year)", re.IGNORECASE)


def _words_near(text: str, start: int, end: int, window: int = 4) -> set[str]:
    before = _WORD.findall(text[max(0, start - 48):start].lower())[-window:]
    after = _WORD.findall(text[end:end + 40].lower())[:window]
    return set(before) | set(after)


def _boost(score: float, text: str, start: int, end: int, cues: frozenset[str], amount: float = 0.1) -> float:
    return min(0.99, score + amount) if _words_near(text, start, end) & cues else score


# --- Names ------------------------------------------------------------------

class _Names:
    def __init__(self, text: str, is_known_term: Callable[[str], bool]):
        self.text = text
        self.known = is_known_term
        self.given = _wordlist("names.txt")
        self.common = _wordlist("common_words.txt")
        self.clinical = _wordlist("clinical_terms.txt")
        self.formulary = _formulary_words()
        self.tokens = [(m.start(), m.end(), m.group(0)) for m in _WORD.finditer(text)]

    def blocked(self, word: str, after_name: bool = False) -> bool:
        """A word that is not part of a name: known to the documents, a medicine,
        a clinical term, a month or weekday, or plain English function."""
        w = word.lower().replace("’", "'")
        base = w[:-2] if w.endswith("'s") else w
        if base in _FUNCTION or base in _MONTH_DAY_WORDS:
            return True
        if after_name and base in _SURNAMES:
            return False
        return (self.known(base) or base in self.formulary or base in self.clinical
                or (len(base) < 2))

    def ambiguous(self, word: str) -> bool:
        return word.lower() in _AMBIGUOUS

    def is_given(self, word: str) -> bool:
        return word.lower() in self.given

    def run_from(self, i: int, limit: int = 3, lower_ok_first: bool = False, after_name: bool = False) -> int:
        """Index just past the run of name-like tokens starting at token i
        (0 when the first is not one). A name word is capitalised, or all
        capitals; the first may be lower case only when allowed."""
        j = i
        text = self.text
        while j < len(self.tokens) and j - i < limit:
            s, e, w = self.tokens[j]
            if j > i and text[self.tokens[j - 1][1]:s] != " ":
                break
            capital = w[0].isupper()
            if not capital and not (j == i and lower_ok_first):
                break
            if self.blocked(w, after_name=after_name or j > i):
                break
            if len(w) <= 3 and w.isupper() and w.lower() not in self.given and w.lower() not in _SURNAMES:
                break
            if j == i and not capital and not (self.is_given(w) or w.lower() in _SURNAMES
                                               or w.lower() not in self.common):
                break
            j += 1
            if w.endswith(("'s", "’s")):
                break
        return j

    def candidates(self) -> list[_Candidate]:
        out: list[_Candidate] = []
        text, toks = self.text, self.tokens
        starts = {t[0]: k for k, t in enumerate(toks)}

        def span_of(i: int, j: int) -> tuple[int, int]:
            end = toks[j - 1][1]
            if text[end - 2:end] in ("'s", "’s"):
                end -= 2
            return toks[i][0], end

        # Titles: any case, so "mrs o'neil" counts. "Miss" is also a verb, so it
        # needs a capitalised name after it.
        for m in _TITLED.finditer(text):
            k = starts.get(m.end())
            if k is None:
                continue
            lower_ok = m.group(0).strip().rstrip(".").lower() != "miss"
            j = self.run_from(k, lower_ok_first=lower_ok, after_name=True)
            if j > k:
                s, e = span_of(k, j)
                out.append(_Candidate(s, e, "name", 0.9, "title"))

        # Cues: "patient", "named", a relative, a nurse, "seen by".
        for m in _CUED.finditer(text):
            k = starts.get(m.end())
            if k is None:
                continue
            strong = bool(re.search(r"(?i)named|called|name", m.group(0)))
            j = self.run_from(k, lower_ok_first=strong)
            if j > k and (toks[k][2][0].isupper() or self.is_given(toks[k][2])):
                s, e = span_of(k, j)
                out.append(_Candidate(s, e, "name", 0.8, "cue"))

        # "SMITH, John".
        for m in _SURNAME_COMMA.finditer(text):
            sur, given = m.group(1), m.group(2)
            if (self.is_given(given) and not self.blocked(sur, after_name=True)
                    and not self.blocked(given, after_name=True)):
                out.append(_Candidate(m.start(), m.end(), "name", 0.85, "surname-comma"))

        # A given name from the list, then a capitalised surname.
        for k, (s, e, w) in enumerate(toks):
            if not w[0].isupper() or not self.is_given(w) or self.blocked(w):
                continue
            nxt = k + 1
            if nxt < len(toks) and text[e:toks[nxt][0]] == " ":
                j = self.run_from(nxt, limit=2, after_name=True)
                if j > nxt:
                    sur = toks[nxt][2]
                    if self.ambiguous(w) and not (sur.lower() in _SURNAMES or self.is_given(sur)
                                                  or sur.lower() not in self.common):
                        continue
                    out.append(_Candidate(*span_of(k, j), "name", 0.75, "given-surname"))
                    continue
            # A lone given name needs company: age, a pronoun, "patient".
            if self.ambiguous(w):
                continue
            score = 0.4
            near = _words_near(text, s, e)
            if near & _PATIENT_WORDS:
                score += 0.2
            if _AGE_COMMA.match(text[e:e + 14]):
                score += 0.25
            if score >= THRESHOLD:
                out.append(_Candidate(s, e, "name", score, "given"))
        return out


# --- NER (optional) ---------------------------------------------------------

_NER_LOADED = False
_NER_FN: Callable[[str], list[tuple[str, int, int, float]]] | None = None


def _load_ner():
    """Presidio, then spaCy, whichever imports and loads; None otherwise. Tried
    once per process."""
    global _NER_LOADED, _NER_FN
    if _NER_LOADED:
        return _NER_FN
    _NER_LOADED = True
    try:
        from presidio_analyzer import AnalyzerEngine

        engine = AnalyzerEngine()
        wanted = {"PERSON": "name", "LOCATION": "place", "DATE_TIME": "date"}

        def run(text: str):
            hits = engine.analyze(text=text, language="en", entities=list(wanted))
            return [(wanted[h.entity_type], h.start, h.end, float(h.score)) for h in hits]

        _NER_FN = run
        return _NER_FN
    except Exception:  # noqa: BLE001
        pass
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        labels = {"PERSON": "name", "GPE": "place", "LOC": "place", "DATE": "date"}

        def run(text: str):
            return [(labels[e.label_], e.start_char, e.end_char, 0.6) for e in nlp(text).ents if e.label_ in labels]

        _NER_FN = run
    except Exception:  # noqa: BLE001
        _NER_FN = None
    return _NER_FN


def _ner_candidates(text: str, names: _Names) -> list[_Candidate]:
    mode = config.DEID_NER
    if mode == "off":
        return []
    fn = _load_ner()
    if fn is None:
        return []
    out = []
    try:
        hits = fn(text)
    except Exception:  # noqa: BLE001
        return []
    for kind, start, end, score in hits:
        span = text[start:end]
        words = [w.group(0) for w in _WORD.finditer(span)]
        if not words or any(names.known(w.lower()) or w.lower() in names.formulary for w in words):
            continue
        if kind == "date" and not (re.search(r"\d", span) and (_DATE_WRITTEN.search(span)
                                                              or _DATE_NUMERIC.search(span))):
            continue
        if kind == "name" and any(names.blocked(w) for w in words):
            continue
        out.append(_Candidate(start, end, kind, min(score, 0.7), "ner"))
    return out


# --- Detection --------------------------------------------------------------

def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _scan(text: str, is_known_term: Callable[[str], bool]) -> list[_Candidate]:
    c: list[_Candidate] = []

    def add(pattern: re.Pattern, kind: str, score: float, detector: str, group: int = 0,
            check: Callable[[re.Match], float | None] | None = None) -> None:
        for m in pattern.finditer(text):
            s = score
            if check is not None:
                adjusted = check(m)
                if adjusted is None:
                    continue
                s = adjusted
            start, end = m.span(group)
            c.append(_Candidate(start, end, kind, s, detector))

    near = lambda m, cues, amount=0.1: _boost(0.0, text, m.start(), m.end(), cues, amount) > 0  # noqa: E731

    add(_EMAIL, "email address", 0.99, "email")
    add(_URL, "web address", 0.95, "url")
    add(_IPV4, "IP address", 0.9, "ip", check=lambda m: 0.9 if _valid_ip(m.group(0)) else None)
    add(_IPV6, "IP address", 0.9, "ip", check=lambda m: 0.9 if _valid_ip(m.group(0)) else None)
    add(_NHS, "NHS number", 0.9, "nhs-mod11",
        check=lambda m: (0.97 if "nhs" in _words_near(text, m.start(), m.end()) else 0.9)
        if _nhs_checksum_ok("".join(m.groups())) else None)
    add(_CARD, "card number", 0.9, "card-luhn",
        check=lambda m: (min(0.99, 0.9 + (0.07 if near(m, _CARD_WORDS) else 0)))
        if 13 <= len(_digits(m.group(0))) <= 19 and len(set(_digits(m.group(0)))) > 1
        and _luhn_ok(_digits(m.group(0))) else None)
    add(_SSN_DASHED, "Social Security number", 0.85, "ssn",
        check=lambda m: _boost(0.85, text, m.start(), m.end(), _SSN_WORDS) if _ssn_ok(_digits(m.group(0))) else None)
    add(_SSN_PLAIN, "Social Security number", 0.85, "ssn-context",
        check=lambda m: 0.85 if _ssn_ok(_digits(m.group(0))) and near(m, _SSN_WORDS, 1) else None)
    add(_NPI, "provider number", 0.9, "npi-luhn",
        check=lambda m: (0.9 if near(m, _NPI_WORDS, 1) else 0.6) if _npi_ok(m.group(0)) else None)
    add(_IBAN, "bank account", 0.9, "iban-mod97", check=lambda m: 0.9 if _iban_ok(m.group(0)) else None)
    add(_NI, "national insurance number", 0.85, "ni")
    add(_LABELLED_ID, "record number", 0.85, "labelled-id", group=1,
        check=lambda m: None if _UNIT_FOLLOWS.match(text, m.end(1)) or len(_digits(m.group(1))) < 3 else 0.85)
    def numeric_date(m: re.Match) -> float | None:
        raw = m.group(0)
        if not _plausible_numeric_date(raw) or _UNIT_FOLLOWS.match(text, m.end()):
            return None
        full_year = bool(re.match(r"\d{4}-", raw)) or len(re.split(r"[/.-]", raw)[-1]) == 4
        return _boost(0.8 if full_year else 0.55, text, m.start(), m.end(), _DATE_WORDS, 0.25)

    def phone(m: re.Match) -> float | None:
        if len(_digits(m.group(0))) < 9 or _UNIT_FOLLOWS.match(text, m.end()):
            return None
        return _boost(0.75, text, m.start(), m.end(), _PHONE_WORDS, 0.15)

    add(_DATE_NUMERIC, "date", 0.8, "date-numeric", check=numeric_date)
    add(_PHONE, "phone number", 0.75, "phone", check=phone)
    add(_DATE_WRITTEN, "date", 0.85, "date-written",
        check=lambda m: _boost(0.85, text, m.start(), m.end(), _DATE_WORDS, 0.1))
    add(_ADDRESS, "street address", 0.8, "address")
    add(_UK_POSTCODE, "postcode", 0.85, "postcode")
    add(_US_ZIP, "ZIP code", 0.8, "zip", group=2)
    for m in _AGE_OVER_89.finditer(text):
        group = 1 if m.group(1) else 2
        c.append(_Candidate(*m.span(group), "age over 89", 0.95, "age", repl="90"))
    add(_LONG_DIGITS, "long number", 0.5, "long-digits")

    names = _Names(text, is_known_term)
    c.extend(names.candidates())
    c.extend(_ner_candidates(text, names))
    return c


def _resolve(candidates: list[_Candidate], text: str) -> list[_Candidate]:
    """Keep the best of any overlapping candidates: score, then length."""
    fixed = [(m.start(), m.end()) for m in PLACEHOLDER.finditer(text)]
    ranked = sorted((x for x in candidates if x.score >= THRESHOLD
                     and not any(x.start < e and s < x.end for s, e in fixed)),
                    key=lambda x: (-x.score, -(x.end - x.start), x.start))
    chosen: list[_Candidate] = []
    for cand in ranked:
        if all(cand.end <= o.start or o.end <= cand.start for o in chosen):
            chosen.append(cand)
    return sorted(chosen, key=lambda x: x.start)


def deidentify(text: str, is_known_term: Callable[[str], bool] | None = None) -> Result:
    """Replace identifiers in text with placeholders. is_known_term tells
    whether a word is in the documents (a drug or condition, not a name)."""
    chosen = _resolve(_scan(text, is_known_term or (lambda _w: False)), text)
    found: Counter = Counter()
    out = text
    for cand in reversed(chosen):
        out = out[:cand.start] + (cand.repl or _PLACEHOLDERS[cand.kind]) + out[cand.end:]
    for cand in chosen:
        found[cand.kind] += 1
    return Result(out, found, [Entity(c.kind, c.start, c.end, round(c.score, 3), c.detector) for c in chosen])
