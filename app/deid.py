"""De-identification of questions before they're searched, sent to a model,
logged or stored.

Each detector finds one kind of identifier and replaces it with a
placeholder such as [NAME] or [DATE], so the question still reads naturally
and nothing identifying leaves this step:

- Names: after a title (Mr, Mrs, Dr...) or a cue ("patient", "named",
  "called", "name:"), skipping words the document index knows, such as drug
  and condition names.
- Dates: day, month and year in numeric and written forms, and a month with
  a day or year. A year on its own is kept.
- Ages over 89 become 90, as HIPAA Safe Harbor requires. Younger ages are
  kept: dosing depends on them.
- NHS numbers (checksum-validated), US Social Security numbers, labelled
  record numbers (MRN, hospital number, patient ID...).
- Phone numbers, email addresses, web addresses and IP addresses.
- UK postcodes, US ZIP codes after a state, and street addresses.
- Any remaining run of seven or more digits.

This is rule-based. It catches identifiers written in these common forms, not
every way a person can be identified: a name with no title or cue, a rare
condition, or a unique event can still identify someone. Treat it as a safety
net, not permission to type patient details into questions."""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

PLACEHOLDER = re.compile(r"\[(?:NAME|DATE|NHS NUMBER|SSN|ID|PHONE|EMAIL|URL|IP ADDRESS|POSTCODE|ZIP|ADDRESS|NUMBER)\]")

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DAY = r"(?:[12]\d|3[01]|0?[1-9])(?:st|nd|rd|th)?"
_UNIT_AFTER = r"(?!\s*(?:mg|mcg|µg|g|kg|ml|mL|l|units?|iu|IU|mmol|%|tablets?|doses?|hours?|hrs?|days?|weeks?)\b)"

_TITLE = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Professor)"
_NAME_WORD = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
_NAME_CUE = r"(?i:patient|pt|named|called|(?:first |sur|fore|last |full )?name(?: is|:)?)"


@dataclass
class Result:
    text: str
    found: Counter = field(default_factory=Counter)

    @property
    def changed(self) -> bool:
        return bool(self.found)

    def summary(self) -> str:
        """For the trace: what kinds were removed, never the values."""
        if not self.found:
            return "no identifiers found"
        parts = [f"{n} {kind}{'' if n == 1 else 's'}" for kind, n in sorted(self.found.items())]
        return "removed " + ", ".join(parts)


def _nhs_checksum_ok(digits: str) -> bool:
    if len(digits) != 10 or len(set(digits)) == 1:
        return False
    total = sum(int(d) * w for d, w in zip(digits[:9], range(10, 1, -1)))
    check = 11 - total % 11
    check = 0 if check == 11 else check
    return check != 10 and check == int(digits[9])


class _Deidentifier:
    def __init__(self, is_known_term: Callable[[str], bool] | None):
        self.is_known_term = is_known_term or (lambda _w: False)
        self.found: Counter = Counter()

    def sub(self, pattern: re.Pattern, text: str, kind: str, placeholder: str,
            accept: Callable[[re.Match], bool] | None = None, group: int = 0) -> str:
        def replace(m: re.Match) -> str:
            if accept is not None and not accept(m):
                return m.group(0)
            self.found[kind] += 1
            if group:
                start, end = m.span(group)
                return m.group(0)[:start - m.start()] + placeholder + m.group(0)[end - m.start():]
            return placeholder
        return pattern.sub(replace, text)


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{1,4}(?![\w:])")
_LABELLED_ID = re.compile(
    r"\b(?:MRN|medical record (?:number|no\.?)|hospital (?:number|no\.?)|patient (?:id|number|no\.?)|"
    r"record (?:number|no\.?)|case (?:number|no\.?)|CHI(?: number)?|account (?:number|no\.?)|NHS (?:number|no\.?))"
    r"\s*(?:is|:|#|=)?\s*((?:[A-Za-z]{1,3} ?)?\d[A-Za-z0-9-]*(?: \d[A-Za-z0-9-]*)*)",
    re.IGNORECASE,
)
_NHS = re.compile(r"(?<![\d-])(\d{3})[ -]?(\d{3})[ -]?(\d{4})(?![\d-])")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
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
_TITLED_NAME = re.compile(rf"\b{_TITLE}\.?\s+({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}})")
_CUED_NAME = re.compile(rf"\b{_NAME_CUE}\s+({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}})")
_LONG_DIGITS = re.compile(rf"(?<!\w)\+?\d(?:[\s().-]?\d){{6,}}(?!\w){_UNIT_AFTER}")


def deidentify(text: str, is_known_term: Callable[[str], bool] | None = None) -> Result:
    """Replace identifiers in text with placeholders. is_known_term tells
    whether a word is in the documents (a drug or condition, not a name)."""
    d = _Deidentifier(is_known_term)
    t = text

    # Structured identifiers first, most specific to least, so a later, looser
    # pattern never sees part of an earlier match.
    t = d.sub(_EMAIL, t, "email address", "[EMAIL]")
    t = d.sub(_URL, t, "web address", "[URL]")
    t = d.sub(_SSN, t, "Social Security number", "[SSN]")
    t = d.sub(_NHS, t, "NHS number", "[NHS NUMBER]", accept=lambda m: _nhs_checksum_ok("".join(m.groups())))
    t = d.sub(_LABELLED_ID, t, "record number", "[ID]", group=1)
    t = d.sub(_IPV4, t, "IP address", "[IP ADDRESS]", accept=lambda m: _valid_ip(m.group(0)))
    t = d.sub(_IPV6, t, "IP address", "[IP ADDRESS]", accept=lambda m: _valid_ip(m.group(0)))
    t = d.sub(_DATE_NUMERIC, t, "date", "[DATE]", accept=lambda m: _plausible_numeric_date(m.group(0)))
    t = d.sub(_PHONE, t, "phone number", "[PHONE]", accept=lambda m: sum(c.isdigit() for c in m.group(0)) >= 9)
    t = d.sub(_DATE_WRITTEN, t, "date", "[DATE]")
    t = d.sub(_ADDRESS, t, "street address", "[ADDRESS]")
    t = d.sub(_UK_POSTCODE, t, "postcode", "[POSTCODE]")
    t = d.sub(_US_ZIP, t, "ZIP code", "[ZIP]", group=2)

    def age(m: re.Match) -> str:
        d.found["age over 89"] += 1
        return m.group(0).replace(m.group(1) or m.group(2), "90", 1)
    t = _AGE_OVER_89.sub(age, t)

    def is_name(m: re.Match) -> bool:
        words = m.group(1).split()
        return not any(d.is_known_term(w.lower()) for w in words) and not any(
            w.lower() in _NOT_NAMES for w in words)
    t = d.sub(_TITLED_NAME, t, "name", "[NAME]", accept=is_name, group=1)
    t = d.sub(_CUED_NAME, t, "name", "[NAME]", accept=is_name, group=1)
    t = d.sub(_LONG_DIGITS, t, "long number", "[NUMBER]")
    return Result(t, d.found)


# Capitalised words that follow a cue like "patient" without being a name.
_NOT_NAMES = {
    "with", "who", "whose", "has", "had", "is", "was", "on", "in", "at", "for", "after", "before", "taking",
    "the", "a", "an", "and", "or", "of", "to", "from", "what", "how", "when", "which", "should", "can",
    "may", "is", "are", "if", "being", "presenting", "admitted", "discharged", "aged", "age", "male", "female",
    "man", "woman", "boy", "girl", "child", "adult", "elderly", "pregnant", "i", "my", "our", "their", "his", "her",
}


def _valid_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def _plausible_numeric_date(text: str) -> bool:
    parts = [int(p) for p in re.split(r"[/.-]", text)]
    if len(str(parts[0])) == 4 or text[:4].isdigit() and len(text.split("-")[0]) == 4:
        _, month, day = parts
        return 1 <= month <= 12 and 1 <= day <= 31
    a, b, _ = parts
    # Either day/month or month/day.
    return (1 <= a <= 31 and 1 <= b <= 12) or (1 <= a <= 12 and 1 <= b <= 31)
