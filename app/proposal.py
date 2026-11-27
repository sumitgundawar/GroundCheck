"""A question that proposes a regimen ("Is 15 mg twice a day okay for X?").

The proposal is read into its parts (dose, frequency, route, duration, a
maximum per day, timing) and each is compared with what the sources say about
that medicine. A part that cannot be read or checked (no unit, an unknown
medicine, two medicines, a source that contradicts itself) makes the whole
question unverifiable, and it is refused rather than guessed; otherwise any
part that differs makes the answer "No" and every part matching makes it "Yes". Nothing here
calls a model.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from . import guards_output, retrieval
from .schemas import LLMClaim, ProposalCheck, ProposalDimension

_NW = guards_output._NUMWORD_ALT
_QTY = rf"(?:\d+(?:\.\d+)?|(?:{_NW})(?:[\s-]+(?:{_NW})){{0,2}})"
_UNIT = (r"(?:milligram(?:me)?s?|mgs?|micrograms?|microgrammes?|mcg|[µμ]g|ug|grams?|grammes?|gm|g|"
         r"millilit(?:re|er)s?|mls?|cc|litres?|liters?|l|units?|iu|%|percent|meq|mmol)")
_PERIOD = (r"(?:(?:/|per\s+|a\s+|each\s+|every\s+)\s*(?:day|d)(?![a-z])|"
           r"(?:/|per\s+|a\s+|each\s+)\s*(?:24\s?h(?:ours?)?|hour|hr|h|week|dose|kg|kilo(?:gram)?s?|m2|m²|"
           r"(?:\d+\s*)?(?:ml|millilit(?:re|er)s?))(?![a-z])|\s*daily(?![a-z]))")
_QU = re.compile(
    rf"(?<![\w.])({_QTY})(?:\s*(?:to|-|–|—|or)\s*({_QTY}))?\s*({_UNIT})(?![a-z0-9])(\s*(?:{_PERIOD}))?", re.I)

_MAXWORD = re.compile(
    r"\b(?:(?:daily |day )?(?:max(?:imum)?|ceiling|cap)(?: daily| per day| a day)?(?: dose| amount| intake| total)?"
    r"|limit(?: of)?|no more than|not (?:more than|to exceed|exceed(?:ing)?|above|over|beyond)|up to|at most|"
    r"exceed(?:ing|s)?|(?:total|overall|combined|cumulative)(?: daily)?(?: dose| amount| intake)?|tops?)\b", re.I)
_GAP = re.compile(r"\s*(?:(?:of|at|is|to|than|about|around|roughly|a|an)\s+)*[:=]?\s*", re.I)
_MAXTAIL = re.compile(r"[\s,]*(?:as (?:a )?)?(?:max(?:imum)?|at most|in total|total|altogether|overall|tops?|limit)\b", re.I)
_DAYWORD = re.compile(r"\b(?:day|daily|24\s?-?\s?h(?:ours?)?|24\s?hrs?|a day|per day)\b|/\s*day", re.I)

_AGE = re.compile(
    rf"\b(?:{_QTY})[\s-]*(?:years?|yrs?|months?|weeks?|days?)[\s-]*old\b|\bage[d]?\s+(?:{_QTY})\b|"
    r"\b\d{1,3}\s*(?:yo|y/o|y\.o\.|yrs?)\b|\b(?:e?gfr|creatinine|age|weight|crcl)\s*(?:of|is|=|:)?\s*\d+(?:\.\d+)?",
    re.I)
_WEIGHT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kg|kgs|kilos?|kilograms?|lbs?|pounds?|stone)\b(?!\s*/)", re.I)

_NUMF = rf"(?:\d+(?:\.\d+)?|(?:{_NW})(?:[\s-]+(?:{_NW}))?)"
_COUNT = {"once": 1, "twice": 2, "thrice": 3}
_ABBR = {"bid": 2, "bd": 2, "bds": 2, "tid": 3, "tds": 3, "qid": 4, "qds": 4, "od": 1, "qd": 1}
_TOD_OF = {"morning": "morning", "evening": "evening", "night": "evening"}

_FREQ: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), kind) for p, kind in (
    (r"\b(?:every\s+(?:other|second)\s+day|on\s+alternate\s+days|alternate\s+days?)\b", "alt"),
    (rf"\bevery\s+({_NUMF})\s*-?\s*(hours?|hrs?|h)\b", "hours"),
    (r"(?<![\w-])q\.?\s?(\d{1,2})\.?\s?h(?:rs?)?\b", "hours"),
    (rf"\b({_NUMF})[\s-]+hourly\b", "hours"),
    (r"\b(?:hourly|every\s+hour|each\s+hour)\b", "hourly"),
    (rf"\bevery\s+({_NUMF})\s*(days?|weeks?)\b", "interval"),
    (rf"\b(once|twice|thrice|{_NUMF}\s*(?:times?|x|×|doses?|applications?|administrations?))\s*"
     rf"(?:(?:a|per|each|every|/)\s*)?(?:(day|week|morning|evening|night)|(daily|weekly|nightly))\b", "count"),
    (rf"\b({_NUMF})\s*(?:x|×)\s*/\s*(day|week)\b", "countx"),
    (r"\b(?:every|each)\s+(?:day|week)\b", "every"),
    (r"\b(?:morning\s+and\s+(?:the\s+)?(?:evening|night|bedtime)|am\s+and\s+pm|"
     r"(?:in\s+the\s+)?morning\s+and\s+(?:in\s+the\s+)?(?:evening|night))\b", "twice"),
    (r"\b(?:every|each)\s+(morning|evening|night)\b|\b(nightly)\b", "tod_daily"),
    (r"(?<![\w.-])(b\.?i\.?d|b\.?d|bds|t\.?i\.?d|t\.?d\.?s|q\.?i\.?d|q\.?d\.?s|o\.?d|q\.?d)(?![\w-])", "abbr"),
    (r"(?<![\w.-])(qhs|nocte|mane)(?![\w-])", "tod_abbr"),
    (r"\b(?:as\s+(?:and\s+when\s+)?(?:needed|required)|when\s+(?:needed|required)|if\s+(?:needed|required)|"
     r"p\.?r\.?n)\b", "prn"),
    (r"\b(?:once\s+only|only\s+once|just\s+once|one[- ]?off|single\s+dose|one\s+time\s+only|a\s+single\s+dose|stat)\b",
     "single"),
    (r"\b(?:weekly|every\s+week|each\s+week)\b", "weekly"),
    (r"\bmonthly\b", "monthly"),
    (r"(?:\bdaily\b|\ba\s+day\b|\bper\s+day\b|/\s*day\b)", "daily"),
)]
_BARE_COUNT = re.compile(rf"\b(?:once|twice|thrice|{_NUMF}\s*(?:times|x|×))\b", re.I)

_DUR_UNIT = {"hour": Fraction(1, 24), "hr": Fraction(1, 24), "h": Fraction(1, 24), "day": Fraction(1),
             "week": Fraction(7), "wk": Fraction(7)}
_DURATION = re.compile(
    rf"\b(?:for\s+|over\s+|during\s+|x\s*|×\s*|course\s+of\s+|up\s+to\s+|within\s+)?({_NUMF}|an?)[\s-]*"
    r"(hours?|hrs?|days?|weeks?|wks?|months?|years?|yrs?|fortnights?)\b", re.I)

_OPEN_BEFORE = re.compile(r"(?:at\s+least|minimum(?:\s+of)?|min\.?|more\s+than|longer\s+than|less\s+than|fewer\s+than|"
                          r"(?:no|not)\s+(?:more|longer)\s+than|under|maximum(?:\s+of)?|up\s+to)\s*(?:of\s+|for\s+)?$", re.I)
_OPEN_AFTER = re.compile(r"\s*(?:\+|plus\b|(?:or|and)\s+(?:longer|more|less|over|up|fewer|shorter|beyond)\b)", re.I)

_ROUTES: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), r) for p, r in (
    (r"\b(?:orally|oral|by\s+mouth|per\s+os|swallow(?:ed)?)\b|(?<![\w.-])po(?![\w-])", "oral"),
    (r"\b(?:topical(?:ly)?|on\s+the\s+skin|to\s+the\s+skin|cutaneous(?:ly)?|transdermal(?:ly)?)\b", "topical"),
    (r"\b(?:subcutaneous(?:ly)?|subcut|sub-cut|under\s+the\s+skin|s\.c\.)\b|(?<![\w.-])(?:sc|sq)(?![\w-])",
     "subcutaneous"),
    (r"\b(?:intravenous(?:ly)?|i\.v\.|drip|infus(?:ion|ed))\b|(?<![\w.-])iv(?![\w-])", "intravenous"),
    (r"\b(?:intramuscular(?:ly)?|i\.m\.)(?!\w)", "intramuscular"),
    (r"\b(?:inhal(?:ed|ation|er)|nebuli[sz]ed?)\b", "inhaled"),
    (r"\b(?:rectal(?:ly)?|suppositor(?:y|ies))\b", "rectal"),
    (r"\bsublingual(?:ly)?\b|\bunder\s+the\s+tongue\b", "sublingual"),
    (r"\b(?:intranasal(?:ly)?|nasal(?:ly)?|nose\s+drops?)\b", "nasal"),
    (r"\b(?:eye\s+drops?|ophthalmic|into\s+the\s+eyes?)\b", "ophthalmic"),
)]
_INJECTION = re.compile(r"\b(?:inject(?:ed|ion|ions|able|ing)?|jab|shot)\b", re.I)
_FORMS = re.compile(
    rf"\b(?:{_NUMF}|an?|half|another|extra|one|two)\s+(?:tablets?|capsules?|pills?|puffs?|drops?|teaspoons?|tsp|"
    r"tbsp|tablespoons?|spoonfuls?|sachets?|patch(?:es)?|vials?|ampoules?|sprays?|doses?|pumps?|sticks?)\b", re.I)
_RELATIVE = re.compile(
    r"\b(?:doubl(?:e|ed|ing)|tripl(?:e|ed|ing)|halv(?:e|ed|ing)|(?:twice|double|triple|half)\s+(?:of\s+)?the\s+"
    r"(?:dose|amount|usual|standard|recommended|normal)|extra\s+dose|another\s+dose|additional\s+dose|"
    r"(?:higher|bigger|larger|lower|smaller|stronger|weaker)\s+dose|skip(?:ping)?\s+(?:a\s+)?dose)\b", re.I)

_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:a\.?m|p\.?m)\b\.?", re.I)
_FOOD = [(re.compile(p, re.I), c, t) for p, c, t in (
    (r"\bwith\s+(?:a\s+|the\s+|some\s+|my\s+)?(?:food|meals?|snack)\b", "with food", None),
    (r"\bwith\s+(?:a\s+|the\s+|my\s+)?breakfast\b", "with food", "morning"),
    (r"\bwith\s+(?:a\s+|the\s+|my\s+)?(?:dinner|supper|evening\s+meal)\b", "with food", "evening"),
    (r"\bwith\s+(?:a\s+|the\s+|my\s+)?(?:lunch|midday\s+meal)\b", "with food", "midday"),
    (r"\bbefore\s+(?:a\s+|the\s+|my\s+)?(?:food|meals?|eating)\b", "before food", None),
    (r"\bbefore\s+(?:a\s+|the\s+|my\s+)?breakfast\b", "before food", "morning"),
    (r"\bbefore\s+(?:a\s+|the\s+|my\s+)?(?:dinner|supper|evening\s+meal)\b", "before food", "evening"),
    (r"\bbefore\s+(?:a\s+|the\s+|my\s+)?(?:lunch|midday\s+meal)\b", "before food", "midday"),
    (r"\bafter\s+(?:a\s+|the\s+|my\s+)?(?:food|meals?|eating)\b", "after food", None),
    (r"\bafter\s+(?:a\s+|the\s+|my\s+)?breakfast\b", "after food", "morning"),
    (r"\bafter\s+(?:a\s+|the\s+|my\s+)?(?:morning\s+meal)\b", "after food", "morning"),
    (r"\bafter\s+(?:a\s+|the\s+|my\s+)?(?:dinner|supper|evening\s+meal)\b", "after food", "evening"),
    (r"\bafter\s+(?:a\s+|the\s+|my\s+)?(?:lunch|midday\s+meal)\b", "after food", "midday"),
    (r"\b(?:on\s+an\s+empty\s+stomach|empty\s+stomach|fasting|without\s+food)\b", "empty stomach", None),
)]
_TOD = [(re.compile(p, re.I), t) for p, t in (
    (r"\bmornings?\b", "morning"),
    (r"\b(?:evenings?|nights?|bedtime|before\s+bed|at\s+bed|tonight)\b", "evening"),
    (r"\b(?:afternoons?|midday|noon|lunchtime)\b", "midday"),
)]

_INVERTED = re.compile(
    r"\b(?:too\s+(?:much|high|low|little|often|long|short|frequent(?:ly)?|many|few)|excessive(?:ly)?|overdos\w*|unsafe|"
    r"dangerous(?:ly)?|harm(?:ful)?|toxic|wrong|incorrect|bad|risky|risk|mistake|hurt|poison\w*|"
    r"under-?dos\w*)\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|no|without|cannot|false|untrue|contraindicated|forbidden|prohibited|invalid|"
                      r"disallowed|unacceptable|inappropriate)\b|n['’]t\b", re.I)
_NEG_OK = re.compile(r"\b(?:no more than|not (?:more than|to exceed|exceed(?:ing)?|above|over|beyond)|"
                     r"without\s+(?:a\s+|the\s+|any\s+)?(?:food|meals?|eating))\b", re.I)
_INFO_START = re.compile(
    r"^\s*(?:what|which|how\s+(?:much|many|long|often|frequently)|when|why|who|where|describe|list|tell\s+me|explain|"
    r"give\s+me|show\s+me|name)\b", re.I)
_MARKER = re.compile(
    r"\b(?:ok|okay|fine|safe|safely|correct|right|acceptable|appropriate|alright|allowed|permitted|recommended|"
    r"advisable|enough|sufficient|suitable|good|oke|what\s+if|what\s+happens|how\s+about|can\s+i|could\s+i|may\s+i|"
    r"should\s+i|would\s+it|is\s+it|am\s+i|are\s+we|can\s+we|should\s+we)\b", re.I)
_FILLER = re.compile(r"^(?:\s*(?:hi|hello|hey|please|quick\s+question|question|doctor|dr\.?|nurse|pharmacist|so|um|ok|okay)"
                     r"\b[\s,.:;!-]*)+", re.I)

_VOCAB = ("twice", "thrice", "daily", "hourly", "weekly", "nightly", "monthly", "orally", "topically", "intravenously",
          "subcutaneously", "intramuscularly", "hours", "weeks", "months", "times", "every", "morning", "evening",
          "milligrams", "micrograms", "millilitres", "milliliters", "grams", "fortnight", "course")

_DIM_ORDER = ["dose", "frequency", "duration", "route", "max_daily", "timing"]


@dataclass
class Quantity:
    value: Decimal
    kind: str                   # mass | volume | <other unit>
    shown: str

    def key(self) -> tuple[str, Decimal]:
        return self.kind, self.value


@dataclass
class Parsed:
    dose: list[Quantity] = field(default_factory=list)
    freq: list[Fraction | str] = field(default_factory=list)
    duration: list[tuple[Fraction, Fraction]] = field(default_factory=list)
    duration_shown: list[str] = field(default_factory=list)
    route: list[str] = field(default_factory=list)
    max_daily: list[Quantity] = field(default_factory=list)
    total_daily: list[Quantity] = field(default_factory=list)
    daily_ambiguous: list[Quantity] = field(default_factory=list)
    tod: set[str] = field(default_factory=set)
    food: set[str] = field(default_factory=set)
    ambiguities: dict[str, list[str]] = field(default_factory=dict)
    relative: bool = False
    consumed: set[str] = field(default_factory=set)
    context: bool = False

    def flag(self, dimension: str, reason: str) -> None:
        self.ambiguities.setdefault(dimension, [])
        if reason not in self.ambiguities[dimension]:
            self.ambiguities[dimension].append(reason)

    def present(self) -> bool:
        return bool(self.dose or self.freq or self.duration or self.route or self.max_daily or self.total_daily
                    or self.daily_ambiguous or self.tod or self.food or self.ambiguities)


def _number(raw: str) -> Decimal | None:
    value = guards_output._word_to_number(raw)
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _quantity(m: re.Match) -> tuple[Quantity | None, str | None]:
    lo = _number(m.group(1))
    if lo is None:
        return None, "a number that could not be read"
    unit = m.group(3).lower()
    if unit.startswith("milligram") or unit in ("mg", "mgs"):
        kind, scale, label = "mass", Decimal(1), "mg"
    elif unit.startswith("microgram") or unit in ("mcg", "µg", "μg", "ug"):
        kind, scale, label = "mass", Decimal("0.001"), "mcg"
    elif unit.startswith("gram") or unit in ("g", "gm"):
        kind, scale, label = "mass", Decimal(1000), "g"
    elif unit.startswith("millilit") or unit in ("ml", "mls", "cc"):
        kind, scale, label = "volume", Decimal(1), "mL"
    elif unit.startswith(("litre", "liter")) or unit == "l":
        kind, scale, label = "volume", Decimal(1000), "L"
    else:
        return None, f"a dose in {unit}, which cannot be compared with a dose in milligrams"
    if m.group(2):
        return None, "a range of doses"
    shown = guards_output._shown(lo) + " " + label
    return Quantity((lo * scale).normalize(), kind, shown), None


def _blank(text: str, m: re.Match) -> str:
    return text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]


def _prepare(text: str, parsed: Parsed) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"(\d)\s*½", r"\1.5", text).replace("½", "0.5")
    text = re.sub(r"(\d)\s*¼", r"\1.25", text).replace("¼", "0.25")
    text = re.sub(r"(\d)\s*¾", r"\1.75", text).replace("¾", "0.75")
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")

    def thousands(m: re.Match) -> str:
        parsed.flag("dose", "a number written with a comma that could be a decimal or a thousands separator")
        return " " * len(m.group(0))

    text = re.sub(r"(?<![\d.,])0,(\d+)(?!\d)", r"0.\1", text)
    text = re.sub(r"(?<![\d.,])[1-9]\d{0,2},\d{3}(?![\d,])", thousands, text)
    text = re.sub(r"(\d),(\d{1,2})(?![\d,])", r"\1.\2", text)
    return text


def parse(text: str) -> Parsed:
    parsed = Parsed()
    t = _prepare(text, parsed)
    if _RELATIVE.search(t):
        parsed.relative = True
        parsed.flag("dose", "a change relative to the usual dose, with no amount stated")
    prepared_words = set(re.findall(r"[a-z]+", t))
    for pattern in (_AGE, _WEIGHT):
        for m in list(pattern.finditer(t)):
            parsed.context = True
            t = _blank(t, m)

    spans: list[tuple[int, int]] = []
    seen_q: set[tuple[int, int]] = set()
    for mw in list(_MAXWORD.finditer(t)):
        tail = _QU.search(t, mw.end(), min(len(t), mw.end() + 45))
        if tail is not None and _GAP.fullmatch(t[mw.end():tail.start()]) and (tail.start(), tail.end()) not in seen_q:
            seen_q.add((tail.start(), tail.end()))
            spans.append((mw.start(), tail.end()))
            _take_max(parsed, tail)
    for qu in list(_QU.finditer(t)):
        tailword = _MAXTAIL.match(t, qu.end())
        if tailword is not None and (qu.start(), qu.end()) not in seen_q:
            seen_q.add((qu.start(), qu.end()))
            spans.append((qu.start(), tailword.end()))
            _take_max(parsed, qu)
    if spans and not _DAYWORD.search(t):
        parsed.flag("max_daily", "a maximum with no period (per day) stated")
    for a, b in spans:
        t = t[:a] + " " * (b - a) + t[b:]

    for qu in list(_QU.finditer(t)):
        quantity, problem = _quantity(qu)
        if quantity is None:
            parsed.flag("dose", problem or "a dose that could not be read")
            t = _blank(t, qu)
            continue
        suffix = (qu.group(4) or "").strip().lower()
        if suffix:
            word = re.sub(r"^(?:/|per|a|each|every)\s*", "", suffix)
            if word in ("daily", "day", "d") or word.startswith("24"):
                if "/" in suffix and suffix.startswith("/"):
                    parsed.total_daily.append(quantity)
                else:
                    parsed.daily_ambiguous.append(quantity)
            elif word == "week":
                parsed.dose.append(quantity)
                parsed.freq.append(Fraction(1, 7))
            elif word == "dose":
                parsed.dose.append(quantity)
            else:
                parsed.flag("dose", "a dose given per kilogram, per hour or as a concentration")
        else:
            parsed.dose.append(quantity)
        t = _blank(t, qu)

    for pattern, kind in _FREQ:
        for m in list(pattern.finditer(t)):
            _take_frequency(parsed, m, kind)
            t = _blank(t, m)
    for m in list(_BARE_COUNT.finditer(t)):
        parsed.flag("frequency", "a frequency with no period (once what?)")
        t = _blank(t, m)

    for m in list(_DURATION.finditer(t)):
        if (_OPEN_BEFORE.search(t[:m.start()]) or _OPEN_AFTER.match(t, m.end())
                or re.match(r"(?:up\s+to|within|over)\b", m.group(0), re.I)):
            parsed.flag("duration", "an open-ended duration (at least, or longer, up to), not a fixed number of days")
        _take_duration(parsed, m)
        t = _blank(t, m)

    injection = _INJECTION.search(t)
    for pattern, route in _ROUTES:
        for m in list(pattern.finditer(t)):
            if route not in parsed.route:
                parsed.route.append(route)
            t = t[:m.start()] + "~" * (m.end() - m.start()) + t[m.end():]
    if re.search(r"(?<![\w'’-])IM(?![\w'’-])", text):
        if "intramuscular" not in parsed.route:
            parsed.route.append("intramuscular")
        t = re.sub(r"(?<![\w'’-])im(?![\w'’-])", "  ", t)
    if injection and not parsed.route:
        parsed.flag("route", "'injection' does not name a route")
    elif injection:
        t = _INJECTION.sub(lambda m: "~" * len(m.group(0)), t)
    if len(parsed.route) > 1:
        parsed.flag("route", "more than one route named")
    for m in list(_CLOCK.finditer(t)):
        parsed.flag("timing", "a clock time, which the sources do not give")
        t = _blank(t, m)
    for m in _FORMS.finditer(t):
        if re.search(r"doses?$", m.group(0)) and (parsed.dose or parsed.daily_ambiguous or parsed.max_daily
                                                  or parsed.total_daily):
            continue
        parsed.flag("dose", "an amount counted in tablets, puffs or doses rather than in milligrams or millilitres")
        break

    for pattern, food, tod in _FOOD:
        for m in list(pattern.finditer(t)):
            parsed.food.add(food)
            if tod:
                parsed.tod.add(tod)
            t = _blank(t, m)
    for pattern, tod in _TOD:
        for m in list(pattern.finditer(t)):
            parsed.tod.add(tod)
            t = _blank(t, m)

    parsed.consumed = prepared_words - set(re.findall(r"[a-z]+", t))
    if re.search(r"\d", t):
        parsed.flag("dose", "a number that could not be placed (no unit, or wording not recognised)")
    if not (parsed.dose or parsed.freq or parsed.duration or parsed.route or parsed.max_daily or parsed.total_daily
            or parsed.daily_ambiguous):
        return parsed
    known = _known_words()
    for word in re.findall(r"[a-z]{5,}", t):
        if word in _VOCAB or word in known or guards_output._is_common(word):
            continue
        if difflib.get_close_matches(word, _VOCAB, n=1, cutoff=0.8):
            parsed.flag("frequency", f"'{word}' looks like a misspelt dosing word")
    return parsed


def _known_words() -> frozenset[str]:
    cache = retrieval._view().cache
    if "proposal_words" not in cache:
        cache["proposal_words"] = frozenset(w for r in retrieval._view().metadata for w in retrieval.tokenize(r["text"]))
    return cache["proposal_words"]


def _take_max(parsed: Parsed, m: re.Match) -> None:
    quantity, problem = _quantity(m)
    if quantity is None:
        parsed.flag("max_daily", problem or "a maximum that could not be read")
        return
    parsed.max_daily.append(quantity)


def _take_frequency(parsed: Parsed, m: re.Match, kind: str) -> None:
    if kind == "alt":
        parsed.freq.append(Fraction(1, 2))
    elif kind == "hours":
        n = _number(m.group(1))
        if n is None or n <= 0:
            parsed.flag("frequency", "an interval that could not be read")
        else:
            parsed.freq.append(Fraction(24) / Fraction(str(n)))
    elif kind == "hourly":
        parsed.freq.append(Fraction(24))
    elif kind == "interval":
        n = _number(m.group(1))
        days = Fraction(7) if m.group(2).lower().startswith("week") else Fraction(1)
        if n is None or n <= 0:
            parsed.flag("frequency", "an interval that could not be read")
        else:
            parsed.freq.append(Fraction(1) / (Fraction(str(n)) * days))
    elif kind == "count":
        word = m.group(1).lower()
        if word in _COUNT:
            count = Fraction(_COUNT[word])
        else:
            n = _number(re.split(r"\s*(?:times?|x|×|doses?|applications?|administrations?)", word)[0])
            count = None if n is None else Fraction(str(n))
        period = (m.group(2) or m.group(3) or "").lower()
        if count is None or count <= 0:
            parsed.flag("frequency", "a count that could not be read")
            return
        if period in _TOD_OF:
            parsed.tod.add(_TOD_OF[period])
        if period in _TOD_OF or period in ("day", "daily", "nightly"):
            parsed.freq.append(count)
        else:
            parsed.freq.append(count / 7)
    elif kind == "countx":
        n = _number(m.group(1))
        if n is None:
            parsed.flag("frequency", "a count that could not be read")
        else:
            parsed.freq.append(Fraction(str(n)) / (7 if m.group(2).lower() == "week" else 1))
    elif kind == "every":
        parsed.freq.append(Fraction(1, 7) if "week" in m.group(0).lower() else Fraction(1))
    elif kind == "twice":
        parsed.freq.append(Fraction(2))
    elif kind == "tod_daily":
        word = (m.group(1) or "night").lower()
        parsed.tod.add(_TOD_OF[word])
        parsed.freq.append(Fraction(1))
    elif kind == "abbr":
        parsed.freq.append(Fraction(_ABBR[re.sub(r"[^a-z]", "", m.group(1).lower())]))
    elif kind == "tod_abbr":
        word = m.group(1).lower()
        parsed.tod.add("morning" if word == "mane" else "evening")
        parsed.freq.append(Fraction(1))
    elif kind == "prn":
        parsed.freq.append("as needed")
    elif kind == "single":
        parsed.freq.append("a single dose")
    elif kind == "weekly":
        parsed.freq.append(Fraction(1, 7))
    elif kind == "monthly":
        parsed.flag("frequency", "'monthly' is not a fixed number of days")
    elif kind == "daily":
        parsed.freq.append(Fraction(1))


def _take_duration(parsed: Parsed, m: re.Match) -> None:
    raw = m.group(1).lower()
    n = Fraction(1) if raw in ("a", "an") else None
    if n is None:
        value = _number(raw)
        if value is None:
            parsed.flag("duration", "a duration that could not be read")
            return
        n = Fraction(str(value))
    unit = re.sub(r"s$", "", m.group(2).lower())
    if unit == "fortnight":
        parsed.duration.append((n * 14, n * 14))
    elif unit in ("month",):
        parsed.duration.append((n * 28, n * 31))
    elif unit in ("year", "yr"):
        parsed.duration.append((n * 365, n * 366))
    else:
        days = n * _DUR_UNIT[unit]
        parsed.duration.append((days, days))
    parsed.duration_shown.append(f"{_plain(n)} {unit}{'' if n == 1 else 's'}")


def _plain(f: Fraction) -> str:
    return str(int(f)) if f.denominator == 1 else str(float(f))


def freq_text(f: Fraction | str) -> str:
    if isinstance(f, str):
        return f
    if f == 1:
        return "once daily"
    if f == 2:
        return "twice daily"
    if f == 3:
        return "three times daily"
    if f == 4:
        return "four times daily"
    if f == Fraction(1, 2):
        return "every other day"
    if f == Fraction(1, 7):
        return "once weekly"
    if f > 1 and (24 / f).denominator == 1:
        return f"every {int(24 / f)} hours"
    if f > 1:
        return f"{float(f):g} times daily"
    if (1 / f).denominator == 1 and 1 / f % 7 == 0:
        return f"once every {int(1 / f // 7)} weeks"
    return f"once every {float(1 / f):g} days"


def _duration_text(d: tuple[Fraction, Fraction], shown: str | None = None) -> str:
    if shown:
        return shown
    lo, hi = d
    return f"{_plain(lo)} days" if lo == hi else f"{_plain(lo)} to {_plain(hi)} days"


# --- Reading a question ----------------------------------------------------------

@dataclass
class Reading:
    parsed: Parsed
    negated: bool
    inverted: bool
    drugs: list[str]
    misspelt: list[tuple[str, str]]


def _drug_topics() -> dict[str, str]:
    cache = retrieval._view().cache
    if "drug_topic_names" not in cache:
        drugs = {r["topic"].lower() for r in retrieval._view().metadata if r.get("kind") == "drug"}
        cache["drug_topic_names"] = {w: t for w, t in retrieval._topic_names().items() if t in drugs}
    return cache["drug_topic_names"]


_ASKING = re.compile(r"^(?:is|are|can|could|should|would|do|does|did|may|am|will|shall|might)\b", re.I)


def is_question(question: str) -> bool:
    """A real question: a tag or marker word ("correct?", "okay"), or a "?" after a yes/no opening. A statement
    or a bare "Caloradine 15,5 mg?" is not one."""
    stripped = _FILLER.sub("", question)
    return bool(_MARKER.search(question)) or ("?" in question and bool(_ASKING.match(stripped)))


def explained(question: str, terms: list[str]) -> bool:
    """True when every term is a word of the regimen itself (a route, a unit, a time of day or food), so that a
    coverage gap on it is the proposal's own to judge."""
    parsed = parse(question)
    return not parsed.context and all(t.lower() in parsed.consumed for t in terms)


_ASKS_DOSE = re.compile(r"\b(?:dose|dosage|dosing|regimen)\b", re.I)


def _is_proposal_question(question: str, parsed: Parsed) -> bool:
    if not parsed.present():
        return False
    stripped = _FILLER.sub("", question)
    if _INFO_START.match(stripped) and not _MARKER.search(question):
        return bool(_ASKS_DOSE.search(question) and (parsed.dose or parsed.freq))
    return True


def read(question: str) -> Reading | None:
    parsed = parse(question)
    if not _is_proposal_question(question, parsed):
        return None
    probe = _NEG_OK.sub(" ", question)
    names = _drug_topics()
    words = set(retrieval.tokenize(question))
    drugs = sorted({names[w] for w in words if w in names})
    misspelt = []
    for w in sorted(words - set(names)):
        if len(w) >= 5 and not guards_output._is_common(w) and not guards_output._is_clinical(w):
            near = difflib.get_close_matches(w, list(names), n=1, cutoff=0.84)
            if near:
                misspelt.append((w, near[0]))
    return Reading(parsed, bool(_NEGATED.search(probe)), bool(_INVERTED.search(question)), drugs, misspelt)


# --- The sources' regimen --------------------------------------------------------

@dataclass
class Documented:
    parsed: Parsed
    sentences: dict[str, list[tuple[str, str]]]   # dimension -> [(source id, sentence)]
    units: list[tuple[str, str, Parsed]] = field(default_factory=list)
    max_stated: bool = False


def documented(topic: str) -> Documented:
    topic_records = [r for r in retrieval.topic_records({topic})]
    name = topic.split()[0]
    pooled = list(topic_records)
    seen = {r["id"] for r in pooled}
    for r in retrieval._view().metadata:
        if r["id"] not in seen and r.get("kind") in ("condition", "procedure") and re.search(rf"\b{name}\b", r["text"], re.I):
            pooled.append(r)
    out = Documented(Parsed(), {})
    for record in pooled:
        text = record["text"]
        for a, b in retrieval.sentence_spans(text):
            sentence = text[a:b]
            if not re.search(rf"\b{name}\b", sentence, re.I) and record not in topic_records:
                continue
            p = parse(sentence)
            if record not in topic_records and not (p.dose and (p.freq or p.duration)):
                continue
            has_regimen = bool(p.dose or p.freq or p.duration or p.total_daily or p.daily_ambiguous)
            if p.max_daily or "max_daily" in p.ambiguities:
                out.max_stated = True
            if has_regimen:
                out.units.append((record["id"], sentence, p))
                for dim in ("dose", "frequency", "duration", "timing"):
                    out.sentences.setdefault(dim, []).append((record["id"], sentence))
                out.sentences.setdefault("max_daily", []).append((record["id"], sentence))
            if p.route:
                out.sentences.setdefault("route", []).append((record["id"], sentence))
            _merge(out.parsed, p)
    return out


def _merge(into: Parsed, p: Parsed) -> None:
    into.dose += [q for q in p.dose if q.key() not in {x.key() for x in into.dose}]
    into.freq += [f for f in p.freq if f not in into.freq]
    into.duration += [d for d in p.duration if d not in into.duration]
    into.duration_shown += [s for s in p.duration_shown if s not in into.duration_shown]
    into.route += [r for r in p.route if r not in into.route]
    into.max_daily += [q for q in p.max_daily if q.key() not in {x.key() for x in into.max_daily}]
    into.tod |= p.tod
    into.food |= p.food
    for k, v in p.ambiguities.items():
        for reason in v:
            into.flag(k, reason)


# --- Comparing -------------------------------------------------------------------

_CODES = (("misspelt", "misspelt_word"), ("comma", "ambiguous_separator"), ("relative", "relative_change"),
          ("tablets", "amount_not_in_units"), ("range", "range"), ("conflicting", "source_conflict"),
          ("two different timings", "source_conflict"), ("more than one regimen", "source_ambiguous"),
          ("more than one", "multiple_values"), ("no medicine", "no_medicine"), ("did you mean", "unknown_medicine"),
          ("is not a medicine", "unknown_medicine"), ("too much", "harm_question"), ("phrased negatively", "negated"), ("clock time", "clock_time"),
          ("kilogram", "unit_not_comparable"), ("cannot be compared", "unit_not_comparable"),
          ("no period", "no_period"), ("could not be placed", "number_without_unit"),
          ("could not be read", "unreadable"), ("does not name a route", "route_not_named"),
          ("state no", "source_missing"), ("do not give a daily total", "source_missing"),
          ("nothing about food", "source_missing"), ("no time of day", "source_missing"),
          ("one dose a day or a total", "ambiguous_daily"), ("not a fixed number", "not_exact"),
          ("a maximum of their own", "source_maximum"), ("not the same instruction", "not_equivalent"))


def _code(status: str, reason: str | None) -> str:
    if status == "match":
        return "match"
    if status == "mismatch":
        return "differs"
    for needle, code in _CODES:
        if reason and needle in reason:
            return code
    return "unverifiable"


def _dim(name: str, status: str, proposed: str | None, doc: str | None = None, source: str | None = None,
         reason: str | None = None) -> ProposalDimension:
    return ProposalDimension(name=name, status=status, proposed=proposed, documented=doc, source_id=source,
                             reason=reason, code=_code(status, reason))


def _src(doc: Documented, dim: str) -> str | None:
    found = doc.sentences.get(dim)
    return found[0][0] if found else None


def _one(values: list, same=lambda a, b: a == b):
    first = values[0]
    return first if all(same(first, v) for v in values[1:]) else None


def _cmp_dose(prop: Quantity, doc: Documented) -> tuple[str, str | None, str | None]:
    src = doc.parsed.dose
    if not src:
        return "unverifiable", None, "the sources state no dose for this medicine"
    if len(src) > 1:
        return "unverifiable", None, "the sources state more than one dose"
    return ("match" if prop.key() == src[0].key() else "mismatch"), src[0].shown, None


def _daily_total(doc: Documented) -> Quantity | None:
    src = doc.parsed
    rates = [f for f in src.freq if isinstance(f, Fraction)]
    if len(src.dose) != 1 or len(set(rates)) != 1 or len(src.freq) != len(rates):
        return None
    d = src.dose[0]
    total = d.value * Decimal(rates[0].numerator) / Decimal(rates[0].denominator)
    return Quantity(total.normalize(), d.kind, _total_shown(d, total))


def _total_shown(d: Quantity, total: Decimal) -> str:
    unit = d.shown.split()[-1]
    scale = {"mg": Decimal(1), "mcg": Decimal("0.001"), "g": Decimal(1000), "mL": Decimal(1), "L": Decimal(1000)}[unit]
    return f"{guards_output._shown(total / scale)} {unit}"


def _regimen_text(doc: Documented) -> str | None:
    src = doc.parsed
    if len(src.dose) != 1 or len({str(f) for f in src.freq}) != 1:
        return None
    return f"{src.dose[0].shown} {freq_text(src.freq[0])}"


def _freq_status(prop: list, doc: Documented) -> tuple[str, str | None, str | None]:
    src = doc.parsed.freq
    if not src:
        return "unverifiable", None, "the sources state no frequency for this medicine"
    if len({str(f) for f in src}) > 1:
        return "unverifiable", None, "the sources give conflicting frequencies"
    return ("match" if prop[0] == src[0] else "mismatch"), freq_text(src[0]), None


def compare_all(reading: Reading, doc: Documented) -> tuple[list[ProposalDimension], Documented]:
    """When the sources give more than one regimen (adults and children, say), the proposal is a match only if one
    regimen has every part of it; otherwise it cannot be said which regimen was meant."""
    signatures = {(tuple(q.key() for q in p.dose), tuple(map(str, p.freq)), tuple(p.duration))
                  for _, _, p in doc.units}
    p = reading.parsed
    asks_regimen = bool(p.dose or p.freq or p.duration or p.max_daily or p.total_daily or p.daily_ambiguous
                        or p.tod or p.food)
    if len(signatures) <= 1 or not asks_regimen:
        return compare(reading, doc), doc
    for source_id, sentence, unit in doc.units:
        sub = Documented(Parsed(), {}, [(source_id, sentence, unit)], doc.max_stated)
        _merge(sub.parsed, unit)
        sub.parsed.route = list(doc.parsed.route)
        for dim in ("dose", "frequency", "duration", "timing", "max_daily"):
            sub.sentences[dim] = [(source_id, sentence)]
        sub.sentences["route"] = doc.sentences.get("route", [])
        dims = compare(reading, sub)
        if dims and all(d.status == "match" for d in dims):
            return dims, sub
    return [_dim("proposal", "unverifiable", None, reason="the sources give more than one regimen and none of them "
                                                           "is the one proposed")], doc


def compare(reading: Reading, doc: Documented) -> list[ProposalDimension]:
    p, src = reading.parsed, doc.parsed
    out: list[ProposalDimension] = []

    for dim, reasons in p.ambiguities.items():
        for reason in reasons:
            out.append(_dim(dim, "unverifiable", None, reason=reason))

    def distinct(values, key):
        seen = {}
        for v in values:
            seen.setdefault(key(v), v)
        return list(seen.values())

    doses = [] if "dose" in p.ambiguities else distinct(p.dose, lambda q: q.key())
    freqs = [] if "frequency" in p.ambiguities else distinct(p.freq, str)
    durs = [] if "duration" in p.ambiguities else distinct(p.duration, lambda d: d)
    if len(doses) > 1:
        out.append(_dim("dose", "unverifiable", None, reason="more than one dose is proposed"))
        doses = []
    if len(freqs) > 1:
        out.append(_dim("frequency", "unverifiable", None, reason="more than one frequency is proposed"))
        freqs = []
    if len(durs) > 1:
        out.append(_dim("duration", "unverifiable", None, reason="more than one duration is proposed"))
        durs = []

    ambiguous_dose_dim = "dose" in p.ambiguities

    if p.daily_ambiguous and "dose" not in p.ambiguities:
        q = p.daily_ambiguous[0]
        as_dose = _cmp_dose(q, doc)
        as_freq = _freq_status([Fraction(1)], doc)
        statuses = {as_dose[0], as_freq[0]}
        a_status = "mismatch" if "mismatch" in statuses else "unverifiable" if "unverifiable" in statuses else "match"
        total = _daily_total(doc)
        b_status, b_doc = ("unverifiable", None) if total is None else (
            "match" if total.key() == q.key() else "mismatch", total.shown + " a day")
        regimen = _regimen_text(doc)
        if a_status == b_status and a_status != "unverifiable":
            out.append(_dim("dose", a_status, q.shown + " a day", regimen, _src(doc, "dose")))
        else:
            out.append(_dim("dose", "unverifiable", q.shown + " a day", regimen, _src(doc, "dose"),
                            f"'{q.shown} a day' could mean one dose a day or a total for the day, and the sources "
                            "answer the two differently"))

    for q in ([] if "max_daily" in p.ambiguities else p.total_daily + p.max_daily):
        name = "max_daily"
        proposed = f"{q.shown} a day"
        stated = list({m.key(): m for m in src.max_daily}.values())
        if not doc.max_stated or not stated:
            out.append(_dim(name, "unverifiable", proposed, None, _src(doc, "dose"),
                            "the sources state no maximum daily dose for this medicine (the usual regimen is not "
                            "a maximum)"))
        elif len(stated) > 1:
            out.append(_dim(name, "unverifiable", proposed, None, _src(doc, "dose"),
                            "the sources give more than one maximum"))
        else:
            out.append(_dim(name, "match" if stated[0].key() == q.key() else "mismatch", proposed,
                            f"{stated[0].shown} a day", _src(doc, "dose")))

    if doses:
        status, shown, reason = _cmp_dose(doses[0], doc)
        out.append(_dim("dose", status, doses[0].shown, shown, _src(doc, "dose"), reason))

    if freqs:
        status, shown, reason = _freq_status(freqs, doc)
        out.append(_dim("frequency", status, freq_text(freqs[0]), shown, _src(doc, "frequency"), reason))

    if durs:
        d = durs[0]
        shown_p = p.duration_shown[p.duration.index(d)] if p.duration.index(d) < len(p.duration_shown) else None
        sd = src.duration
        if not sd:
            out.append(_dim("duration", "unverifiable", _duration_text(d, shown_p), None, _src(doc, "duration"),
                            "the sources state no duration for this medicine"))
        elif len(set(sd)) > 1 or "duration" in src.ambiguities:
            out.append(_dim("duration", "unverifiable", _duration_text(d, shown_p), None, _src(doc, "duration"),
                            "the sources give conflicting or open-ended durations"))
        else:
            lo, hi = d
            sl, sh = sd[0]
            doc_shown = src.duration_shown[0] if src.duration_shown else _duration_text(sd[0])
            if lo == hi:
                status = "match" if lo == sl else "mismatch"
            else:
                status = "unverifiable" if lo <= sl <= hi else "mismatch"
            out.append(_dim("duration", status, _duration_text(d, shown_p), doc_shown, _src(doc, "duration"),
                            "a month is not a fixed number of days" if status == "unverifiable" else None))

    if p.route and len(p.route) == 1 and "route" not in p.ambiguities:
        if not src.route:
            out.append(_dim("route", "unverifiable", p.route[0], None, None, "the sources state no route for this medicine"))
        elif len(set(src.route)) > 1:
            out.append(_dim("route", "unverifiable", p.route[0], None, _src(doc, "route"),
                            "the sources give more than one route"))
        else:
            out.append(_dim("route", "match" if p.route[0] == src.route[0] else "mismatch", p.route[0], src.route[0],
                            _src(doc, "route")))

    if (p.tod or p.food) and "timing" not in p.ambiguities:
        out.extend(_timing(p, src, doc))
    return _order(out)


def _timing(p: Parsed, src: Parsed, doc: Documented) -> list[ProposalDimension]:
    out = []
    shown = timing_text(p)
    doc_shown = timing_text(src) or None
    sid = _src(doc, "timing")
    worst = "match"
    reason = None
    if (p.tod and len(src.tod) > 1) or (p.food and len(src.food) > 1):
        out.append(_dim("timing", "unverifiable", shown, doc_shown, sid, "the sources give two different timings"))
        return out
    for t in p.tod:
        if not src.tod:
            worst, reason = "unverifiable", "the sources state no time of day"
        elif t not in src.tod:
            worst, reason = "mismatch", None
            break
    if worst != "mismatch":
        for f in p.food:
            if f in src.food:
                continue
            if not src.food:
                worst, reason = "unverifiable", "the sources state nothing about food"
            elif {f} | src.food <= {"before food", "empty stomach"}:
                worst, reason = "unverifiable", "'before food' and 'empty stomach' are not the same instruction"
            else:
                worst, reason = "mismatch", None
                break
    out.append(_dim("timing", worst, shown, doc_shown, sid, reason))
    return out


_TOD_TEXT = {"morning": "in the morning", "evening": "in the evening", "midday": "at midday"}
_FOOD_TEXT = {"with food": "with food", "before food": "before food", "after food": "after food",
              "empty stomach": "on an empty stomach"}


def timing_text(p: Parsed) -> str:
    return " ".join([_TOD_TEXT[t] for t in sorted(p.tod)] + [_FOOD_TEXT[f] for f in sorted(p.food)])


def _order(dims: list[ProposalDimension]) -> list[ProposalDimension]:
    rank = {n: i for i, n in enumerate(_DIM_ORDER)}
    return sorted(dims, key=lambda d: rank.get(d.name, 99))


# --- Wording ---------------------------------------------------------------------

def _phrase(dims: list[ProposalDimension], field_: str) -> str:
    parts = []
    for d in dims:
        value = d.proposed if field_ == "proposed" else d.documented
        if value is None:
            continue
        if d.name == "duration":
            value = "for " + value
        elif d.name == "route":
            value = f"by the {value} route" if len(dims) > 1 else f"the {value} route"
        elif d.name == "timing":
            value = "taken " + value if len(dims) > 1 else value
        elif d.name == "max_daily":
            value = ("a daily total of " + value.replace(" a day", "")) if field_ == "documented" \
                else "a maximum of " + value
        parts.append(value)
    return " ".join(parts)


def verdict_text(check: ProposalCheck, hedged: bool, drug: str) -> str:
    checked = [d for d in check.dimensions if d.status != "unverifiable" and d.documented is not None]
    if check.status == "match":
        what = " ".join(x for x in (_phrase([d for d in checked if d.name != "max_daily"], "proposed"),
                                    _phrase([d for d in checked if d.name == "max_daily"], "documented")) if x)
        if hedged:
            return f"Yes — the proposed regimen matches the source: {what}."
        return f"Yes — this matches the source: {what}."
    documented_ = _phrase(checked, "documented")
    proposed = _phrase(checked, "proposed")
    if hedged:
        return f"No — the proposed regimen does not match the source, which gives {documented_}, not {proposed}."
    return f"No — the source gives {documented_}, not {proposed}."


def unverifiable_message(check: ProposalCheck, drug: str | None) -> str:
    reasons = [d.reason for d in check.dimensions if d.status == "unverifiable" and d.reason]
    head = "The proposed regimen cannot be checked against the sources"
    return head + (": " + "; ".join(dict.fromkeys(reasons)) if reasons else "") + "."


# --- The entry point -------------------------------------------------------------

@dataclass
class Outcome:
    check: ProposalCheck
    claims: list[LLMClaim]
    lead: str
    message: str


def _claims(dims: list[ProposalDimension], doc: Documented) -> list[LLMClaim]:
    seen: list[tuple[str, str]] = []
    wanted = [d.name for d in dims if d.status != "unverifiable"]
    if any(n in wanted for n in ("dose", "frequency", "duration", "max_daily", "timing")):
        order = ["dose", "frequency", "duration", "max_daily", "timing"]
    else:
        order = []
    for name in order + (["route"] if "route" in wanted else []):
        if name in wanted:
            for item in doc.sentences.get(name, [])[:2]:
                if item not in seen:
                    seen.append(item)
    return [LLMClaim(text=s, source_ids=[i]) for i, s in seen]


def evaluate(question: str) -> Outcome | None:
    reading = read(question)
    if reading is None:
        return None
    drugs = reading.drugs
    dims: list[ProposalDimension] = []
    doc = None
    drug_name = None
    if len(drugs) > 1:
        dims.append(_dim("drug", "unverifiable", ", ".join(drugs), reason="the question names more than one medicine, "
                                                                           "so the amounts cannot be attributed"))
    elif reading.misspelt and not drugs:
        w, near = reading.misspelt[0]
        dims.append(_dim("drug", "unverifiable", w, reason=f"'{w}' is not a medicine in the sources (did you mean "
                                                           f"'{near}'?)"))
    elif not drugs:
        named = retrieval.topics_mentioned(question).values()
        if not any(r.get("kind") in ("condition", "marker", "procedure") for t in named
                   for r in retrieval.topic_records({t})):
            return None
        dims.append(_dim("drug", "unverifiable", None, reason="no medicine in the sources is named, so there is "
                                                              "nothing to check the regimen against"))
    else:
        drug_name = drugs[0]
        doc = documented(drug_name)
        dims, doc = compare_all(reading, doc)
        if reading.misspelt:
            w, near = reading.misspelt[0]
            dims.append(_dim("drug", "unverifiable", w, reason=f"'{w}' is not a medicine in the sources"))
    if reading.negated:
        dims.append(_dim("proposal", "unverifiable", None,
                         reason="the question is phrased negatively, so a yes or no would be ambiguous; ask it "
                                "without the negative"))
    if reading.inverted:
        dims.append(_dim("proposal", "unverifiable", None,
                         reason="the question asks whether the amount is too much, too little or harmful, which the "
                                "sources do not judge; ask whether the regimen matches the source instead"))
    if not dims:
        dims.append(_dim("proposal", "unverifiable", None, reason="nothing in the question could be checked"))
    status = ("unverifiable" if any(d.status == "unverifiable" for d in dims)
              else "mismatch" if any(d.status == "mismatch" for d in dims) else "match")
    check = ProposalCheck(status=status, dimensions=dims, drug=drug_name)
    if status == "unverifiable":
        return Outcome(check, [], "", unverifiable_message(check, drug_name))
    hedged = reading.negated
    return Outcome(check, _claims(dims, doc), verdict_text(check, hedged, drug_name or ""), "")
