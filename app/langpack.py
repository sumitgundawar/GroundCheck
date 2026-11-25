"""Language packs: the per-language word lists under app/data/<lang>/ and the
translator that turns a question in that language into the English the
pipeline's guards check.

The sources are English and so is every dose in an answer, so a pack never
translates an answer: it only translates the question, with a closed
vocabulary. A word the pack does not know and the index does not know as a name
is not guessed at. The question is refused as unverifiable, and so is any
number whose decimal or thousands separator is ambiguous, or that is followed
by a unit the pack cannot name. A wrong answer needs a wrong dose, and a dose
only ever comes from a source, so the failure to avoid is a question being
read as a different one; the closed vocabulary is what prevents it."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from . import config

PACK_FILES = ("common_words.txt", "clinical_terms.txt", "conversational.txt", "qualifiers.txt",
              "units.txt", "frequencies.txt")
_PLACEHOLDER = re.compile(r"\[[A-Z_ ]+\]")
_SEGMENT_BREAK = re.compile(r"(?<!\d)[,.]|[,.](?!\d)|[;:?!¿¡]")
_TOKEN = re.compile(r"\[[A-Z_ ]+\]|\d+(?:[.,]\d+)*|[^\W\d_]+(?:[/'’-][^\W\d_]+)*|%", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*$")


def available() -> set[str]:
    """Languages with a pack on disk, English included."""
    langs = {"en"}
    root = config.DATA_DIR
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and all((child / name).exists() for name in PACK_FILES):
            langs.add(child.name)
    return langs


def fold(text: str) -> str:
    """Lowercase and drop accents, for people who type without them."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def read_pairs(path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, value = line.partition("\t")
        if not _:
            key, _, value = line.partition(" ")
        rows.append((unicodedata.normalize("NFC", key.strip().lower()), value.strip()))
    return rows


@dataclass
class Pack:
    lang: str
    words: dict[str, str] = field(default_factory=dict)
    folded: dict[str, str] = field(default_factory=dict)
    phrases: dict[tuple[str, ...], str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    qualifiers: set[str] = field(default_factory=set)
    patterns: list[tuple["re.Pattern[str]", str]] = field(default_factory=list)
    longest: int = 1
    time_words: set[str] = field(default_factory=set)
    number_words: set[str] = field(default_factory=set)


@lru_cache(maxsize=8)
def load(lang: str) -> Pack | None:
    folder = config.DATA_DIR / lang
    if lang == "en" or not all((folder / n).exists() for n in PACK_FILES):
        return None
    pack = Pack(lang)
    for name in ("common_words.txt", "frequencies.txt", "conversational.txt", "clinical_terms.txt", "qualifiers.txt"):
        for key, value in read_pairs(folder / name):
            pack.words[key] = value
            if name == "qualifiers.txt":
                pack.qualifiers.add(key)
    pack.units = dict(read_pairs(folder / "units.txt"))
    time = {"hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years", "minute",
            "minutes", "time", "times", "h", "daily", "weekly"}
    numbers = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
               "fifteen", "twenty", "thirty", "forty", "fifty", "hundred", "half", "quarter"}
    for key, value in pack.words.items():
        if value in time:
            pack.time_words.add(fold(key))
        if value in numbers:
            pack.number_words.add(key)
    # Accent-free spellings. A spelling that two different accented words share
    # (se / sé, el / él, si / sí) is only taken when it is itself a word.
    candidates: dict[str, set[str]] = {}
    for key, value in {**pack.words, **pack.units}.items():
        candidates.setdefault(fold(key), set()).add(value)
    for key, values in candidates.items():
        if key in pack.words or key in pack.units:
            continue
        if len(values) == 1:
            pack.folded[key] = next(iter(values))
    if (folder / "phrases.txt").exists():
        for key, value in read_pairs(folder / "phrases.txt"):
            tokens = tuple(fold(t) for t in _TOKEN.findall(key))
            pack.phrases[tokens] = value
            pack.longest = max(pack.longest, len(tokens))
    if (folder / "patterns.txt").exists():
        for line in (folder / "patterns.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                regex, _, template = line.partition("\t")
                pack.patterns.append((re.compile("".join(c for c in unicodedata.normalize("NFD", regex) if not unicodedata.combining(c)), re.IGNORECASE), template.strip()))
    return pack


@dataclass
class Translation:
    english: str
    unverified: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.reasons


def parse_number(token: str) -> tuple[str | None, str | None]:
    """A number written the Spanish way as (English form, None), or (None, why)
    when its separators cannot be read with certainty. A comma is the decimal
    mark, a dot groups thousands; but "1,500" and "1.500" are each a trap: one
    and a half, or fifteen hundred, depending on who wrote it."""
    commas, dots = token.count(","), token.count(".")
    if not commas and not dots:
        return token, None
    if commas and dots:
        last = max(token.rfind(","), token.rfind("."))
        mark = token[last]
        group = "," if mark == "." else "."
        whole, frac = token[:last], token[last + 1:]
        groups = whole.split(group)
        if mark not in whole and all(len(g) == 3 for g in groups[1:]) and 1 <= len(groups[0]) <= 3:
            return "".join(groups) + "." + frac, None
        return None, f"the number {token} is ambiguous"
    sep = "," if commas else "."
    parts = token.split(sep)
    if len(parts) > 2:
        if all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            return "".join(parts), None
        return None, f"the number {token} is ambiguous"
    whole, frac = parts
    if len(frac) == 3 and whole != "0":
        return None, f"the number {token} is ambiguous"
    return f"{whole}.{frac}", None


def _lookup(pack: Pack, word: str) -> str | None:
    low = unicodedata.normalize("NFC", word.lower())
    if low in pack.words:
        return pack.words[low]
    return pack.folded.get(fold(low))


def _word_level(pack: Pack, text: str, is_known, result: Translation, *, start_of_text: bool) -> str:
    raw = _TOKEN.findall(text)
    out: list[str] = []
    i = 0
    n = len(raw)
    after_number = False
    while i < n:
        token = raw[i]
        was_number, after_number = after_number, False
        # Longest multi-word phrase first.
        matched = False
        for size in range(min(pack.longest, n - i), 1, -1):
            key = tuple(fold(t) for t in raw[i:i + size])
            if key in pack.phrases:
                out.append(pack.phrases[key])
                i += size
                matched = True
                break
        if matched:
            continue
        if len(raw[i:i + 1]) and (fold(token),) in pack.phrases:
            out.append(pack.phrases[(fold(token),)])
            i += 1
            continue
        if _PLACEHOLDER.fullmatch(token):
            out.append(token)
        elif _NUMBER.match(token):
            value, why = parse_number(token)
            if value is None:
                result.reasons.append(why or "ambiguous number")
                out.append(token)
            else:
                out.append(value)
                after_number = True
        else:
            low = unicodedata.normalize("NFC", token.lower())
            unit = pack.units.get(low)
            mapped = _lookup(pack, token)
            if was_number and unit is None and fold(token) not in pack.time_words and token.lower() not in pack.number_words:
                result.unverified.append(f"{low} (a unit the pack cannot verify)")
                out.append(token)
            elif unit is not None and mapped is None:
                out.append(unit)
            elif mapped is not None:
                if mapped != "-":
                    out.append(mapped)
                after_number = low in pack.number_words
            elif is_known(low):
                out.append(token)
            elif token[0].isupper() and not (start_of_text and i == 0) and token.lower() not in ("dr", "dra"):
                out.append(token)
            elif low in ("dr", "dra", "sr", "sra"):
                out.append({"dr": "Dr", "dra": "Dr", "sr": "Mr", "sra": "Ms"}[low])
            else:
                result.unverified.append(low)
                out.append(token)
        i += 1
    return " ".join(out)


def translate(text: str, lang: str, is_known=None) -> Translation:
    """The English a question in `lang` says, or why it can't be verified."""
    pack = load(lang)
    result = Translation("")
    if pack is None:
        result.reasons.append(f"no word list for {lang}")
        return result
    if is_known is None:
        from . import retrieval

        is_known = retrieval.is_known_term
    text = unicodedata.normalize("NFC", text)
    pieces: list[str] = []
    last = 0
    for match in _SEGMENT_BREAK.finditer(text):
        pieces.append(("seg", text[last:match.start()]))
        pieces.append(("sep", match.group(0)))
        last = match.end()
    pieces.append(("seg", text[last:]))
    out: list[str] = []
    for kind, piece in pieces:
        if kind == "sep":
            if piece in ("?", "!", ",", ".", ";", ":"):
                out.append(piece)
            continue
        segment = piece.strip()
        if not segment:
            continue
        folded = fold(segment)
        for regex, template in pack.patterns:
            m = regex.match(folded) if len(folded) == len(segment) else None
            if m:
                parts = {k: _word_level(pack, segment[m.start(k):m.end(k)], is_known, result, start_of_text=False)
                         for k in m.groupdict()}
                out.append(template.format(**parts))
                break
        else:
            out.append(_word_level(pack, segment, is_known, result, start_of_text=True))
    english = ""
    for item in out:
        if item in ("?", "!", ",", ".", ";", ":"):
            english += item
        else:
            english += (" " if english and not english.endswith(" ") else "") + item
    result.english = english.strip()
    for number in re.findall(r"\d+", text):
        if number not in re.sub(r"[.,]", "", result.english) and not result.reasons:
            result.reasons.append("a number was lost in translation")
    return result
