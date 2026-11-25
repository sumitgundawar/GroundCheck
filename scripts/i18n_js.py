"""Find the string and template literals in web/app.js that read like prose.

A minimal JavaScript tokenizer (strings, templates with nested ${}, comments,
regex literals) so tests and the extraction helper need no JS toolchain.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
KEYWORD_BEFORE_REGEX = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "else", "do"}


@dataclass
class Literal:
    kind: str          # "str" or "tpl"
    start: int         # index of the opening quote
    end: int           # index just past the closing quote
    parts: list[str] = field(default_factory=list)   # static text pieces
    exprs: list[tuple[int, int]] = field(default_factory=list)   # ${...} source ranges
    in_t_call: bool = False
    line: int = 0

    @property
    def text(self) -> str:
        return "".join(self.parts)


def tokenize(src: str) -> list[Literal]:
    out: list[Literal] = []
    n = len(src)
    last_sig = ""        # last significant character
    last_word = ""

    def line_of(i: int) -> int:
        return src.count("\n", 0, i) + 1

    def read_string(i: int) -> tuple[Literal, int]:
        q = src[i]
        j = i + 1
        buf = []
        while j < n and src[j] != q:
            if src[j] == "\\":
                buf.append(src[j:j + 2])
                j += 2
            else:
                buf.append(src[j])
                j += 1
        return Literal("str", i, j + 1, ["".join(buf)], line=line_of(i)), j + 1

    def read_template(i: int) -> tuple[Literal, int]:
        j = i + 1
        parts: list[str] = []
        exprs: list[tuple[int, int]] = []
        buf: list[str] = []
        while j < n and src[j] != "`":
            if src[j] == "\\":
                buf.append(src[j:j + 2])
                j += 2
            elif src[j] == "$" and src[j + 1:j + 2] == "{":
                parts.append("".join(buf))
                buf = []
                depth = 1
                j += 2
                expr_start = j
                while j < n and depth:
                    c = src[j]
                    if c in "\"'":
                        lit, j = read_string(j)
                        out.append(lit)
                        continue
                    if c == "`":
                        lit, j = read_template(j)
                        out.append(lit)
                        continue
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                    j += 1
                exprs.append((expr_start, j - 1))
            else:
                buf.append(src[j])
                j += 1
        parts.append("".join(buf))
        return Literal("tpl", i, j + 1, parts, exprs, line=line_of(i)), j + 1

    i = 0
    while i < n:
        c = src[i]
        if c == "/" and src[i + 1:i + 2] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and src[i + 1:i + 2] == "*":
            i = src.index("*/", i + 2) + 2
            continue
        if c in "\"'":
            lit, i = read_string(i)
            out.append(lit)
            last_sig = '"'
            continue
        if c == "`":
            lit, i = read_template(i)
            out.append(lit)
            last_sig = '"'
            continue
        if c == "/":
            if last_sig in REGEX_PRECEDERS or last_sig == "" or last_word in KEYWORD_BEFORE_REGEX:
                j = i + 1
                in_class = False
                while j < n:
                    if src[j] == "\\":
                        j += 2
                        continue
                    if src[j] == "[":
                        in_class = True
                    elif src[j] == "]":
                        in_class = False
                    elif src[j] == "/" and not in_class:
                        break
                    j += 1
                j += 1
                while j < n and src[j].isalpha():
                    j += 1
                i = j
                last_sig = '"'
                last_word = ""
                continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            last_word = src[i:j]
            last_sig = "a"
            i = j
            continue
        if not c.isspace():
            last_sig = c
            last_word = ""
        i += 1
    out.sort(key=lambda l: l.start)
    mark_t_calls(src, out)
    return out


T_CALL = re.compile(r"(?<![\w.$])t\(\s*$")


def mark_t_calls(src: str, lits: list[Literal]) -> None:
    """A literal is keyed when it is the first or second argument of t(...)."""
    for lit in lits:
        before = src[max(0, lit.start - 200):lit.start]
        if T_CALL.search(before):
            lit.in_t_call = True
            continue
        m = re.search(r'(?<![\w.$])t\(\s*(?:"[^"\n]*"|`[^`]*`)\s*,\s*$', before)
        if m:
            lit.in_t_call = True


def visible_text(lit: Literal) -> str:
    pieces = []
    for p in lit.parts:
        p = p.replace("\\n", " ").replace('\\"', '"').replace("\\'", "'")
        p = re.sub(r"<[^>]*>", " ", p)
        pieces.append(html.unescape(p))
    return " ".join(" ".join(pieces).split())


WORD = re.compile(r"[A-Za-zÀ-ɏ]{2,}")


def looks_like_prose(text: str) -> bool:
    """Two or more words, or one capitalised word of 4+ letters followed by punctuation/space."""
    words = WORD.findall(text)
    if not words:
        return False
    if re.fullmatch(r"[\w\-./:#\[\]=>*,+~ %()@?&!|]*", text) and " " not in text.strip() and not text[:1].isupper():
        return False  # identifier, selector, path, css value
    return len(words) >= 2 or (text[:1].isupper() and len(words[0]) >= 3)


def prose_literals(src: str) -> list[tuple[Literal, str]]:
    found = []
    for lit in tokenize(src):
        if lit.in_t_call:
            continue
        text = visible_text(lit)
        if text and looks_like_prose(text):
            found.append((lit, text))
    return found


# Strings that are code, not words a person reads: HTTP verbs and headers,
# key names, selectors, CSS, wire values, sample data, brand and model names.
TECHNICAL = {
    "use strict", "POST", "PUT", "PATCH", "DELETE", "Content-Type", "DOMContentLoaded", "Escape", "Enter",
    "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Home", "End", "NFKD", "(prefers-reduced-motion: reduce)",
    "GroundCheckHealth", "ResNet-18", "Small CNN", "SMART on FHIR", "Qdrant", "AUC",
    "Practitioner/example", "Bundle", "MedicationRequest", "Caloradine", "Patient", "Observation",
    "Mendel solution 5 mL", "order-select, order-sign", "DICOM PS3.15 basic profile.",
    "600 10px ui-monospace, monospace", "600 11px ui-monospace, monospace",
    "the and for with what how does can should are was were this that from have has not you your about into when which who whom may much many dose doses give given take taking",
    "neutral", "uncovered", "text-anchor", "end", "middle", "numeric", "short", "img", "aria-label",
    "sentence-transformers/", "sentence-transformers", "qdrant", "running", "failed", "too-large", "in-use", "fit",
    "recommended", "resnet18", "identity", "auto", "closed", "investigating", "status-pending", "critical",
    "refusal_rate", "latency", "ehr_error", "string", "HTTP", "reject", "refused", "draft", "div", "span", "strong",
    "{medicine}", "{ids}", "Local model", "Cloud model", "OK", "GB", "soft tissue", "abdomen", "lung", "bone", "brain",
}

# The prefix of the statement a literal sits in says what the literal is for.
TECH_BEFORE = [
    r"className\b[^;]*$", r"classList\.\w+\([^)]*$", r"querySelector(All)?\([^)]*$", r"getElementById\(\s*$",
    r"addEventListener\(\s*$", r"method:\s*$", r"\bapi\(\s*$", r"\bfetch\(\s*$", r"setAttribute\(\s*$",
    r"\bel\(\s*\"\w+\",\s*$", r"\bel\(\s*\"\w+\",\s*[^,]*\?\s*$", r"\.matches\(\s*$", r"\.closest\(\s*$",
    r"\$\(\s*$", r"\.dataset\.(?!label\b)[^;]*$", r"\.dataset\[[^;]*$", r"\.style\b[^;]*$", r"getPropertyValue\(\s*$", r"\.has\(\s*$",
    r"\.key\s*===?\s*$", r"\.type\s*===?\s*$", r"createElementNS\([^,]*,\s*$", r"createElement\(\s*$",
    r"\bhas\(\s*(\"[^\"]*\",\s*)*$", r"\bfont\s*=\s*$", r"Content-Type", r"\.href\s*=\s*$", r"\.src\s*=\s*$",
    r"\bel\(\s*$", r"plural\([^;]*,\s*$", r"\bcoded\(\s*$", r"\.replace\(\s*$", r"[!=]==?\s*$", r"\bcase\s*$", r"\.includes\(\s*$", r"\bkind\s*[:=]\s*$", r"\bcls\s*[:=]", r"\bcss\s*[:=]",
]
TECH_BEFORE_RE = [re.compile(p) for p in TECH_BEFORE]
TECH_TEXT = [
    re.compile(r"^(/|https?:|#|\.|\[)"), re.compile(r"var\(--"), re.compile(r"^[\w-]*\[.*\]"),
    re.compile(r"^[a-z0-9_-]+( [a-z0-9_-]+)*$"),  # class lists and ids; real words are handled below
]


def technical(src: str, lit: Literal, text: str) -> bool:
    if lit.kind == "str" and text in TECHNICAL:
        return True
    head = src[max(0, lit.start - 3):lit.start].rstrip()
    if src[lit.end:lit.end + 8].lstrip().startswith(":") and head[-1:] in ("{", ",", ""):
        return True  # object key
    before = src[max(0, lit.start - 140):lit.start]
    before = before[before.rfind(";") + 1:] if ";" in before else before
    if any(r.search(before) for r in TECH_BEFORE_RE):
        return True
    t = "".join(lit.parts).strip()
    if lit.kind == "tpl" and t.strip() in {"dot", "fit"}:
        return True
    if re.match(r"^(/|https?:|#|\.|\?|rgba?\(|[0-9a-f]{8}-)", t) and " " not in t:
        return True
    if t in {"2-digit", "numeric", "short", "long"}:
        return True
    if "var(--" in t or re.match(r"^[\w-]*\[.*\]", t):
        return True
    return False


SINK = re.compile(r"(textContent|\.title|\.placeholder|ariaLabel|\.alt|\.label|\.value)\s*=[^=;]*$")
SINK_CALL = re.compile(r"(showMessage\([^,]+,|confirm\(|alert\(|prompt\(|new Error\(|statTile\(|badge\(|"
                       r"\bel\(\s*\"\w+\",\s*[^,]*,\s*|setAttribute\(\s*\"(aria-label|title|placeholder|alt)\",\s*|new Option\(|"
                       r"labelCells\(|termChips\(|Pill\()")


def in_sink(src: str, lit: Literal) -> bool:
    before = src[max(0, lit.start - 160):lit.start]
    before = before[before.rfind(";") + 1:] if ";" in before else before
    return bool(SINK.search(before) or SINK_CALL.search(before))


UNITS = {"MB", "GB", "px", "ms", "s", "min", "h", "mm", "kg", "mg", "ml", "mL", "dl", "id", "ID", "x", "KB", "CT", "MR"}


def template_prose(lit: Literal, text: str) -> bool:
    """A template whose fixed text holds a word is a sentence or a fragment of one."""
    raw = "".join(lit.parts)
    if re.search(r"[\w]-(?=\s|$)", raw) or re.search(r"-\s*$", raw.strip()):
        return False  # a class name built with ${...}
    words = [w for w in WORD.findall(text) if w not in UNITS]
    return bool(words)


def candidates(src: str) -> list[tuple[Literal, str]]:
    """Literals a person could read that are not inside t(...) and not technical."""
    out = []
    for lit in tokenize(src):
        if lit.in_t_call:
            continue
        text = visible_text(lit)
        if not text or technical(src, lit, text):
            continue
        letters = WORD.findall(text)
        if looks_like_prose(text) or (lit.kind == "tpl" and template_prose(lit, text)) or (
                letters and in_sink(src, lit) and not re.fullmatch(r"[a-z]{1,2}", text)):
            out.append((lit, text))
    return out


_cache: dict[int, list[Literal]] = {}


def tokenize_range(src: str, a: int, b: int) -> list[Literal]:
    lits = _cache.get(id(src))
    if lits is None:
        _cache.clear()
        lits = _cache[id(src)] = tokenize(src)
    return [lit for lit in lits if lit.start >= a and lit.end <= b]
