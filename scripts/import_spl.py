"""Turn drug labels in SPL XML (the format DailyMed publishes) into a knowledge pack.

    python scripts/import_spl.py labels/*.xml --out us-labels.gcpack \\
        --name us-labels --version 1.0.0 --publisher "Your organisation" \\
        --licence "Public domain (US FDA labeling); confirm terms" [--sign-key publisher.key]

Each label becomes one document whose headings are the label's standard
sections (named from LOINC section codes: Indications and Usage, Dosage and
Administration, Contraindications and so on), with a "Product Information"
section giving the strength, route and manufacturer. A sub-section's text sits under its parent's heading,
led by its own name ("Pediatric Dosage: ..."), so a whole section is one passage. A small
evaluation set is generated from what each label states.

This script never downloads anything. Give it files you have already
obtained. Whether the content can be redistributed, and on what terms, is for
the owner to confirm before any real label is distributed in a pack.

XML is parsed with defusedxml when it is installed; otherwise with the standard
library, refusing any document type declaration or entity declaration, so
external entities and entity-expansion attacks are not possible either way."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_BYTES = 30 * 1024 * 1024
MAX_DEPTH = 64

# LOINC section codes used in SPL, and the heading each is given.
SECTIONS = {
    "34067-9": "Indications and Usage",
    "34068-7": "Dosage and Administration",
    "43678-2": "Dosage Forms and Strengths",
    "34070-3": "Contraindications",
    "34066-1": "Boxed Warning",
    "34071-1": "Warnings",
    "43685-7": "Warnings and Precautions",
    "34084-4": "Adverse Reactions",
    "34073-7": "Drug Interactions",
    "43684-0": "Use in Specific Populations",
    "42228-7": "Pregnancy",
    "34080-2": "Nursing Mothers",
    "77290-5": "Lactation",
    "34081-0": "Pediatric Use",
    "34082-8": "Geriatric Use",
    "34088-5": "Overdosage",
    "34089-3": "Description",
    "34090-1": "Clinical Pharmacology",
    "34069-5": "How Supplied",
    "34076-0": "Information for Patients",
}
IGNORED_CODES = {"48780-1", "51945-4", "34093-5", "42229-5"}   # listing data, package label, references, unclassified


class SplError(ValueError):
    """The label can't be read. The message is safe to show."""


# --- XML -------------------------------------------------------------------

def _reject_declaration(*_args):
    raise SplError("The file declares a document type or entities, which SPL labels never need; "
                   "refusing it as unsafe.")


def _parse_stdlib(data: bytes) -> ET.Element:
    """Parse with expat directly, so a document type or entity declaration can
    be refused the moment the parser sees it. Names come back as "uri}local"."""
    parser = expat.ParserCreate(namespace_separator="}")
    parser.buffer_text = True
    parser.StartDoctypeDeclHandler = _reject_declaration
    parser.EntityDeclHandler = _reject_declaration
    parser.ExternalEntityRefHandler = lambda *a: _reject_declaration()
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    stack: list[ET.Element] = []
    last: list[ET.Element | None] = []   # the child just closed, whose tail text may follow
    roots: list[ET.Element] = []

    def start(name, attrs):
        el = ET.Element(name, attrs)
        if stack:
            stack[-1].append(el)
        else:
            roots.append(el)
        stack.append(el)
        last.append(None)

    def end(_name):
        done = stack.pop()
        last.pop()
        if last:
            last[-1] = done

    def text(chunk):
        if not stack:
            return
        if last[-1] is None:
            stack[-1].text = (stack[-1].text or "") + chunk
        else:
            last[-1].tail = (last[-1].tail or "") + chunk

    parser.StartElementHandler, parser.EndElementHandler, parser.CharacterDataHandler = start, end, text
    try:
        parser.Parse(data, True)
    except expat.ExpatError as exc:
        raise SplError(f"The file isn't well-formed XML: {expat.ErrorString(exc.code)} "
                       f"(line {exc.lineno})") from None
    if not roots:
        raise SplError("The file has no XML content.")
    return roots[0]


def _parse_defused(data: bytes) -> ET.Element:
    import defusedxml.ElementTree as safe
    from defusedxml import DefusedXmlException

    try:
        return safe.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except DefusedXmlException:
        _reject_declaration()
    except ET.ParseError as exc:
        raise SplError(f"The file isn't well-formed XML: {exc}") from None


def parse_xml(data: bytes) -> ET.Element:
    if not data or not data.strip():
        raise SplError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise SplError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    root = _parse_defused(data) if importlib.util.find_spec("defusedxml") else _parse_stdlib(data)
    stack = [(root, 1)]
    while stack:
        el, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise SplError("The XML is nested too deeply.")
        stack.extend((child, depth + 1) for child in el)
    return root


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(el: ET.Element, name: str):
    return [c for c in el if _local(c.tag) == name]


def _child(el: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in el if _local(c.tag) == name), None)


def _find_all(el: ET.Element, name: str):
    return [e for e in el.iter() if _local(e.tag) == name]


_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    return _WS.sub(" ", text).strip()


# --- Text of a section ------------------------------------------------------

_BLOCKS = {"paragraph", "item", "caption", "br", "list", "table", "thead", "tbody", "tfoot", "tr"}
_SKIP = {"renderMultiMedia", "observationMedia", "footnote"}


def render_text(el: ET.Element) -> list[str]:
    """The paragraphs of an SPL <text> element: paragraphs, list items, and
    table rows (cells joined by " | ")."""
    out: list[str] = []

    def emit(parts: list[str]) -> None:
        text = _clean("".join(parts))
        if text:
            out.append(text)

    def walk(node: ET.Element) -> None:
        buf = [node.text or ""]
        for child in node:
            name = _local(child.tag)
            if name in _SKIP:
                pass
            elif name == "tr":
                emit(buf)
                buf = []
                cells = [_clean("".join(c.itertext())) for c in child if _local(c.tag) in ("td", "th")]
                emit([" | ".join(c for c in cells if c)])
            elif name in _BLOCKS:
                emit(buf)
                buf = []
                walk(child)
            else:
                buf.append(" ".join(_inline(child)))
            buf.append(child.tail or "")
        emit(buf)

    def _inline(node: ET.Element) -> list[str]:
        parts = [node.text or ""]
        for child in node:
            if _local(child.tag) not in _SKIP:
                parts.extend(_inline(child))
            parts.append(child.tail or "")
        return parts

    walk(el)
    return out


# --- Doses, routes and strengths -------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = r"milligrams?|micrograms?|grams?|mcg|mg|µg|μg|ug|mL|ml|g|%|units?|IU"
_DOSE = re.compile(
    rf"(?<![\w.,/])(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?(?![A-Za-z])"
    rf"(?:\s*(?:to|-|–|—|through)\s*(?P<b>{_NUM})\s*(?P<ub>{_UNIT})(?![A-Za-z]))?"
    rf"(?P<per>\s*(?:/|per\s+)\s*(?:\d+(?:\.\d+)?\s*)?(?P<d1>kg|mL|day|dose|m2|m²)(?:\s*/\s*(?P<d2>day|dose|hour))?(?![A-Za-z]))?",
    re.IGNORECASE)
_UNIT_NAMES = {"milligram": "mg", "milligrams": "mg", "mg": "mg", "microgram": "mcg", "micrograms": "mcg",
               "mcg": "mcg", "µg": "mcg", "μg": "mcg", "ug": "mcg", "gram": "g", "grams": "g", "g": "g",
               "ml": "mL", "%": "%", "unit": "units", "units": "units", "iu": "IU"}
_FREQUENCY = [
    (re.compile(r"\b(?:once|1 time)\s+(?:a\s+|per\s+)?(?:day|daily)\b|\bonce-daily\b|\bq\.?d\.?\b", re.I), "once daily"),
    (re.compile(r"\b(?:twice|2 times|two times)\s+(?:a\s+|per\s+)?(?:day|daily)\b|\bb\.?i\.?d\.?\b", re.I), "twice daily"),
    (re.compile(r"\b(?:three|3)\s+times\s+(?:a\s+|per\s+)?(?:day|daily)\b|\bt\.?i\.?d\.?\b", re.I), "three times daily"),
    (re.compile(r"\b(?:four|4)\s+times\s+(?:a\s+|per\s+)?(?:day|daily)\b|\bq\.?i\.?d\.?\b", re.I), "four times daily"),
    (re.compile(r"\bonce\s+(?:a\s+|per\s+)?week(?:ly)?\b|\bweekly\b", re.I), "once weekly"),
    (re.compile(r"\bevery\s+other\s+day\b", re.I), "every other day"),
    (re.compile(r"\bat\s+bedtime\b", re.I), "at bedtime"),
    (re.compile(r"\bas\s+needed\b", re.I), "as needed"),
    (re.compile(r"\bdaily\b", re.I), "daily"),
]
_EVERY_HOURS = re.compile(r"\bevery\s+(\d+)\s*(?:to\s+(\d+)\s*)?hours?\b|\bq\s?(\d+)\s?h\b", re.I)
_ROUTES = [
    (re.compile(r"\b(?:orally|by mouth|oral)\b", re.I), "oral"),
    (re.compile(r"\bintravenous(?:ly)?\b|\bIV\b", re.I), "intravenous"),
    (re.compile(r"\bintramuscular(?:ly)?\b", re.I), "intramuscular"),
    (re.compile(r"\bsubcutaneous(?:ly)?\b", re.I), "subcutaneous"),
    (re.compile(r"\btopical(?:ly)?\b|\bapply\b|\bapplied\b", re.I), "topical"),
    (re.compile(r"\bophthalmic\b|\beye\b", re.I), "ophthalmic"),
    (re.compile(r"\binhal\w*", re.I), "inhalation"),
    (re.compile(r"\brectal(?:ly)?\b", re.I), "rectal"),
    (re.compile(r"\bsublingual(?:ly)?\b", re.I), "sublingual"),
]
_POPULATIONS = [
    (re.compile(r"\b(?:pediatric|paediatric|child|children|infants?|neonates?|adolescents?)\b", re.I), "pediatric"),
    (re.compile(r"\b(?:geriatric|elderly|older adults?)\b", re.I), "geriatric"),
    (re.compile(r"\badults?\b", re.I), "adult"),
]
_SENTENCES = re.compile(r"(?<=[.;!?])\s+(?=[A-Z(])")


@dataclass(frozen=True)
class Dose:
    amount: float
    amount_max: float | None       # set for a range such as "5 to 10 mg"
    unit: str                      # mg, mcg, g, mL, %, units, IU
    per: str | None                # kg, mL, day, dose, m2: "5 mg/kg", "5 mg/mL"
    frequency: str | None
    route: str | None
    population: str                # adult, pediatric, geriatric or unspecified
    section: str
    sentence: str

    @property
    def is_range(self) -> bool:
        return self.amount_max is not None

    @property
    def kind(self) -> str:
        return {"kg": "per_kg", "mL": "concentration", "m2": "per_m2", "day": "daily_total"}.get(self.per or "", "dose")


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _detect(patterns, text: str, default=None):
    for pattern, name in patterns:
        if pattern.search(text):
            return name
    return default


def _frequency(text: str) -> str | None:
    """The frequency named earliest in `text`."""
    found: list[tuple[int, str]] = []
    for pattern, name in _FREQUENCY:
        if m := pattern.search(text):
            found.append((m.start(), name))
    if m := _EVERY_HOURS.search(text):
        low, high, q = m.group(1), m.group(2), m.group(3)
        found.append((m.start(), f"every {low or q}{' to ' + high if high else ''} hours"))
    return min(found, key=lambda f: f[0])[1] if found else None


def extract_doses(text: str, *, section: str = "", route: str | None = None) -> list[Dose]:
    """Dose statements in a passage: an amount (or range) with a unit, what it
    is per, how often, by which route, and for whom."""
    doses: list[Dose] = []
    heading_population = _detect(_POPULATIONS, section, None)
    for sentence in _SENTENCES.split(text):
        for m in _DOSE.finditer(sentence):
            ua, ub = m.group("ua"), m.group("ub")
            if not (ua or ub):
                continue
            unit_text = ub or ua
            low, high = _number(m.group("a")), None
            if ub:
                if ua and _UNIT_NAMES[ua.lower()] != _UNIT_NAMES[ub.lower()]:
                    doses.append(_dose(low, None, ua, m.group("d1"), sentence, m.end(), section, route,
                                       heading_population))
                    low = _number(m.group("b"))
                else:
                    high = _number(m.group("b"))
                    if high < low:
                        low, high = high, low
            doses.append(_dose(low, high, unit_text, m.group("d1"), sentence, m.end(), section, route,
                               heading_population))
    return doses


def _dose(low, high, unit_text, per, sentence, end, section, route, heading_population) -> Dose:
    after = sentence[end:end + 90]
    before = sentence[:end]
    return Dose(
        amount=low, amount_max=high, unit=_UNIT_NAMES[unit_text.lower()],
        per=({"ml": "mL"}.get(per.lower(), per.lower())) if per else None,
        frequency=_frequency(after) or _frequency(before[-60:]),
        route=_detect(_ROUTES, sentence, route),
        population=_detect(_POPULATIONS, sentence, heading_population or "unspecified"),
        section=section, sentence=sentence.strip())


@dataclass(frozen=True)
class Strength:
    ingredient: str
    amount: float
    unit: str
    per_amount: float | None
    per_unit: str | None
    text: str


def _amount_text(num: float, num_unit: str, den: float | None, den_unit: str | None) -> str:
    unit = _UNIT_NAMES.get(num_unit.lower(), num_unit)
    if unit == "%" or (den_unit and den == 100 and num_unit.lower() == den_unit.lower() and unit == "g"):
        return f"{_fmt(num)}%"
    return f"{_fmt(num)} {unit}"


def _fmt(value: float) -> str:
    return f"{value:g}"


def _strength(ingredient: ET.Element, form: str) -> Strength | None:
    quantity = _child(ingredient, "quantity")
    numerator = _child(quantity, "numerator") if quantity is not None else None
    if numerator is None or not numerator.get("value"):
        return None
    try:
        num = _number(numerator.get("value"))
    except ValueError:
        raise SplError(f"An ingredient strength isn't a number: {numerator.get('value')!r}.") from None
    if num <= 0:
        raise SplError(f"An ingredient strength must be positive, not {numerator.get('value')!r}.")
    num_unit = numerator.get("unit") or ""
    denominator = _child(quantity, "denominator")
    den = den_unit = None
    if denominator is not None and denominator.get("value"):
        try:
            den = _number(denominator.get("value"))
        except ValueError:
            raise SplError(f"An ingredient strength denominator isn't a number: {denominator.get('value')!r}.") from None
        den_unit = denominator.get("unit") or "1"
    substance = _child(ingredient, "ingredientSubstance")
    name_el = _child(substance, "name") if substance is not None else None
    name = _clean("".join(name_el.itertext())) if name_el is not None else ""
    head = _amount_text(num, num_unit, den, den_unit)
    if head.endswith("%"):
        text = head
    elif den_unit in (None, "1"):
        text = f"{head} per {form or 'unit'}" if den in (None, 1) else f"{head} per {_fmt(den)} {form or 'units'}"
    else:
        unit = _UNIT_NAMES.get(den_unit.lower(), den_unit)
        text = f"{head}/{unit}" if den == 1 else f"{head}/{_fmt(den)} {unit}"
    return Strength(name, num, _UNIT_NAMES.get(num_unit.lower(), num_unit), den if den_unit != "1" else None,
                    _UNIT_NAMES.get(den_unit.lower(), den_unit) if den_unit and den_unit != "1" else None, text)


# --- A label ------------------------------------------------------------------

@dataclass
class LabelSection:
    code: str
    heading: str            # "Dosage and Administration: Pediatric Dosage" for a sub-section
    paragraphs: list[str]
    top: str = ""           # the top-level section it belongs to

    def __post_init__(self):
        self.top = self.top or self.heading

    @property
    def sub(self) -> str:
        return self.heading[len(self.top) + 2:] if self.heading != self.top else ""


@dataclass
class Label:
    set_id: str
    title: str
    product: str
    generic: str
    organization: str
    effective: str
    routes: list[str]
    forms: list[str]
    strengths: list[Strength]
    sections: list[LabelSection]
    warnings: list[str] = field(default_factory=list)

    def section(self, name: str) -> list[LabelSection]:
        return [s for s in self.sections if s.heading == name or s.heading.startswith(name + ": ")]

    def doses(self) -> list[Dose]:
        route = self.routes[0].lower() if len(self.routes) == 1 else None
        out = []
        for s in self.section("Dosage and Administration"):
            for paragraph in s.paragraphs:
                out.extend(extract_doses(paragraph, section=s.heading, route=route))
        return out


def _date(value: str) -> str:
    if re.fullmatch(r"\d{8}", value or ""):
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value or ""


def _strip_numbering(title: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", title).strip()


def _heading(section: ET.Element, parent_code: str) -> tuple[str, str]:
    """The section's code and heading: the standard name for its LOINC code,
    unless it repeats its parent's code (a sub-section), when its own title says more."""
    code_el = _child(section, "code")
    code = code_el.get("code", "") if code_el is not None else ""
    title_el = _child(section, "title")
    title = _strip_numbering(_clean("".join(title_el.itertext()))) if title_el is not None else ""
    if title.isupper():
        title = title.title()
    standard = SECTIONS.get(code) if code != parent_code else None
    return code, standard or title or SECTIONS.get(code, "")


def _walk_sections(section: ET.Element, parent: str, out: list[LabelSection], parent_code: str = "",
                   top: str = "") -> None:
    code, name = _heading(section, parent_code)
    if code in IGNORED_CODES:
        return
    heading = f"{parent}: {name}" if parent and name else (name or parent or "Other")
    text_el = _child(section, "text")
    paragraphs = render_text(text_el) if text_el is not None else []
    top = top or heading
    if paragraphs:
        out.append(LabelSection(code, heading, paragraphs, top))
    for comp in _children(section, "component"):
        for sub in _children(comp, "section"):
            _walk_sections(sub, heading, out, code, top)


def parse_label(data: bytes) -> Label:
    root = parse_xml(data)
    if _local(root.tag) != "document":
        raise SplError("This isn't an SPL label (the root element isn't <document>).")
    title_el = _child(root, "title")
    title = _clean("".join(title_el.itertext())) if title_el is not None else ""
    set_el = _child(root, "setId")
    set_id = (set_el.get("root", "") if set_el is not None else "").lower()
    eff = _child(root, "effectiveTime")
    organization = ""
    for rep in _find_all(root, "representedOrganization"):
        name_el = _child(rep, "name")
        if name_el is not None:
            organization = _clean("".join(name_el.itertext()))
            break

    product = generic = ""
    forms: list[str] = []
    strengths: list[Strength] = []
    for mp in _find_all(root, "manufacturedProduct"):
        name_el = _child(mp, "name")
        if name_el is not None and not product:
            product = _clean("".join(name_el.itertext()))
        form_el = _child(mp, "formCode")
        form = (form_el.get("displayName") or "").lower().split(",")[0].strip() if form_el is not None else ""
        if form and form not in forms:
            forms.append(form)
        for gm in _find_all(mp, "genericMedicine"):
            g = _child(gm, "name")
            if g is not None and not generic:
                generic = _clean("".join(g.itertext()))
        for ing in _children(mp, "ingredient"):
            if ing.get("classCode", "ACTIB") in ("IACT",):
                continue
            s = _strength(ing, form)
            if s and s not in strengths:
                strengths.append(s)
    routes = []
    for rc in _find_all(root, "routeCode"):
        r = _clean(rc.get("displayName") or "").lower()
        if r and r not in routes:
            routes.append(r)

    sections: list[LabelSection] = []
    body = _child(root, "component")
    body = _child(body, "structuredBody") if body is not None else None
    if body is None:
        raise SplError("The label has no structured body.")
    for comp in _children(body, "component"):
        for sec in _children(comp, "section"):
            _walk_sections(sec, "", sections)

    if not product:
        product = re.split(r"[\(,]", title)[0].strip().title() if title else ""
    if not (title or product):
        raise SplError("The label has no title or product name.")
    label = Label(set_id=set_id, title=title or product, product=product, generic=generic,
                  organization=organization, effective=_date(eff.get("value", "") if eff is not None else ""),
                  routes=routes, forms=forms, strengths=strengths, sections=sections)
    if not sections:
        raise SplError("The label has no readable sections.")
    for needed in ("Indications and Usage", "Dosage and Administration", "Contraindications"):
        if not label.section(needed):
            label.warnings.append(f"{product}: no {needed} section")
    return label


# --- Pack output ----------------------------------------------------------------------

def product_information(label: Label) -> list[str]:
    parts = [f"{label.product} contains {label.generic}." if label.generic else f"{label.product}."]
    if label.strengths:
        parts.append("Strength: " + "; ".join(s.text for s in label.strengths) + ".")
    if label.forms:
        parts.append("Dosage form: " + ", ".join(label.forms) + ".")
    if label.routes:
        parts.append("Route of administration: " + ", ".join(label.routes) + ".")
    if label.organization:
        parts.append(f"Manufacturer: {label.organization.rstrip('.')}.")
    if label.effective:
        parts.append(f"Label effective {label.effective}.")
    return [" ".join(parts)]


def to_markdown(label: Label) -> str:
    """One document. Each top-level section is one heading; a sub-section's
    paragraphs follow its parent's, the first one led by the sub-section's name,
    so a whole section reads (and is retrieved) as one passage."""
    def safe(paragraph: str) -> str:
        return paragraph.lstrip("# ").strip()

    lines = [f"# {label.title}", "", "## Product Information", ""]
    lines += [safe(p) + "\n" for p in product_information(label)]
    current = None
    for s in label.sections:
        if s.top != current:
            lines += [f"## {s.top}", ""]
            current = s.top
        paragraphs = [safe(p) for p in s.paragraphs if safe(p)]
        if s.sub and paragraphs:
            paragraphs[0] = f"{s.sub}: {paragraphs[0]}"
        lines += [p + "\n" for p in paragraphs]
    return "\n".join(lines).rstrip() + "\n"


def _unstated_amount(label: Label, unit: str) -> float:
    text = " ".join(" ".join(s.paragraphs) for s in label.sections) + " " + " ".join(s.text for s in label.strengths)
    present = {float(n.replace(",", "")) for n in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)}
    candidate = 777.0
    while candidate in present:
        candidate += 1
    return candidate


def generate_eval(label: Label) -> list[dict]:
    """Questions each label must answer, and ones it must refuse."""
    name = label.product
    cases: list[dict] = []
    if label.section("Indications and Usage"):
        cases.append({"query": f"What is {name} indicated for?", "expect": "answer"})
    doses = [d for d in label.doses() if d.kind == "dose" and d.population in ("adult", "unspecified")]
    if doses:
        cases.append({"query": f"What is the usual dose of {name}?", "expect": "answer"})
        d = doses[0]
        cases.append({"query": f"Can {name} be given at {_fmt(_unstated_amount(label, d.unit))} {d.unit}?",
                      "expect": "refuse"})
    if label.section("Contraindications"):
        cases.append({"query": f"What are the contraindications of {name}?", "expect": "answer"})
    cases.append({"query": f"What is the usual dose of Quelvarin?", "expect": "refuse"})
    return cases


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
                  ).strip("-")[:60] or "label"


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.rglob("*.xml")) if p.is_dir() else [p])
    return files


def build(files: list[Path], out: Path, meta: dict, private_key: bytes | None = None,
          skip_errors: bool = False) -> dict:
    from app import knowledge, packs

    labels: list[tuple[Path, Label]] = []
    problems: list[str] = []
    for path in files:
        try:
            labels.append((path, parse_label(path.read_bytes())))
        except (SplError, OSError) as exc:
            problems.append(f"{path.name}: {exc}")
    if problems and not skip_errors:
        raise SplError("; ".join(problems))
    if not labels:
        raise SplError("No labels could be read.")
    titles: dict[str, str] = {}
    docs: dict[str, bytes] = {}
    eval_cases: list[dict] = []
    warnings: list[str] = []
    for path, label in labels:
        key = knowledge.document_key(label.title)
        if key in titles:
            raise SplError(f"{path.name}: the title {label.title!r} is already used by another label.")
        titles[key] = label.title
        name = f"documents/{slug(label.product)}.md"
        n = 2
        while name in docs:
            name = f"documents/{slug(label.product)}-{n}.md"
            n += 1
        docs[name] = to_markdown(label).encode("utf-8")
        eval_cases.extend(generate_eval(label))
        warnings.extend(label.warnings)
    files_out = {**docs, "eval.json": json.dumps(eval_cases, indent=1).encode("utf-8")}
    data = packs.assemble(meta, files_out, None, private_key)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {"path": str(out), "labels": len(labels), "eval_cases": len(eval_cases),
            "warnings": warnings + problems, "signed": private_key is not None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a knowledge pack from SPL drug labels (no downloads).")
    ap.add_argument("inputs", nargs="+", type=Path, help="SPL XML files, or folders of them")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--publisher", required=True)
    ap.add_argument("--licence", required=True)
    ap.add_argument("--jurisdiction", default="US")
    ap.add_argument("--description", default="")
    ap.add_argument("--sign-key", type=Path)
    ap.add_argument("--skip-errors", action="store_true", help="Build from the labels that can be read")
    args = ap.parse_args(argv)

    from app import packs

    try:
        key = packs.decode_key(args.sign_key.read_text(encoding="utf-8")) if args.sign_key else None
        meta = {"name": args.name, "version": args.version, "publisher": args.publisher, "licence": args.licence,
                "jurisdiction": args.jurisdiction, "description": args.description}
        result = build(collect(args.inputs), args.out, meta, key, args.skip_errors)
    except (SplError, packs.PackError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {result['path']}: {result['labels']} labels, {result['eval_cases']} evaluation questions, "
          f"{'signed' if result['signed'] else 'unsigned'}.")
    for w in result["warnings"]:
        print(f"  warning: {w}")
    print("Confirm the licence and terms of the source labels before distributing this pack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
