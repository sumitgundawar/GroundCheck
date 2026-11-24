"""The SPL drug label importer: parsing, dose/route/strength extraction,
hostile XML, pack output, and that an imported label is answered, cited and
guarded like any other approved document. No test touches the network."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import import_spl as spl  # noqa: E402

from app import config, db, knowledge, packs, pipeline, retrieval  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "spl"


def label(name: str) -> spl.Label:
    return spl.parse_label((FIXTURES / f"{name}.xml").read_bytes())


def doc(body: str, title="<title>TESTDRUG tablet</title>", head: str = "") -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?>{head}<document xmlns="urn:hl7-org:v3">{title}'
            f'<effectiveTime value="20240101"/><component><structuredBody>{body}</structuredBody></component>'
            f'</document>').encode()


def section(code: str, title: str, text: str) -> str:
    return f'<component><section><code code="{code}"/><title>{title}</title><text>{text}</text></section></component>'


def one(text: str, **kw) -> list[spl.Dose]:
    return spl.extract_doses(text, **kw)


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Parsing and building must never touch the network. The end-to-end tests
    load the embedding model, which is a different matter, so they're exempt."""
    if "labels_live" in request.fixturenames:
        return

    def refuse(*a, **k):
        raise AssertionError("the importer tried to use the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


# --- Fixture labels --------------------------------------------------------------------

def test_zentrilex_metadata_and_headings():
    z = label("zentrilex")
    assert z.product == "Zentrilex" and z.generic == "zentrilex hydrochloride"
    assert z.title.startswith("ZENTRILEX") and z.set_id == "5b1f6f0e-0000-4000-8000-0000000000aa"
    assert z.effective == "2024-01-15" and z.organization == "Fictional Pharma Inc."
    assert z.routes == ["oral"] and z.forms == ["tablet"]
    assert [s.heading for s in z.sections] == [
        "Indications and Usage", "Dosage and Administration", "Dosage and Administration: Adult Dosage",
        "Dosage and Administration: Pediatric Dosage", "Contraindications", "Drug Interactions",
        "Use in Specific Populations: Pediatric Use"]
    assert z.warnings == []


def test_zentrilex_strength_ignores_inactive_ingredients():
    z = label("zentrilex")
    assert [s.text for s in z.strengths] == ["10 mg per tablet"]
    assert (z.strengths[0].ingredient, z.strengths[0].amount, z.strengths[0].unit) == (
        "ZENTRILEX HYDROCHLORIDE", 10.0, "mg")


def test_zentrilex_text_entities_lists_and_tables():
    z = label("zentrilex")
    contra = z.section("Contraindications")[0].paragraphs[0]
    assert "zentrilex hydrochloride & related quorl modulators" in contra
    paediatric = z.section("Dosage and Administration: Pediatric Dosage")[0].paragraphs
    assert paediatric == ["For pediatric patients 6 to 11 years, give 5 mg once daily.",
                          "Patients 12 years and older: 10 mg once daily", "Do not exceed 0.5 mg/kg in a single dose"]
    assert z.section("Drug Interactions")[0].paragraphs == ["Drug | Effect", "Orrin-blockers | Do not combine"]


def test_zentrilex_doses_are_split_by_population():
    doses = label("zentrilex").doses()
    summary = [(d.amount, d.unit, d.per, d.frequency, d.population, d.route) for d in doses]
    assert (10.0, "mg", None, "once daily", "adult", "oral") in summary
    assert (20.0, "mg", "day", None, "adult", "oral") in summary
    assert (5.0, "mg", None, "once daily", "pediatric", "oral") in summary
    assert (0.5, "mg", "kg", None, "pediatric", "oral") in summary
    assert {d.kind for d in doses} == {"dose", "daily_total", "per_kg"}


def test_orvalix_solution_ranges_ml_and_concentration():
    o = label("orvalix")
    assert [s.text for s in o.strengths] == ["5 mg/mL"] and o.routes == ["oral"] and o.forms == ["solution"]
    summary = [(d.amount, d.amount_max, d.unit, d.per, d.frequency, d.population) for d in o.doses()]
    assert (5.0, 10.0, "mL", None, "every 6 hours", "adult") in summary
    assert (25.0, 50.0, "mg", None, "every 6 hours", "adult") in summary
    assert (2.5, None, "mL", None, "every 6 hours", "pediatric") in summary
    assert (12.5, None, "mg", None, "every 6 hours", "pediatric") in summary
    assert (0.25, None, "mg", "kg", None, "pediatric") in summary
    assert any(d.is_range for d in o.doses())


def test_dermaquil_percent_strength_and_dose():
    d = label("dermaquil")
    assert [s.text for s in d.strengths] == ["1%"] and d.routes == ["topical"]
    summary = [(x.amount, x.unit, x.frequency, x.route) for x in d.doses()]
    assert summary == [(1.0, "%", "twice daily", "topical"), (0.5, "%", None, "topical")]
    assert d.warnings == ["Dermaquil: no Contraindications section"]


def test_thyronex_micrograms_ug_and_thousands():
    t = label("thyronex")
    assert [s.text for s in t.strengths] == ["25 mcg per tablet", "1000 mcg per tablet"]
    assert all(s.unit == "mcg" for s in t.strengths)
    summary = [(d.amount, d.amount_max, d.unit, d.frequency, d.population) for d in t.doses()]
    assert summary == [(25.0, None, "mcg", "once daily", "adult"), (25.0, 50.0, "mcg", None, "unspecified")]
    assert t.warnings == ["Thyronex: no Indications and Usage section", "Thyronex: no Contraindications section"]


def test_a_label_without_the_dose_section_has_no_doses_and_a_warning():
    xml = doc(section("34067-9", "INDICATIONS", "<paragraph>Used for tests.</paragraph>"))
    lab = spl.parse_label(xml)
    assert lab.doses() == [] and "no Dosage and Administration section" in " ".join(lab.warnings)
    assert lab.product == "Testdrug Tablet"


# --- Dose extraction table --------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("5 mg", [(5, None, "mg", None, None)]),
    ("5mg", [(5, None, "mg", None, None)]),
    ("0.5 mg", [(0.5, None, "mg", None, None)]),
    ("1,000 mg", [(1000, None, "mg", None, None)]),
    ("5-10 mg", [(5, 10, "mg", None, None)]),
    ("5 – 10 mg", [(5, 10, "mg", None, None)]),
    ("5 to 10 mg", [(5, 10, "mg", None, None)]),
    ("5 mg to 10 mg", [(5, 10, "mg", None, None)]),
    ("5 mg-10 mg", [(5, 10, "mg", None, None)]),
    ("5–10mg", [(5, 10, "mg", None, None)]),
    ("10 to 5 mg", [(5, 10, "mg", None, None)]),
    ("5 mg to 1 g", [(5, None, "mg", None, None), (1, None, "g", None, None)]),
    ("2 mL", [(2, None, "mL", None, None)]),
    ("2 ML", [(2, None, "mL", None, None)]),
    ("100 mcg", [(100, None, "mcg", None, None)]),
    ("100 µg", [(100, None, "mcg", None, None)]),
    ("100 μg", [(100, None, "mcg", None, None)]),
    ("100 ug", [(100, None, "mcg", None, None)]),
    ("100 micrograms", [(100, None, "mcg", None, None)]),
    ("2.5%", [(2.5, None, "%", None, None)]),
    ("10 units", [(10, None, "units", None, None)]),
    ("10 milligrams", [(10, None, "mg", None, None)]),
    ("up to 3 g", [(3, None, "g", None, None)]),
    ("5 mg/kg", [(5, None, "mg", "kg", None)]),
    ("5 mg per kg", [(5, None, "mg", "kg", None)]),
    ("5 mg/kg/day", [(5, None, "mg", "kg", None)]),
    ("1.5 mg/m2", [(1.5, None, "mg", "m2", None)]),
    ("5 mg/mL", [(5, None, "mg", "mL", None)]),
    ("50 mg/5 mL", [(50, None, "mg", "mL", None)]),
    ("5 mg/day", [(5, None, "mg", "day", None)]),
    ("5 mg every 8 hours", [(5, None, "mg", None, "every 8 hours")]),
    ("5 mg every 4 to 6 hours", [(5, None, "mg", None, "every 4 to 6 hours")]),
    ("5 mg q12h", [(5, None, "mg", None, "every 12 hours")]),
    ("5 mg BID", [(5, None, "mg", None, "twice daily")]),
    ("5 mg twice a day", [(5, None, "mg", None, "twice daily")]),
    ("5 mg three times daily", [(5, None, "mg", None, "three times daily")]),
    ("5 mg four times a day", [(5, None, "mg", None, "four times daily")]),
    ("5 mg once daily", [(5, None, "mg", None, "once daily")]),
    ("5 mg once a day", [(5, None, "mg", None, "once daily")]),
    ("5 mg daily", [(5, None, "mg", None, "daily")]),
    ("5 mg at bedtime", [(5, None, "mg", None, "at bedtime")]),
    ("5 mg once weekly", [(5, None, "mg", None, "once weekly")]),
    ("5 mg every other day", [(5, None, "mg", None, "every other day")]),
    ("5 mg/kg every 8 hours", [(5, None, "mg", "kg", "every 8 hours")]),
    ("(5 mg)", [(5, None, "mg", None, None)]),
])
def test_dose_extraction(text, expected):
    got = [(d.amount, d.amount_max, d.unit, d.per, d.frequency) for d in one(text)]
    assert got == expected


@pytest.mark.parametrize("text", ["6 weeks", "12 years", "Section 5.2", "size 5 gauge", "10 gout", "2 to 3",
                                  "mg", "twice daily", "", "version 2.1 of the label", "5 mgs"])
def test_text_without_a_dose_gives_no_doses(text):
    assert one(text) == []


def test_a_dose_is_not_confused_by_a_number_inside_another_token():
    assert one("lot ABC1.5 mg") == []
    assert [d.amount for d in one("Give 5 mg (not 0.5 mg).")] == [5, 0.5]


@pytest.mark.parametrize("text,route", [
    ("Give 5 mg orally.", "oral"), ("Give 5 mg by mouth.", "oral"), ("5 mg intravenously over 30 minutes", "intravenous"),
    ("5 mg IV", "intravenous"), ("5 mg intramuscularly", "intramuscular"), ("5 mg subcutaneously", "subcutaneous"),
    ("Apply 1% to skin", "topical"), ("5 mg rectally", "rectal"), ("5 mg sublingually", "sublingual"),
    ("1 drop of 1% in the eye", "ophthalmic"), ("2 inhalations of 90 mcg", "inhalation"), ("5 mg", None),
])
def test_route_detection(text, route):
    assert one(text)[0].route == route


def test_route_falls_back_to_the_label_route_when_the_text_names_none():
    assert one("Give 5 mg.", route="oral")[0].route == "oral"
    assert one("Give 5 mg intravenously.", route="oral")[0].route == "intravenous"


@pytest.mark.parametrize("text,section,population", [
    ("Give 5 mg.", "", "unspecified"), ("Children 2 to 5 years: 5 mg.", "", "pediatric"),
    ("Neonates: 1 mg.", "", "pediatric"), ("Infants receive 1 mg.", "", "pediatric"),
    ("Adolescents: 5 mg.", "", "pediatric"), ("Elderly patients: 2 mg.", "", "geriatric"),
    ("Adults: 10 mg.", "", "adult"), ("Give 5 mg.", "Dosage and Administration: Pediatric Dosage", "pediatric"),
    ("Adults: 10 mg.", "Dosage and Administration: Pediatric Dosage", "adult"),
    ("Give 5 mg.", "Dosage and Administration: Geriatric Use", "geriatric"),
])
def test_population_detection(text, section, population):
    assert one(text, section=section)[0].population == population


def test_each_sentence_gets_its_own_population_and_frequency():
    doses = one("Adults: 10 mg once daily. Children: 5 mg twice daily.")
    assert [(d.amount, d.population, d.frequency) for d in doses] == [
        (10, "adult", "once daily"), (5, "pediatric", "twice daily")]


# --- Missing and malformed content ----------------------------------------------------------

def test_label_with_no_sections_is_an_error():
    with pytest.raises(spl.SplError, match="no readable sections"):
        spl.parse_label(doc(""))


def test_empty_sections_are_dropped():
    xml = doc(section("34067-9", "X", "") + section("34068-7", "D", "<paragraph>Give 5 mg.</paragraph>"))
    assert [s.heading for s in spl.parse_label(xml).sections] == ["Dosage and Administration"]


def test_unknown_sections_use_their_own_title_and_untitled_ones_are_called_other():
    xml = doc(section("99999-9", "8.9 SPECIAL NOTES", "<paragraph>Note text.</paragraph>")
              + section("99999-8", "", "<paragraph>Anonymous.</paragraph>"))
    assert [s.heading for s in spl.parse_label(xml).sections] == ["Special Notes", "Other"]


def test_ignored_sections_do_not_appear():
    xml = doc(section("51945-4", "PACKAGE LABEL", "<paragraph>Principal display panel.</paragraph>")
              + section("34067-9", "I", "<paragraph>Used for tests.</paragraph>"))
    assert [s.heading for s in spl.parse_label(xml).sections] == ["Indications and Usage"]


def test_media_and_footnotes_are_skipped_but_inline_markup_is_kept():
    text = ('<paragraph>Dose is <content styleCode="bold">5 mg</content> <sub>x</sub>daily'
            '<renderMultiMedia referencedObject="m1"><caption>FIGURE</caption></renderMultiMedia> now.</paragraph>')
    xml = doc(section("34068-7", "D", text))
    assert spl.parse_label(xml).sections[0].paragraphs == ["Dose is 5 mg xdaily now."]


def test_text_without_paragraph_elements_is_kept():
    xml = doc(section("34068-7", "D", 'Take <content>5 mg</content> daily.<br/>Second line.'))
    assert spl.parse_label(xml).sections[0].paragraphs == ["Take 5 mg daily.", "Second line."]


@pytest.mark.parametrize("data,message", [
    (b"", "empty"), (b"   \n ", "empty"), (b"not xml", "well-formed"), (b"<document", "well-formed"),
    (b"<a><b></a>", "well-formed"), (b"<?xml version='1.0'?>", "well-formed"),
    (b"<html><body/></html>", "isn't an SPL label"), (b"<document xmlns='urn:hl7-org:v3'><title>T</title></document>",
                                                     "no structured body"),
    (b"\xff\xfe\x00\x00garbage", "well-formed"),
])
def test_malformed_input_raises_a_clear_error(data, message):
    with pytest.raises(spl.SplError, match=message):
        spl.parse_label(data)


def test_truncated_real_label_is_an_error_not_a_partial_label():
    data = (FIXTURES / "zentrilex.xml").read_bytes()
    for cut in (50, len(data) // 3, len(data) // 2, len(data) - 20):
        with pytest.raises(spl.SplError):
            spl.parse_label(data[:cut])


def test_no_title_and_no_product_is_an_error():
    xml = (b'<document xmlns="urn:hl7-org:v3"><component><structuredBody><component><section><code code="34067-9"/>'
           b'<text><paragraph>x</paragraph></text></section></component></structuredBody></component></document>')
    with pytest.raises(spl.SplError, match="no title"):
        spl.parse_label(xml)


@pytest.mark.parametrize("value", ["abc", "-5", "0"])
def test_bad_ingredient_strengths_are_errors(value):
    product = (f'<component><section><code code="48780-1"/><subject><manufacturedProduct><manufacturedProduct>'
               f'<name>X</name><ingredient classCode="ACTIB"><quantity><numerator value="{value}" unit="mg"/>'
               f'<denominator value="1" unit="1"/></quantity><ingredientSubstance><name>X</name></ingredientSubstance>'
               f'</ingredient></manufacturedProduct></manufacturedProduct></subject></section></component>')
    xml = doc(product + section("34067-9", "I", "<paragraph>Used.</paragraph>"))
    with pytest.raises(spl.SplError, match="strength"):
        spl.parse_label(xml)


def test_strength_without_a_quantity_is_skipped_not_invented():
    product = ('<component><section><code code="48780-1"/><subject><manufacturedProduct><manufacturedProduct>'
               '<name>X</name><ingredient classCode="ACTIB"><ingredientSubstance><name>X</name></ingredientSubstance>'
               '</ingredient></manufacturedProduct></manufacturedProduct></subject></section></component>')
    lab = spl.parse_label(doc(product + section("34067-9", "I", "<paragraph>Used.</paragraph>")))
    assert lab.strengths == []
    assert "Strength" not in spl.to_markdown(lab)


def test_unicode_and_bom_are_handled():
    xml = b"\xef\xbb\xbf" + doc(section("34067-9", "I", "<paragraph>Für Patienten: Dosis 5 mg, 日本語 ✓.</paragraph>"),
                                 title="<title>ZÉPHYROL tablet</title>")
    lab = spl.parse_label(xml)
    assert lab.title == "ZÉPHYROL tablet" and "日本語 ✓" in lab.sections[0].paragraphs[0]


def test_utf16_documents_are_read():
    xml = doc(section("34067-9", "I", "<paragraph>Used for tests.</paragraph>")).decode().replace("UTF-8", "UTF-16")
    assert spl.parse_label(b"\xff\xfe" + xml.encode("utf-16-le")).sections[0].paragraphs == ["Used for tests."]


def test_very_deep_nesting_is_refused():
    deep = "<component><section>" * 100 + "</section></component>" * 100
    with pytest.raises(spl.SplError, match="nested too deeply"):
        spl.parse_label(doc(deep))


def test_oversized_files_are_refused(monkeypatch):
    monkeypatch.setattr(spl, "MAX_BYTES", 100)
    with pytest.raises(spl.SplError, match="larger"):
        spl.parse_label((FIXTURES / "zentrilex.xml").read_bytes())


def test_a_very_large_section_is_parsed():
    paragraphs = "".join(f"<paragraph>Paragraph {i} gives {i + 1} mg daily.</paragraph>" for i in range(5000))
    lab = spl.parse_label(doc(section("34068-7", "D", paragraphs)))
    assert len(lab.sections[0].paragraphs) == 5000 and len(lab.doses()) == 5000


# --- Entities and hostile XML ----------------------------------------------------------------

def test_standard_and_numeric_entities_are_decoded():
    text = "<paragraph>A &amp; B &lt; C &gt; D &quot;E&quot; &apos;F&apos; &#176;C &#x2265; 5 mg &#8211; 10 mg</paragraph>"
    para = spl.parse_label(doc(section("34068-7", "D", text))).sections[0].paragraphs[0]
    assert para == "A & B < C > D \"E\" 'F' °C ≥ 5 mg – 10 mg"


def test_escaped_markup_stays_text_and_double_escaping_is_preserved():
    text = "<paragraph>&lt;b&gt;bold&lt;/b&gt; and &amp;lt; literal</paragraph>"
    assert spl.parse_label(doc(section("34068-7", "D", text))).sections[0].paragraphs == [
        "<b>bold</b> and &lt; literal"]


def test_html_named_entities_are_not_defined_in_xml_and_give_a_clean_error():
    with pytest.raises(spl.SplError, match="well-formed"):
        spl.parse_label(doc(section("34068-7", "D", "<paragraph>5&nbsp;mg &copy; label</paragraph>")))


XXE_FILE = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            b'<document xmlns="urn:hl7-org:v3"><title>&x;</title></document>')
XXE_HTTP = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "http://127.0.0.1:9/evil">]>'
            b'<document xmlns="urn:hl7-org:v3"><title>&x;</title></document>')
XXE_PARAMETER = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY % p SYSTEM "http://127.0.0.1:9/x.dtd"> %p;]>'
                 b'<document xmlns="urn:hl7-org:v3"/>')
EXTERNAL_DTD = (b'<?xml version="1.0"?><!DOCTYPE document SYSTEM "http://127.0.0.1:9/spl.dtd">'
                b'<document xmlns="urn:hl7-org:v3"/>')
INTERNAL_ONLY = b'<?xml version="1.0"?><!DOCTYPE document [<!ELEMENT document ANY>]><document xmlns="urn:hl7-org:v3"/>'
LOLZ = (b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
        b'<!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">]>'
        b'<document xmlns="urn:hl7-org:v3"><title>&lol4;</title></document>')
ATTRIBUTE_ENTITY = (b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "x">]><document xmlns="urn:hl7-org:v3" id="&a;"/>')


@pytest.mark.parametrize("payload", [XXE_FILE, XXE_HTTP, XXE_PARAMETER, EXTERNAL_DTD, INTERNAL_ONLY, LOLZ,
                                     ATTRIBUTE_ENTITY], ids=["file", "http", "parameter", "dtd", "internal", "lolz",
                                                              "attribute"])
def test_xxe_and_entity_attacks_are_refused(payload):
    start = time.time()
    with pytest.raises(spl.SplError, match="unsafe") as info:
        spl.parse_label(payload)
    assert time.time() - start < 2
    assert "root:" not in str(info.value)


def test_a_secret_file_is_never_read_into_the_label(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-VALUE")
    payload = (f'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file://{secret}">]>'
               f'<document xmlns="urn:hl7-org:v3"><title>&x;</title></document>').encode()
    with pytest.raises(spl.SplError) as info:
        spl.parse_label(payload)
    assert "TOP-SECRET-VALUE" not in str(info.value)


def test_doctype_is_refused_after_a_label_prefix_too():
    payload = (b'<?xml version="1.0"?><!-- c --><!DOCTYPE document [<!ENTITY x "y">]><document xmlns="urn:hl7-org:v3"/>')
    with pytest.raises(spl.SplError, match="unsafe"):
        spl.parse_label(payload)


def test_xinclude_and_processing_instructions_are_inert(tmp_path):
    secret = tmp_path / "s.txt"
    secret.write_text("TOP-SECRET-VALUE")
    text = (f'<paragraph>Before.<xi:include xmlns:xi="http://www.w3.org/2001/XInclude" href="file://{secret}" '
            f'parse="text"/><?php echo 1;?> After 5 mg.</paragraph>')
    paragraphs = spl.parse_label(doc(section("34068-7", "D", text))).sections[0].paragraphs
    assert "TOP-SECRET-VALUE" not in " ".join(paragraphs) and "5 mg" in paragraphs[0]


def test_the_standard_library_path_is_used_without_defusedxml(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    with pytest.raises(spl.SplError, match="unsafe"):
        spl.parse_label(XXE_FILE)
    assert spl.parse_label((FIXTURES / "orvalix.xml").read_bytes()).product == "Orvalix"


@pytest.mark.skipif(importlib.util.find_spec("defusedxml") is None, reason="defusedxml isn't installed")
def test_the_defusedxml_path_refuses_the_same_attacks():
    for payload in (XXE_FILE, XXE_HTTP, EXTERNAL_DTD, LOLZ):
        with pytest.raises(spl.SplError):
            spl._parse_defused(payload)
    assert spl._parse_defused((FIXTURES / "orvalix.xml").read_bytes()) is not None


def test_both_parsers_agree_on_the_fixtures():
    for f in sorted(FIXTURES.glob("*.xml")):
        root = spl._parse_stdlib(f.read_bytes())
        import xml.etree.ElementTree as ET
        reference = ET.fromstring(f.read_bytes())
        assert [(spl._local(e.tag), (e.text or "").strip(), (e.tail or "").strip()) for e in root.iter()] == [
            (spl._local(e.tag), (e.text or "").strip(), (e.tail or "").strip()) for e in reference.iter()]


# --- Markdown and pack output ------------------------------------------------------------------

def test_markdown_has_the_standard_headings_and_product_information():
    md = spl.to_markdown(label("zentrilex"))
    lines = md.splitlines()
    assert lines[0] == "# ZENTRILEX (zentrilex hydrochloride) tablet, film coated"
    assert [l for l in lines if l.startswith("## ")] == [
        "## Product Information", "## Indications and Usage", "## Dosage and Administration",
        "## Contraindications", "## Drug Interactions", "## Use in Specific Populations"]
    assert "Adult Dosage: The recommended adult dose is 10 mg once daily." in md
    assert "Pediatric Dosage: For pediatric patients 6 to 11 years, give 5 mg once daily." in md
    assert "Pediatric Use: Safety and effectiveness have not been established" in md
    assert "Strength: 10 mg per tablet." in md and "Route of administration: oral." in md
    assert "Manufacturer: Fictional Pharma Inc. Label" in md


def test_a_paragraph_cannot_turn_into_a_heading_in_markdown():
    xml = doc(section("34068-7", "D", "<paragraph>## Injected heading with 5 mg</paragraph>"))
    md = spl.to_markdown(spl.parse_label(xml))
    assert "## Injected" not in md and "Injected heading with 5 mg" in md
    from app import documents
    assert [s.heading for s in documents.parse("x.md", md.encode()).sections] == ["Product Information",
                                                                                 "Dosage and Administration"]


def test_generated_eval_covers_answerable_wrong_dose_and_unknown_drug():
    cases = spl.generate_eval(label("zentrilex"))
    assert [c["expect"] for c in cases] == ["answer", "answer", "refuse", "answer", "refuse"]
    wrong = cases[2]["query"]
    assert wrong == "Can Zentrilex be given at 777 mg?"
    text = " ".join(" ".join(s.paragraphs) for s in label("zentrilex").sections)
    assert "777" not in text


def test_the_wrong_dose_is_never_a_number_the_label_states():
    z = label("zentrilex")
    z.sections[0].paragraphs.append("The odd value 777 and 778 appear here.")
    assert spl._unstated_amount(z, "mg") == 779


def build_fixture_pack(tmp_path, names=("zentrilex", "orvalix", "dermaquil", "thyronex"), key=None):
    files = [FIXTURES / f"{n}.xml" for n in names]
    meta = {"name": "us-labels-fixtures", "version": "1.0.0", "publisher": "Test Publisher", "licence": "Synthetic",
            "jurisdiction": "US", "synthetic": True}
    out = tmp_path / "labels.gcpack"
    return out, spl.build(files, out, meta, key)


def test_build_makes_an_inspectable_pack(tmp_path):
    out, result = build_fixture_pack(tmp_path)
    assert result["labels"] == 4 and result["eval_cases"] == 17
    info = packs.inspect_pack(out)
    assert [f["path"] for f in info["files"]] == [
        "documents/dermaquil.md", "documents/orvalix.md", "documents/thyronex.md", "documents/zentrilex.md", "eval.json"]
    assert info["verification"]["integrity"] and info["jurisdiction"] == "US"
    assert any("no Contraindications" in w for w in result["warnings"])


def test_build_is_reproducible(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _ = build_fixture_pack(tmp_path / "a")
    b, _ = build_fixture_pack(tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_a_bad_label_stops_the_build_unless_errors_are_skipped(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_bytes(XXE_FILE)
    meta = {"name": "x-pack", "version": "1.0.0", "publisher": "P", "licence": "L", "jurisdiction": "US"}
    with pytest.raises(spl.SplError, match="bad.xml"):
        spl.build([FIXTURES / "orvalix.xml", bad], tmp_path / "o.gcpack", meta)
    assert not (tmp_path / "o.gcpack").exists()
    result = spl.build([FIXTURES / "orvalix.xml", bad], tmp_path / "o.gcpack", meta, skip_errors=True)
    assert result["labels"] == 1 and any("bad.xml" in w for w in result["warnings"])
    with pytest.raises(spl.SplError, match="No labels"):
        spl.build([bad], tmp_path / "p.gcpack", meta, skip_errors=True)


def test_two_labels_with_the_same_title_are_refused(tmp_path):
    copy = tmp_path / "copy.xml"
    copy.write_bytes((FIXTURES / "orvalix.xml").read_bytes())
    meta = {"name": "x-pack", "version": "1.0.0", "publisher": "P", "licence": "L", "jurisdiction": "US"}
    with pytest.raises(spl.SplError, match="already used"):
        spl.build([FIXTURES / "orvalix.xml", copy], tmp_path / "o.gcpack", meta)


def test_command_line(tmp_path, capsys):
    out = tmp_path / "cli.gcpack"
    args = [str(FIXTURES), "--out", str(out), "--name", "us-labels", "--version", "1.2.3", "--publisher", "Pub",
            "--licence", "Synthetic"]
    assert spl.main(args) == 0
    text = capsys.readouterr().out
    assert "4 labels" in text and "Confirm the licence" in text
    assert packs.inspect_pack(out)["version"] == "1.2.3"
    assert spl.main([str(tmp_path / "missing.xml"), "--out", str(out), "--name", "us-labels", "--version", "1.0.0",
                     "--publisher", "P", "--licence", "L"]) == 1
    assert spl.main([str(FIXTURES / "orvalix.xml"), "--out", str(out), "--name", "Bad Name", "--version", "1.0.0",
                     "--publisher", "P", "--licence", "L"]) == 1
    assert capsys.readouterr().err.count("Error:") == 2


def test_cli_signs_the_pack(tmp_path):
    private, public = packs.generate_keypair()
    key = tmp_path / "k"
    key.write_text(private)
    out = tmp_path / "s.gcpack"
    assert spl.main([str(FIXTURES / "orvalix.xml"), "--out", str(out), "--name", "us-labels", "--version", "1.0.0",
                     "--publisher", "Pub", "--licence", "L", "--sign-key", str(key)]) == 0
    assert packs.inspect_pack(out, packs.trusted_keys(f"Pub={public}"))["verification"]["trusted"]


# --- End to end: an imported label flows through retrieval and the guards -----------------------

@pytest.fixture()
def labels_live(database, tmp_path, monkeypatch):
    monkeypatch.setenv("PACK_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "INCLUDE_DEMO_CORPUS", False)
    monkeypatch.setattr(config, "VECTOR_STORE", "local")
    out, _ = build_fixture_pack(tmp_path)
    result = packs.import_pack(out, keys={}, override=True, override_reason="synthetic fixture labels")
    assert all(d["status"] == "pending" for d in result["documents"])
    for d in result["documents"]:
        knowledge.review(d["id"], "approve", None, allow_self_approval=True)
    knowledge.rebuild_index()
    yield result
    monkeypatch.undo()
    db.reset_engine()
    retrieval.load_index()


def ask(q: str):
    return pipeline.run(q, review=False)


def cited_titles(response) -> set[str]:
    ids = {sid for c in response.claims for sid in c.source_ids}
    return {s.title.split(":")[0] for s in response.sources if s.id in ids}


def test_the_generated_eval_set_passes_end_to_end(labels_live):
    cases = packs.release_gate_cases()
    assert len(cases) >= 15
    result = packs.run_eval(cases)
    assert result["failures"] == [] and result["ok"] and result["unsafe_answers"] == 0


def test_the_usual_dose_is_answered_from_the_imported_label_and_cited(labels_live):
    r = ask("What is the usual dose of Zentrilex?")
    assert r.decision == "answer" and "10 mg" in r.answer_text
    assert any("ZENTRILEX" in t.upper() for t in cited_titles(r))
    assert all(c.grounded for c in r.claims)


def test_each_fixture_label_answers_about_its_own_drug(labels_live):
    for question, needle, title in [
        ("What is Orvalix indicated for?", "tremor", "ORVALIX"),
        ("What are the contraindications of Zentrilex?", "hypersensitivity", "ZENTRILEX"),
        ("What is the usual dose of Thyronex?", "25 mcg", "THYRONEX"),
    ]:
        r = ask(question)
        assert r.decision == "answer", question
        assert needle in r.answer_text and any(title in t.upper() for t in cited_titles(r)), question


def test_a_wrong_dose_is_still_refused(labels_live):
    for question in ["Can Zentrilex be given at 50 mg?", "Can Zentrilex be given at 777 mg?",
                     "Can Thyronex be given at 500 mcg?", "Can Orvalix be given at 100 mL?"]:
        r = ask(question)
        assert r.decision == "refuse", question
    assert ask("Can Zentrilex be given at 50 mg?").reason_code in (
        "stated_value_unsupported", "dose_unsupported", "claim_ungrounded", "no_relevant_source")


def test_a_drug_that_is_not_in_any_label_is_refused(labels_live):
    for question in ["What is the usual dose of Quelvarin?", "What is Quelvarin indicated for?"]:
        assert ask(question).decision == "refuse"


def test_a_stated_correct_dose_is_not_refused(labels_live):
    r = ask("Can Zentrilex be given at 10 mg once daily?")
    assert r.decision == "answer"
