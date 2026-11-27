"""CDS Hooks cards in the caller's language: chosen by the hook's own language, else Accept-Language, else
English; every card stays valid CDS Hooks (summary within 140 characters) and every number, unit and
medicine name is copied exactly."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_cds_conformance import SERVICE, call, check_card, draft, ehr, request  # noqa: E402,F401

NUMBER = re.compile(r"\d+(?:\.\d+)?")
ENGLISH = re.compile(r"\b(?:The patient|is contraindicated|should be avoided|formulary gives|is not in the formulary|Not checked|"
                     r"Partly checked|was not checked|not a statement)\b")


def _cards(ehr, body, **kw):
    response = call(ehr, body, **kw)
    assert response.status_code == 200, response.text
    return response.json()["cards"]


def _unsafe(**over):
    return request(egfr=25, allergies=["Quorl"], orders=[draft(dose=900)], **over)


def test_english_is_the_default_and_unchanged(ehr):
    cards = _cards(ehr, _unsafe())
    assert cards and not any(re.search(r"\b(?:Evitar|paciente|TFGe|orden)\b", c["summary"] + c["detail"]) for c in cards)
    assert all(c["source"]["label"].startswith("GroundCheckHealth formulary") for c in cards)


@pytest.mark.parametrize("how", ["accept", "context", "top"])
def test_spanish_cards_are_chosen_by_accept_language_or_the_hooks_own_language(ehr, how):
    english = _cards(ehr, _unsafe())
    body = _unsafe()
    headers = {}
    if how == "accept":
        headers = {"Accept-Language": "es-ES,es;q=0.9,en;q=0.5"}
    elif how == "context":
        body["context"]["language"] = "es"
    else:
        body["locale"] = "es-MX"
    cards = _cards(ehr, body, headers=headers)
    assert len(cards) == len(english)
    for es, en in zip(cards, english):
        check_card(es)
        assert len(es["summary"]) <= 140
        assert es["source"]["label"].startswith("Formulario de GroundCheckHealth")
        assert es["source"]["topic"] == en["source"]["topic"] and es["indicator"] == en["indicator"]
        assert NUMBER.findall(es["summary"] + es["detail"]) == NUMBER.findall(en["summary"] + en["detail"]), (en["detail"], es["detail"])
        assert "Caloradine" in es["detail"]
    assert any("TFGe" in c["detail"] for c in cards)
    blocking = [c for c in cards if c["indicator"] == "critical"]
    assert blocking and all(o["display"] in ("Clínicamente adecuado para este paciente", "Los datos del paciente están desactualizados")
                            for c in blocking for o in c["overrideReasons"])


def test_a_dose_amount_and_unit_survive_in_spanish(ehr):
    cards = _cards(ehr, _unsafe(), headers={"Accept-Language": "es"})
    text = " ".join(c["detail"] for c in cards)
    assert "900 mg" in text and "15 mg una vez al día" in text
    assert not ENGLISH.search(text)


def test_the_hooks_own_language_beats_accept_language(ehr):
    body = _unsafe()
    body["context"]["language"] = "en"
    cards = _cards(ehr, body, headers={"Accept-Language": "es"})
    assert all(c["source"]["label"].startswith("GroundCheckHealth formulary") for c in cards)


def test_an_unsupported_language_gets_english(ehr):
    cards = _cards(ehr, _unsafe(), headers={"Accept-Language": "fr-FR,de;q=0.8"})
    assert all(c["source"]["label"].startswith("GroundCheckHealth formulary") for c in cards)


def test_the_not_in_formulary_card_and_the_missing_data_card_are_spanish(ehr):
    cards = _cards(ehr, request(orders=[draft(medicine="Unlistedine")], observations=None), headers={"Accept-Language": "es"})
    summaries = [c["summary"] for c in cards]
    assert "No comprobado: Unlistedine no está en el formulario" in summaries
    detail = next(c["detail"] for c in cards if c["source"]["topic"]["code"] == "not_in_formulary")
    assert "**Unlistedine** no se comprobó" in detail and "Esto no significa que la orden sea segura" in detail
    for c in cards:
        check_card(c)


def test_a_very_long_spanish_finding_is_cut_to_the_limit(ehr):
    cards = _cards(ehr, _unsafe(), headers={"Accept-Language": "es"})
    assert all(len(c["summary"]) <= 140 for c in cards)
    from app import messages

    long = "x" * 300
    assert len(messages.message("msg.cds.not_checked.summary", "es", name=long)) > 140  # why the card cuts it
