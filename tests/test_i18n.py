"""The dashboard's strings: web/i18n/en.json is the source, es.json a locale
that may lack keys (they show English). These checks need no browser."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import i18n_extract  # noqa: E402

WEB = ROOT / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")
EN = json.loads((WEB / "i18n" / "en.json").read_text(encoding="utf-8"))
ES = json.loads((WEB / "i18n" / "es.json").read_text(encoding="utf-8"))
PLACEHOLDER = re.compile(r"\{\w+\}")


def html_keys() -> set[str]:
    keys = set(re.findall(r'data-i18n="([^"]+)"', INDEX))
    for pairs in re.findall(r'data-i18n-attr="([^"]+)"', INDEX):
        keys |= {pair.partition(":")[2] for pair in pairs.split(";")}
    return keys


def script_keys() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for m in re.finditer(r'\bt\("([\w.]+)"(?:,\s*"((?:[^"\\]|\\.)*)")?', APP_JS):
        if not m.group(1).endswith((".", "_")):  # "reason." + code, "js.role_" + role: checked per code
            found[m.group(1)] = m.group(2)
    return found


def test_the_html_and_the_english_file_are_in_step():
    rewritten, strings = i18n_extract.extract(INDEX)
    assert rewritten == INDEX, "visible text without a data-i18n key; run scripts/i18n_extract.py"
    missing = {k for k in strings if EN.get(k) != strings[k]}
    assert not missing, sorted(missing)[:10]


def test_no_visible_text_is_hard_coded_in_the_page():
    """Every text node with a letter sits inside an element that has a key,
    except code samples, scripts and the icon sprite."""
    root = i18n_extract.parse(INDEX)
    offenders = []
    for node in i18n_extract.walk(root):
        if node.tag == "#root" or node.opaque():
            continue
        for a, b in node.texts:
            if not (node.open_end <= a and b <= node.close_start):
                continue
            raw = INDEX[a:b]
            if i18n_extract.HAS_LETTER.search(raw) and "{" not in raw:
                if "data-i18n" not in node.attrs:
                    offenders.append((node.tag, node.attr("id"), raw.strip()[:60]))
    assert offenders == []


def test_visible_attributes_are_keyed_too():
    root = i18n_extract.parse(INDEX)
    unkeyed = []
    for node in i18n_extract.walk(root):
        if node.tag == "#root" or node.opaque():
            continue
        keyed = node.attr("data-i18n-attr") or ""
        for name in i18n_extract.ATTRS:
            value = node.attr(name)
            if value and i18n_extract.HAS_LETTER.search(value) and "{" not in value and f"{name}:" not in keyed:
                unkeyed.append((node.tag, name, value))
    assert unkeyed == []


def test_every_key_the_page_uses_exists_in_english():
    assert html_keys() - set(EN) == set()


def test_every_key_a_script_uses_exists_in_both_languages():
    used = script_keys()
    assert len(used) >= 20
    assert set(used) - set(EN) == set()
    assert set(used) - set(ES) == set()


def test_a_scripts_default_text_is_the_english_text():
    for key, default in script_keys().items():
        if default is not None:
            assert EN[key] == default.replace('\\"', '"').replace("\\'", "'"), key


def test_every_reason_code_has_both_translations():
    schemas = pytest.importorskip("app.schemas")
    for code in get_args(schemas.ReasonCode):
        assert f"reason.{code}" in EN and f"reason.{code}" in ES, code
    assert "reason.fallback" in EN and "reason.fallback" in ES


def test_the_english_reasons_match_the_scripts_defaults():
    block = re.search(r"const REASON_EXPLAIN = \{(.*?)\n\};", APP_JS, re.S).group(1)
    for code, text in re.findall(r'^\s+(\w+): "(.*)",?$', block, re.M):
        assert EN[f"reason.{code}"] == text.replace('\\"', '"'), code


def test_the_spanish_file_adds_no_key_english_lacks():
    assert set(ES) - set(EN) == set()


def test_no_value_is_empty_or_untranslated_by_accident():
    assert all(isinstance(v, str) and v.strip() for v in EN.values())
    assert all(isinstance(v, str) and v.strip() for v in ES.values())
    same = [k for k in ES if ES[k] == EN[k]]
    allowed = {"auth_screen.groundcheckhealth", "app.title", "sidebar.embeddings", "pt_sex.other", "query.how_is_veltris_syndrome",
               "js.hooks", "js.hardware", "view_training.hardware", "cds_title.cds_hooks", "view_ehr.cds_hooks_request",
               "dataset_note.dailymed", "dataset_note.medquad", "model_dialog.auc", "patient_box.alt", "patient_box.inr", "patient_box.qtc",
               "train_dataset.data_datasets_organamnist_64", "view_users.2fa", "imaging_series_title.series", "js.child_pugh",
               "js.compatible", "js.considerable", "js.control", "js.duration_h_min", "js.duration_min_s", "js.groundcheckhealth", "js.http",
               "js.local_numpy", "js.mm", "js.px_2", "patient_box.mmol_l", "plural.series.other", "pt_liver.child_pugh_a",
               "pt_liver.child_pugh_b", "pt_liver.child_pugh_c"}
    brand_or_unit = {k for k in same if not re.search(r"[a-z]{4,}", EN[k])}
    assert set(same) <= allowed | brand_or_unit | {k for k in same if EN[k] in {"Total", "Local AI"}}, sorted(set(same) - allowed)


def test_placeholders_survive_translation():
    """Every {name} in English appears in Spanish, and no new one. {s} only marks
    an English plural ending, so a translation may do without it."""
    for key, value in ES.items():
        want = {p for p in PLACEHOLDER.findall(EN[key]) if not re.fullmatch(r"\{s\d*\}", p)}
        got = set(PLACEHOLDER.findall(value))
        assert want <= got <= want | {p for p in PLACEHOLDER.findall(EN[key])}, key


def test_the_main_views_are_translated():
    prefixes = ("auth_", "sidebar.", "nav_admin.", "view_ask.", "ask_", "patient_box.", "feedback_", "tuning_",
                "claim_rows.", "refuse_callout.", "sources_panel.", "trace_panel.", "audit_", "login_form.", "reason.", "js.")
    main = [k for k in EN if k.startswith(prefixes)]
    assert len(main) >= 100
    assert [k for k in main if k not in ES] == []


def test_spanish_has_every_key():
    assert [k for k in EN if k not in ES] == []


def test_no_visible_literal_in_the_script_is_unkeyed():
    """Every string literal in app.js that a person could read goes through t(),
    apart from code (selectors, wire values, key names) and the reason table,
    whose texts are keyed under reason.* and checked against it."""
    import i18n_js

    reasons = re.search(r"const REASON_EXPLAIN = \{.*?\n\};\nconst REASON_FALLBACK = [^\n]*", APP_JS, re.S)
    offenders = []
    for lit, text in i18n_js.candidates(APP_JS):
        if reasons and reasons.start() <= lit.start < reasons.end():
            continue
        offenders.append((lit.line, text[:70]))
    assert offenders == []


def test_the_scanner_catches_an_unkeyed_string():
    import i18n_js

    found = [text for _, text in i18n_js.candidates('el.textContent = "Dose not stated";\nel.textContent = t("js.x", "Fine text");\nx = `${n} of ${m} done`;')]
    assert found == ["Dose not stated", "of done"]


def test_every_plural_word_has_both_forms():
    calls = list(re.finditer(r"\bplural\(", APP_JS))
    assert len(calls) >= 20
    for m in calls:
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(APP_JS[i], 0)
            i += 1
        word = re.search(r',\s*"([^"]+)"\s*$', APP_JS[m.end():i - 1])
        assert word, APP_JS[m.end():i - 1][:60]
        for form in ("one", "other"):
            key = f"plural.{word.group(1).replace(' ', '_')}.{form}"
            assert key in EN and key in ES, key


def test_every_server_code_the_page_labels_has_a_translation():
    governance = pytest.importorskip("app.governance")
    from app import compliance_controls

    def slug(code):
        return re.sub(r"[^a-z0-9]+", "_", str(code), flags=re.I)

    wanted = [f"outcome.{slug(k)}" for k in governance.OUTCOMES]
    wanted += [f"control_status.{slug(k)}" for k in compliance_controls.STATUSES]
    wanted += [f"refusal_category.{slug(v)}" for v in governance._CODE_CATEGORY.values()]
    wanted += [f"case_event.{k}" for k in ("assigned", "commented", "escalated", "flagged", "opened", "reopened", "resolved")]
    wanted += [f"hazard_status.{k}" for k in ("open", "mitigated", "closed")]
    wanted += [f"js.role_{k}" for k in ("clinician", "reviewer", "admin")]
    wanted += [f"js.stage_{k}" for k in ("decision", "generate", "retrieve")]
    assert [k for k in wanted if k not in EN or k not in ES] == []


def test_no_translation_key_is_orphaned():
    used = set(re.findall(r'\bt\("([\w.]+)"', APP_JS)) | html_keys()
    dynamic = ("plural.", "outcome.", "control_status.", "refusal_category.", "case_event.", "hazard_status.",
               "js.role_", "reason.", "feedback_reason.", "app.")
    orphans = [k for k in EN if k.startswith("js.") and k not in used and not k.startswith(dynamic)]
    assert orphans == []


def test_every_nav_entry_is_translated():
    for key in (k for k in EN if k.startswith("sidebar.") or k.startswith("nav_admin.")):
        assert key in ES, key


def test_clinical_words_keep_their_meaning():
    assert ES["patient_box.pregnant"] == "Embarazada"
    assert ES["patient_box.breastfeeding"] == "Lactancia"
    assert ES["patient_box.allergies"] == "Alergias"
    assert "no " in ES["patient_box.answers_are_checked_against"].lower()
    assert "150" not in json.dumps(ES)
    assert ES["feedback_down.not_useful"].lower().startswith("no ")
    assert ES["js.answer"] != ES["js.refused"]


def test_the_locale_switch_is_in_the_page_and_the_scripts_are_ordered():
    assert INDEX.count('class="select locale-select"') == 2
    assert INDEX.index('/i18n.js') < INDEX.index('/app.js')
    assert 'lang="en"' in INDEX
    for option in ('value="en"', 'value="es"'):
        assert option in INDEX


def test_the_locale_files_are_served():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        for name in ("en", "es"):
            r = client.get(f"/i18n/{name}.json")
            assert r.status_code == 200 and r.json()
        assert client.get("/i18n.js").status_code == 200
        assert client.get("/i18n/xx.json").status_code == 404


def test_the_switch_does_not_reintroduce_inline_html():
    js = (WEB / "i18n.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert banned not in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_locale_handling_in_a_stubbed_browser():
    out = subprocess.run(["node", "i18n_unit.mjs"], cwd=ROOT / "tests" / "ui", capture_output=True, text=True, timeout=60)
    results = json.loads(out.stdout.strip().splitlines()[-1])
    failed = [r for r in results if not r["ok"]]
    assert out.returncode == 0 and failed == [] and len(results) >= 12
