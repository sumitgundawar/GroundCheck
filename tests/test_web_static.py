"""Static checks on the dashboard's JavaScript and HTML that need no browser."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEB = Path(__file__).resolve().parent.parent / "web"
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")
INDEX = (WEB / "index.html").read_text(encoding="utf-8")


def test_every_element_the_script_looks_up_exists():
    html_ids = set(re.findall(r'\bid="([^"$]+)"', INDEX)) | set(re.findall(r'\.id = "([^"$]+)"', APP_JS))
    wanted = set(re.findall(r'\$\("([^"]+)"\)', APP_JS))
    missing = sorted(wanted - html_ids)
    assert not missing, f"app.js looks up ids that index.html lacks: {missing}"


def test_every_reason_code_has_an_explanation():
    schemas = pytest.importorskip("app.schemas")
    if not hasattr(schemas, "ReasonCode"):
        pytest.skip("schemas.ReasonCode is not defined")
    from typing import get_args

    block = re.search(r"const REASON_EXPLAIN = \{(.*?)\n\};", APP_JS, re.S)
    assert block, "REASON_EXPLAIN is missing from app.js"
    keys = set(re.findall(r"^\s+(\w+):", block.group(1), re.M))
    missing = [code for code in get_args(schemas.ReasonCode) if code not in keys]
    assert not missing, f"no explanation for reason codes: {missing}"



def test_untrusted_text_is_never_assigned_through_innerhtml():
    offenders = []
    for number, line in enumerate(APP_JS.splitlines(), 1):
        match = re.search(r"\.(innerHTML|outerHTML)\s*\+?=\s*(.+)", line)
        if not match:
            continue
        value = match.group(2).strip().rstrip(";")
        if value in ('""', "''", "``"):
            continue
        if re.match(r'await \(await fetch\("/assets/[\w./-]+"\)\)\.text\(\)', value):
            continue
        offenders.append(f"{number}: {line.strip()}")
    assert not offenders, "innerHTML assigned from non-constant values:\n" + "\n".join(offenders)
    assert "insertAdjacentHTML" not in APP_JS
    assert "document.write" not in APP_JS


def test_no_outdated_verbatim_wording():
    assert "verbatim" not in APP_JS.lower()
    assert "character for character" not in APP_JS
    assert '"cosine"' not in APP_JS


def test_live_region_and_focus_styles_exist():
    assert re.search(r'id="decision-panel"[^>]*aria-live="polite"', INDEX)
    assert ":focus-visible" in (WEB / "styles.css").read_text(encoding="utf-8")


# --- Feedback, adoption, controls and telemetry panels -------------------------------------------

def _function_source(name: str) -> str:
    match = re.search(rf"^(?:async )?function {name}\(.*?^}}", APP_JS, re.S | re.M)
    assert match, f"{name} is missing from app.js"
    return match.group(0)


def test_the_feedback_reasons_in_the_page_are_the_ones_the_server_accepts():
    from app import feedback

    select = re.search(r'<select id="feedback-reason">(.*?)</select>', INDEX, re.S).group(1)
    offered = [v for v in re.findall(r'<option value="(\w*)"', select) if v]
    assert offered == list(feedback.REASONS)


def test_feedback_controls_are_buttons_with_labels_and_state():
    for ident in ("feedback-up", "feedback-down", "feedback-cancel"):
        assert re.search(rf'<button type="button"[^>]*id="{ident}"', INDEX), ident
    assert re.search(r'id="feedback-send"', INDEX) and 'type="submit"' in INDEX.split('id="feedback-form"')[1].split("</form>")[0]
    for ident in ("feedback-up", "feedback-down"):
        assert re.search(rf'id="{ident}"[^>]*aria-pressed="false"', INDEX)
    for target in ("feedback-reason", "feedback-comment"):
        assert f'for="{target}"' in INDEX
    assert re.search(r'id="feedback-message"[^>]*role="status"', INDEX)
    assert 'maxlength="500"' in INDEX.split('id="feedback-comment"')[1][:80] or 'id="feedback-comment" rows="2" maxlength="500"' in INDEX


def test_the_comment_box_warns_about_patient_details():
    block = INDEX.split('for="feedback-comment"')[1][:300]
    assert "removed" in block and "patient details" in block


def test_feedback_code_builds_no_html_and_talks_to_the_right_endpoints():
    for name in ("resetFeedback", "setRated", "sendFeedback", "wireFeedback"):
        source = _function_source(name)
        assert "innerHTML" not in source and "insertAdjacentHTML" not in source
    assert "/api/audit/${auditId}/feedback" in APP_JS
    assert 'sendFeedback({ rating: "up" })' in APP_JS and 'rating: "down"' in APP_JS
    # The feedback row follows every answer or refusal through the one place that resets the flag row.
    assert "resetFeedback(auditId);" in _function_source("resetFlag")
    assert "wireFeedback();" in APP_JS


def test_every_endpoint_the_new_panels_call_is_a_real_route():
    from app.main import app

    templates = {route.path for route in app.routes if hasattr(route, "methods")}

    def known(url: str) -> bool:
        url = url.split("?")[0]
        norm = re.sub(r"\$\{[^}]+\}", "{x}", url)
        return any(re.fullmatch(re.sub(r"\{[^}]+\}", "[^/]+", t), norm) for t in templates)

    for url in ("/api/audit/${auditId}/feedback", "/api/metrics/adoption?days=${days}", "/api/telemetry",
                "/api/governance/controls", "/api/governance/controls.csv", "/api/governance/pilot-report?days=${days}",
                "/api/governance/evidence-pack"):
        assert known(url), url
        assert url.split("?")[0].split("${")[0] in APP_JS + INDEX, url
    for url in re.findall(r'href="(/api/[^"?]+)', INDEX):
        assert known(url), url


def test_the_review_tabs_agree_between_the_page_and_the_script():
    tabs = re.findall(r'role="tab" class="review-tab" id="tab-(\w+)" aria-controls="review-(\w+)"', INDEX)
    assert all(a == b for a, b in tabs) and [a for a, _ in tabs] == ["queue", "hazards", "report", "controls"]
    for name in ("selectTab",):
        assert '["queue", "hazards", "report", "controls"]' in _function_source(name)
    assert 'const tabs = ["queue", "hazards", "report", "controls"];' in APP_JS
    for name, _ in tabs:
        assert f'id="review-{name}" aria-labelledby="tab-{name}"' in INDEX


def test_the_controls_page_uses_the_servers_banner_and_statuses():
    from app import compliance_controls as cc

    options = re.findall(r'<option value="(\w*)"', INDEX.split('id="controls-status"')[1].split("</select>")[0])
    assert [o for o in options if o] == list(cc.STATUSES)
    assert re.search(r'id="controls-banner" role="note"', INDEX)
    load = _function_source("loadControls")
    assert 'textContent = data.banner' in load and "innerHTML" not in load
    assert "innerHTML" not in _function_source("renderControls")
    # The banner is not a literal in the script: it cannot be edited out of the data and left standing.
    assert "no certification" not in APP_JS.lower()


def test_the_controls_table_scrolls_inside_its_wrapper_on_a_phone():
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    assert re.search(r"\.users-table-wrap\s*\{[^}]*overflow-x:\s*auto", css)
    assert 'className = "users-table-wrap"' in _function_source("renderControls")
    assert re.search(r"\.controls-table td\s*\{[^}]*overflow-wrap:\s*anywhere", css)
    assert re.search(r"\.feedback-row\s*\{[^}]*flex-wrap:\s*wrap", css)


def test_adoption_is_admin_only_and_hidden_until_the_server_allows_it():
    assert re.search(r'<section class="panel" id="adoption-panel"[^>]*hidden', INDEX)
    source = _function_source("loadAdoption")
    assert "panel.hidden = true; return;" in source and "innerHTML" not in source
    assert "/api/metrics/adoption?days=${days}" in source and "/api/governance/pilot-report?days=${days}" in source
    assert "loadAdoption();" in _function_source("loadUsage")


def test_telemetry_is_shown_read_only_with_every_field_before_anything_is_enabled():
    source = _function_source("loadTelemetry")
    assert "innerHTML" not in source and "t.fields" in source and "t.sample" in source and "Off. Nothing is sent." in source
    assert 'id="telemetry-fields"' in INDEX and 'id="evidence-download"' in INDEX
    assert not re.search(r'id="telemetry[^"]*"[^>]*type="checkbox"', INDEX)       # no toggle: the environment decides
    assert "loadTelemetry();" in _function_source("loadProtection")
