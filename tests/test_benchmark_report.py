"""The benchmark report: a fixed caveat block with no switch, signing and
verification, tamper detection anywhere on the page, HTML-safe output, and a
snapshot."""

from __future__ import annotations

import conftest  # noqa: F401

import html.parser
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ["GROQ_API_KEY"] = ""
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark as bm  # noqa: E402

from app import config, encryption  # noqa: E402
from app.reports import benchmark as rb  # noqa: E402

SNAPSHOT = Path(__file__).parent / "fixtures" / "benchmark_report_snapshot.html"


def arm(cases, evaluated, correct, must, unsafe, ans, over):
    rate = lambda n, d: round(n / d, 4) if d else None  # noqa: E731
    return {"cases": cases, "evaluated": evaluated, "not_evaluated": cases - evaluated, "correct": correct,
            "accuracy": rate(correct, evaluated), "must_refuse": must, "unsafe_answers": unsafe,
            "unsafe_rate": rate(unsafe, must), "answerable": ans, "over_refusals": over, "over_refusal_rate": rate(over, ans)}


RESULT = {
    "benchmark_version": 1, "created_at": "2026-11-24T12:00:00Z", "provider": {"name": "anthropic", "model": "example-model-1"},
    "seed": 7, "thresholds": {"RETRIEVAL_MIN_SCORE": 0.3, "GROUNDING_MIN": 0.45, "TOP_K": 4},
    "cases": {"golden": 50, "adversarial": 16},
    "scores": {"suites": {
        "golden": {"with_guards": arm(50, 50, 50, 25, 0, 25, 0), "without_guards": arm(50, 49, 40, 24, 9, 25, 0)},
        "adversarial": {"with_guards": arm(16, 16, 16, 16, 0, 0, 0), "without_guards": arm(16, 16, 9, 16, 7, 0, 0)}},
        "overall": {"with_guards": arm(66, 66, 66, 41, 0, 25, 0), "without_guards": arm(66, 65, 49, 40, 16, 25, 0)}},
    "scores_sha256": "0" * 64, "latency_ms": {"on": {"median": 812.5, "p95": 1990.0}, "off": {"median": 790.0, "p95": 1800.0}},
    "usage": {"prompt_tokens": 123456, "completion_tokens": 7890, "calls_without_usage": 0,
              "price_per_million": {"input": 3.0, "output": 15.0}, "estimated_cost": 0.488658},
    "rows": [],
}


@pytest.fixture()
def unsigned(monkeypatch):
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", "")


@pytest.fixture()
def signed(monkeypatch):
    key = encryption.generate_key()
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", key)
    return key


def page(result=RESULT):
    return rb.render_html(result, rb.sign(result))


# --- Caveats --------------------------------------------------------------------------------------

def test_the_caveat_block_is_always_there(unsigned):
    for result in (RESULT, {}, {"provider": {}}, {"scores": {}}, {**RESULT, "scores": {"suites": {}, "overall": {}}}):
        text = page(result)
        assert 'id="caveats"' in text and "What this report is not" in text
        for caveat in rb.CAVEATS:
            assert caveat.split(",")[0][:50] in text
    assert "not a certification, clearance, approval" in page(RESULT)


def test_the_caveat_block_has_no_switch(unsigned):
    params = inspect.signature(rb.render_html).parameters
    assert list(params) == ["result", "signature"]
    with pytest.raises(TypeError):
        rb.render_html(RESULT, rb.sign(RESULT), caveats=False)
    with pytest.raises(TypeError):
        rb.render_html(RESULT, rb.sign(RESULT), show_caveats=False)
    # Nor can the data ask for it.
    sneaky = {**RESULT, "caveats": False, "show_caveats": False, "hide_caveats": True}
    assert 'id="caveats"' in page(sneaky)


def test_removing_the_caveats_from_a_page_is_detected(unsigned):
    text = page()
    stripped = re.sub(r'<section id="caveats".*?</section>', "", text, flags=re.S)
    assert stripped != text and not rb.verify_page(stripped)["ok"]
    assert rb.verify_page(text)["ok"]


# --- Content -------------------------------------------------------------------------------------------

def test_results_appear_as_computed(unsigned):
    text = page()
    assert "example-model-1" in text and "seed 7" in text
    assert "16 of 40 (40.0%)" in text                    # overall unsafe, guards off
    assert "0 of 41 (0.0%)" in text                      # overall unsafe, guards on
    assert "9 of 24 (37.5%)" in text and "7 of 16 (43.8%)" in text
    assert "65 of 66" in text and "median 812.5 ms" in text and "123456 prompt" in text and "0.488658" in text
    assert "A held-out split is included" not in text and "No held-out split is included" in text
    assert "A held-out split is included" in page({**RESULT, "cases": {**RESULT["cases"], "heldout": 10}})


def test_nothing_to_measure_is_said_not_shown_as_zero(unsigned):
    empty = {**RESULT, "scores": {"suites": {}, "overall": {"with_guards": arm(4, 4, 4, 0, 0, 4, 0),
                                                            "without_guards": arm(4, 0, 0, 0, 0, 0, 0)}}}
    text = page(empty)
    assert "none to measure" in text and "n/a" in text


def test_no_cost_when_no_prices(unsigned):
    result = {**RESULT, "usage": {**RESULT["usage"], "estimated_cost": None}}
    assert "not estimated (no prices given)" in page(result)


class Collector(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.scripts, self.title, self.tags = [], [], [], "", set()
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag not in ("meta", "br"):
            self.stack.append(tag)
        if tag == "script":
            self.scripts.append(dict(attrs))
        self._in_title = tag == "title"

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.errors.append(tag)
        self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def test_the_page_is_well_formed_and_titled(unsigned):
    c = Collector()
    c.feed(page())
    assert c.errors == [] and c.stack == []
    assert c.title == "Model safety benchmark: example-model-1"
    assert [s["id"] for s in c.scripts] == ["benchmark-data", "benchmark-signature"]
    assert all(s["type"] == "application/json" for s in c.scripts)


HOSTILE = '"><script>alert(1)</script><img src=x onerror=alert(2)>&amp;'


def test_hostile_values_cannot_inject_markup(unsigned):
    result = {**RESULT, "provider": {"name": HOSTILE, "model": HOSTILE}, "created_at": HOSTILE, "seed": HOSTILE,
              "cases": {HOSTILE: 3}, "scores": {**RESULT["scores"], "suites": {HOSTILE: RESULT["scores"]["suites"]["golden"]}}}
    text = page(result)
    body = re.sub(r'<script type="application/json".*?</script>', "", text, flags=re.S)
    assert "<script" not in body and "<img" not in body
    c = Collector()
    c.feed(text)
    assert "img" not in c.tags and "b" not in c.tags
    assert [s["id"] for s in c.scripts] == ["benchmark-data", "benchmark-signature"]      # nothing broke out of the data block
    assert rb.verify_page(text)["ok"], "hostile text still round-trips through the embedded data"


def test_a_script_close_tag_in_the_data_cannot_end_the_data_block(unsigned):
    result = {**RESULT, "provider": {"name": "x</script><b>boom</b><!--", "model": "m"}}
    text = page(result)
    assert text.count("</script>") == 2 and "<b>boom</b>" not in re.sub(r'<script.*?</script>', "", text, flags=re.S)
    assert rb.extract(text)[0]["provider"]["name"] == "x</script><b>boom</b><!--"


def test_rendering_is_deterministic_and_takes_nothing_from_the_clock(unsigned):
    assert page() == page()
    assert "2026-11-24T12:00:00Z" in page()


def test_the_page_matches_a_committed_snapshot(unsigned):
    from app import __version__

    text = page().replace(f"GroundCheckHealth {__version__}", "GroundCheckHealth X")
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(text, encoding="utf-8")
    assert text == SNAPSHOT.read_text(encoding="utf-8")


# --- Signing and verification -----------------------------------------------------------------------------

def test_unsigned_reports_carry_a_hash_and_say_they_are_unsigned(unsigned):
    sig = rb.sign(RESULT)
    assert sig["signed"] is False and sig["signature"] is None and sig["alg"] == "sha256"
    assert len(sig["content_sha256"]) == 64
    text = page()
    assert "not signed" in text and sig["content_sha256"] in text
    verdict = rb.verify_page(text)
    assert verdict["ok"] and "hash only, not signed" in verdict["message"]


def test_the_hash_is_over_the_canonical_results(unsigned):
    import hashlib

    reordered = json.loads(json.dumps(RESULT, sort_keys=True))
    assert rb.sign(reordered)["content_sha256"] == rb.sign(RESULT)["content_sha256"] == hashlib.sha256(
        json.dumps(RESULT, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert rb.sign({**RESULT, "seed": 8})["content_sha256"] != rb.sign(RESULT)["content_sha256"]


def test_a_signed_report_verifies_with_the_key(signed):
    sig = rb.sign(RESULT)
    assert sig["signed"] and sig["alg"].startswith("hmac-sha256:") and sig["key_id"] == sig["alg"].split(":")[1]
    text = page()
    assert sig["signature"] in text and sig["key_id"] in text and "Signed with HMAC-SHA256" in text
    verdict = rb.verify_page(text)
    assert verdict["ok"] and f"signed with key {sig['key_id']}" in verdict["message"]


def test_a_signed_report_does_not_verify_without_the_key(signed, monkeypatch):
    text = page()
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", encryption.generate_key())
    verdict = rb.verify_page(text)
    assert not verdict["ok"] and "key this installation does not have" in verdict["message"]
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", "")
    assert not rb.verify_page(text)["ok"]


def test_a_rotated_key_still_verifies_older_reports(signed, monkeypatch):
    text = page()
    monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", f"{encryption.generate_key()},{signed}")
    assert rb.verify_page(text)["ok"]


@pytest.mark.parametrize("name,edit", [
    ("a visible number", lambda t: t.replace("16 of 40 (40.0%)", "1 of 40 (2.5%)")),
    ("a visible model name", lambda t: t.replace("<strong>example-model-1</strong>", "<strong>other-model</strong>")),
    ("the embedded results", lambda t: t.replace('"seed":7', '"seed":8').replace('"seed": 7', '"seed": 8')),
    ("the embedded results number", lambda t: t.replace('"unsafe_answers": 16', '"unsafe_answers": 1')),
    ("the shown hash", lambda t: re.sub(r"(SHA-256 of the results: <code>)[0-9a-f]", r"\g<1>f", t)),
    ("extra text appended", lambda t: t.replace("</body>", "<p>Approved by a regulator.</p></body>")),
    ("the method text", lambda t: t.replace("Temperature 0", "Temperature 1")),
    ("truncation", lambda t: t[: len(t) // 2]),
    ("the signature block removed", lambda t: re.sub(r'<script type="application/json" id="benchmark-signature">.*?</script>', "", t, flags=re.S)),
    ("the results block removed", lambda t: re.sub(r'<script type="application/json" id="benchmark-data">.*?</script>', "", t, flags=re.S)),
])
def test_tampering_anywhere_is_detected_signed_or_not(name, edit, monkeypatch):
    for key in ("", encryption.generate_key()):
        monkeypatch.setattr(config, "AUDIT_SIGNING_KEYS", key)
        original = page()
        assert rb.verify_page(original)["ok"]
        edited = edit(original)
        assert edited != original, name
        assert rb.verify_page(edited)["ok"] is False, (name, bool(key))


def test_the_signature_value_cannot_be_forged_by_editing_it(signed):
    text = page()
    sig = rb.sign(RESULT)["signature"]
    forged = text.replace(sig, "0" * len(sig))
    assert not rb.verify_page(forged)["ok"]


def test_a_signature_block_cannot_be_downgraded_to_unsigned(signed):
    text = page()
    _, sig = rb.extract(text)
    sig.update(signed=False, signature=None, alg="sha256", key_id=None)
    downgraded = rb.render_html(RESULT, sig)
    assert rb.verify_page(downgraded)["ok"]          # a hash-only page is valid as a hash-only page,
    assert "not signed" in downgraded and "Signed with HMAC" not in downgraded    # and says so plainly


@pytest.mark.parametrize("garbage", ["", "hello", "<html></html>", '<script type="application/json" id="benchmark-data">{}</script>',
                                     '<script type="application/json" id="benchmark-data">not json</script>'
                                     '<script type="application/json" id="benchmark-signature">{}</script>'])
def test_garbage_is_not_a_report(garbage):
    assert rb.verify_page(garbage)["ok"] is False


def test_verify_file_and_the_cli_command(unsigned, tmp_path, capsys):
    path = tmp_path / "report.html"
    path.write_text(page(), encoding="utf-8")
    assert rb.verify_file(path)["ok"]
    assert bm.main(["verify-report", str(path)]) == 0
    path.write_text(path.read_text().replace("0.0%", "9.9%", 1), encoding="utf-8")
    assert bm.main(["verify-report", str(path)]) == 1
    assert "NOT VERIFIED" in capsys.readouterr().out
    assert "scripts/benchmark.py verify-report" in page()


def test_a_report_made_from_a_real_run_verifies_and_carries_the_caveats(tmp_path, monkeypatch, unsigned):
    import httpx

    from app import providers

    monkeypatch.setattr(config, "LLM_API_KEY", "k" * 20)
    body = {"choices": [{"message": {"content": json.dumps({"insufficient_context": True, "claims": []})}, "finish_reason": "stop"}]}
    monkeypatch.setattr(providers, "_client", lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))))
    result = bm.run("openai_compatible", tmp_path / "r.json", seed=3, suites=["golden"], golden_per_decision=3)
    text = rb.render_html(result, rb.sign(result))
    assert rb.verify_page(text)["ok"] and 'id="caveats"' in text
    assert "6 cases" in text or "(6 cases)" in text
