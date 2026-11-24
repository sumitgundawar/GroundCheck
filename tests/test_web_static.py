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
