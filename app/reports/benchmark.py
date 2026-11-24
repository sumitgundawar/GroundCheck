"""A benchmark result as a document a vendor can hand to procurement.

The page states the method, the suites, the results and the caveats, and
carries the results as JSON with a hash and, when an audit signing key is
configured, an HMAC over them. The caveat block is part of every page and has
no switch. `verify_file` re-checks the hash and the signature and re-renders
the page from its own embedded data, so an edit anywhere, including to the
visible numbers, is found.

An HMAC proves the report came from an installation holding the key; only a
holder can check it. Without a key the page carries a hash only and says so."""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
from pathlib import Path

from .. import __version__, integrity

CONTEXT = "groundcheck-benchmark"
VERIFY = "python scripts/benchmark.py verify-report REPORT.html"
CAVEATS = (
    "This report is not a certification, clearance, approval or regulatory submission, and GroundCheckHealth is "
    "not a certified medical device.",
    "The suites were written by the people who built the guards. They catch regressions; they do not forecast "
    "performance on real clinical questions. Where a held-out split is reported it is the only part not seen "
    "while the guards were tuned, and it is still derived from the same synthetic corpus.",
    "The corpus and formulary the questions are drawn from are synthetic. Results say how a model behaved with "
    "these guards on these questions on this date, in English, with sampled cases, and nothing about any real "
    "medicine, patient or deployment.",
    "A low unsafe-answer rate with the guards on describes the guards and this pipeline, not the model alone. "
    "The figure without the guards describes the model alone.",
    "A rate over few cases is a rough estimate. Read the case counts beside every rate.",
)


def canonical(result: dict) -> bytes:
    return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sign(result: dict) -> dict:
    """The hash of the results and, when a signing key is configured, an HMAC."""
    content = canonical(result)
    sig = {"content_sha256": hashlib.sha256(content).hexdigest(), "alg": "sha256", "signature": None,
           "key_id": None, "signed": False, "generator": f"GroundCheckHealth {__version__}"}
    if integrity.signing_summary()["enabled"]:
        alg, digest = integrity.compute(CONTEXT, content)
        sig.update(alg=alg, signature=digest, key_id=alg.split(":", 1)[1], signed=True)
    return sig


def check(result: dict, sig: dict) -> list[str]:
    """Problems with a result and its signature block; empty if both hold."""
    problems = []
    content = canonical(result)
    if sig.get("content_sha256") != hashlib.sha256(content).hexdigest():
        problems.append("The results do not match their hash.")
    if sig.get("signed"):
        try:
            _, expected = integrity.compute(CONTEXT, content, sig.get("alg"))
            if not hmac.compare_digest(expected, str(sig.get("signature") or "")):
                problems.append("The signature does not match the results.")
        except KeyError:
            problems.append("The report was signed with a key this installation does not have.")
    return problems


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _pct(rate) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _frac(n, d, rate) -> str:
    return f"{n} of {d} ({_pct(rate)})" if d else "none to measure"


def _arm_rows(a: dict, b: dict) -> list[tuple[str, str, str]]:
    return [
        ("Cases evaluated", f"{a['evaluated']} of {a['cases']}", f"{b['evaluated']} of {b['cases']}"),
        ("Decision accuracy", _pct(a["accuracy"]), _pct(b["accuracy"])),
        ("Unsafe answers (must-refuse cases answered)", _frac(a["unsafe_answers"], a["must_refuse"], a["unsafe_rate"]),
         _frac(b["unsafe_answers"], b["must_refuse"], b["unsafe_rate"])),
        ("Over-refusals (answerable cases refused)", _frac(a["over_refusals"], a["answerable"], a["over_refusal_rate"]),
         _frac(b["over_refusals"], b["answerable"], b["over_refusal_rate"])),
    ]


def _table(a: dict, b: dict) -> str:
    rows = "".join(f"<tr><th scope=\"row\">{_e(n)}</th><td>{_e(x)}</td><td>{_e(y)}</td></tr>" for n, x, y in _arm_rows(a, b))
    return ("<table><thead><tr><th></th><th scope=\"col\">Model alone (guards off)</th>"
            f"<th scope=\"col\">With GroundCheck guards</th></tr></thead><tbody>{rows}</tbody></table>")


def _script_json(value: dict) -> str:
    """JSON for a data block, in insertion order so a round trip re-renders the same page, with the
    sequences that could end or confuse the block escaped (still valid JSON)."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\u0021--")


STYLE = ("body{font:16px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;color:#1b1f23;background:#fff}"
         "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #c8ccd0;padding:.4rem .6rem;text-align:left}"
         "thead th{background:#f1f3f5}h1{font-size:1.6rem}h2{margin-top:2rem}"
         ".caveats{border:2px solid #8a1c1c;padding:.5rem 1rem;background:#fdf1f1}code{background:#f1f3f5;padding:0 .25rem}"
         "@media (prefers-color-scheme:dark){body{background:#14171a;color:#e6e8ea}th,td{border-color:#3a4047}"
         "thead th{background:#20252a}.caveats{background:#2a1618;border-color:#d96b6b}code{background:#20252a}}")


def render_html(result: dict, signature: dict) -> str:
    """The report page. Deterministic: only the result and signature appear in it."""
    scores = result.get("scores", {})
    overall = scores.get("overall", {})
    provider = result.get("provider", {})
    suites = scores.get("suites", {})
    cases = result.get("cases", {})
    lat, usage = result.get("latency_ms", {}), result.get("usage", {})
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        f"<title>Model safety benchmark: {_e(provider.get('model', 'unknown'))}</title><style>{STYLE}</style></head><body>",
        "<h1>Model safety benchmark report</h1>",
        f"<p>Model: <strong>{_e(provider.get('model', 'unknown'))}</strong> (provider: {_e(provider.get('name', 'unknown'))}). "
        f"Run on {_e(result.get('created_at', 'unknown date'))} with seed {_e(result.get('seed', 'n/a'))}. "
        f"GroundCheckHealth {_e(__version__)}, benchmark format {_e(result.get('benchmark_version', 'n/a'))}.</p>",
        "<section id=\"caveats\" class=\"caveats\"><h2>What this report is not</h2><ul>"
        + "".join(f"<li>{_e(c)}</li>" for c in CAVEATS) + "</ul></section>",
        "<h2>Method</h2><ul>",
        "<li>Each question is put to the model through GroundCheck's pipeline twice. <strong>With guards</strong>: the "
        "normal pipeline, including input checks, the retrieval gate, source-coverage, grounding and dosage checks "
        "and, for patient cases, the patient's own data. <strong>Guards off</strong>: none of those; the model sees "
        "the passages retrieved for the question and its answer stands. The decision to answer or refuse is then "
        "compared with the one the case expects.</li>",
        "<li>An <strong>unsafe answer</strong> is a must-refuse case that was answered. An <strong>over-refusal</strong> "
        "is an answerable case that was refused. Where the model call failed and extractive generation answered, "
        "the guards-off result is not evaluated and is counted apart.</li>",
        "<li>Temperature 0; the optional model judge is off. Cases are sampled by the seed above, so the same seed "
        "gives the same cases.</li>",
        "<li>Suites: " + (", ".join(f"{_e(s)} ({_e(n)} cases)" for s, n in cases.items()) or "none") + ". "
        + ("A held-out split is included." if "heldout" in cases else "No held-out split is included.") + "</li></ul>",
    ]
    if overall:
        parts += ["<h2>Results, all suites</h2>", _table(overall["without_guards"], overall["with_guards"])]
    for name, pair in suites.items():
        parts += [f"<h3>{_e(name)} ({_e(cases.get(name, 0))} cases)</h3>", _table(pair["without_guards"], pair["with_guards"])]
    on, off = lat.get("on", {}), lat.get("off", {})
    cost = usage.get("estimated_cost")
    parts += [
        "<h2>Latency and cost</h2><ul>",
        f"<li>Per question, with guards: median {_e(on.get('median', 'n/a'))} ms, 95th percentile {_e(on.get('p95', 'n/a'))} ms. "
        f"Guards off: median {_e(off.get('median', 'n/a'))} ms, 95th percentile {_e(off.get('p95', 'n/a'))} ms.</li>",
        f"<li>Tokens: {_e(usage.get('prompt_tokens', 0))} prompt, {_e(usage.get('completion_tokens', 0))} completion; "
        f"{_e(usage.get('calls_without_usage', 0))} call(s) reported no usage.</li>",
        "<li>Estimated cost: " + ("not estimated (no prices given)" if cost is None else _e(cost)) + ".</li></ul>",
        "<h2>Integrity</h2>",
        f"<p>SHA-256 of the results: <code>{_e(signature.get('content_sha256', ''))}</code></p>",
    ]
    if signature.get("signed"):
        parts.append(f"<p>Signed with HMAC-SHA256 under audit signing key <code>{_e(signature.get('key_id'))}</code>: "
                     f"<code>{_e(signature.get('signature'))}</code>. Only a holder of that key can check the signature.</p>")
    else:
        parts.append("<p>This report is <strong>not signed</strong>: no audit signing key was configured, so it carries a "
                     "hash only, which shows the results were not changed after the report was made but not who made it.</p>")
    parts += [f"<p>Check it with <code>{_e(VERIFY)}</code>.</p>",
              f"<script type=\"application/json\" id=\"benchmark-data\">{_script_json(result)}</script>",
              f"<script type=\"application/json\" id=\"benchmark-signature\">{_script_json(signature)}</script>",
              "</body></html>"]
    return "\n".join(parts) + "\n"


_BLOCK = re.compile(r'<script type="application/json" id="(benchmark-data|benchmark-signature)">(.*?)</script>', re.S)


def extract(page: str) -> tuple[dict, dict]:
    blocks = {name: body for name, body in _BLOCK.findall(page)}
    if set(blocks) != {"benchmark-data", "benchmark-signature"}:
        raise ValueError("This is not a benchmark report.")
    return json.loads(blocks["benchmark-data"]), json.loads(blocks["benchmark-signature"])


def verify_page(page: str) -> dict:
    try:
        result, sig = extract(page)
    except (ValueError, KeyError) as exc:
        return {"ok": False, "message": f"Not verified: {exc}"}
    problems = check(result, sig)
    if not problems and render_html(result, sig) != page:
        problems.append("The page was edited: it does not match the results it carries.")
    if problems:
        return {"ok": False, "message": "NOT VERIFIED: " + " ".join(problems)}
    how = f"signed with key {sig['key_id']}" if sig.get("signed") else "hash only, not signed"
    return {"ok": True, "message": f"Verified ({how}); SHA-256 {sig['content_sha256']}."}


def verify_file(path: Path) -> dict:
    return verify_page(Path(path).read_text(encoding="utf-8"))
