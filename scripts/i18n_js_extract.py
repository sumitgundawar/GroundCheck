"""Move the visible string literals in web/app.js into web/i18n/en.json.

    python scripts/i18n_js_extract.py            rewrite app.js and add keys to en.json
    python scripts/i18n_js_extract.py --dry-run  list what would change

A plain string becomes t("js.key", "text"); a template literal becomes
t("js.key", "text {name}", { name: expr }). The English text stays as the
default argument, so the page reads correctly if a locale file is missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i18n_js  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "web" / "app.js"
EN = ROOT / "web" / "i18n" / "en.json"


def decode(raw: str) -> str:
    out, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            n = raw[i + 1]
            if n == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", raw[i + 2:i + 6]):
                out.append(chr(int(raw[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append({"n": "\n", "t": "\t"}.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def js_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", re.sub(r"\{\w+\}", " ", text).lower())
    return "_".join(words[:5]) or "text"


def placeholder_name(expr: str, used: set[str]) -> str:
    if re.search(r'\?\s*(""|\'\')\s*:\s*"(s|es)"', expr):
        base_s = "s"
        n = base_s
        k = 2
        while n in used:
            n = f"s{k}"
            k += 1
        used.add(n)
        return n
    bare = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', '""', expr)
    bare = re.sub(r"`[^`]*`", "``", bare)
    skip = {"toLocaleString", "length", "toFixed", "join", "map", "filter", "true", "false", "null", "undefined", "toUpperCase", "slice", "round", "Math", "String", "Number", "toString", "toLowerCase"}
    ids = [i for i in re.findall(r"[A-Za-z_]\w*(?=\s*(?:[^\w(\s]|\s|$))", bare) if i not in skip]
    name = ids[0] if ids and "." not in bare else (ids[-1] if ids else "n")
    if "." in bare:
        chain = re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", bare)
        if chain:
            name = chain[0].split(".")[-1]
            if name in skip:
                name = chain[0].split(".")[-2]
    base, k = name, 2
    while name in used:
        name = f"{base}{k}"
        k += 1
    used.add(name)
    return name


LAZY_TABLES = ["PIPELINE_STAGES", "STAGE_LABELS", "REDACTION_LABEL", "KIND_LABEL", "STATUS_LABELS", "KIND_LABELS",
               "SEVERITY", "LIKELIHOOD", "FINDING_LABEL", "RELEASE_STATUS", "INCIDENT_CATEGORIES", "INCIDENT_HARM",
               "INCIDENT_STATUS", "RULE_NOW", "AGREEMENT_LABELS"]


def make_lazy(src: str) -> str:
    """A table of labels built at load time would be fixed in English: make it a function."""
    lines = src.split("\n")
    for name in LAZY_TABLES:
        start = next((i for i, ln in enumerate(lines) if ln.startswith(f"const {name} = ")), None)
        if start is None or f"const {name} = () =>" in lines[start]:
            continue
        end = start
        while not lines[end].rstrip().endswith(";"):
            end += 1
        lines[start] = lines[start].replace(f"const {name} = ", f"const {name} = () => (", 1)
        lines[end] = lines[end].rstrip()[:-1] + ");"
    out = "\n".join(lines)
    for name in LAZY_TABLES:
        out = re.sub(rf"(?<![\w.]){name}\b(?!\s*=\s*\(\)|\()", f"{name}()", out)
    return out


def run(dry: bool) -> None:
    src = make_lazy(APP.read_text(encoding="utf-8"))
    en = json.loads(EN.read_text(encoding="utf-8"))
    by_text = {v: k for k, v in en.items() if k.startswith("js.")}
    reasons = re.search(r"const REASON_EXPLAIN = \{.*?\n\};\nconst REASON_FALLBACK = [^\n]*", src, re.S)
    cands = [(lit, text) for lit, text in i18n_js.candidates(src)
             if not (reasons and reasons.start() <= lit.start < reasons.end())]
    chosen = {lit.start: lit for lit, _ in cands}
    new_keys: dict[str, str] = {}

    def key_for(text: str) -> str:
        if text in by_text:
            return by_text[text]
        base = "js." + slug(text)
        key, k = base, 2
        while key in en or key in new_keys:
            key = f"{base}_{k}"
            k += 1
        new_keys[key] = text
        by_text[text] = key
        return key

    def render(a: int, b: int) -> str:
        """Source between a and b with every chosen literal inside rewritten."""
        out, pos = [], a
        for lit in i18n_js.tokenize_range(src, a, b):
            if lit.start < pos:
                continue
            out.append(src[pos:lit.start])
            out.append(rewrite(lit))
            pos = lit.end
        out.append(src[pos:b])
        return "".join(out)

    def rewrite(lit) -> str:
        if lit.kind == "str":
            if lit.start not in chosen:
                return src[lit.start:lit.end]
            text = decode(src[lit.start + 1:lit.end - 1])
            return f"t({js_string(key_for(text))}, {js_string(text)})"
        pieces = [decode(p) for p in lit.parts]
        exprs = [render(a, b) for a, b in lit.exprs]
        if lit.start not in chosen:
            body = pieces[0] if False else ""
            out = []
            for i, piece in enumerate(lit.parts):
                out.append(piece)
                if i < len(exprs):
                    out.append("${" + exprs[i] + "}")
            return "`" + "".join(out) + "`"
        used: set[str] = set()
        names = [placeholder_name(src[a:b], used) for a, b in lit.exprs]
        text = pieces[0]
        for i, n in enumerate(names):
            text += "{" + n + "}" + pieces[i + 1]
        call = f"t({js_string(key_for(text))}, {js_string(text)}"
        if names:
            call += ", { " + ", ".join(f"{n}: {e}" if n != e else n for n, e in zip(names, exprs)) + " }"
        return call + ")"

    new_src = render(0, len(src))
    print(f"{len(cands)} literals, {len(new_keys)} new keys")
    if dry:
        for k, v in list(new_keys.items())[:0]:
            print(k, repr(v))
        return
    en.update(new_keys)
    EN.write_text(json.dumps(dict(sorted(en.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    APP.write_text(new_src, encoding="utf-8")


if __name__ == "__main__":
    run("--dry-run" in sys.argv)
