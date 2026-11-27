"""Keep web/index.html and web/i18n/en.json in step.

Every piece of visible text in index.html carries a `data-i18n` key (or
`data-i18n-attr` for placeholder, title, aria-label and alt), and web/i18n.js
swaps it for the chosen locale's string. The English text stays in the HTML as
the default, so the page reads correctly before the locale file has loaded.

    python scripts/i18n_extract.py            add keys to the HTML, rewrite en.json
    python scripts/i18n_extract.py --check    exit 1 if the HTML and en.json disagree
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "index.html"
EN = ROOT / "web" / "i18n" / "en.json"

VOID = {"input", "br", "img", "meta", "link", "hr", "area", "base", "col", "embed", "source", "track", "wbr"}
OPAQUE = {"script", "style", "svg", "textarea", "title", "code", "pre", "kbd", "samp", "select-none"}
ATTRS = ("placeholder", "title", "aria-label", "alt")
TOKEN = re.compile(r"<!--.*?-->|<!doctype[^>]*>|<(/?)([a-zA-Z][\w:-]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>", re.S | re.I)
HAS_LETTER = re.compile(r"[^\W\d_]")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def slug(text: str, limit: int = 4) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:limit]
    return "_".join(words) or "text"


class Node:
    def __init__(self, tag: str, start: int, open_end: int, attrs: str, parent: "Node | None"):
        self.tag, self.start, self.open_end, self.attrs, self.parent = tag, start, open_end, attrs, parent
        self.children: list[Node] = []
        self.texts: list[tuple[int, int]] = []
        self.close_start = open_end

    def attr(self, name: str) -> str | None:
        m = re.search(rf'(?<![\w-]){re.escape(name)}\s*=\s*"([^"]*)"', self.attrs)
        return m.group(1) if m else None

    def opaque(self) -> bool:
        node = self
        while node is not None:
            if node.tag in OPAQUE or "data-i18n-skip" in node.attrs:
                return True
            node = node.parent
        return False

    def section(self) -> str:
        node = self
        while node is not None:
            ident = node.attr("id")
            if ident and not ident.startswith(("i-",)):
                return ident
            node = node.parent
        return "page"


def parse(source: str) -> Node:
    root = Node("#root", 0, 0, "", None)
    top = root
    pos = 0
    for m in TOKEN.finditer(source):
        if m.start() > pos:
            top.texts.append((pos, m.start()))
        pos = m.end()
        if m.group(0).startswith("<!"):
            continue
        closing, tag, attrs = m.group(1), m.group(2).lower(), m.group(3)
        if closing:
            node = top
            while node is not None and node.tag != tag:
                node = node.parent
            if node is not None and node.parent is not None:
                node.close_start = m.start()
                top = node.parent
            continue
        node = Node(tag, m.start(), m.end(), attrs, top)
        top.children.append(node)
        if tag in VOID or attrs.rstrip().endswith("/"):
            continue
        top = node
        if tag in ("script", "style"):
            end = re.compile(rf"</{tag}\s*>", re.I).search(source, m.end())
            if end:
                node.close_start = end.start()
                pos = end.end()
                top = node.parent
    if pos < len(source):
        top.texts.append((pos, len(source)))
    return root


def walk(node: Node):
    yield node
    for child in node.children:
        yield from walk(child)


def extract(source: str) -> tuple[str, dict[str, str]]:
    root = parse(source)
    strings: dict[str, str] = {}
    by_text: dict[str, str] = {}
    existing = {m.group(1) for m in re.finditer(r'data-i18n(?:-attr)?="([^"]+)"', source)}

    def key_for(node: Node, text: str) -> str:
        if text in by_text:
            return by_text[text]
        base = f"{node.section()}.{slug(text)}".replace("-", "_")
        key, n = base, 2
        while key in strings and strings[key] != text:
            key = f"{base}_{n}"
            n += 1
        strings[key] = text
        by_text[text] = key
        return key

    edits: list[tuple[int, int, str]] = []
    for node in walk(root):
        if node.tag == "#root":
            continue
        # Text content.
        if not node.opaque():
            live = [(a, b) for a, b in node.texts if node is not root and node.open_end <= a and b <= node.close_start]
            letters = [(a, b) for a, b in live if HAS_LETTER.search(source[a:b]) and "{" not in source[a:b]]
            existing_key = node.attr("data-i18n")
            if existing_key:
                text = squash(source[node.open_end:node.close_start]) if not node.children else None
                if text is not None:
                    strings.setdefault(existing_key, text)
                    by_text.setdefault(text, existing_key)
                continue
            if letters and not node.children and len(letters) == 1:
                a, b = letters[0]
                text = squash(source[a:b])
                edits.append((node.open_end - 1, node.open_end - 1, f' data-i18n="{key_for(node, text)}"'))
            elif letters:
                for a, b in letters:
                    raw = source[a:b]
                    text = squash(raw)
                    lead = raw[:len(raw) - len(raw.lstrip())]
                    trail = raw[len(raw.rstrip()):]
                    inner = raw.strip()
                    edits.append((a, b, f'{lead}<span data-i18n="{key_for(node, text)}">{inner}</span>{trail}'))
        # Attributes.
        found = []
        for name in ATTRS:
            value = node.attr(name)
            if value and HAS_LETTER.search(value) and "{" not in value and not node.opaque():
                found.append((name, value))
        if found and "data-i18n-attr" not in node.attrs:
            parts = []
            for name, value in found:
                text = squash(value)
                parts.append(f"{name}:{key_for(node, text)}")
            edits.append((node.open_end - 1, node.open_end - 1, f' data-i18n-attr="{";".join(parts)}"'))
        elif "data-i18n-attr" in node.attrs:
            for pair in (node.attr("data-i18n-attr") or "").split(";"):
                name, _, key = pair.partition(":")
                value = node.attr(name)
                if value is not None:
                    strings.setdefault(key, squash(value))
    # Apply from the end so offsets stay valid. A tag's own edit lands before
    # its closing '>', and a self-closing slash is kept after it.
    out = source
    for a, b, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        if a == b and out[a - 1:a] == "/":
            a = b = a - 1
        out = out[:a] + text + out[b:]
    del existing
    return out, strings


KEPT_PREFIXES = ("reason", "js", "app", "plural", "control_status", "outcome", "case_event", "hazard_status", "refusal_category",
                 "srv", "msg", "example", "data")


def main() -> int:
    check = "--check" in sys.argv
    source = INDEX.read_text(encoding="utf-8")
    rewritten, strings = extract(source)
    current = json.loads(EN.read_text(encoding="utf-8")) if EN.exists() else {}
    # Keys that the scripts use (not the HTML) are kept as they are.
    # A key the page uses on an element that has child elements has no single text to read back; keep its string.
    used = set(re.findall(r'data-i18n="([^"]+)"', source))
    for pairs in re.findall(r'data-i18n-attr="([^"]+)"', source):
        used |= {pair.partition(":")[2] for pair in pairs.split(";")}
    kept = {k: v for k, v in current.items() if k not in strings and (k.split(".")[0] in KEPT_PREFIXES or k in used)}
    merged = {**strings, **kept}
    text = json.dumps(dict(sorted(merged.items())), indent=1, ensure_ascii=False) + "\n"
    if check:
        ok = rewritten == source and EN.exists() and EN.read_text(encoding="utf-8") == text
        print("in step" if ok else "web/index.html and web/i18n/en.json are out of step; run scripts/i18n_extract.py")
        return 0 if ok else 1
    INDEX.write_text(rewritten, encoding="utf-8")
    EN.parent.mkdir(parents=True, exist_ok=True)
    EN.write_text(text, encoding="utf-8")
    print(f"{len(strings)} strings from the page, {len(kept)} kept from the scripts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
