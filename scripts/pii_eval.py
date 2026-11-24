#!/usr/bin/env python3
"""Measure the de-identifier against sentences with known identifiers.

    python scripts/pii_eval.py

eval/pii_cases.json holds synthetic sentences with gold spans (every value is
invented) and a group of sentences with nothing to remove, including doses,
ages and ordinary words that look like names. The de-identifier is judged on
its output alone: where the text changed, and to which placeholder. A removed
span is correct when it overlaps a gold span of the same kind; a gold span is
found when a removed span of the same kind overlaps it. Reported per kind and
overall, with the 95th percentile time per sentence.

Targets: recall >= 0.95 and precision >= 0.97."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CASES = ROOT / "eval" / "pii_cases.json"
KIND = {"[NAME]": "NAME", "[DATE]": "DATE", "[NHS NUMBER]": "NHS NUMBER", "[SSN]": "SSN", "[ID]": "ID",
        "[PHONE]": "PHONE", "[EMAIL]": "EMAIL", "[URL]": "URL", "[IP ADDRESS]": "IP ADDRESS",
        "[POSTCODE]": "POSTCODE", "[ZIP]": "ZIP", "[ADDRESS]": "ADDRESS", "[NUMBER]": "NUMBER", "[CARD]": "CARD",
        "90": "AGE"}
TARGET_RECALL, TARGET_PRECISION = 0.95, 0.97


def removed_spans(original: str, cleaned: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) in the original for every placeholder in the output,
    found by locating the unchanged text between placeholders."""
    marks = "|".join(re.escape(p) for p in KIND if p != "90")
    if "90" not in original:
        marks += r"|(?<!\d)90(?!\d)"
    pieces = re.split(f"({marks})", cleaned)
    out, pos, pending = [], 0, None
    for n, piece in enumerate(pieces):
        if n % 2:
            pending = (pos, KIND[piece])
            continue
        start = pos + 1 if pending is not None else pos
        if n == len(pieces) - 1 and not piece:
            at = len(original)
        else:
            at = original.find(piece, start) if piece else start
            if at < 0:
                at = start
        if pending is not None:
            out.append((pending[0], at, pending[1]))
            pending = None
        pos = at + len(piece)
    if pending is not None:
        out.append((pending[0], len(original), pending[1]))
    return out


def evaluate(path: Path = CASES) -> dict:
    from app import deid, retrieval

    try:
        retrieval.load_index()
    except FileNotFoundError:
        pass
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    misses, extras = [], []
    times = []
    for case in cases:
        text = case["text"]
        t0 = time.perf_counter()
        cleaned = deid.deidentify(text, retrieval.is_known_term).text
        times.append((time.perf_counter() - t0) * 1000)
        pred = removed_spans(text, cleaned)
        gold = [(g["start"], g["end"], g["kind"]) for g in case["spans"]]
        for a, b, kind in pred:
            if any(k == kind and a < e and s < b for s, e, k in gold):
                tp[kind] += 1
            else:
                fp[kind] += 1
                extras.append({"text": text, "removed": kind, "output": cleaned})
        for s, e, kind in gold:
            if not any(k == kind and a < e and s < b for a, b, k in pred):
                fn[kind] += 1
                misses.append({"text": text, "missed": kind, "output": cleaned})
    kinds = sorted(set(tp) | set(fp) | set(fn))

    def prf(t, f, n):
        p = t / (t + f) if t + f else 1.0
        r = t / (t + n) if t + n else 1.0
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": t, "fp": f, "fn": n}

    per_kind = {k: prf(tp[k], fp[k], fn[k]) for k in kinds}
    overall = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    times.sort()
    return {
        "cases": len(cases), "spans": sum(len(c["spans"]) for c in cases),
        "ner": os.environ.get("DEID_NER", "auto"),
        "overall": overall, "per_kind": per_kind,
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 3), "mean_ms": round(sum(times) / len(times), 3),
        "targets_met": overall["recall"] >= TARGET_RECALL and overall["precision"] >= TARGET_PRECISION,
        "misses": misses[:25], "false_positives": extras[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()
    result = evaluate()
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{result['cases']} sentences, {result['spans']} gold spans, NER={result['ner']}\n")
    print(f"{'kind':14} {'precision':>9} {'recall':>7} {'f1':>6} {'tp':>4} {'fp':>4} {'fn':>4}")
    for kind, m in result["per_kind"].items():
        print(f"{kind:14} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
    o = result["overall"]
    print(f"{'OVERALL':14} {o['precision']:>9.3f} {o['recall']:>7.3f} {o['f1']:>6.3f} {o['tp']:>4} {o['fp']:>4} {o['fn']:>4}")
    print(f"\nlatency: mean {result['mean_ms']} ms, p95 {result['p95_ms']} ms")
    for m in result["misses"][:12]:
        print(f"  MISSED {m['missed']:10} {m['text']!r} -> {m['output']!r}")
    for m in result["false_positives"][:12]:
        print(f"  WRONG  {m['removed']:10} {m['text']!r} -> {m['output']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
