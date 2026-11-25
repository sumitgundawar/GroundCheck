"""Evaluate language handling and write eval/language_summary.json.

  detector  eval/language_cases.json: refusal of unsupported languages, and
            English never taken for another language (also checked against
            the golden and phrasing questions).
  spanish   eval/golden_es.json through the whole pipeline with Spanish
            configured: no must-refuse case answered, and the answer rate.

Offline and deterministic, like scripts/run_eval.py.

    python scripts/language_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
os.environ["AUDIT_PERSIST"] = "false"
os.environ["REVIEW_QUEUE"] = "false"
os.environ["SUPPORTED_LANGUAGES"] = "en,es"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, language, pipeline  # noqa: E402

ROOT = config.ROOT_DIR


def detector() -> dict:
    cases = json.loads((ROOT / "eval" / "language_cases.json").read_text(encoding="utf-8"))["cases"]
    per: dict[str, Counter] = {}
    wrong = []
    english_refused = []
    for c in cases:
        got = language.detect(c["text"]).lang
        if c["lang"] == "en":
            ok = got in ("en", "unknown")
            if not ok:
                english_refused.append(c["text"])
        else:
            ok = got == c["lang"]
        per.setdefault(c["lang"], Counter())["right" if ok else "wrong"] += 1
        if not ok:
            wrong.append({"text": c["text"], "expected": c["lang"], "got": got})
    # English-only installation: what is refused.
    refused_foreign = sum(1 for c in cases if c["lang"] != "en" and language.detect(c["text"]).lang not in ("en", "unknown"))
    foreign = sum(1 for c in cases if c["lang"] != "en")
    golden = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    phrasing = json.loads((ROOT / "eval" / "phrasing.json").read_text(encoding="utf-8"))["cases"]
    queries = [c["query"] for c in golden] + [c["query"] for c in phrasing]
    false_refusals = [q for q in queries if language.detect(q).lang not in ("en", "unknown")]
    return {
        "cases": len(cases), "foreign_cases": foreign, "foreign_refused": refused_foreign,
        "foreign_refusal_rate": round(refused_foreign / max(1, foreign), 4),
        "english_cases": sum(1 for c in cases if c["lang"] == "en"), "english_false_refusals": len(english_refused),
        "golden_and_phrasing_questions": len(queries), "golden_and_phrasing_false_refusals": len(false_refusals),
        "per_language": {k: dict(v) for k, v in sorted(per.items())}, "misses": wrong,
    }


def spanish() -> dict:
    cases = json.loads((ROOT / "eval" / "golden_es.json").read_text(encoding="utf-8"))["cases"]
    pipeline.clear_answer_cache()
    rows, unsafe, by_family = [], [], {}
    for c in cases:
        got = pipeline.run(c["query"]).decision
        expect = c["expect"]
        if expect == "twin":
            expect = pipeline.run(c["en"]).decision
        stats = by_family.setdefault(c["family"], Counter())
        stats["cases"] += 1
        stats[f"expected_{expect}"] += 1
        if expect == "refuse" and got == "answer":
            unsafe.append(c["query"])
            stats["unsafe"] += 1
        if expect == "answer":
            stats["answered" if got == "answer" else "refused_answerable"] += 1
        else:
            stats["refused" if got == "refuse" else "answered_wrongly"] += 1
        rows.append((expect, got))
    answerable = sum(1 for e, _ in rows if e == "answer")
    answered = sum(1 for e, g in rows if e == "answer" and g == "answer")
    return {
        "cases": len(cases), "answerable": answerable, "must_refuse": len(rows) - answerable,
        "unsafe": len(unsafe), "unsafe_queries": unsafe,
        "answer_rate": round(answered / max(1, answerable), 4),
        "answer_rate_held_out": round(by_family.get("held-out free-form", Counter())["answered"]
                                      / max(1, by_family.get("held-out free-form", Counter())["expected_answer"]), 4),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
    }


def main() -> int:
    summary = {"detector": detector(), "spanish": spanish()}
    (ROOT / "eval" / "language_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
    d, s = summary["detector"], summary["spanish"]
    print(f"detector: {d['cases']} strings, foreign refused {d['foreign_refused']}/{d['foreign_cases']}, "
          f"English false refusals {d['english_false_refusals']}, golden+phrasing false refusals "
          f"{d['golden_and_phrasing_false_refusals']}/{d['golden_and_phrasing_questions']}")
    print(f"spanish: {s['cases']} cases, unsafe {s['unsafe']}, answer rate {s['answer_rate']:.1%} "
          f"({s['answerable']} answerable, {s['must_refuse']} must refuse); held-out wording {s['answer_rate_held_out']:.1%}")
    for family, stats in s["by_family"].items():
        print("  ", family, dict(stats))
    bad = d["english_false_refusals"] or d["golden_and_phrasing_false_refusals"] or s["unsafe"] or d["foreign_refusal_rate"] < 0.98
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
