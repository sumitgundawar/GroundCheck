#!/usr/bin/env python3
"""Run eval/verify_cases.json through the verifier and report how often it is wrong.

    python scripts/verify_eval.py [--cases PATH] [--limit N] [--show 20]

wrong_pass   an answer that is wrong (expect fail or not_pass) but was passed.
             The unsafe error. The target is zero, and any one fails the run.
false_fail   a correct answer (expect pass) that was failed. Target: 2% or less.
false_review a correct answer sent to a person instead of passed. Reported.
missed_fail  a case expected to fail that was sent to review: safe, but less
             precise than it could be. Reported.

Cases are built from the corpus by scripts/build_verify_cases.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.update({"GROQ_API_KEY": "", "FORCE_EXTRACTIVE": "true", "AUDIT_PERSIST": "false", "REVIEW_QUEUE": "false"})
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import verify  # noqa: E402
from app.schemas import PatientContext  # noqa: E402

CASES = ROOT / "eval" / "verify_cases.json"
SUMMARY = ROOT / "eval" / "verify_summary.json"
MAX_FALSE_FAIL = 0.02


def evaluate(cases: list[dict]) -> dict:
    rows, latencies = [], []
    for c in cases:
        patient = PatientContext(**c["patient"]) if c.get("patient") else None
        started = time.perf_counter()
        try:
            response, _ = verify.verify(c["question"], c["answer"], c.get("sources"), patient)
            got, code = response.decision, response.reason_code
            redacted = bool(response.redaction.kinds)
        except verify.SourceNotFound:
            got, code, redacted = "error", "source_not_found", False
        latencies.append((time.perf_counter() - started) * 1000)
        expect = c["expect"]
        wrong_pass = expect != "pass" and got == "pass"
        false_fail = expect == "pass" and got == "fail"
        false_review = expect == "pass" and got == "review"
        missed_fail = expect == "fail" and got == "review"
        redaction_missed = bool(c.get("expect_redaction")) and not redacted
        rows.append({"id": c["id"], "category": c["category"], "expect": expect, "got": got, "code": code,
                     "wrong_pass": wrong_pass, "false_fail": false_fail, "false_review": false_review,
                     "missed_fail": missed_fail, "redaction_missed": redaction_missed})
    latencies.sort()
    correct = sum(r["expect"] == "pass" for r in rows)
    by_category: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_category[r["category"]]["total"] += 1
        for key in ("wrong_pass", "false_fail", "false_review", "missed_fail", "redaction_missed"):
            by_category[r["category"]][key] += int(r[key])
    summary = {
        "cases": len(rows), "correct_answers": correct,
        "wrong_answers": len(rows) - correct,
        "wrong_pass": sum(r["wrong_pass"] for r in rows),
        "false_fail": sum(r["false_fail"] for r in rows),
        "false_fail_rate": round(sum(r["false_fail"] for r in rows) / correct, 4) if correct else 0.0,
        "false_review": sum(r["false_review"] for r in rows),
        "missed_fail": sum(r["missed_fail"] for r in rows),
        "redaction_missed": sum(r["redaction_missed"] for r in rows),
        "p50_ms": round(latencies[len(latencies) // 2], 1), "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
    }
    summary["targets_met"] = (summary["wrong_pass"] == 0 and summary["false_fail_rate"] <= MAX_FALSE_FAIL
                              and summary["redaction_missed"] == 0)
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.limit:
        cases = cases[:args.limit]
    result = evaluate(cases)
    s, rows = result["summary"], result["rows"]
    by_id = {c["id"]: c for c in cases}
    print(f"{s['cases']} cases: wrong_pass={s['wrong_pass']} false_fail={s['false_fail']} "
          f"({s['false_fail_rate']:.2%} of {s['correct_answers']} correct) false_review={s['false_review']} "
          f"missed_fail={s['missed_fail']} redaction_missed={s['redaction_missed']} "
          f"p50={s['p50_ms']}ms p95={s['p95_ms']}ms")
    for key in ("wrong_pass", "false_fail", "redaction_missed"):
        bad = [r for r in rows if r[key]]
        for r in bad[:args.show]:
            c = by_id[r["id"]]
            print(f"  {key} {r['id']} [{r['category']}] expect={r['expect']} got={r['got']}/{r['code']}\n"
                  f"     Q: {c['question']}\n     A: {c['answer'][:300]}")
    if not args.no_write and not args.limit:
        SUMMARY.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    print("targets met" if s["targets_met"] else "TARGETS NOT MET")
    return 0 if s["targets_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
