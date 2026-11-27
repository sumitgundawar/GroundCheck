#!/usr/bin/env python3
"""Run eval/proposal_cases.json (questions that propose a regimen) through the
pipeline in extractive mode and write eval/proposal_summary.json.

    python scripts/proposal_eval.py [--family tuned|heldout] [--show N]

Hard failures, each of which fails the run:
  wrong_answer    a mismatching proposal was answered without a leading "No"
  wrong_no        a matching proposal was answered "No"
  guessed         an unverifiable proposal was answered
  patient         a patient the sources do not cover was given an answer
Reported, not failed: a proposal refused where a verdict was expected (a
stricter guard got there first or the wording was not read), by reason code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
os.environ["AUDIT_PERSIST"] = "false"
os.environ["REVIEW_QUEUE"] = "false"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import pipeline  # noqa: E402
from app.schemas import PatientContext  # noqa: E402

CASES = ROOT / "eval" / "proposal_cases.json"
SUMMARY = ROOT / "eval" / "proposal_summary.json"


def judge(case: dict, r) -> tuple[str, str]:
    """(outcome, failure) where failure is '' unless a hard failure."""
    expect = case["expect"]
    answered = r.decision == "answer"
    text = r.answer_text
    said = "no" if text.startswith("No ") else "yes" if text.startswith("Yes ") else ""
    check = r.proposal_check
    if expect == "none":
        return ("answered" if answered else "refused"), ("proposal check on a plain question" if check else "")
    if expect == "match":
        if answered and said == "no":
            return "no", "wrong_no"
        if answered and said == "yes" and check and check.status == "match":
            return "yes", ""
        return ("answered_unmarked" if answered else "refused:" + str(r.reason_code)), ("wrong_answer" if answered else "")
    if expect == "mismatch":
        if answered:
            return ("no", "") if said == "no" and check and check.status == "mismatch" else ("unmarked", "wrong_answer")
        return "refused:" + str(r.reason_code), ""
    if expect in ("unverifiable", "unverifiable_or_refuse", "refuse_any"):
        if answered:
            if expect == "unverifiable_or_refuse" and said == "no" and check and check.status == "mismatch":
                return "no", ""
            return "answered", "guessed"
        return "refused:" + str(r.reason_code), ""
    if expect == "refuse_patient":
        return ("answered", "patient") if answered else ("refused:" + str(r.reason_code), "")
    if expect == "match_or_patient":
        if answered and said == "no":
            return "no", "wrong_no"
        return ("yes", "") if answered else ("refused:" + str(r.reason_code), "")
    if expect == "mismatch_or_patient":
        if answered and said != "no":
            return "unmarked", "wrong_answer"
        return ("no", "") if answered else ("refused:" + str(r.reason_code), "")
    return "?", f"unknown expectation {expect}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["tuned", "heldout"])
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    cases = json.loads(CASES.read_text())["cases"]
    if args.family:
        cases = [c for c in cases if c["family"] == args.family]
    table: dict = defaultdict(Counter)
    failures, over_refused = [], []
    for c in cases:
        patient = PatientContext(**c["patient"]) if c.get("patient") else None
        r = pipeline.run(c["query"], patient=patient, review=False)
        outcome, failure = judge(c, r)
        key = f'{c["family"]}/{c["kind"]}/{c["expect"]}'
        table[key][outcome] += 1
        if failure:
            failures.append({"id": c["id"], "query": c["query"], "expect": c["expect"], "failure": failure,
                             "answer": r.answer_text[:200]})
        elif c["expect"] in ("match", "match_or_patient") and outcome.startswith("refused"):
            over_refused.append({"id": c["id"], "query": c["query"], "outcome": outcome})
    kinds = Counter(f["failure"] for f in failures)
    total = len(cases)
    summary = {
        "total": total, "failures": dict(kinds), "wrong_answer": kinds.get("wrong_answer", 0),
        "wrong_no": kinds.get("wrong_no", 0), "guessed": kinds.get("guessed", 0), "patient": kinds.get("patient", 0),
        "over_refused_matches": len(over_refused),
        "by_group": {k: dict(v) for k, v in sorted(table.items())},
    }
    print(f"{total} cases: wrong_answer {summary['wrong_answer']}, wrong_no {summary['wrong_no']}, "
          f"guessed {summary['guessed']}, patient {summary['patient']}, matches refused {len(over_refused)}")
    for key, counts in sorted(table.items()):
        print(f"  {key:60s} {dict(counts)}")
    for f in failures[:args.show]:
        print("FAIL", f)
    for f in over_refused[:args.show]:
        print("REFUSED MATCH", f)
    if not args.no_write and not args.family:
        SUMMARY.write_text(json.dumps(summary, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
