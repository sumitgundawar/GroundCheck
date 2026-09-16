"""Run the golden evaluation set through the pipeline in extractive mode and
write eval/eval_summary.json. Deterministic and needs no API key.

Run from the repository root:
    python scripts/run_eval.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

# Force extractive mode so the evaluation is deterministic and offline, and lift
# the per-process rate limit, which exists to protect a live endpoint and would
# otherwise throttle a batch run of thousands of cases.
os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"  # also ignores any selected local model
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
# Evaluation questions aren't real use: keep them out of the audit trail and
# the review queue.
os.environ["AUDIT_PERSIST"] = "false"
os.environ["REVIEW_QUEUE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app import pipeline  # noqa: E402

GOLDEN_PATH = config.ROOT_DIR / "eval" / "golden.json"
ADVERSARIAL_PATH = config.ROOT_DIR / "eval" / "adversarial.json"


def run_adversarial() -> dict:
    """Run the hand-written adversarial cases. These are curated probes (leading
    questions, injection, near-miss spellings, missing-context traps) with the
    intended decision. Reported honestly and separately; they do NOT gate the
    build, so a known limitation is shown rather than hidden."""
    if not ADVERSARIAL_PATH.exists():
        return {}
    with open(ADVERSARIAL_PATH, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    rows, passed = [], 0
    for c in cases:
        got = pipeline.run(c["query"]).decision
        ok = got == c["expect"]
        passed += int(ok)
        rows.append({"query": c["query"], "expect": c["expect"], "got": got,
                     "ok": ok, "note": c.get("note", "")})
    return {"total": len(cases), "passed": passed, "cases": rows}


def main() -> None:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        cases = json.load(fh)

    results = []
    passed = 0
    answerable_correct = 0
    must_refuse_correct = 0
    for case in cases:
        response = pipeline.run(case["query"])
        got = response.decision
        ok = got == case["expect"]
        passed += int(ok)
        if case["expect"] == "answer" and ok:
            answerable_correct += 1
        if case["expect"] == "refuse" and ok:
            must_refuse_correct += 1
        results.append({
            "query": case["query"],
            "expect": case["expect"],
            "got": got,
            "ok": ok,
        })

    total_answerable = sum(1 for c in cases if c["expect"] == "answer")
    total_refuse = sum(1 for c in cases if c["expect"] == "refuse")

    # The corpus and golden set are large, so store only a representative sample
    # of cases for the UI, plus the headline aggregates. The full run still gates
    # the build via the exit code below. We skip the leading canonical cases
    # (which use longer clinical phrasing) and show a diverse, plainly-worded mix
    # of answers and refusals, plus any failures.
    failing = [r for r in results if not r["ok"]]
    answers = [r for r in results if r["expect"] == "answer"]
    refuses = [r for r in results if r["expect"] == "refuse"]
    sample = answers[4:14] + refuses[4:14] + failing
    # De-duplicate by query, preserving order, and cap the list.
    seen, sample_unique = set(), []
    for r in sample:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        sample_unique.append(r)
    sample = sample_unique[:24]

    adversarial = run_adversarial()

    summary = {
        "total": len(cases),
        "passed": passed,
        "answerable_total": total_answerable,
        "must_refuse_total": total_refuse,
        "answerable_correct": answerable_correct,
        "must_refuse_correct": must_refuse_correct,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_cases": sample,
        "cases": sample,
        "adversarial": adversarial,
    }

    config.EVAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Evaluation: {passed}/{len(cases)} cases correct "
          f"({answerable_correct} answerable, {must_refuse_correct} must-refuse)")
    for r in results:
        if not r["ok"]:
            print(f"  XX  expect={r['expect']:<7} got={r['got']:<7} {r['query']}")
    if adversarial:
        print(f"Adversarial probes: {adversarial['passed']}/{adversarial['total']} "
              f"behaved as intended (diagnostic, non-gating).")
        for r in adversarial["cases"]:
            if not r["ok"]:
                print(f"  limitation: expect={r['expect']} got={r['got']} :: {r['query']}")

    # Build gate: fail only on the UNSAFE direction, a must-refuse case that was
    # answered. Over-refusing an answerable case is the safe direction and does
    # not fail the build; it is reported honestly in the summary instead.
    false_answers = total_refuse - must_refuse_correct
    over_refusals = total_answerable - answerable_correct
    if over_refusals:
        print(f"Note: {over_refusals} answerable case(s) were over-refused "
              f"(safe direction, not a build failure).")
    if false_answers:
        print(f"FAIL: {false_answers} must-refuse case(s) were answered "
              f"(unsafe direction).")
        sys.exit(1)


if __name__ == "__main__":
    main()
