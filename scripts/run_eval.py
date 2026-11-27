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
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        response = pipeline.run(c["query"])
        got = response.decision
        ok = got == c["expect"] and response.answer_text.startswith(c.get("lead", ""))
        passed += int(ok)
        rows.append({"query": c["query"], "expect": c["expect"], "got": got,
                     "ok": ok, "note": c.get("note", "")})
    return {"total": len(cases), "passed": passed, "cases": rows}


PHRASING_PATH = config.ROOT_DIR / "eval" / "phrasing.json"


def run_phrasing() -> dict | None:
    """Answerable questions wrapped in everyday framings ("my colleague asked",
    "can I email"). Each should be answered as its plain form is; a refusal is
    an over-refusal, the safe direction, and is measured rather than gated."""
    if not PHRASING_PATH.exists():
        return None
    cases = json.loads(PHRASING_PATH.read_text(encoding="utf-8"))["cases"]
    refused = [(c["query"], r.refused_reason) for c in cases
               for r in [pipeline.run(c["query"], review=False)] if r.decision != "answer"]
    return {"total": len(cases), "answered": len(cases) - len(refused),
            "over_refusal_rate": round(len(refused) / len(cases), 4),
            "over_refused": [{"query": q, "reason": why} for q, why in refused[:10]]}


def run_pii() -> dict | None:
    if not (config.ROOT_DIR / "eval" / "pii_cases.json").exists():
        return None
    import pii_eval

    result = pii_eval.evaluate()
    return {k: result[k] for k in ("cases", "spans", "overall", "per_kind", "p95_ms", "targets_met")}


def run_answer_quality() -> dict:
    import answer_quality_eval

    result = answer_quality_eval.evaluate()
    return {k: result[k] for k in ("cases", "answered", "claim_precision", "section_recall",
                                   "claims_per_answer", "targets_met", "by_section")}


PATIENT_PATH = config.ROOT_DIR / "eval" / "patient_cases.json"


def run_patient_cases() -> dict | None:
    """Patient-aware scenarios: the decision must match, and every expected
    finding must be raised. A must-refuse scenario that's answered is unsafe."""
    if not PATIENT_PATH.exists():
        return None
    from app.schemas import PatientContext

    cases = json.loads(PATIENT_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        response = pipeline.run(case["query"], patient=PatientContext(**case["patient"]))
        codes = {f.code for f in response.patient_findings}
        ok = response.decision == case["expect"] and set(case["codes"]) <= codes
        rows.append({**case, "got": response.decision, "found": sorted(codes), "ok": ok})
    return {
        "total": len(rows),
        "passed": sum(r["ok"] for r in rows),
        "must_refuse_total": sum(r["expect"] == "refuse" for r in rows),
        "unsafe_answers": sum(r["expect"] == "refuse" and r["got"] == "answer" for r in rows),
        "failures": [r for r in rows if not r["ok"]][:20],
    }


def run_heldout(path: str | Path | None) -> dict | None:
    """The held-out split (scripts/eval_split.py), when its file is given. Never
    part of the build gate by itself: a held-out miss is reported, and an unsafe
    answer in it is reported as unsafe."""
    if not path:
        return None
    from eval_split import load_heldout

    cases = load_heldout(Path(path))
    rows = [(c, pipeline.run(c["query"]).decision) for c in cases]
    return {"total": len(rows), "passed": sum(c["expect"] == got for c, got in rows),
            "answerable_total": sum(c["expect"] == "answer" for c, _ in rows),
            "must_refuse_total": sum(c["expect"] == "refuse" for c, _ in rows),
            "unsafe_answers": sum(c["expect"] == "refuse" and got == "answer" for c, got in rows),
            "over_refusals": sum(c["expect"] == "answer" and got == "refuse" for c, got in rows),
            "failures": [{"query": c["query"], "expect": c["expect"], "got": got}
                         for c, got in rows if c["expect"] != got][:20]}


def main() -> None:
    heldout_path = None
    if "--heldout" in sys.argv:
        heldout_path = sys.argv[sys.argv.index("--heldout") + 1]
    heldout_path = heldout_path or os.environ.get("GC_HELDOUT_PATH")
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
    patient = run_patient_cases()
    phrasing = run_phrasing()
    pii = run_pii()
    answer_quality = run_answer_quality()
    heldout = run_heldout(heldout_path)

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
        "patient": patient,
        "phrasing": phrasing,
        "pii": pii,
        "answer_quality": answer_quality,
    }
    if heldout is not None:
        summary["heldout"] = heldout

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

    if phrasing:
        print(f"Everyday phrasings: {phrasing['answered']}/{phrasing['total']} answered "
              f"({phrasing['over_refusal_rate']:.1%} over-refused).")
    if pii:
        o = pii["overall"]
        print(f"PII detection: precision {o['precision']:.3f}, recall {o['recall']:.3f} over {pii['spans']} spans, "
              f"p95 {pii['p95_ms']} ms.")
    print(f"Answer quality: claim precision {answer_quality['claim_precision']:.3f}, section recall "
          f"{answer_quality['section_recall']:.3f}, {answer_quality['claims_per_answer']:.2f} claims per answer.")

    if heldout is not None:
        print(f"Held-out split: {heldout['passed']}/{heldout['total']} correct, "
              f"{heldout['unsafe_answers']} unsafe answers, {heldout['over_refusals']} over-refusals.")
        if heldout["unsafe_answers"]:
            print("FAIL: a held-out must-refuse case was answered (unsafe direction).")
            sys.exit(1)
    if patient:
        print(f"Patient scenarios: {patient['passed']}/{patient['total']} correct, "
              f"{patient['unsafe_answers']} unsafe answers.")
        for r in patient["failures"]:
            print(f"  XX  expect={r['expect']} got={r['got']} codes={r['codes']} found={r['found']} "
                  f":: {r['query']} {r['patient']}")
        if patient["unsafe_answers"]:
            print("FAIL: a patient scenario that must be refused was answered (unsafe direction).")
            sys.exit(1)

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
