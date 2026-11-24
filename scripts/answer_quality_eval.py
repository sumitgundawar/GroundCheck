#!/usr/bin/env python3
"""Measure whether the extractive answers cite the passages that answer.

    python scripts/answer_quality_eval.py            # every answerable golden question
    python scripts/answer_quality_eval.py --sample 300

Every answerable golden question was built from a template that names the
corpus section holding its answer ("What are the symptoms of X?" is answered by
X's Symptoms passage). So the gold set for a question is the passages of that
topic and section, and the check needs no labelling:

  claim precision   share of claims that cite a gold passage
  section recall    share of gold passages that some claim cites
  claims per answer mean number of claims in an answered question

Targets: precision >= 0.90, recall >= 0.95, at most 2.0 claims per answer."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
os.environ.setdefault("AUDIT_PERSIST", "false")
os.environ.setdefault("REVIEW_QUEUE", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TARGETS = {"precision": 0.90, "recall": 0.95, "claims": 2.0}


def _templates() -> list[tuple[str, re.Pattern]]:
    from generate_golden import ANSWER_TEMPLATES

    out = []
    for forms in ANSWER_TEMPLATES.values():
        for section, template in forms:
            pattern = re.escape(template).replace(re.escape("{name}"), r"(?P<name>.+)")
            out.append((section, re.compile(f"^{pattern}$")))
    return out


def evaluate(sample: int = 0, seed: int = 20260924) -> dict:
    from app import pipeline, retrieval

    retrieval.load_index()
    records = retrieval.all_metadata()
    by_topic_section: dict[tuple[str, str], set[str]] = {}
    for r in records:
        by_topic_section.setdefault((r["topic"].lower(), r.get("section", "")), set()).add(r["id"])
    templates = _templates()
    cases = []
    for case in json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))[8:]:
        if case["expect"] != "answer":
            continue
        for section, pattern in templates:
            m = pattern.match(case["query"])
            if m:
                gold = by_topic_section.get((m.group("name").lower(), section))
                if gold:
                    cases.append({"query": case["query"], "section": section, "gold": gold})
                break
    if sample and sample < len(cases):
        cases = random.Random(seed).sample(cases, sample)

    claims_total = relevant = answered = 0
    gold_total = gold_found = 0
    refused, wrong = [], []
    by_section: dict[str, Counter] = {}
    for case in cases:
        response = pipeline.run(case["query"], review=False)
        row = by_section.setdefault(case["section"], Counter())
        row["cases"] += 1
        gold_total += len(case["gold"])
        if response.decision != "answer":
            refused.append(case["query"])
            continue
        answered += 1
        cited = {sid for c in response.claims for sid in c.source_ids}
        gold_found += len(cited & case["gold"])
        row["found"] += len(cited & case["gold"])
        row["gold"] += len(case["gold"])
        for c in response.claims:
            claims_total += 1
            if set(c.source_ids) & case["gold"]:
                relevant += 1
                row["relevant"] += 1
            elif len(wrong) < 15:
                wrong.append({"query": case["query"], "wanted": case["section"], "cited": c.source_ids,
                              "claim": c.text[:110]})
            row["claims"] += 1
    return {
        "cases": len(cases), "answered": answered,
        "claim_precision": round(relevant / claims_total, 4) if claims_total else 0.0,
        "section_recall": round(gold_found / gold_total, 4) if gold_total else 0.0,
        "claims_per_answer": round(claims_total / answered, 3) if answered else 0.0,
        "targets_met": bool(claims_total and relevant / claims_total >= TARGETS["precision"]
                            and gold_found / gold_total >= TARGETS["recall"]
                            and claims_total / max(answered, 1) <= TARGETS["claims"]),
        "by_section": {k: dict(v) for k, v in sorted(by_section.items())},
        "refused": refused[:10], "imprecise_examples": wrong,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.sample)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"{result['cases']} questions, {result['answered']} answered")
    print(f"claim precision   {result['claim_precision']:.3f}  (target >= {TARGETS['precision']})")
    print(f"section recall    {result['section_recall']:.3f}  (target >= {TARGETS['recall']})")
    print(f"claims per answer {result['claims_per_answer']:.2f}  (target <= {TARGETS['claims']})")
    for section, row in result["by_section"].items():
        p = row.get("relevant", 0) / row["claims"] if row.get("claims") else 0
        r = row.get("found", 0) / row["gold"] if row.get("gold") else 0
        print(f"  {section:28} cases {row['cases']:4}  precision {p:.2f}  recall {r:.2f}")
    for w in result["imprecise_examples"][:8]:
        print(f"  off-section: {w['query']!r} wanted {w['wanted']}, cited {w['cited']}: {w['claim']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
