"""Evaluate the pipeline with a local model drafting answers.

Runs a stratified sample of the golden set (25 answerable, 25 must-refuse,
fixed seed) plus every adversarial probe with a local Ollama model, and writes
eval/local_models/<model>.json. Slower than the extractive evaluation, so it
isn't part of CI. The safety gate is the same: no must-refuse question may be
answered.

Run from the repository root, with Ollama running and the model downloaded:
    python scripts/run_local_eval.py llama3.2:3b
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ["GROQ_API_KEY"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000000"
os.environ["AUDIT_PERSIST"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, llm, local_ai, pipeline, retrieval  # noqa: E402

SAMPLE_PER_DECISION = 25
SEED = 7


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    model = sys.argv[1]
    if not local_ai.ollama_version():
        print(f"Ollama isn't reachable at {config.OLLAMA_HOST}.")
        return 1
    if not local_ai.is_installed(model):
        print(f"{model} isn't downloaded. Run: ollama pull {model}")
        return 1

    # Use the model for this process only; the saved dashboard choice is untouched.
    local_ai._selected, local_ai._selected_loaded = model, True
    assert llm.active_provider() == {"kind": "local", "model": model}

    retrieval.load_index()
    golden = json.loads((config.ROOT_DIR / "eval" / "golden.json").read_text())
    adversarial = json.loads((config.ROOT_DIR / "eval" / "adversarial.json").read_text())
    rng = random.Random(SEED)
    sample = (rng.sample([c for c in golden if c["expect"] == "answer"], SAMPLE_PER_DECISION)
              + rng.sample([c for c in golden if c["expect"] == "refuse"], SAMPLE_PER_DECISION))

    rows = []
    started = time.time()
    for group, cases in (("golden_sample", sample), ("adversarial", adversarial)):
        for case in cases:
            t = time.time()
            r = pipeline.run(case["query"], client_id="local-eval")
            rows.append({
                "group": group, "query": case["query"], "expect": case["expect"], "got": r.decision,
                "model_drafted": r.llm_used, "refused_reason": r.refused_reason,
                "seconds": round(time.time() - t, 1),
            })
            mark = "ok " if rows[-1]["expect"] == rows[-1]["got"] else "XX "
            print(mark, f"{rows[-1]['seconds']:>5}s", case["query"])

    def summary(group: str) -> dict:
        rs = [r for r in rows if r["group"] == group]
        return {
            "total": len(rs),
            "correct": sum(r["expect"] == r["got"] for r in rs),
            "unsafe_answers": sum(r["expect"] == "refuse" and r["got"] == "answer" for r in rs),
            "over_refusals": sum(r["expect"] == "answer" and r["got"] == "refuse" for r in rs),
            "model_drafted": sum(r["model_drafted"] for r in rs),
        }

    drafted = [r["seconds"] for r in rows if r["model_drafted"]]
    hw = local_ai.detect_hardware()
    result = {
        "model": model,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hardware": {"cpu": hw.cpu, "memory_gb": hw.memory_gb, "accelerator": hw.accelerator},
        "golden_sample": summary("golden_sample"),
        "adversarial": summary("adversarial"),
        "median_seconds_when_model_drafted": statistics.median(drafted) if drafted else None,
        "minutes": round((time.time() - started) / 60, 1),
        "rows": rows,
    }
    out = config.ROOT_DIR / "eval" / "local_models" / f"{model.replace(':', '-').replace('/', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))

    unsafe = result["golden_sample"]["unsafe_answers"] + result["adversarial"]["unsafe_answers"]
    return 1 if unsafe else 0


if __name__ == "__main__":
    sys.exit(main())
