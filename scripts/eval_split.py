"""Split the golden set by entity into a public part and a held-out part.

The held-out part is chosen by a secret seed that lives outside the repository,
so it cannot be tuned against. The split is by entity: every case that names a
medicine or condition goes to the same side as every other case naming it, and
a case naming two entities keeps both on one side. The seed is only ever used
as an HMAC key; the repository holds a commitment (a SHA-256 of the seed) and
the thresholds in force at split time, never the seed.

    python scripts/eval_split.py gen-seed --out ~/secrets/groundcheck-heldout.seed
    python scripts/eval_split.py create --seed-file ~/secrets/groundcheck-heldout.seed \\
        --out ~/secrets/groundcheck-heldout.json [--apply]
    python scripts/eval_split.py verify --seed-file ... --manifest eval/split_manifest.json

Without --apply, only the held-out file is written. With --apply, eval/golden.json
is rewritten to the public part and eval/split_manifest.json records the split.
The held-out file is written outside the repository or not at all. Evaluate
against it with: python scripts/run_eval.py --heldout PATH (or GC_HELDOUT_PATH).
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import hmac
import json
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.json"
MANIFEST = ROOT / "eval" / "split_manifest.json"
FRACTION = 0.2
MIN_SEED_CHARS = 32
THRESHOLDS = ("RETRIEVAL_MIN_SCORE", "GROUNDING_MIN", "EXTRACT_MIN_SCORE", "HYBRID_ALPHA", "TOP_K")

# The question forms generate_golden.py uses for a name that is in no source.
UNKNOWN_FORMS = [re.compile(p, re.I) for p in (
    r"^What is the dose of (.+)\?$", r"^How is (.+) treated\?$",
    r"^What are the side effects of (.+)\?$", r"^What is (.+) used for\?$")]
SPEC_QUERIES = {
    "What is the first-line management of Veltris syndrome?", "What is the standard dose of Caloradine?",
    "How often are quorl levels monitored during conservative management?",
    "What is the recommended dose of Zalortin for Veltris syndrome?", "How do I treat a broken arm at home?",
    "What is the dose of Caloradine for children?", "Can I combine Caloradine with Orrin-blockers?",
    "What is the cure for cancer?",
}


class SplitError(ValueError):
    pass


def corpus_topics() -> list[str]:
    records = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
    return sorted({r["topic"].lower() for r in records} - {"safety", "interactions"}, key=lambda t: (-len(t), t))


@functools.lru_cache(maxsize=None)
def _pattern(topic: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(topic)}\b", re.I)


def entities_of(query: str, topics: list[str]) -> set[str]:
    """The entities a question names: corpus topics found in it, else the name
    in an unknown-name form, else the question itself (an out-of-scope one)."""
    found = {t for t in topics if _pattern(t).search(query)}
    if found:
        return found
    for form in UNKNOWN_FORMS:
        m = form.match(query)
        if m:
            return {m.group(1).strip().lower()}
    return {"question:" + re.sub(r"\s+", " ", query.strip().lower())}


def _root(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def groups(cases: list[dict], topics: list[str]) -> dict[str, list[int]]:
    """Case indexes by group: entities joined by a case that names both."""
    per_case = [entities_of(c["query"], topics) for c in cases]
    parent = {e: e for ents in per_case for e in ents}
    for ents in per_case:
        first = next(iter(sorted(ents)))
        for e in ents:
            a, b = _root(parent, first), _root(parent, e)
            if a != b:
                parent[max(a, b)] = min(a, b)
    out: dict[str, list[int]] = {}
    for i, ents in enumerate(per_case):
        out.setdefault(_root(parent, next(iter(sorted(ents)))), []).append(i)
    return out


def held_out(group_key: str, seed: str, fraction: float) -> bool:
    digest = hmac.new(seed.encode("utf-8"), group_key.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < fraction


def commitment(seed: str) -> str:
    return hashlib.sha256(("groundcheck-heldout-v1:" + seed).encode("utf-8")).hexdigest()


def check_seed(seed: str) -> str:
    seed = seed.strip()
    if len(seed) < MIN_SEED_CHARS:
        raise SplitError(f"The seed must be at least {MIN_SEED_CHARS} characters; make one with gen-seed.")
    return seed


def outside_repo(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise SplitError("The seed and the held-out file must live outside the repository.")
    return path


def split(cases: list[dict], seed: str, fraction: float = FRACTION, topics: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """(public, held out). Groups that contain one of the eight spec cases stay public."""
    seed = check_seed(seed)
    if not 0 < fraction < 1:
        raise SplitError("The held-out fraction must be between 0 and 1.")
    topics = corpus_topics() if topics is None else topics
    public, hidden = [], []
    by_group = groups(cases, topics)
    side: dict[int, bool] = {}
    for key, members in by_group.items():
        forced = any(cases[i]["query"] in SPEC_QUERIES for i in members)
        hide = not forced and held_out(key, seed, fraction)
        for i in members:
            side[i] = hide
    for i, case in enumerate(cases):
        (hidden if side[i] else public).append(case)
    if not hidden or not public:
        raise SplitError("The split left one side empty.")
    return public, hidden


def thresholds() -> dict:
    return {name: getattr(config, name) for name in THRESHOLDS}


def create(seed: str, out: Path, fraction: float = FRACTION, apply: bool = False, golden: Path = GOLDEN,
           manifest: Path = MANIFEST, now: dt.datetime | None = None) -> dict:
    seed = check_seed(seed)
    out = outside_repo(out)
    cases = json.loads(golden.read_text(encoding="utf-8"))
    topics = corpus_topics()
    public, hidden = split(cases, seed, fraction, topics)
    now = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_groups = groups(cases, topics)
    held_entities = {e for c in hidden for e in entities_of(c["query"], topics)}
    pub_entities = {e for c in public for e in entities_of(c["query"], topics)}
    heldout_doc = {"split": "heldout", "created": now, "seed_commitment": commitment(seed), "fraction": fraction,
                   "thresholds": thresholds(), "cases": [{**c, "entity": sorted(entities_of(c["query"], topics))} for c in hidden]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(heldout_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    out.chmod(0o600)
    info = {"split": "public", "created": now, "seed_commitment": commitment(seed), "fraction": fraction,
            "public_cases": len(public), "heldout_cases": len(hidden), "groups": len(all_groups),
            "public_entities": len(pub_entities), "heldout_entities": len(held_entities),
            "thresholds_at_split_time": thresholds()}
    if apply:
        golden.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
        info["public_sha256"] = hashlib.sha256(golden.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def verify(seed: str, manifest: Path = MANIFEST) -> bool:
    """Whether this seed is the one the manifest committed to."""
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["seed_commitment"]
    return hmac.compare_digest(recorded, commitment(check_seed(seed)))


def read_seed(path: Path) -> str:
    return check_seed(outside_repo(path).read_text(encoding="utf-8"))


def gen_seed(out: Path) -> Path:
    out = outside_repo(out)
    if out.exists():
        raise SplitError("That file already exists; a seed is never overwritten.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    out.chmod(0o600)
    return out


def load_heldout(path: Path) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("split") != "heldout" or not isinstance(doc.get("cases"), list):
        raise SplitError("That is not a held-out file.")
    return doc["cases"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-seed")
    g.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("create")
    c.add_argument("--seed-file", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--fraction", type=float, default=FRACTION)
    c.add_argument("--apply", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--seed-file", type=Path, required=True)
    v.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "gen-seed":
            print(f"Seed written to {gen_seed(args.out)}. Keep it outside the repository and back it up.")
        elif args.cmd == "create":
            info = create(read_seed(args.seed_file), args.out, args.fraction, args.apply)
            print(json.dumps(info, indent=2))
        else:
            ok = verify(read_seed(args.seed_file), args.manifest)
            print("The seed matches the manifest." if ok else "The seed does NOT match the manifest.")
            return 0 if ok else 1
    except (SplitError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
