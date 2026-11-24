"""SOUP list (IEC 62304) and machine-learning model register.

Builds the list of third-party software in use from requirements.txt and either
a CycloneDX SBOM (--sbom sbom.json) or the installed environment, adds what each
direct dependency is used for from compliance/soup.yaml, and checks the model
register compliance/models.yaml.

Fails (exit status 1) when:
- a dependency has no licence, in the SBOM or package metadata or compliance/soup.yaml;
- a direct dependency has no entry in compliance/soup.yaml, or a requirement is not installed or in the SBOM;
- a model in app/config.py or the Local AI catalogue is missing from the model register, or an entry lacks a field.

Warnings (printed, exit 0): a copyleft licence that compliance/soup.yaml doesn't mark as reviewed, and
notes for packages that are no longer required.

    python scripts/soup_list.py --sbom sbom.json --markdown build/soup.md --json build/soup.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
SOUP_NOTES = ROOT / "compliance" / "soup.yaml"
MODELS = ROOT / "compliance" / "models.yaml"
COPYLEFT = re.compile(r"(?<![a-z])[al]?gpl|(?<![a-z])(?:mpl|epl|cddl|eupl|osl|sspl)(?![a-z])|cc-by-sa|general public licen[cs]e|"
                      r"mozilla public|eclipse public", re.IGNORECASE)
MODEL_FIELDS = ("id", "kind", "name", "version", "provider", "licence", "intended_use", "not_for", "notes")
MODEL_KINDS = {"embedding", "generation", "judge", "ner", "imaging"}
CLASSIFIER_LICENCES = {"MIT License": "MIT", "BSD License": "BSD", "Apache Software License": "Apache-2.0",
                       "ISC License (ISCL)": "ISC", "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
                       "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0", "Python Software Foundation License": "PSF-2.0",
                       "The Unlicense (Unlicense)": "Unlicense", "Zope Public License": "ZPL-2.1",
                       "GNU General Public License v3 (GPLv3)": "GPL-3.0",
                       "GNU Affero General Public License v3": "AGPL-3.0", "Public Domain": "Public Domain"}


class CheckError(Exception):
    pass


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise CheckError("PyYAML is needed (pip install pyyaml).") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CheckError(f"Can't read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise CheckError(f"{path} isn't valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckError(f"{path} must be a mapping.")
    return data


# --- Requirements -------------------------------------------------------------------------------------

def read_requirements(path: Path = REQUIREMENTS) -> list[dict]:
    """The direct dependencies: [{"name", "spec", "extras"}]. Comment lines and blank lines are skipped."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-") or re.match(r"^(git\+|https?://)", line):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?\s*(.*)$", line)
        if not match:
            raise CheckError(f"Can't read requirement {line!r}.")
        name, extras, spec = match.groups()
        out.append({"name": name, "spec": spec.strip(), "extras": [e.strip() for e in (extras or "").split(",") if e.strip()]})
    return out


# --- Licences -----------------------------------------------------------------------------------------

def licence_from_metadata(meta) -> str:
    """A licence string from a distribution's metadata: the SPDX expression, a short License field, or a classifier."""
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    field = (meta.get("License") or "").strip()
    if field and field.upper() != "UNKNOWN" and len(field) <= 100 and "\n" not in field:
        return field
    found = []
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            last = classifier.split("::")[-1].strip()
            found.append(CLASSIFIER_LICENCES.get(last, last))
    found = [f for f in found if f not in ("OSI Approved",)]
    return " OR ".join(dict.fromkeys(found))


def licence_from_sbom(component: dict) -> str:
    parts = []
    for entry in component.get("licenses") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("expression"):
            parts.append(str(entry["expression"]))
        elif isinstance(entry.get("license"), dict):
            lic = entry["license"]
            parts.append(str(lic.get("id") or lic.get("name") or ""))
    return " AND ".join(p for p in parts if p)


def from_sbom(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckError(f"Can't read the SBOM {path}: {exc}") from exc
    components = data.get("components") if isinstance(data, dict) else None
    if not isinstance(components, list):
        raise CheckError("The SBOM has no `components` list (expected CycloneDX JSON).")
    out = {}
    for c in components:
        if isinstance(c, dict) and c.get("name"):
            out[normalise(str(c["name"]))] = {"name": str(c["name"]), "version": str(c.get("version") or ""),
                                              "licence": licence_from_sbom(c), "source": "SBOM"}
    return out


def from_environment(direct: list[dict]) -> dict[str, dict]:
    """The direct dependencies and everything they need that is installed, with metadata licences."""
    out: dict[str, dict] = {}
    try:
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover
        Requirement = None
    queue = [(d["name"], tuple(d["extras"])) for d in direct]
    while queue:
        name, extras = queue.pop()
        key = normalise(name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen_extras = set(out.get(key, {}).get("extras", ()))
        if key in out and set(extras) <= seen_extras:
            continue
        out[key] = {"name": dist.metadata["Name"] or name, "version": dist.version, "licence": licence_from_metadata(dist.metadata),
                    "source": "environment", "extras": sorted(seen_extras | set(extras))}
        for req in dist.requires or []:
            if Requirement is None:
                if "extra ==" in req:
                    continue
                queue.append((re.split(r"[ ;<>=!~\[(]", req, 1)[0], ()))
                continue
            try:
                r = Requirement(req)
            except Exception:  # noqa: BLE001 - an unparseable requirement is skipped, not fatal
                continue
            if r.marker is not None and not any(r.marker.evaluate({"extra": e}) for e in (list(extras) or [""])):
                continue
            queue.append((r.name, tuple(r.extras)))
    return out


# --- The SOUP list --------------------------------------------------------------------------------------

def build_soup(direct: list[dict], installed: dict[str, dict], notes: dict) -> dict:
    """Pure: the list and what is wrong with it."""
    problems: list[str] = []
    warnings: list[str] = []
    entries_notes = {normalise(k): v for k, v in (notes.get("direct") or {}).items()}
    overrides = {normalise(k): v for k, v in (notes.get("licence_overrides") or {}).items()}
    reviewed = {normalise(k): v for k, v in (notes.get("reviewed_licences") or {}).items()}
    direct_keys = {normalise(d["name"]): d for d in direct}
    rows = []
    for key in sorted(set(installed) | set(direct_keys)):
        item = installed.get(key)
        d = direct_keys.get(key)
        if item is None:
            problems.append(f"{d['name']}: required but not found in the {'SBOM' if installed else 'installed environment'}, so its "
                            "version and licence are unknown.")
            item = {"name": d["name"], "version": "", "licence": "", "source": "missing"}
        licence = (overrides.get(key) or {}).get("licence") or item["licence"]
        if not licence:
            problems.append(f"{item['name']}: no licence found (add one under licence_overrides in compliance/soup.yaml if it has one).")
        elif COPYLEFT.search(licence) and key not in reviewed:
            warnings.append(f"{item['name']}: licence {licence} needs review (add it under reviewed_licences in compliance/soup.yaml).")
        note = entries_notes.get(key)
        if d is not None and not (note and str(note.get("use", "")).strip()):
            problems.append(f"{d['name']}: a direct dependency with no intended-use entry in compliance/soup.yaml.")
        rows.append({"name": item["name"], "version": item["version"], "licence": licence or "UNKNOWN", "direct": d is not None,
                     "spec": d["spec"] if d else "", "use": (note or {}).get("use", "") if d is not None else "",
                     "failure": (note or {}).get("failure", "") if d is not None else "", "source": item["source"],
                     "reviewed": reviewed.get(key, "")})
    for key in entries_notes:
        if key not in direct_keys:
            warnings.append(f"compliance/soup.yaml describes {key}, which is no longer in requirements.txt.")
    return {"rows": rows, "problems": problems, "warnings": warnings}


# --- Model register -----------------------------------------------------------------------------------------

def load_models(path: Path = MODELS) -> list[dict]:
    data = _yaml(path)
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise CheckError("The model register needs a `models:` list.")
    return models


def check_models(models: list[dict], config_models: dict[str, str], catalogue: list[str]) -> list[str]:
    """Problems with the register: fields, duplicates, and models in use that aren't listed."""
    problems: list[str] = []
    ids: set[str] = set()
    names: set[str] = set()
    for m in models:
        if not isinstance(m, dict):
            problems.append(f"A model entry must be a mapping, not {m!r}.")
            continue
        label = str(m.get("id") or "(no id)")
        for field in MODEL_FIELDS:
            if not isinstance(m.get(field), str) or not m[field].strip():
                problems.append(f"Model {label}: missing `{field}`.")
        if not (isinstance(m.get("kind"), str) and m["kind"] in MODEL_KINDS):
            problems.append(f"Model {label}: kind must be one of {', '.join(sorted(MODEL_KINDS))}.")
        if not isinstance(m.get("data_leaves_machine"), bool):
            problems.append(f"Model {label}: `data_leaves_machine` must be true or false.")
        if str(m.get("id")) in ids:
            problems.append(f"Model {label} is listed twice.")
        ids.add(str(m.get("id")))
        names.add(str(m.get("name", "")))
        if m.get("kind") == "generation" and m.get("data_leaves_machine") is True and "de-identified" not in str(m.get("intended_use", "")):
            problems.append(f"Model {label}: a model that data leaves the machine for must say its input is de-identified.")
    for key, name in config_models.items():
        if name and name not in names:
            problems.append(f"The default {key} model {name!r} (app/config.py) isn't in the model register.")
    for tag in catalogue:
        if tag not in names:
            problems.append(f"The Local AI catalogue model {tag!r} isn't in the model register.")
    return problems


def config_and_catalogue() -> tuple[dict[str, str], list[str]]:
    sys.path.insert(0, str(ROOT))
    from app import config, local_ai

    return ({"EMBED_MODEL": config.EMBED_MODEL, "GEN_MODEL": config.GEN_MODEL, "JUDGE_MODEL": config.JUDGE_MODEL},
            [m.name for m in local_ai.CATALOGUE])


def unregistered_library_models(models: list[dict], library: Path = ROOT / "models" / "library") -> list[str]:
    """Trained models present on disk that the register doesn't mention (informational: libraries are local)."""
    if not library.is_dir():
        return []
    text = json.dumps(models).lower()
    out = []
    for card in sorted(library.glob("*/model.json")):
        try:
            name = str(json.loads(card.read_text(encoding="utf-8")).get("name", "")).lower()
        except (OSError, ValueError):
            continue
        if name and name.split(" (")[0] not in text:
            out.append(name)
    return out


# --- Output ------------------------------------------------------------------------------------------------------

def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def to_markdown(soup: dict, models: list[dict]) -> str:
    rows = soup["rows"]
    direct = [r for r in rows if r["direct"]]
    out = ["# Third-party software (SOUP) and model register", "",
           "Generated by `scripts/soup_list.py`. Versions are those in the SBOM or environment it was run against. "
           "Known vulnerabilities are checked separately by `pip-audit`.", "",
           f"## Direct dependencies ({len(direct)})", "", "| Package | Version | Licence | Intended use | If it fails |",
           "| --- | --- | --- | --- | --- |"]
    out += [f"| {r['name']} | {r['version']} | {_cell(r['licence'])} | {_cell(r['use'])} | {_cell(r['failure'])} |" for r in direct]
    out += ["", f"## Everything else ({len(rows) - len(direct)})", "", "| Package | Version | Licence |", "| --- | --- | --- |"]
    out += [f"| {r['name']} | {r['version']} | {_cell(r['licence'])} |" for r in rows if not r["direct"]]
    out += ["", f"## Models ({len(models)})", "",
            "| Id | Kind | Name | Version | Licence | Intended use | Not for | Data leaves the machine |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    out += [f"| {m['id']} | {m['kind']} | {_cell(m['name'])} | {_cell(m['version'])} | {_cell(m['licence'])} | "
            f"{_cell(m['intended_use'])} | {_cell(m['not_for'])} | {'yes' if m.get('data_leaves_machine') else 'no'} |" for m in models]
    return "\n".join(out) + "\n"


def to_csv(soup: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["name", "version", "licence", "direct", "intended_use", "failure_effect", "source"])
    for r in soup["rows"]:
        writer.writerow([r["name"], r["version"], r["licence"], "yes" if r["direct"] else "no", r["use"], r["failure"], r["source"]])
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS)
    parser.add_argument("--sbom", type=Path, help="CycloneDX JSON; without it the installed environment is used")
    parser.add_argument("--notes", type=Path, default=SOUP_NOTES)
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    try:
        direct = read_requirements(args.requirements)
        installed = from_sbom(args.sbom) if args.sbom else from_environment(direct)
        soup = build_soup(direct, installed, _yaml(args.notes))
        models = load_models(args.models)
        config_models, catalogue = config_and_catalogue()
        model_problems = check_models(models, config_models, catalogue)
    except CheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    problems = soup["problems"] + model_problems
    for path, content in ((args.markdown, to_markdown(soup, models)),
                          (args.json, json.dumps({"soup": soup["rows"], "models": models}, indent=1, default=str) + "\n"),
                          (args.csv, to_csv(soup))):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if not (args.markdown or args.json or args.csv):
        print(to_markdown(soup, models))
    for w in soup["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    for name in unregistered_library_models(models):
        print(f"warning: the model {name!r} in models/library isn't in the register.", file=sys.stderr)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
