"""The control mapping matrix: HIPAA Security Rule, GDPR, SOC 2 and ISO 27001
Annex A controls mapped to product features and evidence, read from
compliance/controls.yaml.

Each row is "supported" (the product implements a technical control that
helps), "customer" (the organisation running it must meet it) or
"not_addressed". Each carries a list of evidence files, or the word none. A
row marked supported must point at evidence that exists in the repository. The
banner is fixed and is part of every output."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from . import config

PATH = config.ROOT_DIR / "compliance" / "controls.yaml"
STATUSES = {"supported": "Supported by product", "customer": "Customer responsibility",
            "not_addressed": "Not addressed"}
BANNER = ("GroundCheckHealth holds no certification, attestation or clearance against HIPAA, GDPR, SOC 2 or "
          "ISO 27001. This matrix maps product features to controls to help an organisation prepare its own "
          "assessment. It is not a statement that any requirement is met.")


def load(path: Path | None = None) -> dict:
    import yaml

    return yaml.safe_load((path or PATH).read_text(encoding="utf-8"))


def validate(data: dict, root: Path | None = None) -> list[str]:
    """Problems with the mapping: a missing status or evidence, evidence that
    isn't a file in the repository, a supported row with no evidence, a
    duplicate id."""
    root = root or config.ROOT_DIR
    problems: list[str] = []
    frameworks = data.get("frameworks")
    if not isinstance(frameworks, dict) or not frameworks:
        return ["no frameworks"]
    for key, framework in frameworks.items():
        seen: set[str] = set()
        for row in framework.get("controls", []):
            where = f"{key} {row.get('id', '?')}"
            if not row.get("id") or not row.get("title") or not row.get("how"):
                problems.append(f"{where}: needs an id, title and how")
            if row.get("id") in seen:
                problems.append(f"{where}: duplicate id")
            seen.add(row.get("id"))
            if row.get("status") not in STATUSES:
                problems.append(f"{where}: status must be one of {sorted(STATUSES)}")
            evidence = row.get("evidence")
            if evidence == "none":
                if row.get("status") == "supported":
                    problems.append(f"{where}: supported needs evidence")
            elif isinstance(evidence, list) and evidence and all(isinstance(e, str) for e in evidence):
                for pointer in evidence:
                    if not (root / pointer).exists():
                        problems.append(f"{where}: evidence {pointer} does not exist")
            else:
                problems.append(f"{where}: evidence must be a list of files or the word none")
    return problems


def matrix(path: Path | None = None) -> dict:
    data = load(path)
    out = []
    totals = {s: 0 for s in STATUSES}
    for key, framework in data["frameworks"].items():
        counts = {s: 0 for s in STATUSES}
        controls = []
        for row in framework["controls"]:
            counts[row["status"]] += 1
            totals[row["status"]] += 1
            controls.append({"id": str(row["id"]), "title": row["title"], "status": row["status"],
                             "status_label": STATUSES[row["status"]], "how": row["how"],
                             "evidence": [] if row["evidence"] == "none" else list(row["evidence"])})
        out.append({"key": key, "name": framework["name"], "counts": counts, "controls": controls})
    return {"banner": BANNER, "statuses": STATUSES, "frameworks": out, "totals": totals,
            "problems": validate(data)}


def to_csv(m: dict) -> str:
    buffer = io.StringIO()
    buffer.write("# " + m["banner"] + "\n")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["Framework", "Control", "Title", "Status", "How the product helps", "Evidence"])
    for framework in m["frameworks"]:
        for c in framework["controls"]:
            writer.writerow([framework["name"], c["id"], c["title"], c["status_label"], c["how"],
                             "; ".join(c["evidence"]) or "none"])
    return buffer.getvalue()


def to_markdown(m: dict) -> str:
    lines = ["# GroundCheckHealth control mapping matrix", "", f"> {m['banner']}", ""]
    t = m["totals"]
    lines += [f"Supported by product: {t['supported']}. Customer responsibility: {t['customer']}. "
              f"Not addressed: {t['not_addressed']}.", ""]
    for framework in m["frameworks"]:
        lines += [f"## {framework['name']}", "", "| Control | Title | Status | How | Evidence |",
                  "| --- | --- | --- | --- | --- |"]
        for c in framework["controls"]:
            cells = [c["id"], c["title"], c["status_label"], c["how"], ", ".join(c["evidence"]) or "none"]
            lines.append("| " + " | ".join(x.replace("|", "/") for x in cells) + " |")
        lines.append("")
    return "\n".join(lines)
