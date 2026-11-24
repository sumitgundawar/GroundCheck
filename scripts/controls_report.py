"""Print or check the control mapping matrix (compliance/controls.yaml).

    python scripts/controls_report.py              # Markdown to stdout
    python scripts/controls_report.py --csv        # CSV
    python scripts/controls_report.py --json
    python scripts/controls_report.py --check      # exit 1 on any problem

The check fails when a row lacks a status or evidence, when evidence points at
a file that does not exist, or when a supported row has none."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import compliance_controls as cc  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    m = cc.matrix()
    if args.check:
        for problem in m["problems"]:
            print("problem:", problem, file=sys.stderr)
        rows = sum(len(f["controls"]) for f in m["frameworks"])
        print(f"{rows} controls, {len(m['problems'])} problem(s)")
        return 1 if m["problems"] else 0
    sys.stdout.write(cc.to_csv(m) if args.csv else json.dumps(m, indent=2) + "\n" if args.json else cc.to_markdown(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
