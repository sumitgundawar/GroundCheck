#!/usr/bin/env python3
"""Write openapi/v1.json, the committed contract for /api/v1.

    python scripts/export_openapi.py          write the file
    python scripts/export_openapi.py --check  exit 1 if the file is out of date

tests/test_api_contract.py fails on any change that would break a client of
the committed contract. A compatible change (a new endpoint, a new optional
field) is recorded by running this script and committing the result.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import api_v1  # noqa: E402

PATH = ROOT / "openapi" / "v1.json"


def render() -> str:
    return json.dumps(api_v1.openapi_document(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if PATH.exists() and PATH.read_text(encoding="utf-8") == text else 1)
    PATH.write_text(text, encoding="utf-8")
    print("wrote", PATH.relative_to(ROOT))
