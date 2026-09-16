"""Embed the corpus into the configured vector store and write the metadata.

Run from the repository root:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the app package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, retrieval  # noqa: E402


def main() -> None:
    corpus = retrieval.load_corpus()
    print(f"Embedding {len(corpus)} corpus records with {config.EMBED_MODEL} ...")
    retrieval.build_index()
    print(f"Index written to {config.INDEX_DIR}")
    # Sanity check: reload and run a quick search.
    retrieval.load_index()
    results = retrieval.search("first-line management of Veltris syndrome", 3)
    print("Sanity search top results:")
    for record, score in results:
        print(f"  {record['id']:<10} {score:.3f}  {record['title']}")


if __name__ == "__main__":
    main()
