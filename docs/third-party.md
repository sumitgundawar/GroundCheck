# Third-party software

GroundCheck itself is MIT-licensed (`LICENSE`). It depends on open-source
packages, whose licences you inherit when you deploy it. This page says how to
produce the two artefacts a buyer or an auditor usually asks for, and lists the
licences worth knowing about.

## Bill of materials (SBOM)

```bash
pip install cyclonedx-bom
cyclonedx-py environment .venv -o sbom.json --output-format json   # CycloneDX 1.6
```

Produce one for the exact environment you deploy (the container, not a
developer's laptop):

```bash
docker run --rm --entrypoint sh groundcheck:1.0.0 -c 'pip install -q cyclonedx-bom >/dev/null && cyclonedx-py environment -o - --output-format json' > sbom.json
```

## Licences

```bash
pip install pip-licenses
pip-licenses --format=csv --with-urls > third-party-licences.csv
```

At the time of writing, a runtime environment has about 130 packages: mostly
MIT, BSD and Apache-2.0. The ones to be aware of:

| Package | Licence | Why it matters |
| --- | --- | --- |
| `psycopg`, `psycopg-binary` | LGPL-3.0-only | The optional PostgreSQL driver. Used as an unmodified library through its Python API, which the LGPL allows in commercial and closed products; if you modify it, the LGPL's terms apply to those changes. Install it only when you use PostgreSQL. |
| `certifi` | MPL-2.0 | CA certificate bundle, unmodified. |
| `hypothesis` | MPL-2.0 | Test-only; not installed in the runtime image unless you install the test extras. |
| `torch`, `torchvision` | BSD-3-Clause | Also ship NVIDIA runtime components in GPU builds; check their terms if you distribute a GPU image. |
| `sentence-transformers`, `transformers` | Apache-2.0 | The embedding model `all-MiniLM-L6-v2` is Apache-2.0 too. |

Models you download yourself carry their own licences. The Local AI page shows
each model's licence and leaves out models that forbid commercial use; imaging
models you train belong to you, but the datasets you train on may not (the
public ones GroundCheck fetches are CC BY, and their citations are written into
`DATASET.md` beside the images).

## Keeping it current

Re-generate both files for every release, and run `pip-audit -r
requirements.txt` to check that nothing in them has a known vulnerability. CI
does this on every push and weekly.
