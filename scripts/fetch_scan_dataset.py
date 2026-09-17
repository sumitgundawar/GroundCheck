"""Download a public medical imaging dataset for trying the training studio.

    python scripts/fetch_scan_dataset.py organamnist            # abdominal CT, 11 organs
    python scripts/fetch_scan_dataset.py pneumoniamnist --size 128  # chest X-ray, 2 classes

The images come from MedMNIST v2 (https://medmnist.com), released under
CC BY 4.0, and are written as PNG files in the layout the training studio
reads:

    data/datasets/<name>-<size>/train/<class>/*.png
    data/datasets/<name>-<size>/val/<class>/*.png
    data/datasets/<name>-<size>/test/<class>/*.png

A DATASET.md with the source, licence and citations is written alongside.
These are research datasets: a model trained on them is not a medical device."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://zenodo.org/records/10519652/files"
MEDMNIST_CITATION = (
    "Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. "
    "MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. "
    "Scientific Data, 2023."
)
DATASETS = {
    "organamnist": {
        "title": "OrganAMNIST",
        "modality": "Abdominal CT, axial slices",
        "task": "Which organ a CT slice shows (11 classes)",
        "classes": ["bladder", "femur-left", "femur-right", "heart", "kidney-left", "kidney-right", "liver",
                    "lung-left", "lung-right", "pancreas", "spleen"],
        "source": "Liver Tumor Segmentation Benchmark (LiTS), processed by MedMNIST v2",
        "source_citation": "Patrick Bilic et al. The Liver Tumor Segmentation Benchmark (LiTS). "
                           "Medical Image Analysis, 2023.",
    },
    "pneumoniamnist": {
        "title": "PneumoniaMNIST",
        "modality": "Paediatric chest X-ray",
        "task": "Normal or pneumonia (2 classes)",
        "classes": ["normal", "pneumonia"],
        "source": "Kermany et al. chest X-ray images, processed by MedMNIST v2",
        "source_citation": "Daniel S. Kermany et al. Identifying Medical Diagnoses and Treatable Diseases by "
                           "Image-Based Deep Learning. Cell, 2018.",
    },
}
SIZES = (28, 64, 128, 224)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GroundCheck dataset fetcher"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        chunks, done = [], 0
        while chunk := response.read(1024 * 1024):
            chunks.append(chunk)
            done += len(chunk)
            if total:
                print(f"\r  downloading {done / 1e6:6.1f} of {total / 1e6:.1f} MB", end="", flush=True)
        print()
    return b"".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--size", type=int, default=64, choices=SIZES, help="image size in pixels (default 64)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "datasets")
    args = parser.parse_args(argv)

    spec = DATASETS[args.dataset]
    suffix = "" if args.size == 28 else f"_{args.size}"
    url = f"{BASE_URL}/{args.dataset}{suffix}.npz?download=1"
    target = args.out / f"{args.dataset}-{args.size}"
    if target.exists():
        print(f"{target} already exists. Delete it to download again.")
        return 1

    print(f"{spec['title']} ({spec['modality']}), {args.size}px, from {url}")
    data = download(url)
    sha256 = hashlib.sha256(data).hexdigest()
    arrays = np.load(io.BytesIO(data))
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images, labels = arrays[f"{split}_images"], arrays[f"{split}_labels"].reshape(-1)
        if labels.max() >= len(spec["classes"]):
            print(f"Unexpected label {labels.max()} in {split}.", file=sys.stderr)
            return 1
        for name in spec["classes"]:
            (target / split / name).mkdir(parents=True, exist_ok=True)
        for i, (image, label) in enumerate(zip(images, labels)):
            name = spec["classes"][int(label)]
            Image.fromarray(image).save(target / split / name / f"{split}-{i:05d}.png", optimize=False)
            counts.setdefault(name, {}).setdefault(split, 0)
            counts[name][split] += 1
        print(f"  {split}: {len(images):,} images")

    rows = "\n".join(f"| {c} | {counts.get(c, {}).get('train', 0):,} | {counts.get(c, {}).get('val', 0):,} "
                     f"| {counts.get(c, {}).get('test', 0):,} |" for c in spec["classes"])
    (target / "DATASET.md").write_text(f"""# {spec['title']}, {args.size}px

- Modality: {spec['modality']}
- Task: {spec['task']}
- Source: {spec['source']}
- Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- Downloaded from: {url}
- SHA-256 of the download: {sha256}

For research and evaluation. A model trained on this dataset is not a medical device.

| Class | Train | Validation | Test |
| --- | --- | --- | --- |
{rows}

## Citation

{MEDMNIST_CITATION}

{spec['source_citation']}
""", encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
