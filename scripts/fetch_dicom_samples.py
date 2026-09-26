"""Download two public DICOM series for trying CT and MRI imaging.

    python scripts/fetch_dicom_samples.py

- An abdominal CT (181 slices, 95 MB) from Pancreas-CT
- A prostate MRI (20 slices, 3 MB) from Prostate-Diagnosis

Both are from The Cancer Imaging Archive under CC BY 3.0 and are already
de-identified by TCIA; GroundCheckHealth de-identifies again when importing. They
are written to data/imaging-samples/, with a DATASET.md giving the sources,
licence and citations. For research and evaluation only."""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
TCIA_CITATION = ("Clark K, Vendt B, Smith K, et al. The Cancer Imaging Archive (TCIA): Maintaining and Operating a "
                 "Public Information Repository. Journal of Digital Imaging, 2013; 26(6):1045-1057.")
SERIES = [
    {
        "name": "pancreas-ct-0080", "uid": "1.2.826.0.1.3680043.2.1125.1.41202274843063370955090296887703130",
        "description": "Abdominal CT, portal venous phase (Pancreas-CT, PANCREAS_0080)",
        "citation": "Roth HR, Farag A, Turkbey EB, Lu L, Liu J, Summers RM. Data From Pancreas-CT. The Cancer Imaging "
                    "Archive, 2016. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU",
    },
    {
        "name": "prostate-mr-0003", "uid": "1.3.6.1.4.1.14519.5.2.1.4792.2002.239745692969312974757081064337",
        "description": "Prostate MRI, T2-weighted coronal (Prostate-Diagnosis, ProstateDx-01-0003)",
        "citation": "Bloch BN, Jain A, Jaffe CC. Data From Prostate-Diagnosis. The Cancer Imaging Archive, 2015. "
                    "https://doi.org/10.7937/K9/TCIA.2015.FOQEUJVT",
    },
]


def main() -> int:
    out = ROOT / "data" / "imaging-samples"
    out.mkdir(parents=True, exist_ok=True)
    for series in SERIES:
        target = out / series["name"]
        if target.exists():
            print(f"{target} already exists.")
            continue
        print(f"Downloading {series['description']}…")
        with urllib.request.urlopen(API + series["uid"], timeout=300) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".dcm")]
            target.mkdir()
            for name in names:
                (target / Path(name).name).write_bytes(archive.read(name))
        print(f"  {len(names)} files in {target.relative_to(ROOT)}")
    (out / "DATASET.md").write_text(
        "# DICOM samples\n\nFrom The Cancer Imaging Archive, licensed under CC BY 3.0 "
        "(https://creativecommons.org/licenses/by/3.0/). For research and evaluation only.\n\n"
        + "\n".join(f"## {s['name']}\n\n{s['description']}\n\nSeries Instance UID: {s['uid']}\n\n{s['citation']}\n"
                    for s in SERIES)
        + f"\n## The Cancer Imaging Archive\n\n{TCIA_CITATION}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
