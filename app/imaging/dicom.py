"""Reading DICOM uploads into series of CT or MR slices.

Accepts .dcm files and zip archives of them, with limits on archive size and
entry count so an upload can't exhaust memory or disk. Only CT and MR image
storage is accepted. Slices are grouped by series, de-identified
(deid.py), ordered along the patient axis, and converted to physical values
(Hounsfield units for CT)."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from . import deid

CT = {"1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.1"}
MR = {"1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.4.1"}
MAX_FILES = 3000
MAX_TOTAL_BYTES = 2 * 1024**3
MAX_FILE_BYTES = 200 * 1024**2
WINDOWS = {
    "CT": {"abdomen": (400, 50), "soft tissue": (350, 40), "lung": (1500, -600), "bone": (1800, 400), "brain": (80, 40)},
    "MR": {"auto": None},
}


class DicomError(ValueError):
    """An upload can't be used. The message is safe to show."""


@dataclass
class Series:
    key: str                         # de-identified SeriesInstanceUID
    study_key: str
    modality: str
    description: str
    body_part: str
    rows: int
    columns: int
    pixel_spacing: list[float]
    slice_thickness: float | None
    volume: np.ndarray               # (slices, rows, columns) float32, physical values
    display: dict = field(default_factory=dict)            # how stored slices were turned into display orientation
    instances: list[dict] = field(default_factory=list)   # de-identified per-slice references
    patient: dict = field(default_factory=dict)            # retained characteristics: age, sex, size, weight
    warnings: list[str] = field(default_factory=list)
    deid_report: dict = field(default_factory=dict)
    # Identifiers before de-identification. Only kept for series retrieved
    # from a PACS, so a signed report can be sent back to the right patient.
    original: dict = field(default_factory=dict)


def _files(uploads: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    total = 0
    for name, data in uploads:
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    if info.is_dir() or info.filename.startswith("__MACOSX"):
                        continue
                    if info.file_size > MAX_FILE_BYTES:
                        raise DicomError(f"{info.filename} is larger than {MAX_FILE_BYTES // 1024**2} MB.")
                    total += info.file_size
                    if total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                        raise DicomError("The upload is too large: up to 3,000 files and 2 GB.")
                    files.append((info.filename, archive.read(info)))
        else:
            total += len(data)
            if total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                raise DicomError("The upload is too large: up to 3,000 files and 2 GB.")
            files.append((name, data))
    return files


def pixels(ds) -> np.ndarray:
    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return array * slope + intercept


def window(image: np.ndarray, modality: str, name: str | None = None) -> np.ndarray:
    """Physical values to 8-bit grey for display or a model."""
    preset = WINDOWS.get(modality, {}).get(name or ("abdomen" if modality == "CT" else "auto"))
    if preset is None:
        low, high = np.percentile(image, [1, 99])
    else:
        width, level = preset
        low, high = level - width / 2, level + width / 2
    scaled = (np.clip(image, low, high) - low) / max(high - low, 1e-6)
    return (scaled * 255).astype(np.uint8)


def display_transform(orientation) -> dict:
    """How to turn stored slices into standard radiological display: axial
    with the patient's right on the image left and anterior at the top,
    coronal and sagittal with the head at the top (sagittal: anterior left).
    Returns {"transpose", "flip_rows", "flip_columns", "plane"}."""
    if orientation is None or len(orientation) != 6:
        return {"plane": "unknown", "transpose": False, "flip_rows": False, "flip_columns": False}
    row, col = np.array(orientation[:3], float), np.array(orientation[3:], float)
    normal = np.abs(np.cross(row, col))
    plane = ["sagittal", "coronal", "axial"][int(np.argmax(normal))]
    # The patient axis (0 x left, 1 y posterior, 2 z head) each image direction should follow, and its sign.
    want = {"axial": ((0, 1), (1, 1)), "coronal": ((0, 1), (2, -1)), "sagittal": ((1, 1), (2, -1))}[plane]
    transpose = int(np.argmax(np.abs(row))) != want[0][0]
    if transpose:
        row, col = col, row
    return {"plane": plane, "transpose": bool(transpose),
            "flip_columns": bool(np.sign(row[want[0][0]]) != want[0][1]),   # left-right within each row
            "flip_rows": bool(np.sign(col[want[1][0]]) != want[1][1])}      # top-bottom


def to_display(volume: np.ndarray, transform: dict) -> np.ndarray:
    out = volume.transpose(0, 2, 1) if transform.get("transpose") else volume
    if transform.get("flip_rows"):
        out = out[:, ::-1, :]
    if transform.get("flip_columns"):
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _position(ds) -> float:
    """Where a slice sits along the series, in mm, increasing in reading
    order: head to feet for axial, front to back for coronal, right to left
    for sagittal. Without position data, the instance number."""
    position = getattr(ds, "ImagePositionPatient", None)
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if position is not None and orientation is not None and len(orientation) == 6:
        normal = np.cross(np.array(orientation[:3], float), np.array(orientation[3:], float))
        axis = int(np.argmax(np.abs(normal)))
        normal = normal * np.sign(normal[axis])     # point along +x (left), +y (posterior) or +z (head)
        along = float(np.dot(normal, np.array(position, float)))
        return -along if axis == 2 else along
    return float(getattr(ds, "InstanceNumber", 0) or 0)


IDENTIFIERS = ("PatientName", "PatientID", "IssuerOfPatientID", "PatientBirthDate", "PatientSex", "AccessionNumber",
               "StudyID", "StudyDate", "StudyTime", "ReferringPhysicianName", "StudyDescription")


def _identifiers(ds) -> dict:
    return {"study_instance_uid": str(ds.get("StudyInstanceUID", "")),
            "series_instance_uid": str(ds.get("SeriesInstanceUID", "")),
            "sop_instance_uid": str(ds.get("SOPInstanceUID", "")),
            **{k: str(ds.get(k)) for k in IDENTIFIERS if ds.get(k) not in (None, "")}}


def _spacing(ds, transform: dict) -> list[float]:
    """[row spacing, column spacing] in mm, in display orientation."""
    spacing = [float(v) for v in ds.get("PixelSpacing", [1, 1])]
    return spacing[::-1] if transform.get("transpose") else spacing


def read(uploads: list[tuple[str, bytes]]) -> list[Series]:
    groups: dict[str, list] = {}
    skipped = 0
    refused: list[str] = []
    for name, data in _files(uploads):
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=False)
        except (InvalidDicomError, Exception):  # noqa: BLE001 - not DICOM
            skipped += 1
            continue
        sop = str(ds.get("SOPClassUID", ""))
        if sop not in CT | MR:
            skipped += 1
            continue
        if "PixelData" not in ds:
            skipped += 1
            continue
        original = _identifiers(ds)
        try:
            report = deid.deidentify(ds)
        except deid.DeidError as exc:
            refused.append(str(exc))
            continue
        report["_original"] = original
        groups.setdefault(str(ds.SeriesInstanceUID), []).append((ds, report))
    if not groups:
        if refused:
            raise DicomError(refused[0])
        raise DicomError("No CT or MR images were found. Upload .dcm files, or a zip of them.")

    series_list = []
    for key, items in groups.items():
        items.sort(key=lambda item: _position(item[0]))
        first = items[0][0]
        rows, columns = int(first.Rows), int(first.Columns)
        warnings = []
        usable = [(ds, rep) for ds, rep in items if int(ds.Rows) == rows and int(ds.Columns) == columns]
        if len(usable) < len(items):
            warnings.append(f"{len(items) - len(usable)} slices had a different size and were left out.")
        try:
            volume = np.stack([pixels(ds) for ds, _ in usable])
        except Exception as exc:  # noqa: BLE001 - unsupported compression, most often
            raise DicomError("These images use a compression GroundCheckHealth can't read. Export them uncompressed.") from exc
        report = {k: sum(rep[k] for _, rep in usable) for k in usable[0][1] if k != "_original"}
        first_original = usable[0][1]["_original"]
        modality = "CT" if str(first.SOPClassUID) in CT else "MR"
        transform = display_transform(first.get("ImageOrientationPatient"))
        if transform["plane"] == "unknown":
            warnings.append("The images have no orientation, so they're shown as stored.")
        volume = to_display(volume, transform)
        series_list.append(Series(
            key=key, study_key=str(first.StudyInstanceUID), modality=modality,
            description=str(first.get("SeriesDescription", "") or first.get("StudyDescription", "")),
            body_part=str(first.get("BodyPartExamined", "")),
            rows=int(volume.shape[1]), columns=int(volume.shape[2]),
            pixel_spacing=_spacing(first, transform),
            slice_thickness=float(first.SliceThickness) if first.get("SliceThickness") else None,
            volume=volume.astype(np.float32), display=transform,
            instances=[{"sop_instance_uid": str(ds.SOPInstanceUID), "sop_class_uid": str(ds.SOPClassUID),
                        "position": _position(ds)} for ds, _ in usable],
            patient={k: str(first.get(k)) for k in ("PatientAge", "PatientSex", "PatientSize", "PatientWeight")
                     if first.get(k)},
            warnings=warnings + ([f"{skipped} files weren't CT or MR images and were skipped."] if skipped else [])
                     + ([f"{len(refused)} images were refused: {refused[0]}"] if refused else []),
            deid_report=report,
            original={**first_original, "instances": [rep["_original"]["sop_instance_uid"] for _, rep in usable]},
        ))
    return series_list
