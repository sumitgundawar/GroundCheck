"""Imported series, model analyses and clinician reports.

Pixels are stored in IMAGING_DIR, one file per series, compressed and, with
DATA_ENCRYPTION_KEYS set, encrypted with the file name bound to the
ciphertext. Descriptions, labels, reports and any identifiers kept for a
PACS are encrypted columns in the database.

Analyses run one at a time on a background thread; the series page polls
for progress. A signed report can't be edited: amending it creates a new
report that supersedes it."""

from __future__ import annotations

import io
import secrets
import threading
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sqlalchemy import select

from .. import config, db, encryption
from ..db import ImagingAnalysis, ImagingReport, ImagingSeries
from . import analysis, dicom

AGREEMENTS = ("agree", "partly", "disagree", "not_used")
_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_SIZE = 3
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imaging-analysis")


class ImagingError(ValueError):
    """Something the person can fix. The message is safe to show."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# ---------------------------------------------------------------- files

def _path(name: str):
    return config.IMAGING_DIR / name


def _write_volume(volume: np.ndarray) -> str:
    name = f"{secrets.token_hex(16)}.vol"
    small = volume.min() >= -32768 and volume.max() <= 32767 and np.allclose(volume, np.round(volume))
    array = volume.astype(np.int16 if small else np.float32)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    data = encryption.keyring().encrypt_bytes(zlib.compress(buffer.getvalue(), 1), f"imaging/{name}")
    config.IMAGING_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(_path(name))
    return name


def _read_volume(name: str) -> np.ndarray:
    data = encryption.keyring().decrypt_bytes(_path(name).read_bytes(), f"imaging/{name}")
    return np.load(io.BytesIO(zlib.decompress(data)), allow_pickle=False)


def volume(series_id: int) -> np.ndarray:
    with _cache_lock:
        if series_id in _cache:
            _cache.move_to_end(series_id)
            return _cache[series_id]
    with db.session() as s:
        row = s.get(ImagingSeries, series_id)
        if row is None:
            raise LookupError(series_id)
        name = row.file
    array = _read_volume(name)
    with _cache_lock:
        _cache[series_id] = array
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return array


def reencrypt_files() -> int:
    """Rewrite series files under the current primary key (or unencrypted,
    when encryption is off). Returns how many were rewritten."""
    ring, count = encryption.keyring(), 0
    with db.session() as s:
        names = list(s.scalars(select(ImagingSeries.file)))
    for name in names:
        path = _path(name)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        current = (raw.startswith(encryption.BYTES_PREFIX + ring.primary.encode()) if ring.enabled
                   else not raw.startswith(encryption.BYTES_PREFIX))
        if current:
            continue
        plain = ring.decrypt_bytes(raw, f"imaging/{name}")
        tmp = _path(name + ".tmp")
        tmp.write_bytes(ring.encrypt_bytes(plain, f"imaging/{name}"))
        tmp.replace(path)
        count += 1
    return count


# ---------------------------------------------------------------- series

def _series_summary(row: ImagingSeries) -> dict:
    latest = row.analyses[-1] if row.analyses else None
    signed = [r for r in row.reports if r.status == "signed"]
    return {
        "id": row.id, "label": row.label, "source": row.source, "modality": row.modality,
        "description": row.description, "body_part": row.body_part, "slices": row.slices,
        "rows": row.rows, "columns": row.columns, "created_at": _iso(row.created_at),
        "patient": row.meta.get("patient", {}), "plane": row.meta.get("display", {}).get("plane"),
        "analysis": None if latest is None else {"id": latest.id, "status": latest.status, "model_name": latest.model_name,
                                                 "abstained": (latest.result or {}).get("summary", {}).get("abstained")},
        "report": "signed" if signed else ("draft" if row.reports else None),
    }


def import_series(uploads: list[tuple[str, bytes]], source: str = "upload", label: str = "",
                  user_id: int | None = None) -> dict:
    try:
        found = dicom.read(uploads)
    except dicom.DicomError as exc:
        raise ImagingError(str(exc)) from exc
    added, existing = [], []
    for series in found:
        with db.session() as s:
            row = s.scalar(select(ImagingSeries).where(ImagingSeries.uid == series.key))
            if row is not None:
                existing.append(_series_summary(row))
                continue
        name = _write_volume(series.volume)
        meta = {"pixel_spacing": series.pixel_spacing, "slice_thickness": series.slice_thickness,
                "display": series.display, "patient": series.patient, "warnings": series.warnings,
                "deid": series.deid_report, "instances": series.instances}
        with db.session() as s:
            row = ImagingSeries(uid=series.key, study_uid=series.study_key, source=source, label=label[:200],
                                modality=series.modality, description=series.description[:200],
                                body_part=series.body_part[:64], slices=int(series.volume.shape[0]),
                                rows=series.rows, columns=series.columns, meta=meta,
                                original=series.original if source == "pacs" else None,
                                file=name, created_by=user_id)
            s.add(row)
            s.flush()
            added.append({**_series_summary(row), "warnings": series.warnings})
    return {"added": added, "existing": existing}


def list_series() -> list[dict]:
    with db.session() as s:
        rows = s.scalars(select(ImagingSeries).order_by(ImagingSeries.id.desc())).all()
        return [_series_summary(r) for r in rows]


def _report(row: ImagingReport) -> dict:
    return {"id": row.id, "status": row.status, "agreement": row.agreement, "findings": row.findings,
            "impression": row.impression, "author_name": row.author_name, "analysis_id": row.analysis_id,
            "created_at": _iso(row.created_at), "updated_at": _iso(row.updated_at), "signed_at": _iso(row.signed_at),
            "replaces_id": row.replaces_id, "sr_uid": row.sr_uid or None, "sent_at": _iso(row.sent_at)}


def _analysis(row: ImagingAnalysis, detail: bool = True) -> dict:
    out = {"id": row.id, "model_id": row.model_id, "model_name": row.model_name, "status": row.status,
           "progress": row.progress, "error": row.error or None, "created_at": _iso(row.created_at),
           "finished_at": _iso(row.finished_at)}
    if detail and row.result:
        out["summary"] = row.result.get("summary")
        out["slices"] = row.result.get("slices")
    return out


def get_series(series_id: int) -> dict:
    with db.session() as s:
        row = s.get(ImagingSeries, series_id)
        if row is None:
            raise LookupError(series_id)
        meta = row.meta
        return {**_series_summary(row), "pixel_spacing": meta.get("pixel_spacing"),
                "slice_thickness": meta.get("slice_thickness"), "slice_spacing": _slice_spacing(meta),
                "display": meta.get("display"),
                "warnings": meta.get("warnings", []), "deid": meta.get("deid", {}),
                "windows": list(dicom.WINDOWS.get(row.modality, {})),
                "can_send_to_pacs": bool(row.original) and bool(config.DICOMWEB_URL),
                "analyses": [_analysis(a) for a in row.analyses],
                "reports": [_report(r) for r in row.reports]}


def _slice_spacing(meta: dict) -> float | None:
    """The distance between neighbouring slices, in mm, from their positions."""
    if (meta.get("display") or {}).get("plane") in (None, "unknown"):
        return None   # positions are instance numbers, not millimetres
    positions = [i.get("position") for i in meta.get("instances", []) if i.get("position") is not None]
    if len(positions) < 2:
        return None
    gaps = np.abs(np.diff(positions))
    return round(float(np.median(gaps)), 2) if gaps.size and np.median(gaps) > 0 else None


def delete_series(series_id: int) -> None:
    with db.session() as s:
        row = s.get(ImagingSeries, series_id)
        if row is None:
            raise LookupError(series_id)
        if any(a.status in ("queued", "running") for a in row.analyses):
            raise ImagingError("An analysis of this series is still running. Try again when it finishes.")
        name = row.file
        s.delete(row)
    _path(name).unlink(missing_ok=True)
    with _cache_lock:
        _cache.pop(series_id, None)


def slice_png(series_id: int, index: int, window: str | None = None) -> bytes:
    array = volume(series_id)
    if not 0 <= index < array.shape[0]:
        raise ImagingError(f"This series has {array.shape[0]} slices.")
    with db.session() as s:
        modality = s.get(ImagingSeries, series_id).modality
    if window and window not in dicom.WINDOWS.get(modality, {}):
        raise ImagingError(f"Choose one of: {', '.join(dicom.WINDOWS.get(modality, {}))}.")
    grey = dicom.window(array[index], modality, window)
    buffer = io.BytesIO()
    Image.fromarray(grey).save(buffer, "PNG", compress_level=1)
    return buffer.getvalue()


# ---------------------------------------------------------------- analysis

def imaging_models() -> list[dict]:
    from ..training import library

    return [{"id": m["id"], "name": m["name"], "modality": (m.get("imaging") or {}).get("modality"),
             "classes": m.get("classes"), "target_met": m.get("target_met"), "imaging": m.get("imaging")}
            for m in library.list_models() if m.get("imaging")]


def _run(analysis_id: int) -> None:
    with db.session() as s:
        row = s.get(ImagingAnalysis, analysis_id)
        row.status = "running"
        series_id, model_id = row.series_id, row.model_id
        series = s.get(ImagingSeries, series_id)
        spacing, modality = series.meta.get("pixel_spacing") or [1.0, 1.0], series.modality

    last = [0]

    def progress(done: int, total: int) -> None:
        percent = int(100 * done / max(1, total))
        if percent - last[0] >= 5 or done == total:
            last[0] = percent
            with db.session() as s2:
                s2.get(ImagingAnalysis, analysis_id).progress = percent

    status, result, error = "done", None, ""
    try:
        model = analysis.LibraryModel.load(model_id)
        result = model.analyse(volume(series_id), spacing, modality, progress=progress)
    except analysis.AnalysisError as exc:
        status, error = "refused", str(exc)
    except Exception as exc:  # noqa: BLE001 - recorded for the person who asked
        status, error = "failed", f"The analysis failed: {type(exc).__name__}."
    with db.session() as s:
        row = s.get(ImagingAnalysis, analysis_id)
        if row is None:
            return
        row.status, row.result, row.error, row.finished_at = status, result, error, _now()
        if status == "done":
            row.progress = 100


def start_analysis(series_id: int, model_id: str, user_id: int | None = None, wait: bool = False) -> dict:
    from ..training import library

    try:
        card = library.get_model(model_id)
    except library.LibraryError as exc:
        raise ImagingError(str(exc)) from exc
    if not card.get("imaging"):
        raise ImagingError("This model has no imaging settings yet. An administrator can add them on the Models page.")
    with db.session() as s:
        series = s.get(ImagingSeries, series_id)
        if series is None:
            raise LookupError(series_id)
        if any(a.status in ("queued", "running") for a in series.analyses):
            raise ImagingError("This series is already being analysed.")
        row = ImagingAnalysis(series_id=series_id, model_id=model_id, model_name=card["name"], requested_by=user_id)
        s.add(row)
        s.flush()
        analysis_id = row.id
    future = _executor.submit(_run, analysis_id)
    if wait:
        future.result()
    with db.session() as s:
        return _analysis(s.get(ImagingAnalysis, analysis_id))


def recover_interrupted() -> int:
    """Analyses that were running when the app stopped won't finish."""
    with db.session() as s:
        rows = s.scalars(select(ImagingAnalysis).where(ImagingAnalysis.status.in_(("queued", "running")))).all()
        for row in rows:
            row.status, row.error, row.finished_at = "failed", "Stopped when GroundCheck restarted. Run it again.", _now()
        return len(rows)


# ---------------------------------------------------------------- reports

def save_report(series_id: int, *, findings: str, impression: str, agreement: str, analysis_id: int | None,
                sign: bool, report_id: int | None = None, author_id: int | None = None,
                author_name: str = "") -> dict:
    findings, impression = findings.strip(), impression.strip()
    if agreement not in AGREEMENTS:
        raise ImagingError("Say whether you agree with the model, or that you didn't use it.")
    if len(findings) > 20000 or len(impression) > 5000:
        raise ImagingError("The report is too long.")
    if sign and not impression:
        raise ImagingError("Write an impression before signing.")
    with db.session() as s:
        series = s.get(ImagingSeries, series_id)
        if series is None:
            raise LookupError(series_id)
        if analysis_id is not None:
            chosen = s.get(ImagingAnalysis, analysis_id)
            if chosen is None or chosen.series_id != series_id or chosen.status != "done":
                raise ImagingError("That analysis isn't a finished analysis of this series.")
        elif agreement != "not_used":
            raise ImagingError("Choose the analysis you're agreeing or disagreeing with.")
        replaces = None
        if report_id is not None:
            row = s.get(ImagingReport, report_id)
            if row is None or row.series_id != series_id:
                raise LookupError(report_id)
            if row.status == "signed":
                # Amending a signed report: a new report supersedes it.
                replaces, row = row, None
            elif row.status == "superseded":
                raise ImagingError("This report has been replaced by an amendment.")
        else:
            row = None
        if row is None:
            row = ImagingReport(series_id=series_id, author_id=author_id, author_name=author_name[:200],
                                replaces_id=replaces.id if replaces else None)
            s.add(row)
        row.findings, row.impression, row.agreement, row.analysis_id = findings, impression, agreement, analysis_id
        row.updated_at = _now()
        if sign:
            from pydicom.uid import generate_uid

            row.status, row.signed_at = "signed", _now()
            row.author_id, row.author_name = author_id, author_name[:200] or row.author_name
            row.sr_uid = generate_uid()
            if replaces is not None:
                replaces.status = "superseded"
        s.flush()
        return _report(row)


def report_sr(report_id: int, identified: bool = False) -> tuple[bytes, dict]:
    from . import report

    with db.session() as s:
        row = s.get(ImagingReport, report_id)
        if row is None:
            raise LookupError(report_id)
        if row.status == "draft":
            raise ImagingError("Sign the report before exporting it.")
        series = row.series
        chosen = s.get(ImagingAnalysis, row.analysis_id) if row.analysis_id else None
        previous = s.get(ImagingReport, row.replaces_id) if row.replaces_id else None
        if identified and not series.original:
            raise ImagingError("This series wasn't retrieved from a PACS, so there's no patient to send the report to.")
        ds = report.build(series=series, report=row, analysis=chosen, previous=previous, identified=identified)
        buffer = io.BytesIO()
        ds.save_as(buffer, enforce_file_format=True)
        return buffer.getvalue(), {"sop_instance_uid": str(ds.SOPInstanceUID), "study": str(ds.StudyInstanceUID)}


def mark_sent(report_id: int) -> None:
    with db.session() as s:
        s.get(ImagingReport, report_id).sent_at = _now()
