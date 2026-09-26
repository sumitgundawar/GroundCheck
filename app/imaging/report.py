"""Signed reports as DICOM Structured Reports (Comprehensive SR), which a
PACS stores alongside the images and viewers display.

The report references every image in the series as evidence and holds the
clinician's findings and impression, who signed it and when, and, when a
model was used, which model, what it reported, and whether the clinician
agreed. By default it references the de-identified series; for a series
retrieved from a PACS it can carry the original patient and study
identifiers instead, so the PACS files it with the right study."""

from __future__ import annotations

import highdicom as hd
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from .. import config

AGREEMENT_TEXT = {"agree": "Agrees with the model's output", "partly": "Partly agrees with the model's output",
                  "disagree": "Disagrees with the model's output", "not_used": "Model output not used"}
PATIENT_KEYS = ("PatientName", "PatientID", "IssuerOfPatientID", "PatientBirthDate", "PatientSex")
STUDY_KEYS = ("AccessionNumber", "StudyID", "StudyDate", "StudyTime", "ReferringPhysicianName")


def _concept(value: str, scheme: str, meaning: str) -> hd.sr.CodedConcept:
    return hd.sr.CodedConcept(value, scheme, meaning)


def _text(value: str, scheme: str, meaning: str, text: str) -> hd.sr.TextContentItem:
    return hd.sr.TextContentItem(name=_concept(value, scheme, meaning), value=text[:10000] or "None",
                                 relationship_type=hd.sr.RelationshipTypeValues.CONTAINS)


def _evidence(series, identified: bool) -> list[Dataset]:
    original = series.original or {}
    instances = series.meta.get("instances", [])
    uids = original.get("instances", []) if identified else [i["sop_instance_uid"] for i in instances]
    out = []
    for inst, uid in zip(instances, uids):
        ds = Dataset()
        ds.StudyInstanceUID = original["study_instance_uid"] if identified else series.study_uid
        ds.SeriesInstanceUID = original["series_instance_uid"] if identified else series.uid
        ds.SOPInstanceUID = uid
        ds.SOPClassUID = inst["sop_class_uid"]
        for key in PATIENT_KEYS + STUDY_KEYS:
            setattr(ds, key, original.get(key, "") if identified else "")
        if not identified:
            sex = (series.meta.get("patient") or {}).get("PatientSex", "")
            ds.PatientSex = sex if sex in ("M", "F", "O") else ""
        out.append(ds)
    return out


def _model_lines(analysis) -> str:
    summary = (analysis.result or {}).get("summary") or {}
    if summary.get("abstained"):
        share = summary.get("unfamiliar_share", 0)
        return f"The model abstained: {share:.0%} of the image regions were unlike its training images."
    labels = summary.get("labels") or []
    if not labels:
        return "The model reported no regions."
    parts = []
    for item in labels:
        ranges = ", ".join(f"{a + 1}" if a == b else f"{a + 1}-{b + 1}" for a, b in item["ranges"])
        parts.append(f"{item['label']} on slices {ranges} (highest confidence {item['max_confidence']:.0%})")
    return "; ".join(parts)


def _person_name(name: str) -> str:
    """A DICOM person name (family^given) from a display name."""
    parts = (name or "").replace("^", " ").split()
    if not parts:
        return "Unknown^"
    if len(parts) == 1:
        return f"{parts[0]}^"
    return f"{parts[-1]}^{' '.join(parts[:-1])}"


def build(*, series, report, analysis, previous, identified: bool) -> Dataset:
    items = [
        _text("121071", "DCM", "Finding", report.findings),
        _text("121073", "DCM", "Impression", report.impression),
    ]
    if analysis is not None:
        items += [
            _text("111001", "DCM", "Algorithm Name", analysis.model_name),
            _text("111003", "DCM", "Algorithm Version", analysis.model_id),
            _text("121106", "DCM", "Comment",
                  f"Model output: {_model_lines(analysis)} Clinician: {AGREEMENT_TEXT[report.agreement]}. "
                  "The model is for research and evaluation, not a diagnosis."),
        ]
    else:
        items.append(_text("121106", "DCM", "Comment", AGREEMENT_TEXT["not_used"] + "."))
    if previous is not None and previous.signed_at:
        items.append(_text("121106", "DCM", "Comment",
                           f"Amends the report signed on {previous.signed_at:%Y-%m-%d %H:%M} UTC."))
    root = hd.sr.ContainerContentItem(name=_concept("18748-4", "LN", "Diagnostic imaging report"), template_id="2000")
    root.ContentSequence = items
    evidence = _evidence(series, identified)
    ds = hd.sr.ComprehensiveSR(
        evidence=evidence, content=root,
        series_instance_uid=generate_uid(entropy_srcs=[series.uid, "groundcheck report series"]),
        series_number=990, sop_instance_uid=report.sr_uid, instance_number=report.id,
        manufacturer="GroundCheckHealth", is_complete=True, is_final=True, is_verified=True,
        verifying_observer_name=_person_name(report.author_name),
        verifying_organization=config.ORGANISATION_NAME or "GroundCheckHealth",
    )
    ds.SeriesDescription = "GroundCheckHealth report"
    if report.signed_at:
        ds.ContentDate = report.signed_at.strftime("%Y%m%d")
        ds.ContentTime = report.signed_at.strftime("%H%M%S")
    if not identified:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "GroundCheckHealth de-identified series"
    return ds
