"""De-identifying DICOM at import, following the DICOM PS3.15 Basic
Application Level Confidentiality Profile with the Retain Patient
Characteristics and Clean Descriptors options.

- Names, IDs, dates, times, addresses, institutions, physicians, operators,
  devices' serial numbers and free-text comments are emptied or removed
- Private tags are removed
- UIDs are replaced with new ones, consistently within this deployment, so a
  series stays a series and slices keep their order and references
- Age (ages over 89 become 90), sex, size and weight are kept: models and dose
  checks need them
- Descriptions are kept after removing the patient's own name and IDs from
  them, then text de-identification (app/deid.py)
- Images marked as having burned-in annotations are refused, because text in
  the pixels can't be removed reliably

The dataset records what was done (PatientIdentityRemoved, DeidentificationMethod)."""

from __future__ import annotations

import hashlib
import re

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from .. import config
from ..deid import deidentify as deidentify_text

# Emptied (type 2 attributes that must stay present) or removed.
EMPTY = {
    "PatientName", "PatientID", "PatientBirthDate", "AccessionNumber", "StudyID", "ReferringPhysicianName",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime", "SeriesTime", "AcquisitionTime",
    "ContentTime", "InstanceCreationDate", "InstanceCreationTime",
}
REMOVE = {
    "PatientBirthTime", "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence", "PatientBirthName",
    "PatientMotherBirthName", "PatientAddress", "PatientTelephoneNumbers", "PatientTelecomInformation",
    "MilitaryRank", "BranchOfService", "EthnicGroup", "Occupation", "AdditionalPatientHistory", "PatientComments",
    "PatientReligiousPreference", "MedicalRecordLocator", "InsurancePlanIdentification", "PatientInsurancePlanCodeSequence",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "InstitutionCodeSequence",
    "StationName", "OperatorsName", "OperatorIdentificationSequence", "PerformingPhysicianName",
    "PerformingPhysicianIdentificationSequence", "NameOfPhysiciansReadingStudy", "PhysiciansOfRecord",
    "PhysiciansReadingStudyIdentificationSequence", "RequestingPhysician", "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers", "ReferringPhysicianIdentificationSequence", "DeviceSerialNumber",
    "PlateID", "RequestAttributesSequence", "ImageComments", "StudyComments", "InterpretationText",
    "PerformedProcedureStepID", "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime",
    "PerformedProcedureStepDescription", "ScheduledProcedureStepID", "RequestedProcedureID", "AdmissionID",
    "IssuerOfAdmissionID", "IssuerOfPatientID", "CurrentPatientLocation", "PatientState", "ResponsibleOrganization",
    "ResponsiblePerson", "OriginalAttributesSequence", "ModifiedAttributesSequence", "DigitalSignaturesSequence",
    "ContributingEquipmentSequence", "AcquisitionDateTime", "FrameReferenceDateTime",
}
UID_KEYWORDS = {"StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID",
                "ReferencedSOPInstanceUID", "MediaStorageSOPInstanceUID", "SynchronizationFrameOfReferenceUID",
                "IrradiationEventUID", "DimensionOrganizationUID", "ConcatenationUID"}
DESCRIPTORS = {"StudyDescription", "SeriesDescription", "ProtocolName", "BodyPartExamined", "PerformedProcedureStepDescription"}
METHOD = "PS3.15 Basic Profile; Retain Patient Chars; Clean Descriptors"


class DeidError(ValueError):
    """An instance can't be de-identified safely. The message is safe to show."""


def _salt() -> bytes:
    material = config.DATA_ENCRYPTION_KEYS or config.AUDIT_SIGNING_KEYS or config.DATABASE_URL
    return hashlib.sha256(b"groundcheck dicom uid\x00" + material.encode()).digest()


def new_uid(original: str) -> str:
    """A replacement UID: the same for the same original in this deployment,
    and unlinkable to the original without the deployment's secret."""
    return generate_uid(entropy_srcs=[_salt().hex(), original])


def _age(value: str) -> str:
    match = re.fullmatch(r"(\d{3})([DWMY])", value or "")
    if match and match.group(2) == "Y" and int(match.group(1)) > 89:
        return "090Y"
    return value


def _known_identifiers(ds: Dataset) -> list[tuple[re.Pattern, str]]:
    """The patient's own name parts and identifiers, to remove from free text
    such as a study description, where a general rule might miss them."""
    found: list[tuple[str, str]] = []
    name = str(ds.get("PatientName", "") or "")
    for part in re.split(r"[\^\s,]+", name):
        if len(part) >= 2 and part.isalpha():
            found.append((part, "[NAME]"))
    for keyword in ("PatientID", "AccessionNumber", "OtherPatientIDs"):
        value = str(ds.get(keyword, "") or "").strip()
        if len(value) >= 3:
            found.append((value, "[ID]"))
    return [(re.compile(rf"(?<![\w]){re.escape(v)}(?![\w])", re.IGNORECASE), label)
            for v, label in sorted(found, key=lambda f: -len(f[0]))]


def _scrub(text: str, known: list[tuple[re.Pattern, str]]) -> str:
    for pattern, label in known:
        text = pattern.sub(label, text)
    return deidentify_text(text).text


def _clean(ds: Dataset, report: dict, known: list[tuple[re.Pattern, str]]) -> None:
    for elem in list(ds):
        keyword = elem.keyword
        if elem.tag.is_private:
            del ds[elem.tag]
            report["private_removed"] += 1
            continue
        if keyword in REMOVE:
            del ds[elem.tag]
            report["removed"] += 1
            continue
        if keyword in EMPTY:
            elem.value = ""
            report["emptied"] += 1
            continue
        if keyword in UID_KEYWORDS and elem.value:
            elem.value = [new_uid(str(v)) for v in elem.value] if elem.VM > 1 else new_uid(str(elem.value))
            report["uids_replaced"] += 1
            continue
        if keyword == "PatientAge":
            elem.value = _age(str(elem.value))
            continue
        if keyword in DESCRIPTORS and isinstance(elem.value, str) and elem.value:
            cleaned = _scrub(elem.value, known)
            if cleaned != elem.value:
                elem.value = cleaned[:64]
                report["descriptors_cleaned"] += 1
            continue
        if elem.VR == "SQ":
            for item in elem.value:
                _clean(item, report, known)
        elif elem.VR in ("PN",) and elem.value:
            elem.value = ""
            report["emptied"] += 1
        elif elem.VR in ("DA", "DT", "TM") and elem.value and keyword not in ("PatientAge",):
            elem.value = ""
            report["emptied"] += 1


def deidentify(ds: Dataset) -> dict:
    """De-identify a dataset in place and return what was done."""
    if str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
        raise DeidError("This image says it has patient details burned into the pixels, which can't be removed safely.")
    report = {"removed": 0, "emptied": 0, "private_removed": 0, "uids_replaced": 0, "descriptors_cleaned": 0}
    _clean(ds, report, _known_identifiers(ds))
    if getattr(ds, "file_meta", None) is not None and "MediaStorageSOPInstanceUID" in ds.file_meta:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = METHOD
    codes = []
    for value, meaning in (("113100", "Basic Application Confidentiality Profile"),
                           ("113108", "Retain Patient Characteristics Option"),
                           ("113105", "Clean Descriptors Option")):
        item = Dataset()
        item.CodeValue, item.CodingSchemeDesignator, item.CodeMeaning = value, "DCM", meaning
        codes.append(item)
    ds.DeidentificationMethodCodeSequence = Sequence(codes)
    return report
