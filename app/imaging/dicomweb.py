"""A PACS or VNA over DICOMweb: search (QIDO-RS), retrieve (WADO-RS) and
store (STOW-RS).

Only DICOMWEB_URL is contacted, with DICOMWEB_AUTHORIZATION when set. Search
results show patient details to the signed-in clinician choosing a study and
aren't stored. Retrieved series are de-identified on import like uploads,
with the original identifiers kept encrypted so a signed report can be sent
back to the same study."""

from __future__ import annotations

import re
from email.parser import BytesParser
from email.policy import HTTP

import httpx

from .. import config

# Tests replace this with a mock transport; None uses the network.
transport: httpx.BaseTransport | None = None
MAX_RETRIEVE_BYTES = 2 * 1024**3
STUDY_FIELDS = {
    "0020000D": "study_uid", "00100010": "patient_name", "00100020": "patient_id", "00100030": "birth_date",
    "00080020": "study_date", "00080050": "accession", "00081030": "description", "00080061": "modalities",
    "00201208": "instances",
}
SERIES_FIELDS = {"0020000E": "series_uid", "00080060": "modality", "0008103E": "description",
                 "00200011": "number", "00201209": "instances", "00180015": "body_part"}
_DATE = re.compile(r"^\d{8}(-\d{8})?$|^-\d{8}$|^\d{8}-$")


class PacsError(Exception):
    """The PACS couldn't be reached or refused. The message is safe to show."""


def enabled() -> bool:
    return bool(config.DICOMWEB_URL)


def _client(timeout: float = 30.0) -> httpx.Client:
    if not enabled():
        raise PacsError("No PACS is configured. Set DICOMWEB_URL.")
    headers = {"Authorization": config.DICOMWEB_AUTHORIZATION} if config.DICOMWEB_AUTHORIZATION else {}
    return httpx.Client(base_url=config.DICOMWEB_URL, timeout=timeout, transport=transport, headers=headers,
                        follow_redirects=False)


def _value(item: dict, tag: str):
    values = (item.get(tag) or {}).get("Value") or []
    if not values:
        return None
    first = values[0]
    if isinstance(first, dict) and "Alphabetic" in first:
        return first["Alphabetic"].replace("^", " ").strip()
    if len(values) > 1:
        return [str(v) for v in values]
    return first


def _json(response: httpx.Response) -> list[dict]:
    if response.status_code == 204:
        return []
    if response.status_code in (401, 403):
        raise PacsError("The PACS refused GroundCheck's credentials. Check DICOMWEB_AUTHORIZATION.")
    if response.status_code >= 400:
        raise PacsError(f"The PACS returned an error ({response.status_code}).")
    try:
        data = response.json()
    except ValueError as exc:
        raise PacsError("The PACS didn't return DICOM JSON.") from exc
    return data if isinstance(data, list) else []


def search_studies(patient_id: str = "", patient_name: str = "", accession: str = "", study_date: str = "",
                   modality: str = "", limit: int = 25) -> list[dict]:
    params: dict[str, str] = {"limit": str(max(1, min(limit, 100))), "includefield": "00081030,00080061,00201208"}
    if patient_id.strip():
        params["PatientID"] = patient_id.strip()[:64]
    if patient_name.strip():
        params["PatientName"] = patient_name.strip()[:64] + "*"
        params["fuzzymatching"] = "true"
    if accession.strip():
        params["AccessionNumber"] = accession.strip()[:16]
    if study_date.strip():
        if not _DATE.match(study_date.strip()):
            raise PacsError("Dates are YYYYMMDD, or a range such as 20260101-20260131.")
        params["StudyDate"] = study_date.strip()
    if modality.strip():
        params["ModalitiesInStudy"] = modality.strip().upper()[:4]
    if len(params) == 2:
        raise PacsError("Search by patient ID, name, accession number or date.")
    try:
        with _client() as client:
            rows = _json(client.get("/studies", params=params, headers={"Accept": "application/dicom+json"}))
    except httpx.HTTPError as exc:
        raise PacsError("The PACS couldn't be reached.") from exc
    return [{name: _value(row, tag) for tag, name in STUDY_FIELDS.items()} for row in rows]


def _uid(value: str) -> str:
    if not re.fullmatch(r"[0-9.]{1,64}", value or ""):
        raise PacsError("That isn't a DICOM UID.")
    return value


def search_series(study_uid: str) -> list[dict]:
    try:
        with _client() as client:
            rows = _json(client.get(f"/studies/{_uid(study_uid)}/series", headers={"Accept": "application/dicom+json"}))
    except httpx.HTTPError as exc:
        raise PacsError("The PACS couldn't be reached.") from exc
    series = [{name: _value(row, tag) for tag, name in SERIES_FIELDS.items()} for row in rows]
    return sorted(series, key=lambda s: int(s.get("number") or 0))


def _multipart(content_type: str, body: bytes) -> list[bytes]:
    message = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    if not message.is_multipart():
        raise PacsError("The PACS didn't return DICOM instances.")
    return [part.get_payload(decode=True) for part in message.iter_parts()]


def retrieve_series(study_uid: str, series_uid: str) -> list[tuple[str, bytes]]:
    path = f"/studies/{_uid(study_uid)}/series/{_uid(series_uid)}"
    try:
        with _client(timeout=300.0) as client, client.stream(
                "GET", path, headers={"Accept": 'multipart/related; type="application/dicom"; transfer-syntax=*'}) as r:
            if r.status_code in (401, 403):
                raise PacsError("The PACS refused GroundCheck's credentials. Check DICOMWEB_AUTHORIZATION.")
            if r.status_code == 404:
                raise PacsError("The PACS has no such series.")
            if r.status_code >= 400:
                raise PacsError(f"The PACS returned an error ({r.status_code}).")
            chunks, size = [], 0
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > MAX_RETRIEVE_BYTES:
                    raise PacsError("The series is larger than 2 GB.")
                chunks.append(chunk)
            content_type = r.headers.get("content-type", "")
    except httpx.HTTPError as exc:
        raise PacsError("The PACS couldn't be reached.") from exc
    parts = _multipart(content_type, b"".join(chunks))
    return [(f"{i}.dcm", part) for i, part in enumerate(parts) if part]


def store(instances: list[bytes], study_uid: str | None = None) -> dict:
    boundary = "groundcheck-" + __import__("secrets").token_hex(12)
    body = b"".join(b"--" + boundary.encode() + b"\r\nContent-Type: application/dicom\r\n\r\n" + data + b"\r\n"
                    for data in instances) + b"--" + boundary.encode() + b"--\r\n"
    path = f"/studies/{_uid(study_uid)}" if study_uid else "/studies"
    try:
        with _client(timeout=120.0) as client:
            response = client.post(path, content=body, headers={
                "Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}',
                "Accept": "application/dicom+json"})
    except httpx.HTTPError as exc:
        raise PacsError("The PACS couldn't be reached.") from exc
    if response.status_code in (401, 403):
        raise PacsError("The PACS refused GroundCheck's credentials. Check DICOMWEB_AUTHORIZATION.")
    if response.status_code == 409 or response.status_code >= 400:
        raise PacsError(f"The PACS didn't store the report ({response.status_code}).")
    try:
        result = response.json() if response.content else {}
    except ValueError:
        result = {}
    failed = (result.get("00081198") or {}).get("Value") if isinstance(result, dict) else None
    if failed:
        raise PacsError("The PACS rejected the report.")
    return {"status": response.status_code}
