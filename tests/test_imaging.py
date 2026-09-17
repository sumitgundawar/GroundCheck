"""CT and MRI imaging: DICOM de-identification, orientation and slice order,
upload limits, running a model over a series with abstention, signed reports
as DICOM SR, and a PACS over DICOMweb (mocked)."""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config, encryption  # noqa: E402
from app.imaging import analysis, deid, dicom, dicomweb, store  # noqa: E402
from app.training import architectures, novelty  # noqa: E402

CT_CLASS, MR_CLASS = "1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.4"
STUDY, SERIES = "1.2.826.0.1.3680043.8.498.1", "1.2.826.0.1.3680043.8.498.2"


def phantom(rows: int = 64) -> np.ndarray:
    """A body-like slice in Hounsfield units: air around a soft-tissue disc
    with a bright marker in the patient's right anterior quadrant."""
    y, x = np.mgrid[:rows, :rows]
    image = np.full((rows, rows), -1000.0)
    image[(x - rows / 2) ** 2 + (y - rows / 2) ** 2 < (rows * 0.42) ** 2] = 40
    image[rows // 5:rows // 3, rows // 5:rows // 3] = 700
    return image


def instance(z: float, number: int, modality_class: str = CT_CLASS, orientation=(1, 0, 0, 0, 1, 0),
             pixels: np.ndarray | None = None, series: str = SERIES, burned: bool = False) -> bytes:
    pixels = phantom() if pixels is None else pixels
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = modality_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = modality_class, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = STUDY, series, "1.2.3.4"
    ds.Modality = "CT" if modality_class == CT_CLASS else "MR"
    ds.PatientName, ds.PatientID, ds.PatientBirthDate = "Smith^John", "NHS9434765919", "19480312"
    ds.PatientAge, ds.PatientSex, ds.AccessionNumber = "093Y", "M", "ACC123"
    ds.StudyDate, ds.StudyTime, ds.InstitutionName = "20260301", "101500", "St Elsewhere Hospital"
    ds.ReferringPhysicianName, ds.StudyDescription = "Jones^Mary", "CT abdomen for John Smith"
    ds.SeriesDescription = "Axial 1mm"
    ds.add_new(0x00111001, "LO", "private note about the patient")
    ds.InstanceNumber = number
    ds.ImagePositionPatient = [0, 0, z]
    ds.ImageOrientationPatient = list(orientation)
    ds.PixelSpacing = [4.0, 4.0]
    ds.SliceThickness = 5
    if burned:
        ds.BurnedInAnnotation = "YES"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
    ds.PixelData = (pixels + 1024).astype(np.int16).tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def ct_series(slices: int = 4, **kwargs) -> list[tuple[str, bytes]]:
    # Stored in a shuffled order, positions 0, -5, -10... (head at z=0).
    order = [2, 0, 3, 1][:slices] if slices == 4 else list(range(slices))
    return [(f"{i}.dcm", instance(-5.0 * i, i + 1, **kwargs)) for i in order]


# --- De-identification and reading ----------------------------------------------

def test_dicom_is_de_identified_on_import():
    [series] = dicom.read(ct_series())
    first = pydicom.dcmread(io.BytesIO(instance(0, 1)))
    report = deid.deidentify(first)
    assert first.PatientName == "" and first.PatientID == "" and first.PatientBirthDate == ""
    assert first.AccessionNumber == "" and first.StudyDate == "" and "InstitutionName" not in first
    assert first.ReferringPhysicianName == "" and not any(e.tag.is_private for e in first)
    assert first.PatientAge == "090Y" and first.PatientSex == "M"
    assert "John Smith" not in first.StudyDescription
    assert first.StudyInstanceUID != STUDY and first.StudyInstanceUID == deid.new_uid(STUDY)
    assert first.file_meta.MediaStorageSOPInstanceUID == first.SOPInstanceUID
    assert first.PatientIdentityRemoved == "YES" and len(first.DeidentificationMethod) <= 64
    assert report["private_removed"] == 1 and report["uids_replaced"] >= 3
    assert series.key == deid.new_uid(SERIES) and series.patient == {"PatientAge": "090Y", "PatientSex": "M"}
    # Originals are captured for PACS write-back only.
    assert series.original["PatientID"] == "NHS9434765919" and len(series.original["instances"]) == 4


def test_burned_in_annotations_are_refused():
    with pytest.raises(dicom.DicomError, match="burned into the pixels"):
        dicom.read([("a.dcm", instance(0, 1, burned=True))])


def test_slices_are_ordered_head_first_and_displayed_in_standard_orientation():
    [series] = dicom.read(ct_series())
    assert series.volume.shape == (4, 64, 64)
    assert [i["position"] for i in series.instances] == sorted(i["position"] for i in series.instances)
    assert series.display == {"plane": "axial", "transpose": False, "flip_rows": False, "flip_columns": False}
    # Stored upside down (columns run towards the head... here: posterior to anterior).
    [flipped] = dicom.read([("a.dcm", instance(0, 1, orientation=(1, 0, 0, 0, -1, 0), pixels=phantom()[::-1]))])
    assert flipped.display["flip_rows"] and np.array_equal(flipped.volume[0], series.volume[0])
    # Coronal and sagittal planes: head at the top.
    assert dicom.display_transform([1, 0, 0, 0, 0, -1])["plane"] == "coronal"
    assert dicom.display_transform([0, 1, 0, 0, 0, -1]) == {"plane": "sagittal", "transpose": False,
                                                             "flip_rows": False, "flip_columns": False}
    assert dicom.display_transform([0, 0, -1, 1, 0, 0])["transpose"]


def test_uploads_are_limited_and_non_dicom_is_skipped(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in ct_series():
            archive.writestr(f"study/{name}", data)
        archive.writestr("notes.txt", "hello")
    [series] = dicom.read([("study.zip", buffer.getvalue())])
    assert series.volume.shape[0] == 4 and any("skipped" in w for w in series.warnings)
    monkeypatch.setattr(dicom, "MAX_FILES", 3)
    with pytest.raises(dicom.DicomError, match="too large"):
        dicom.read([("study.zip", buffer.getvalue())])
    with pytest.raises(dicom.DicomError, match="No CT or MR"):
        dicom.read([("a.txt", b"not dicom")])


# --- Unfamiliar images ---------------------------------------------------------

def test_novelty_flags_features_far_from_every_class():
    rng = np.random.default_rng(0)
    features = np.concatenate([rng.normal(0, 1, (200, 8)), rng.normal(5, 1, (200, 8))])
    labels = np.array([0] * 200 + [1] * 200)
    shallow = [features, features[:, :4] * 2]
    stats, cutoffs = novelty.fit_layers(shallow, labels, 2, shallow)
    familiar, _ = novelty.flagged([rng.normal(5, 1, (50, 8)), rng.normal(10, 2, (50, 4))], stats, cutoffs)
    # Normal at the deep block but strange at the shallow one: still flagged.
    strange, worst = novelty.flagged([rng.normal(5, 1, (50, 8)), rng.normal(-40, 1, (50, 4))], stats, cutoffs)
    assert (~familiar).mean() > 0.85 and strange.all() and (worst > 1).all()


# --- A model over a series -------------------------------------------------------

@pytest.fixture()
def imaging(database, tmp_path, monkeypatch):
    """An imaging library model: a small CNN whose unfamiliar-image check was
    fitted on patches of the phantom, so real phantom slices are familiar and
    noise isn't."""
    import torch
    from safetensors.torch import save_file

    from app.training import library, preprocess

    monkeypatch.setattr(config, "IMAGING_DIR", tmp_path / "imaging")
    monkeypatch.setattr(config, "MODEL_LIBRARY_DIR", tmp_path / "library")
    library._loaded.clear()
    torch.manual_seed(0)
    model = architectures.build("small-cnn", 2, 1).eval()
    folder = tmp_path / "library" / "phantom-model"
    folder.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(folder / "weights.safetensors"))
    rng = np.random.default_rng(1)
    grey = dicom.window(phantom(), "CT", "abdomen")
    patches = []
    for i in range(120):
        noise = rng.normal(0, 3, grey.shape) if i % 2 else 0
        shifted = np.roll(grey, rng.integers(-3, 4, 2), axis=(0, 1)).astype(float) + noise
        from PIL import Image
        patches.append(preprocess.to_array(Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8)), 32, 1))
    x = preprocess.normalise(torch.from_numpy(np.stack(patches)), [0.5], [0.25], 1)
    _, feats = novelty.forward(model, "small-cnn", x)
    labels = np.array([0, 1] * 60)
    stats, cutoffs = novelty.fit_layers([f[:80] for f in feats], labels[:80], 2, [f[80:] for f in feats])
    novelty.save(folder / "novelty.safetensors", stats)
    card = {
        "id": "phantom-model", "name": "Phantom model", "task": "image-classification", "architecture": "small-cnn",
        "pretrained": False, "classes": ["body", "marker"],
        "input": {"image_size": 32, "channels": 1, "model_channels": 1, "mean": [0.5], "std": [0.25]},
        "threshold": {"threshold": 0.5, "met": True, "target": 0.95},
        "novelty": {"method": "mahalanobis-per-block", "cutoffs": [c * 1.5 for c in cutoffs], "flag_rate": 0.02},
        "validation": {"accuracy": 1.0}, "dataset": {"name": "phantom", "total": 120},
        "created_at": "2026-09-17T00:00:00+00:00", "hardware": {},
        "imaging": {"modality": "CT", "window": "abdomen", "orientation": "identity", "patch_mm": None},
    }
    (folder / "model.json").write_text(json.dumps(card))
    from app.main import app

    return TestClient(app)


def _wait(client, series_id):
    deadline = time.time() + 60
    while time.time() < deadline:
        series = client.get(f"/api/imaging/series/{series_id}").json()["series"]
        if series["analyses"] and series["analyses"][-1]["status"] not in ("queued", "running"):
            return series
        time.sleep(0.2)
    raise AssertionError("analysis didn't finish")


def test_upload_view_analyse_and_report(imaging):
    client = imaging
    files = [("files", (name, data, "application/dicom")) for name, data in ct_series()]
    added = client.post("/api/imaging/upload", files=files, data={"label": "Phantom 1"}).json()["added"]
    assert len(added) == 1 and added[0]["slices"] == 4 and added[0]["label"] == "Phantom 1"
    series_id = added[0]["id"]
    again = client.post("/api/imaging/upload", files=files).json()
    assert again["added"] == [] and again["existing"][0]["id"] == series_id

    home = client.get("/api/imaging").json()
    assert [m["id"] for m in home["models"]] == ["phantom-model"] and home["series"][0]["id"] == series_id
    png = client.get(f"/api/imaging/series/{series_id}/slices/0.png?window=lung")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.get(f"/api/imaging/series/{series_id}/slices/9.png").status_code == 400
    assert client.get(f"/api/imaging/series/{series_id}/slices/0.png?window=nope").status_code == 400
    stored = next((config.IMAGING_DIR).iterdir()).read_bytes()
    assert b"Smith" not in stored

    started = client.post(f"/api/imaging/series/{series_id}/analyses", json={"model_id": "phantom-model"})
    assert started.status_code == 200, started.text
    series = _wait(client, series_id)
    result = series["analyses"][-1]
    assert result["status"] == "done", result
    assert result["summary"]["totals"]["patches"] == 4 and not result["summary"]["abstained"]
    assert result["summary"]["labels"] and len(result["slices"]) == 4

    draft = client.post(f"/api/imaging/series/{series_id}/reports", json={
        "findings": "Soft tissue disc with a dense focus.", "impression": "", "agreement": "partly",
        "analysis_id": result["id"]}).json()["report"]
    assert draft["status"] == "draft"
    refused = client.post(f"/api/imaging/series/{series_id}/reports", json={
        "report_id": draft["id"], "findings": "x", "impression": "", "agreement": "agree",
        "analysis_id": result["id"], "sign": True})
    assert "impression" in refused.json()["detail"]
    assert client.get(f"/api/imaging/reports/{draft['id']}/sr.dcm").status_code == 400
    signed = client.post(f"/api/imaging/series/{series_id}/reports", json={
        "report_id": draft["id"], "findings": "Soft tissue disc with a dense focus.",
        "impression": "Dense focus, likely calcification.", "agreement": "partly", "analysis_id": result["id"],
        "sign": True}).json()["report"]
    assert signed["id"] == draft["id"] and signed["status"] == "signed" and signed["sr_uid"]

    sr = pydicom.dcmread(io.BytesIO(client.get(f"/api/imaging/reports/{signed['id']}/sr.dcm").content))
    assert sr.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.33" and sr.VerificationFlag == "VERIFIED"
    assert sr.StudyInstanceUID == deid.new_uid(STUDY) and sr.PatientName == "" and sr.PatientIdentityRemoved == "YES"
    texts = {item.ConceptNameCodeSequence[0].CodeMeaning: item.TextValue for item in sr.ContentSequence}
    assert texts["Impression"] == "Dense focus, likely calcification." and texts["Algorithm Name"] == "Phantom model"
    assert "Partly agrees" in texts["Comment"]
    evidence = sr.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0]
    assert len(evidence.ReferencedSOPSequence) == 4

    amended = client.post(f"/api/imaging/series/{series_id}/reports", json={
        "report_id": signed["id"], "findings": "Unchanged.", "impression": "Calcified granuloma.",
        "agreement": "not_used", "sign": True}).json()["report"]
    assert amended["id"] != signed["id"] and amended["replaces_id"] == signed["id"]
    reports = {r["id"]: r for r in client.get(f"/api/imaging/series/{series_id}").json()["series"]["reports"]}
    assert reports[signed["id"]]["status"] == "superseded" and reports[signed["id"]]["impression"].startswith("Dense")
    sr2 = pydicom.dcmread(io.BytesIO(client.get(f"/api/imaging/reports/{amended['id']}/sr.dcm").content))
    assert any("Amends the report" in i.TextValue for i in sr2.ContentSequence if hasattr(i, "TextValue"))

    assert client.post(f"/api/imaging/reports/{amended['id']}/send").status_code == 400  # not from a PACS
    assert client.delete(f"/api/imaging/series/{series_id}").json() == {"deleted": series_id}
    assert list(config.IMAGING_DIR.iterdir()) == []
    assert client.get(f"/api/imaging/series/{series_id}").status_code == 404


def test_wrong_modality_is_refused_and_unfamiliar_series_abstain(imaging):
    client = imaging
    mr = [("files", (f"{i}.dcm", instance(-5.0 * i, i + 1, modality_class=MR_CLASS, series="1.2.3.9"), "x"))
          for i in range(2)]
    mr_id = client.post("/api/imaging/upload", files=mr).json()["added"][0]["id"]
    client.post(f"/api/imaging/series/{mr_id}/analyses", json={"model_id": "phantom-model"})
    refused = _wait(client, mr_id)["analyses"][-1]
    assert refused["status"] == "refused" and "trained on CT" in refused["error"]

    rng = np.random.default_rng(5)
    noise = [("files", (f"{i}.dcm", instance(-5.0 * i, i + 1, series="1.2.3.10",
                                             pixels=rng.normal(0, 900, (64, 64)).clip(-1000, 3000)), "x"))
             for i in range(3)]
    noise_id = client.post("/api/imaging/upload", files=noise).json()["added"][0]["id"]
    client.post(f"/api/imaging/series/{noise_id}/analyses", json={"model_id": "phantom-model"})
    result = _wait(client, noise_id)["analyses"][-1]
    assert result["status"] == "done" and result["summary"]["abstained"] and result["summary"]["labels"] == []
    assert all(s["findings"] == [] for s in result["slices"])
    assert client.post(f"/api/imaging/series/{noise_id}/analyses", json={"model_id": "missing"}).status_code == 400


def test_series_files_are_encrypted_when_keys_are_set(imaging, monkeypatch):
    monkeypatch.setattr(config, "DATA_ENCRYPTION_KEYS", encryption.generate_key())
    encryption.reset()
    try:
        added = store.import_series(ct_series(), label="Encrypted")["added"][0]
        name = next(config.IMAGING_DIR.iterdir())
        assert name.read_bytes().startswith(encryption.BYTES_PREFIX)
        store._cache.clear()
        assert store.volume(added["id"]).shape == (4, 64, 64)
        tampered = bytearray(name.read_bytes())
        tampered[-1] ^= 1
        name.write_bytes(bytes(tampered))
        store._cache.clear()
        with pytest.raises(encryption.EncryptionError, match="integrity"):
            store.volume(added["id"])
    finally:
        monkeypatch.undo()
        encryption.reset()


def test_model_imaging_settings_are_validated(imaging):
    client = imaging
    ok = client.put("/api/models/phantom-model/imaging",
                    json={"modality": "MR", "orientation": "transpose", "patch_mm": 120})
    assert ok.status_code == 200 and ok.json()["imaging"]["modality"] == "MR"
    assert client.put("/api/models/phantom-model/imaging", json={"modality": "XR"}).status_code == 400
    assert client.put("/api/models/phantom-model/imaging", json={"modality": "CT", "window": "x"}).status_code == 400
    assert client.put("/api/models/phantom-model/imaging",
                      json={"modality": "CT", "patch_mm": 5}).status_code == 400
    assert client.delete("/api/models/phantom-model/imaging").json() == {"imaging": None}
    assert client.get("/api/imaging").json()["models"] == []


# --- PACS over DICOMweb ----------------------------------------------------------

class FakePacs:
    def __init__(self):
        self.instances = [data for _, data in ct_series()]
        self.stored: list[bytes] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer pacs-token"
        path = request.url.path
        if request.method == "GET" and path == "/dicom-web/studies":
            assert request.url.params["PatientID"] == "NHS9434765919"
            return httpx.Response(200, json=[{
                "0020000D": {"vr": "UI", "Value": [STUDY]}, "00100010": {"vr": "PN", "Value": [{"Alphabetic": "Smith^John"}]},
                "00100020": {"vr": "LO", "Value": ["NHS9434765919"]}, "00080020": {"vr": "DA", "Value": ["20260301"]},
                "00080061": {"vr": "CS", "Value": ["CT", "SR"]}}])
        if request.method == "GET" and path == f"/dicom-web/studies/{STUDY}/series":
            return httpx.Response(200, json=[{"0020000E": {"vr": "UI", "Value": [SERIES]},
                                              "00080060": {"vr": "CS", "Value": ["CT"]},
                                              "00201209": {"vr": "IS", "Value": [4]}}])
        if request.method == "GET" and path == f"/dicom-web/studies/{STUDY}/series/{SERIES}":
            boundary = "b0undary"
            body = b"".join(b"--" + boundary.encode() + b"\r\nContent-Type: application/dicom\r\n\r\n" + d + b"\r\n"
                            for d in self.instances) + b"--" + boundary.encode() + b"--\r\n"
            return httpx.Response(200, content=body, headers={
                "content-type": f'multipart/related; type="application/dicom"; boundary={boundary}'})
        if request.method == "POST" and path == f"/dicom-web/studies/{STUDY}":
            boundary = request.headers["content-type"].split("boundary=")[1]
            self.stored += dicomweb._multipart(request.headers["content-type"], request.content)
            assert boundary
            return httpx.Response(200, json={})
        return httpx.Response(404)


def test_pacs_search_retrieve_and_send_the_signed_report(imaging, monkeypatch):
    client, pacs = imaging, FakePacs()
    monkeypatch.setattr(config, "DICOMWEB_URL", "https://pacs.example/dicom-web")
    monkeypatch.setattr(config, "DICOMWEB_AUTHORIZATION", "Bearer pacs-token")
    monkeypatch.setattr(dicomweb, "transport", httpx.MockTransport(pacs.handler))

    assert client.get("/api/imaging/pacs/studies").status_code == 502  # no search terms
    assert client.get("/api/imaging/pacs/studies?study_date=March").status_code == 502
    [study] = client.get("/api/imaging/pacs/studies?patient_id=NHS9434765919").json()["studies"]
    assert study["patient_name"] == "Smith John" and study["modalities"] == ["CT", "SR"]
    [series] = client.get(f"/api/imaging/pacs/studies/{STUDY}/series").json()["series"]
    assert series["series_uid"] == SERIES and series["instances"] == 4
    assert client.get("/api/imaging/pacs/studies/not-a-uid/series").status_code == 502

    added = client.post("/api/imaging/pacs/retrieve", json={"study_uid": STUDY, "series_uid": SERIES}).json()["added"]
    series_id = added[0]["id"]
    assert added[0]["source"] == "pacs"
    detail = client.get(f"/api/imaging/series/{series_id}").json()["series"]
    assert detail["can_send_to_pacs"] and "original" not in detail

    report = client.post(f"/api/imaging/series/{series_id}/reports", json={
        "findings": "Normal.", "impression": "No abnormality.", "agreement": "not_used", "sign": True}).json()["report"]
    sent = client.post(f"/api/imaging/reports/{report['id']}/send")
    assert sent.status_code == 200, sent.text
    [stored] = pacs.stored
    sr = pydicom.dcmread(io.BytesIO(stored))
    assert sr.StudyInstanceUID == STUDY and sr.PatientID == "NHS9434765919" and sr.PatientName == "Smith^John"
    assert sr.AccessionNumber == "ACC123" and "PatientIdentityRemoved" not in sr
    referenced = sr.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0]
    assert referenced.SeriesInstanceUID == SERIES
    # The downloaded copy stays de-identified.
    download = pydicom.dcmread(io.BytesIO(client.get(f"/api/imaging/reports/{report['id']}/sr.dcm").content))
    assert download.PatientID == "" and download.StudyInstanceUID != STUDY
    assert client.get(f"/api/imaging/series/{series_id}").json()["series"]["reports"][0]["sent_at"]


def test_pacs_errors_are_explained(imaging, monkeypatch):
    monkeypatch.setattr(config, "DICOMWEB_URL", "https://pacs.example/dicom-web")
    monkeypatch.setattr(dicomweb, "transport", httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(dicomweb.PacsError, match="credentials"):
        dicomweb.search_studies(patient_id="1")
    monkeypatch.setattr(config, "DICOMWEB_URL", "")
    with pytest.raises(dicomweb.PacsError, match="No PACS"):
        dicomweb.search_studies(patient_id="1")


@pytest.mark.skipif(not os.environ.get("RUN_LIVE_PACS"), reason="set RUN_LIVE_PACS=1 to use the public Orthanc demo server")
def test_live_search_and_retrieve_from_the_orthanc_demo(imaging, monkeypatch):
    monkeypatch.setattr(config, "DICOMWEB_URL", "https://orthanc.uclouvain.be/demo/dicom-web")
    monkeypatch.setattr(dicomweb, "transport", None)
    [study] = dicomweb.search_studies(patient_id="HN_P001")
    assert "CT" in study["modalities"]
    ct = [s for s in dicomweb.search_series(study["study_uid"]) if s["modality"] == "CT"]
    assert ct
    result = imaging.post("/api/imaging/pacs/retrieve", json={"study_uid": study["study_uid"], "series_uid": ct[0]["series_uid"]})
    assert result.status_code == 200, result.text
    [added] = result.json()["added"]
    assert added["modality"] == "CT" and added["slices"] == int(ct[0]["instances"]) and added["source"] == "pacs"
    detail = imaging.get(f"/api/imaging/series/{added['id']}").json()["series"]
    assert detail["plane"] == "axial" and detail["deid"]["uids_replaced"] > 0
