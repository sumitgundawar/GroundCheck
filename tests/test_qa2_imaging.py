"""Red-team round 2, imaging: free text that survives de-identification, and files built to hurt the reader."""

from __future__ import annotations

import io
import os
import struct
import sys
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.imaging import deid, store  # noqa: E402
from test_imaging import instance  # noqa: E402


def edited(change) -> bytes:
    ds = pydicom.dcmread(io.BytesIO(instance(0, 1)))
    change(ds)
    out = io.BytesIO()
    ds.save_as(out)
    return out.getvalue()


def phi(ds):
    ds.add_new(0x00231010, "UT", "Call 555-123-4567 about John Smith")
    ds.add_new(0x00091001, "LO", "John Smith MRN 99887766")
    ds.ContentSequence = Sequence([Dataset()])
    ds.ContentSequence[0].TextValue = "John Smith 12/03/1948"
    ds.ContentSequence[0].add_new(0x00091002, "LO", "Smith J")
    ds.add_new(0x00101010, "LO", "John Smith 077")
    ds.add_new(0x00101020, "LO", "NHS 943 476 5919")
    ds.add_new(0x00101030, "LO", "call 555-123-4567")
    ds.add_new(0x00100040, "LO", "john@x.org")
    ds.add_new(0x00321030, "LO", "CT for john@example.org")


def test_free_text_in_unknown_attributes_and_sequence_items_does_not_survive():
    ds = pydicom.dcmread(io.BytesIO(edited(phi)))
    deid.deidentify(ds)
    text = " ".join(str(e.value) for e in ds.iterall() if e.VR in ("LT", "ST", "UT", "LO", "SH", "AS", "CS"))
    for leaked in ("John", "Smith", "555-123", "943 476", "john@", "example.org"):
        assert leaked not in text, (leaked, text)
    assert not any(e.tag.is_private for e in ds.iterall())


def test_the_patient_fields_kept_for_dose_checks_hold_only_what_they_should(database):
    [added] = store.import_series([("a.dcm", edited(phi))])["added"]
    assert added["patient"] == {}


def test_valid_age_sex_size_and_weight_are_kept():
    def valid(ds):
        ds.PatientAge, ds.PatientSex, ds.PatientSize, ds.PatientWeight = "042Y", "F", "1.7", "64.5"

    ds = pydicom.dcmread(io.BytesIO(edited(valid)))
    deid.deidentify(ds)
    assert (ds.PatientAge, ds.PatientSex, float(ds.PatientSize), float(ds.PatientWeight)) == ("042Y", "F", 1.7, 64.5)


def test_an_upload_label_is_de_identified(database):
    out = store.import_series([("a.dcm", edited(lambda ds: None))], label="Scan for NHS 943 476 5919 mail john@x.org")
    label = out["added"][0]["label"]
    assert "943 476" not in label and "john@x.org" not in label
    assert "943 476" not in store.list_series()[0]["label"]


def nifti(dims, dtype=4, bits=16, body=b"\0" * 100):
    h = bytearray(348)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, *dims)
    struct.pack_into("<h", h, 70, dtype)
    struct.pack_into("<h", h, 72, bits)
    struct.pack_into("<8f", h, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<f", h, 108, 352.0)
    h[344:348] = b"n+1\0"
    return bytes(h) + b"\0\0\0\0" + body


@pytest.mark.parametrize("dims", [[3, 30000, 30000, 1000, 1, 1, 1, 1], [4, 512, 512, 512, 2000, 1, 1, 1], [3, -5, 10, 10, 1, 1, 1, 1],
                                  [3, 0, 10, 10, 1, 1, 1, 1], [3, 32767, 32767, 32767, 1, 1, 1, 1]])
def test_a_tiny_file_claiming_huge_or_invalid_dimensions_is_refused_without_allocating(database, dims):
    with pytest.raises(store.ImagingError):
        store.import_series([("x.nii", nifti(dims))])


@pytest.mark.parametrize("change", [lambda ds: setattr(ds, "Rows", 65535) or setattr(ds, "Columns", 65535),
                                    lambda ds: setattr(ds, "NumberOfFrames", "100000"),
                                    lambda ds: setattr(ds, "BitsAllocated", 64), lambda ds: setattr(ds, "SamplesPerPixel", 3)])
def test_dicom_headers_that_lie_about_the_pixels_are_refused_not_crashed(database, change):
    with pytest.raises(store.ImagingError):
        store.import_series([("r.dcm", edited(change))])


def test_polyglots_and_mislabelled_files_are_refused(database):
    with pytest.raises(store.ImagingError):
        store.import_series([("p.png", b"\x89PNG\r\n\x1a\n" + b"\0" * 100 + instance(0, 1))])
    with pytest.raises(store.ImagingError):
        store.import_series([("x.nii", instance(0, 1))])
    with pytest.raises(store.ImagingError):
        store.import_series([("x.dcm", nifti([3, 4, 4, 4, 1, 1, 1, 1], body=b"\0" * 128))])
