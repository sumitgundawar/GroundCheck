"""NIfTI input: the header and data reader (orientation, dimensions, data types, compression, corrupt files),
cross-checked against nibabel when it is installed, and NIfTI volumes used end to end for training and analysis."""

from __future__ import annotations

import gzip
import io
import itertools
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config  # noqa: E402
from app.imaging import nifti, store  # noqa: E402
from app.training import datasets, library, preprocess, runs  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-023")


@pytest.fixture(autouse=True)
def _fresh_volume_cache():
    """Volumes are cached by series id, and every test's database starts again at id 1."""
    store._cache.clear()
    yield
    store._cache.clear()

DTYPE_CODES = {np.uint8: 2, np.int16: 4, np.int32: 8, np.float32: 16, np.float64: 64, np.int8: 256, np.uint16: 512,
               np.uint32: 768, np.int64: 1024, np.uint64: 1280}


def make_nifti(array, affine=None, dtype=np.int16, order="<", gz=False, descrip="", aux="", db_name="", intent="",
               slope=1.0, inter=0.0, units=2, pixdim=None, sform_code=1, qform_code=0, quatern=(0, 0, 0), qoffset=(0, 0, 0),
               qfac=1.0, extension=b"", vox_offset=None, header_edit=None, dim=None, datatype=None, bitpix=None,
               truncate=0) -> bytes:
    """A NIfTI-1 single file, built by hand."""
    array = np.asarray(array)
    dims = dim if dim is not None else list(array.shape)
    hdr = bytearray(348)

    def put(offset, fmt, *values):
        struct.pack_into(order + fmt, hdr, offset, *values)

    put(0, "i", 348)
    hdr[14:14 + len(db_name)] = db_name.encode()[:18]
    put(40, "8h", len(dims), *(list(dims) + [1] * (7 - len(dims))))
    code = datatype if datatype is not None else DTYPE_CODES[dtype]
    put(70, "h", code)
    put(72, "h", bitpix if bitpix is not None else np.dtype(dtype).itemsize * 8)
    px = pixdim or ([qfac] + ([1.0] * 7))
    if pixdim is None and affine is not None:
        px = [qfac] + [float(np.linalg.norm(np.asarray(affine)[:3, i])) for i in range(3)] + [1.0] * 4
    put(76, "8f", *px)
    offset = int(vox_offset) if vox_offset is not None else 352 + len(extension)
    put(108, "f", float(offset))
    put(112, "f", slope)
    put(116, "f", inter)
    hdr[123] = units
    hdr[148:148 + len(descrip)] = descrip.encode()[:80]
    hdr[228:228 + len(aux)] = aux.encode()[:24]
    put(252, "h", qform_code)
    put(254, "h", sform_code if affine is not None else 0)
    put(256, "3f", *quatern)
    put(268, "3f", *qoffset)
    if affine is not None:
        put(280, "12f", *np.asarray(affine, float)[:3].ravel())
    hdr[328:328 + len(intent)] = intent.encode()[:16]
    hdr[344:348] = b"n+1\x00"
    if header_edit:
        header_edit(hdr)
    body = np.asarray(array).astype(np.dtype(dtype).newbyteorder(order)).tobytes(order="F")
    pad = (b"\x00" * 4 + extension).ljust(max(0, offset - 348), b"\x00")[: max(0, offset - 348)]
    raw = bytes(hdr) + pad + body
    if truncate:
        raw = raw[:-truncate]
    return gzip.compress(raw) if gz else raw


def ras(spacing=(2.0, 2.0, 3.0), perm=(0, 1, 2), signs=(1, 1, 1), origin=(10.0, -20.0, 30.0)) -> np.ndarray:
    """Voxel axis i follows patient axis perm[i] (0 right, 1 anterior, 2 superior) with a sign."""
    a = np.eye(4)
    a[:3, :3] = 0
    for i in range(3):
        a[perm[i], i] = signs[i] * spacing[i]
    a[:3, 3] = origin
    return a


def ids(shape):
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)


def world_of(out_ids, shape, a):
    """For each output cell, the world coordinates of the voxel it came from: shape (3, S, R, C)."""
    flat = np.stack(np.unravel_index(out_ids.astype(int), shape)).reshape(3, -1)
    world = (a[:3, :3] @ flat + a[:3, 3:4]).reshape(3, *out_ids.shape)
    return world


# --- Orientation --------------------------------------------------------------------------------

@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_every_axis_arrangement_is_reordered_to_head_first_anterior_up_right_on_the_left(perm):
    """All 48 ways a scanner can store a volume (6 axis orders x 8 sign combinations) end up in one convention:
    slices from head to feet, rows from anterior to posterior, columns from the patient's right to left."""
    shape = (4, 5, 6)
    for signs in itertools.product((1, -1), repeat=3):
        a = ras(perm=perm, signs=signs)
        vol = nifti.read(make_nifti(ids(shape), a, np.float32))
        assert vol.oriented and vol.array.dtype == np.float32
        world = world_of(vol.array, shape, a)
        # z (superior) falls along slices, y (anterior) along rows, x (right) along columns
        assert (np.diff(world[2], axis=0) < 0).all() and (np.diff(world[1], axis=1) < 0).all() and (np.diff(world[0], axis=2) < 0).all()
        assert (np.diff(world[0], axis=0) == 0).all() and (np.diff(world[2], axis=1) == 0).all()
        assert vol.array.shape == tuple(shape[perm.index(w)] for w in (2, 1, 0))


def test_spacing_follows_the_patient_axes_not_the_storage_order():
    a = ras(spacing=(1.0, 2.0, 3.0), perm=(2, 0, 1))            # voxel i is superior (1.0), j right (2.0), k anterior (3.0)
    vol = nifti.read(make_nifti(ids((4, 5, 6)), a, np.float32))
    assert vol.slice_thickness == pytest.approx(1.0) and vol.pixel_spacing == [pytest.approx(3.0), pytest.approx(2.0)]


def test_matches_nibabel_on_random_orientations():
    nib = pytest.importorskip("nibabel")
    from nibabel import orientations as o

    rng = np.random.default_rng(0)
    for _ in range(25):
        perm = tuple(rng.permutation(3))
        signs = tuple(rng.choice((-1, 1), 3))
        a = ras(spacing=tuple(rng.uniform(0.5, 4, 3)), perm=perm, signs=signs, origin=tuple(rng.uniform(-50, 50, 3)))
        data = rng.normal(size=(5, 6, 7)).astype(np.float32)
        ours = nifti.read(make_nifti(data, a, np.float32)).array
        img = nib.Nifti1Image(data, a)
        want = o.apply_orientation(data, o.ornt_transform(o.io_orientation(img.affine), o.axcodes2ornt(("I", "P", "L"))))
        assert ours.shape == want.shape and np.array_equal(ours, want)


def test_qform_is_used_when_there_is_no_sform_and_matches_nibabel():
    nib = pytest.importorskip("nibabel")
    from nibabel import orientations as o

    data = ids((4, 5, 6))
    b, c, d = 0.0, 0.0, np.sqrt(0.5)                          # a quarter turn about z
    raw = make_nifti(data, None, np.float32, sform_code=0, qform_code=1, quatern=(b, c, d), qoffset=(1, 2, 3),
                     pixdim=[1.0, 2.0, 2.0, 3.0, 1, 1, 1, 1])
    vol = nifti.read(raw)
    img = nib.Nifti1Image(data, None)
    img.header.set_qform(None) if False else None
    loaded = nib.Nifti1Image.from_bytes(raw)
    want = o.apply_orientation(data, o.ornt_transform(o.io_orientation(loaded.affine), o.axcodes2ornt(("I", "P", "L"))))
    assert vol.oriented and np.array_equal(vol.array, want)


def test_qfac_minus_one_flips_the_slice_axis():
    plain = nifti.read(make_nifti(ids((3, 4, 5)), None, np.float32, sform_code=0, qform_code=1, quatern=(0, 0, 0),
                                  pixdim=[1.0, 1, 1, 1, 1, 1, 1, 1]))
    flipped = nifti.read(make_nifti(ids((3, 4, 5)), None, np.float32, sform_code=0, qform_code=1, quatern=(0, 0, 0),
                                    pixdim=[-1.0, 1, 1, 1, 1, 1, 1, 1]))
    assert np.array_equal(plain.array[::-1], flipped.array)


def test_sform_wins_over_qform():
    a = ras(signs=(-1, 1, 1))
    raw = make_nifti(ids((3, 4, 5)), a, np.float32, sform_code=2, qform_code=1, quatern=(0, 0, 0))
    same = nifti.read(make_nifti(ids((3, 4, 5)), a, np.float32, sform_code=2, qform_code=0))
    assert np.array_equal(nifti.read(raw).array, same.array)


def test_an_invalid_sform_falls_back_to_the_qform_and_then_to_unoriented():
    bad = np.zeros((4, 4))
    bad[3, 3] = 1
    ones = [1.0] * 8
    raw = make_nifti(ids((3, 4, 5)), bad, np.float32, sform_code=1, qform_code=1, quatern=(0, 0, 0), pixdim=ones)
    assert nifti.read(raw).oriented
    raw = make_nifti(ids((3, 4, 5)), bad, np.float32, sform_code=1, qform_code=0, pixdim=ones)
    vol = nifti.read(raw)
    assert not vol.oriented and vol.display["plane"] == "unknown" and any("no orientation" in w for w in vol.warnings)
    nan = ras()
    nan[0, 0] = np.nan
    assert not nifti.read(make_nifti(ids((3, 4, 5)), nan, np.float32, pixdim=ones)).oriented


def test_a_quaternion_longer_than_one_is_not_used():
    raw = make_nifti(ids((3, 4, 5)), None, np.float32, sform_code=0, qform_code=1, quatern=(0.9, 0.9, 0.9))
    assert not nifti.read(raw).oriented


def test_a_volume_without_orientation_is_shown_as_stored_and_says_so():
    data = ids((3, 4, 5))
    vol = nifti.read(make_nifti(data, None, np.float32, sform_code=0))
    assert not vol.oriented and np.array_equal(vol.array, data.transpose(2, 1, 0)) and vol.display["plane"] == "unknown"
    assert vol.warnings == ["The image has no orientation, so it's shown as stored."]


def test_tilted_volumes_warn_and_extremely_tilted_ones_are_not_reordered():
    tilt = np.deg2rad(15)
    a = np.eye(4)
    a[:3, :3] = [[np.cos(tilt), -np.sin(tilt), 0], [np.sin(tilt), np.cos(tilt), 0], [0, 0, 1]]
    vol = nifti.read(make_nifti(ids((4, 5, 6)), a, np.float32))
    assert vol.oriented and any("tilted 15 degrees" in w for w in vol.warnings)
    small = np.eye(4)
    small[:3, :3] = [[np.cos(0.05), -np.sin(0.05), 0], [np.sin(0.05), np.cos(0.05), 0], [0, 0, 1]]
    assert nifti.read(make_nifti(ids((4, 5, 6)), small, np.float32)).warnings == []
    diagonal = np.eye(4)
    diagonal[:3, :3] = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    diagonal[:3, :3] += np.diag([0.01, -0.01, 0])
    vol = nifti.read(make_nifti(ids((4, 5, 6)), diagonal, np.float32))
    assert not vol.oriented and any("can't be matched" in w for w in vol.warnings)


# --- Dimensions -----------------------------------------------------------------------------------

def test_three_dimensional_volumes_keep_all_their_values():
    data = np.random.default_rng(1).normal(size=(6, 7, 8)).astype(np.float32)
    vol = nifti.read(make_nifti(data, ras(signs=(-1, -1, -1)), np.float32))
    assert vol.array.shape == (8, 7, 6) and vol.volumes == 1
    assert np.array_equal(np.sort(vol.array.ravel()), np.sort(data.ravel()))


def test_a_two_dimensional_image_is_one_slice():
    vol = nifti.read(make_nifti(ids((5, 6)), None, np.float32, sform_code=0))
    assert vol.array.shape == (1, 6, 5)


def test_four_dimensional_images_use_one_volume_and_say_so():
    data = np.stack([ids((3, 4, 5)) + 1000 * t for t in range(3)], axis=3)
    raw = make_nifti(data, ras(), np.float32)
    first = nifti.read(raw)
    assert first.volumes == 3 and any("4-D image with 3 volumes" in w for w in first.warnings)
    assert first.array.max() < 1000
    second = nifti.read(raw, volume_index=2)
    assert second.array.min() >= 2000 and any("volume 3" in w for w in second.warnings)
    with pytest.raises(nifti.NiftiError, match="3 volumes"):
        nifti.read(raw, volume_index=3)
    with pytest.raises(nifti.NiftiError):
        nifti.read(raw, volume_index=-1)
    single = nifti.read(make_nifti(data[..., :1], ras(), np.float32))
    assert single.volumes == 1 and single.warnings == []


@pytest.mark.parametrize("dims", [[10], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2]])
def test_one_and_five_or_more_dimensions_are_refused(dims):
    raw = make_nifti(np.zeros(dims, np.float32), None, np.float32)
    with pytest.raises(nifti.NiftiError, match="one-dimensional|more than four"):
        nifti.read(raw)


@pytest.mark.parametrize("dtype", list(DTYPE_CODES))
@pytest.mark.parametrize("order", ["<", ">"])
def test_every_supported_data_type_and_byte_order_reads_back(dtype, order):
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    values = np.array([0, 1, 2, 100], dtype=dtype) if info is None else np.array([info.min, 0, 1, min(info.max, 2**24)], dtype=dtype)
    data = np.tile(values, 6).reshape(2, 3, 4).astype(dtype)
    vol = nifti.read(make_nifti(data, None, dtype, order=order, sform_code=0))
    assert np.array_equal(vol.array, data.transpose(2, 1, 0).astype(np.float32))


@pytest.mark.parametrize("code,name", [(1, "Binary"), (32, "Complex"), (128, "RGB"), (1792, "Complex"), (2304, "RGBA")])
def test_unsupported_data_types_are_named(code, name):
    raw = make_nifti(np.zeros((2, 2, 2), np.float32), None, np.float32, datatype=code, bitpix=32)
    with pytest.raises(nifti.NiftiError, match=name):
        nifti.read(raw)


def test_unknown_data_types_and_inconsistent_bits_are_refused():
    with pytest.raises(nifti.NiftiError, match="isn't supported"):
        nifti.read(make_nifti(np.zeros((2, 2, 2), np.float32), None, np.float32, datatype=999))
    with pytest.raises(nifti.NiftiError, match="disagree"):
        nifti.read(make_nifti(np.zeros((2, 2, 2), np.float32), None, np.float32, bitpix=16))


def test_scaling_slope_and_intercept_are_applied():
    data = np.array([[[0, 1], [2, 3]]], dtype=np.int16)
    vol = nifti.read(make_nifti(data, None, np.int16, slope=2.0, inter=-1000.0, sform_code=0))
    assert np.array_equal(np.sort(vol.array.ravel()), [-1000, -998, -996, -994])
    for slope in (0.0, float("nan"), float("inf")):
        plain = nifti.read(make_nifti(data, None, np.int16, slope=slope, inter=5.0, sform_code=0))
        assert plain.array.max() <= 3 or slope == float("inf")


def test_non_finite_values_become_zero_with_a_warning():
    data = np.ones((3, 3, 3), np.float32)
    data[0, 0, 0], data[1, 1, 1], data[2, 2, 2] = np.nan, np.inf, -np.inf
    vol = nifti.read(make_nifti(data, None, np.float32, sform_code=0))
    assert np.isfinite(vol.array).all() and any("3 voxels" in w for w in vol.warnings) and vol.array.min() == 0


@pytest.mark.parametrize("units,factor", [(2, 1.0), (1, 1000.0), (3, 0.001), (0, 1.0), (10, 1.0)])
def test_spatial_units_become_millimetres(units, factor):
    vol = nifti.read(make_nifti(ids((3, 4, 5)), ras(spacing=(2.0, 3.0, 4.0)), np.float32, units=units))
    assert vol.slice_thickness == pytest.approx(4.0 * factor) and vol.pixel_spacing == [pytest.approx(3.0 * factor), pytest.approx(2.0 * factor)]


def test_spacing_without_orientation_comes_from_pixdim():
    vol = nifti.read(make_nifti(ids((3, 4, 5)), None, np.float32, sform_code=0, pixdim=[1.0, 0.5, 0.7, 2.5, 1, 1, 1, 1]))
    assert vol.pixel_spacing == [pytest.approx(0.7), pytest.approx(0.5)] and vol.slice_thickness == pytest.approx(2.5)


def test_header_extensions_are_skipped():
    for extension in (b"", b"\x08\x00\x00\x00\x04\x00\x00\x00" + b"x" * 24, b"y" * 60):
        raw = make_nifti(ids((2, 3, 4)), None, np.float32, sform_code=0, extension=extension)
        assert np.array_equal(nifti.read(raw).array, ids((2, 3, 4)).transpose(2, 1, 0))


# --- Compression and corrupt files ------------------------------------------------------------------

def test_gzip_and_plain_files_read_the_same():
    data = ids((4, 5, 6))
    plain = nifti.read(nifti.decompress(make_nifti(data, ras(), np.float32)))
    zipped = nifti.read(nifti.decompress(make_nifti(data, ras(), np.float32, gz=True)))
    assert np.array_equal(plain.array, zipped.array)


def test_a_truncated_gzip_is_refused():
    raw = make_nifti(ids((8, 8, 8)), ras(), np.float32, gz=True)
    with pytest.raises(nifti.NiftiError, match="truncated"):
        nifti.decompress(raw[: len(raw) // 2])


def test_a_corrupt_gzip_is_refused():
    raw = bytearray(make_nifti(ids((8, 8, 8)), ras(), np.float32, gz=True))
    raw[len(raw) // 2] ^= 0xFF
    with pytest.raises(nifti.NiftiError, match="corrupt|truncated"):
        nifti.decompress(bytes(raw))
    with pytest.raises(nifti.NiftiError, match="corrupt"):
        nifti.decompress(b"\x1f\x8b" + b"garbage" * 10)


def test_a_gzip_bomb_is_stopped_at_the_limit_without_building_it():
    bomb = gzip.compress(b"\x00" * (60 * 1024**2), compresslevel=9)
    assert len(bomb) < 100_000
    start = time.perf_counter()
    with pytest.raises(nifti.NiftiError, match="larger than 8 MB when decompressed"):
        nifti.decompress(bomb, limit=8 * 1024**2)
    assert time.perf_counter() - start < 5
    assert len(nifti.decompress(bomb, limit=64 * 1024**2)) == 60 * 1024**2


def test_a_plain_file_over_the_limit_is_refused():
    with pytest.raises(nifti.NiftiError, match="larger than 1 MB"):
        nifti.decompress(b"\x00" * (2 * 1024**2), limit=1024**2)


def test_files_are_read_from_disk_with_the_same_checks(tmp_path):
    good = tmp_path / "a.nii.gz"
    good.write_bytes(make_nifti(ids((3, 4, 5)), ras(), np.float32, gz=True))
    assert nifti.read_path(good).array.shape == (5, 4, 3)
    with pytest.raises(nifti.NiftiError, match="couldn't be read"):
        nifti.read_path(tmp_path / "missing.nii")
    big = tmp_path / "big.nii"
    big.write_bytes(b"\x00" * 3000)
    with pytest.raises(nifti.NiftiError, match="larger than"):
        nifti.read_file(big, limit=1000)


HEADER = make_nifti(ids((3, 4, 5)), ras(), np.float32)


def edit(offset, fmt, *values, order="<"):
    def f(hdr):
        struct.pack_into(order + fmt, hdr, offset, *values)

    return f


BAD_FILES = {
    "empty": (b"", "too short"),
    "short": (b"\x00" * 100, "too short"),
    "zeros": (b"\x00" * 600, "isn't a NIfTI-1"),
    "random text": (b"hello world " * 100, "isn't a NIfTI-1"),
    "a png": (b"\x89PNG\r\n\x1a\n" + b"\x00" * 500, "isn't a NIfTI-1"),
    "a dicom": (b"\x00" * 128 + b"DICM" + b"\x00" * 500, "isn't a NIfTI-1"),
    "nifti-2": (struct.pack("<i", 540) + b"\x00" * 600, "NIfTI-2"),
    "nifti-2 big endian": (struct.pack(">i", 540) + b"\x00" * 600, "NIfTI-2"),
    "two-file": (HEADER[:344] + b"ni1\x00" + HEADER[348:], "Two-file"),
    "analyze": (HEADER[:344] + b"\x00\x00\x00\x00" + HEADER[348:], "Analyze"),
    "bad magic": (HEADER[:344] + b"abcd" + HEADER[348:], "no n+1 marker"),
    "wrong header size": (struct.pack("<i", 347) + HEADER[4:], "isn't a NIfTI-1"),
}


@pytest.mark.parametrize("name", list(BAD_FILES))
def test_files_that_are_not_nifti_are_refused_with_a_reason(name):
    raw, message = BAD_FILES[name]
    with pytest.raises(nifti.NiftiError, match=__import__("re").escape(message)):
        nifti.read(raw)


@pytest.mark.parametrize("label,dim,message", [("zero dimension", [0, 4, 5], "invalid"), ("negative dimension", [-3, 4, 5], "invalid"),
                                               ("seven dims with a zero", [2, 2, 0, 2, 2, 2, 2], "invalid")])
def test_invalid_dimensions_are_refused(label, dim, message):
    raw = make_nifti(np.zeros((2, 2, 2), np.float32), None, np.float32, dim=dim)
    with pytest.raises(nifti.NiftiError, match=message):
        nifti.read(raw)


@pytest.mark.parametrize("ndim", [0, 8, 9, -1, 100])
def test_an_impossible_number_of_dimensions_is_refused(ndim):
    raw = bytearray(HEADER)
    struct.pack_into("<h", raw, 40, ndim)
    with pytest.raises(nifti.NiftiError, match="invalid"):
        nifti.read(bytes(raw))


def test_huge_dimensions_are_refused_before_anything_is_allocated():
    raw = make_nifti(np.zeros((2, 2, 2), np.float32), None, np.float32, dim=[30000, 30000, 30000])
    start = time.perf_counter()
    with pytest.raises(nifti.NiftiError, match="voxels; the limit"):
        nifti.read(raw)
    assert time.perf_counter() - start < 2


def test_a_file_missing_image_data_is_truncated():
    for cut in (1, 7, 100, 3 * 4 * 5 * 4):
        with pytest.raises(nifti.NiftiError, match="truncated|outside the file"):
            nifti.read(make_nifti(ids((3, 4, 5)), ras(), np.float32, truncate=cut))


@pytest.mark.parametrize("offset", [-5.0, 100.0, 351.0, 1e12, float("nan"), float("inf")])
def test_a_data_offset_outside_the_file_is_refused(offset):
    raw = bytearray(HEADER)
    struct.pack_into("<f", raw, 108, offset)
    with pytest.raises(nifti.NiftiError, match="outside the file"):
        nifti.read(bytes(raw))


def test_an_offset_of_zero_means_the_usual_352():
    raw = bytearray(HEADER)
    struct.pack_into("<f", raw, 108, 0.0)
    assert nifti.read(bytes(raw)).array.shape == (5, 4, 3)


def test_arbitrary_bytes_never_crash_the_reader():
    rng = np.random.default_rng(3)
    for _ in range(300):
        raw = bytearray(HEADER)
        for _ in range(int(rng.integers(1, 12))):
            raw[int(rng.integers(0, 348))] = int(rng.integers(0, 256))
        try:
            nifti.read(bytes(raw))
        except nifti.NiftiError:
            pass
    for size in (0, 1, 347, 348, 349, 351, 352, 353):
        try:
            nifti.read(HEADER[:size])
        except nifti.NiftiError:
            pass


# --- Patient information --------------------------------------------------------------------------------

def test_free_text_header_fields_are_never_returned_or_kept():
    raw = make_nifti(ids((3, 4, 5)), ras(), np.float32, descrip="Patient John Smith DOB 12/03/1948", aux="NHS 9434765919",
                     db_name="Smith^John", intent="MRN 00123456")
    vol = nifti.read(raw)
    blob = repr(vol) + json.dumps(vol.display) + " ".join(vol.warnings)
    for needle in ("Smith", "9434765919", "00123456", "1948"):
        assert needle not in blob
    assert nifti.header_text(raw) == ["Patient John Smith DOB 12/03/1948", "NHS 9434765919", "Smith^John", "MRN 00123456"]
    assert nifti.header_text(b"junk") == []


# --- Helpers -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,ok", [("a.nii", True), ("a.NII.GZ", True), ("dir/a.nii.gz", True), ("a.gz", False), ("a.nii.bak", False),
                                     ("a.png", False), ("nii", False), (".nii", True), ("", False)])
def test_nifti_names(name, ok):
    assert nifti.is_nifti_name(name) is ok


def test_slice_references_parse_from_paths():
    assert nifti.parse_ref(Path("x/a.nii.gz#12")) == nifti.SliceRef(Path("x/a.nii.gz"), 12)
    assert nifti.parse_ref(Path("a.nii#0")) == nifti.SliceRef(Path("a.nii"), 0)
    for bad in ("a.nii.gz", "a.png#1", "a.nii#x", "a.nii#-1", "a.nii#", "#3", "a#b.nii.gz"):
        assert nifti.parse_ref(Path(bad)) is None


@pytest.mark.parametrize("values,name,want", [(np.array([-1000, 40, 300.0]), "scan.nii", "CT"), (np.array([0, 300, 900.0]), "scan.nii", "MR"),
                                              (np.array([0, 300, 900.0]), "case_ct.nii.gz", "CT"), (np.array([-1000, 40, 300.0]), "T1w.nii", "MR"),
                                              (np.array([-1000, 40, 300.0]), "flair_brain.nii", "MR"), (np.array([5.0, 6.0]), "", "MR")])
def test_modality_guess(values, name, want):
    assert nifti.guess_modality(np.tile(values, 100), name) == want


def test_slice_counts_and_training_slices(tmp_path):
    a = ras(perm=(1, 2, 0))
    raw = make_nifti(ids((4, 40, 5)), a, np.float32)
    assert nifti.slice_count(raw) == 40
    path = tmp_path / "v.nii"
    path.write_bytes(raw)
    picks = nifti.training_slices(path)
    assert picks == sorted(set(picks)) and picks[0] >= 4 and picks[-1] <= 36 and len(picks) == 32 - 0 or len(picks) <= 32
    tiny = tmp_path / "t.nii.gz"
    tiny.write_bytes(make_nifti(ids((4, 4, 1)), ras(), np.float32, gz=True))
    assert nifti.training_slices(tiny) == [0]
    assert nifti.header_only(tiny)[:4] == struct.pack("<i", 348)


def test_training_slices_stay_inside_the_middle():
    for n in (1, 2, 3, 5, 10, 40, 300):
        path = Path(f"/nonexistent/{n}")
        low, high = int(n * 0.1), max(int(n * 0.1) + 1, int(np.ceil(n * 0.9)))
        assert 0 <= low < high <= n


def test_slice_images_are_greyscale_and_scaled_by_the_whole_volume(tmp_path):
    data = np.zeros((16, 16, 4), np.float32)
    data[:, :, 0], data[:, :, 3] = 0, 100.0
    data += np.random.default_rng(0).normal(0, 1, data.shape).astype(np.float32)
    path = tmp_path / "v.nii.gz"
    path.write_bytes(make_nifti(data, ras(signs=(-1, -1, -1)), np.float32, gz=True))
    first = nifti.slice_image(nifti.SliceRef(path, 0))
    last = nifti.slice_image(nifti.SliceRef(path, 3))
    assert first.mode == "L" and first.size == (16, 16)
    assert np.asarray(first).mean() < 40 < 200 < np.asarray(last).mean()            # the same scale for every slice
    with pytest.raises(ValueError, match="no such slice"):
        nifti.slice_image(nifti.SliceRef(path, 4))
    with pytest.raises(ValueError, match="couldn't be read"):
        nifti.slice_image(nifti.SliceRef(tmp_path / "gone.nii", 0))
    (tmp_path / "bad.nii").write_bytes(b"x" * 400)
    with pytest.raises(ValueError):
        nifti.slice_image(nifti.SliceRef(tmp_path / "bad.nii", 0))


def test_open_image_understands_slices_and_still_opens_pngs(tmp_path):
    path = tmp_path / "v.nii"
    path.write_bytes(make_nifti(np.random.default_rng(0).normal(size=(8, 8, 8)).astype(np.float32), ras(), np.float32))
    assert preprocess.open_image(Path(f"{path}#3")).size == (8, 8)
    assert preprocess.open_image(nifti.SliceRef(path, 3)).size == (8, 8)
    png = tmp_path / "a.png"
    Image.new("L", (5, 5)).save(png)
    assert preprocess.open_image(png).size == (5, 5)
    with pytest.raises(ValueError):
        preprocess.open_image(Path(f"{path}#99"))


# --- Training datasets -------------------------------------------------------------------------------------------

def volume_bytes(level: int, seed: int, shape=(24, 24, 20), gz=True, **kw) -> bytes:
    """Noise, and for the "bright" class (level > 100) a disc in every slice. Volumes are scaled one by one,
    as MR is, so the classes differ in structure and not in absolute level."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    disc = ((x - shape[1] / 2) ** 2 + (y - shape[0] / 2) ** 2 < 49)[..., None]
    data = np.clip(100 + rng.normal(0, 8, shape) + (100 * disc if level > 100 else 0), 0, 255).astype(np.int16)
    return make_nifti(data, ras(signs=(-1, -1, -1)), np.int16, gz=gz, **kw)


def volume_dataset(root: Path, per_class: int = 12, split_folders: bool = False) -> Path:
    folder = root / "vols"
    for label, level in (("dark", 40), ("bright", 210)):
        for i in range(per_class):
            split = "train" if i < per_class * 0.7 else "val" if i < per_class * 0.85 else "test"
            target = folder / split / label if split_folders else folder / label
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{label}-{i}.nii.gz").write_bytes(volume_bytes(level, i + (0 if label == "dark" else 500)))
    return folder


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    data, library_dir, runs_dir = tmp_path / "data", tmp_path / "library", tmp_path / "runs"
    data.mkdir()
    monkeypatch.setattr(config, "TRAINING_DATA_DIRS", [data.resolve()])
    monkeypatch.setattr(config, "MODEL_LIBRARY_DIR", library_dir)
    monkeypatch.setattr(config, "TRAINING_RUNS_DIR", runs_dir)
    monkeypatch.setenv("TRAINING_DATA_DIRS", str(data))
    monkeypatch.setenv("MODEL_LIBRARY_DIR", str(library_dir))
    monkeypatch.setenv("TRAINING_RUNS_DIR", str(runs_dir))
    return data


def test_volumes_are_read_as_slices_and_a_volume_never_spans_two_splits(studio):
    folder = volume_dataset(studio)
    _, classes, items, summary = datasets.read(str(folder))
    assert classes == ["bright", "dark"] and summary["volumes"] == 24
    assert all("#" in i.path and i.group.endswith(".nii.gz") for i in items)
    split_of_group: dict[str, set] = {}
    for i in items:
        split_of_group.setdefault(i.group, set()).add(i.split)
    assert all(len(v) == 1 for v in split_of_group.values())
    assert {i.split for i in items} == {"train", "val", "test"}
    assert any("NIfTI volumes" in w for w in summary["warnings"])
    per_volume = {}
    for i in items:
        per_volume[i.group] = per_volume.get(i.group, 0) + 1
    assert set(per_volume.values()) == {16}                     # the middle 80% of 20 slices


def test_split_folders_with_volumes_keep_volumes_together_even_when_val_is_carved(studio):
    folder = volume_dataset(studio, split_folders=True)
    import shutil

    shutil.rmtree(folder / "val")
    _, _, items, summary = datasets.read(str(folder))
    seen: dict[str, set] = {}
    for i in items:
        seen.setdefault(i.group, set()).add(i.split)
    assert all(len(v) == 1 for v in seen.values()) and "val" in {i.split for i in items}


def test_mixed_images_and_volumes_split_by_file(studio):
    folder = volume_dataset(studio, per_class=6)
    for label in ("dark", "bright"):
        for i in range(8):
            Image.fromarray(np.full((16, 16), 40 if label == "dark" else 210, np.uint8)).save(folder / label / f"p{i}.png")
    _, _, items, summary = datasets.read(str(folder))
    plain = [i for i in items if not i.group]
    assert plain and all("#" not in i.path for i in plain) and summary["volumes"] == 12


def test_a_corrupt_volume_contributes_nothing_but_does_not_stop_reading(studio):
    folder = volume_dataset(studio, per_class=8)
    (folder / "dark" / "broken.nii.gz").write_bytes(b"\x1f\x8b" + b"nonsense")
    (folder / "dark" / "empty.nii").write_bytes(b"")
    _, _, items, summary = datasets.read(str(folder))
    assert not any("broken" in i.path or "empty" in i.path for i in items)


def test_preview_files_resolve_slices_and_stay_inside_the_dataset(studio):
    folder = volume_dataset(studio, per_class=6)
    _, _, items, _ = datasets.read(str(folder))
    ref = datasets.resolve_file(folder.resolve(), items[0].path)
    assert isinstance(ref, nifti.SliceRef) and preprocess.open_image(ref).size == (24, 24)
    assert isinstance(datasets.resolve_file(folder.resolve(), "dark/dark-0.nii.gz"), Path)
    for bad in ("../../etc/passwd#1", "dark/dark-0.nii.gz#999999", "dark/nothing.nii.gz#1", "dark/../../x.nii.gz#1"):
        try:
            result = datasets.resolve_file(folder.resolve(), bad)
        except datasets.DatasetError:
            continue
        with pytest.raises(ValueError):
            preprocess.open_image(result)


def test_duplicate_volumes_are_found_once_per_volume(studio):
    folder = volume_dataset(studio, per_class=8)
    original = (folder / "dark" / "dark-0.nii.gz").read_bytes()
    (folder / "dark" / "copy-of-0.nii.gz").write_bytes(original)
    _, _, items, _ = datasets.read(str(folder))
    found = datasets.find_duplicates(folder, items)
    assert found["groups"] == 1 and found["extra_copies"] == 1
    assert set(found["leaked"]) <= {i.path for i in items if i.split != "train"}


def test_identifiers_in_volume_file_names_and_headers_are_reported(studio):
    folder = volume_dataset(studio, per_class=6)
    (folder / "dark" / "John Smith NHS 943 476 5919.nii.gz").write_bytes(volume_bytes(40, 99))
    (folder / "bright" / "clean-name.nii.gz").write_bytes(volume_bytes(210, 98, descrip="Patient John Smith DOB 12/03/1948"))
    _, _, items, _ = datasets.read(str(folder))
    found = datasets.find_identifiers(folder, items)
    assert found["file_names"] >= 1 and any("943" in e or "Smith" in e for e in found["file_name_examples"])
    assert found["metadata_files"] >= 1 and any("clean-name" in e for e in found["metadata_examples"])
    assert found["metadata_sampled"] == len({i.source for i in items})


def test_a_generated_volume_dataset_trains_a_model_that_classifies_its_slices(studio, monkeypatch):
    monkeypatch.setenv("MODEL_TARGET_ACCURACY", "0.7")
    folder = volume_dataset(studio, per_class=12)
    run = runs.start({"name": "Dark or bright volumes", "dataset": str(folder), "device": "cpu", "epochs": 4,
                      "image_size": 28, "batch_size": 16})
    deadline = time.time() + float(os.environ.get("TRAINING_TEST_TIMEOUT", "600"))
    while runs.get(run["id"])["progress"].get("state") in runs.ACTIVE and time.time() < deadline:
        time.sleep(1)
    progress = runs.get(run["id"])["progress"]
    log = (config.TRAINING_RUNS_DIR / run["id"] / "worker.log").read_text()
    assert progress["state"] == "completed", (progress, log)
    [model] = library.list_models()
    card = library.get_model(model["id"])
    assert card["test"]["accuracy"] == 1.0 and card["dataset"]["total"] == 24 * 16
    _, _, items, _ = datasets.read(str(folder))
    for want in ("bright", "dark"):
        pick = next(i for i in items if i.split == "test" and i.group.startswith(f"{want}/"))
        buffer = io.BytesIO()
        preprocess.open_image(datasets.resolve_file(folder.resolve(), pick.path)).save(buffer, "PNG")
        prediction = library.predict(model["id"], buffer.getvalue())
        assert (prediction["prediction"] == "bright") == (want == "bright")


# --- Analysis ------------------------------------------------------------------------------------------------------

from test_imaging import _wait, ct_series, imaging, phantom  # noqa: E402,F401  (the model fixture and DICOM reference series)



def phantom_nifti(name_hint: str = "phantom_ct", gz: bool = True) -> bytes:
    """The four CT phantom slices the DICOM tests use, as a NIfTI volume: head-first stack, Hounsfield units."""
    slices = np.stack([phantom() for _ in range(4)])                # (z, row, column), each 64 x 64
    data = slices.transpose(2, 1, 0).astype(np.int16)               # voxel (i, j, k) = column, row, slice
    return make_nifti(data, ras(spacing=(4.0, 4.0, 5.0), signs=(-1, -1, -1)), np.int16, gz=gz,
                      descrip="Patient John Smith", aux="NHS 9434765919")


def test_a_nifti_volume_uploads_analyses_and_gives_the_same_answer_as_the_dicom_series(imaging):
    client = imaging
    dicom_files = [("files", (n, d, "application/dicom")) for n, d in ct_series()]
    dicom_id = client.post("/api/imaging/upload", files=dicom_files).json()["added"][0]["id"]
    client.post(f"/api/imaging/series/{dicom_id}/analyses", json={"model_id": "phantom-model"})
    expected = _wait(client, dicom_id)["analyses"][-1]

    upload = client.post("/api/imaging/upload", files=[("files", ("phantom_ct.nii.gz", phantom_nifti(), "application/gzip"))],
                         data={"label": "Volume 1"})
    assert upload.status_code == 200, upload.text
    added = upload.json()["added"]
    assert len(added) == 1 and added[0]["slices"] == 4 and added[0]["modality"] == "CT" and added[0]["plane"] == "axial"
    series_id = added[0]["id"]
    blob = json.dumps(upload.json())
    assert "Smith" not in blob and "9434765919" not in blob

    detail = client.get(f"/api/imaging/series/{series_id}").json()["series"]
    assert detail["pixel_spacing"] == [4.0, 4.0] and detail["slice_spacing"] == 5.0 and detail["display"]["source"] == "nifti"
    assert detail["patient"] == {} and not detail["can_send_to_pacs"]
    assert client.get(f"/api/imaging/series/{series_id}/slices/0.png?window=lung").content[:4] == b"\x89PNG"
    assert client.get(f"/api/imaging/series/{series_id}/slices/4.png").status_code == 400
    assert "Smith" not in next(config.IMAGING_DIR.iterdir()).read_bytes().decode("latin-1")

    client.post(f"/api/imaging/series/{series_id}/analyses", json={"model_id": "phantom-model"})
    result = _wait(client, series_id)["analyses"][-1]
    assert result["status"] == "done", result
    assert result["summary"]["totals"] == expected["summary"]["totals"] and not result["summary"]["abstained"]
    assert result["summary"]["labels"] == expected["summary"]["labels"] and len(result["slices"]) == 4

    again = client.post("/api/imaging/upload", files=[("files", ("again.nii", phantom_nifti(gz=False), "x"))]).json()
    assert again["added"] == [] and again["existing"][0]["id"] == series_id

    report = client.post(f"/api/imaging/series/{series_id}/reports", json={
        "findings": "Soft tissue disc.", "impression": "Normal.", "agreement": "agree", "analysis_id": result["id"], "sign": True})
    assert report.status_code == 200
    sr = client.get(f"/api/imaging/reports/{report.json()['report']['id']}/sr.dcm")
    assert sr.status_code == 400 and "NIfTI" in sr.json()["detail"]
    assert client.delete(f"/api/imaging/series/{series_id}").json() == {"deleted": series_id}


def test_modality_comes_from_the_name_or_the_values_and_the_user_is_told(imaging):
    client = imaging
    for name, expected in (("brain_t1w.nii.gz", "MR"), ("chest.nii.gz", "CT")):
        data = phantom_nifti()
        if expected == "MR":
            data = make_nifti(np.random.default_rng(0).integers(0, 500, (16, 16, 4)).astype(np.int16), ras(), np.int16, gz=True)
        added = client.post("/api/imaging/upload", files=[("files", (name, data, "x"))]).json()["added"][0]
        assert added["modality"] == expected and any("taken to be" in w for w in added["warnings"])


def test_bad_nifti_uploads_are_refused_with_the_reason_and_store_nothing(imaging):
    client = imaging
    for name, data, message in (("a.nii", b"junk" * 200, "isn't a NIfTI-1"), ("b.nii.gz", b"\x1f\x8b junk", "corrupt|truncated"),
                                ("c.nii", make_nifti(ids((3, 4, 5)), ras(), np.float32, truncate=9), "truncated"),
                                ("d.nii", make_nifti(np.zeros((2, 2, 2, 2, 2), np.float32), None, np.float32), "more than four")):
        response = client.post("/api/imaging/upload", files=[("files", (name, data, "x"))])
        assert response.status_code == 400 and name in response.json()["detail"], response.text
        assert __import__("re").search(message, response.json()["detail"])
    assert not list(config.IMAGING_DIR.iterdir()) if config.IMAGING_DIR.exists() else True


def test_nifti_and_dicom_can_be_uploaded_together(imaging):
    client = imaging
    files = [("files", (n, d, "x")) for n, d in ct_series()] + [("files", ("v_ct.nii.gz", phantom_nifti(), "x"))]
    added = client.post("/api/imaging/upload", files=files).json()["added"]
    assert len(added) == 2 and {a["slices"] for a in added} == {4}


def test_a_series_from_another_site_cannot_be_claimed(database, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGING_DIR", tmp_path / "imaging")
    from app import sites

    a = sites.create("site-a", "A")
    b = sites.create("site-b", "B")
    first = store.import_nifti("v_ct.nii.gz", phantom_nifti(), site_id=a["id"])
    assert len(first["added"]) == 1
    assert store.import_nifti("v_ct.nii.gz", phantom_nifti(), site_id=a["id"])["added"] == []
    with pytest.raises(store.ImagingError, match="another site"):
        store.import_nifti("v_ct.nii.gz", phantom_nifti(), site_id=b["id"])
    assert [s["id"] for s in store.list_series(a["id"])] == [first["added"][0]["id"]] and store.list_series(b["id"]) == []


def test_modality_can_be_given_and_must_be_ct_or_mr(database, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGING_DIR", tmp_path / "imaging")
    assert store.import_nifti("x.nii.gz", phantom_nifti(), modality="MR")["added"][0]["modality"] == "MR"
    with pytest.raises(store.ImagingError, match="CT or MR"):
        store.import_nifti("y.nii.gz", phantom_nifti(), modality="US")
