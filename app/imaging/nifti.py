"""Reading NIfTI-1 volumes (.nii and .nii.gz) for imaging analysis and training.

Pure numpy: no nibabel needed. (If nibabel is installed the tests cross-check
this reader against it.) NIfTI-1 single files with real-valued or integer data
of 2 to 4 dimensions are understood; NIfTI-2, two-file (.hdr/.img) images,
complex and RGB data are refused with a clear message.

- Orientation: voxel axes are mapped to the patient axes through the sform or
  qform and the array is reordered, without resampling, to the viewer's
  convention (see dicom.display_transform): axial slices stacked head first,
  patient's right on the image left, anterior at the top. A volume tilted by
  more than 10 degrees is reordered to the nearest axes and a warning says so.
  A volume without orientation is shown as stored, and says so.
- 4-D images (time, diffusion): one volume is used (the first), with a warning.
- Patient information: nothing from the header's free-text fields (description,
  auxiliary file, database name, intent name) is kept or returned.
- Limits: decompressed size and voxel count are capped before anything large is
  allocated, so a small compressed file can't exhaust memory."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_BYTES = 2 * 1024**3
MAX_VOXELS = 400_000_000
HEADER_SIZE = 348
TILT_WARNING_DEGREES = 10.0
MAX_SLICES_PER_VOLUME = 32

DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8, 512: np.uint16,
          768: np.uint32, 1024: np.int64, 1280: np.uint64}
UNSUPPORTED = {1: "Binary", 32: "Complex", 128: "RGB", 1792: "Complex", 2048: "Complex", 2304: "RGBA"}
_UNITS = {1: 1000.0, 2: 1.0, 3: 0.001}


class NiftiError(ValueError):
    """The file can't be used. The message is safe to show."""


@dataclass
class Header:
    byte_order: str
    dims: tuple[int, ...]
    datatype: int
    bitpix: int
    pixdim: tuple[float, ...]
    vox_offset: float
    slope: float
    inter: float
    qform_code: int
    sform_code: int
    quatern: tuple[float, float, float]
    qoffset: tuple[float, float, float]
    srow: np.ndarray
    units: int
    text: list[str] = field(default_factory=list)


@dataclass
class Volume:
    array: np.ndarray                     # (slices, rows, columns) float32, display orientation
    pixel_spacing: list[float]            # [row, column] mm
    slice_thickness: float | None
    volumes: int                          # volumes in the file (1 unless 4-D)
    oriented: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def display(self) -> dict:
        return {"plane": "axial" if self.oriented else "unknown", "transpose": False, "flip_rows": False,
                "flip_columns": False, "source": "nifti"}


@dataclass(frozen=True)
class SliceRef:
    """One slice of a volume file, as training and the dataset browser open it."""
    path: Path
    index: int


# --- Bytes -------------------------------------------------------------------------------------

def decompress(data: bytes, limit: int = MAX_BYTES) -> bytes:
    """The file's bytes, un-gzipped when it is gzip. Reads in chunks and stops at `limit`."""
    if data[:2] != b"\x1f\x8b":
        if len(data) > limit:
            raise NiftiError(f"The file is larger than {limit // 1024**2} MB.")
        return data
    unpacker = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = bytearray()
    view = memoryview(data)
    try:
        for start in range(0, len(data), 1 << 20):
            out += unpacker.decompress(view[start:start + (1 << 20)], max(1, limit + 1 - len(out)))
            while unpacker.unconsumed_tail:
                if len(out) > limit:
                    break
                out += unpacker.decompress(unpacker.unconsumed_tail, max(1, limit + 1 - len(out)))
            if len(out) > limit:
                raise NiftiError(f"The file is larger than {limit // 1024**2} MB when decompressed.")
        out += unpacker.flush()
    except zlib.error as exc:
        raise NiftiError("The gzip file is corrupt.") from exc
    if not unpacker.eof:
        raise NiftiError("The gzip file is truncated.")
    if len(out) > limit:
        raise NiftiError(f"The file is larger than {limit // 1024**2} MB when decompressed.")
    return bytes(out)


def read_file(path: Path, limit: int = MAX_BYTES) -> bytes:
    try:
        with open(path, "rb") as fh:
            data = fh.read(limit + 1)
    except OSError as exc:
        raise NiftiError("The file couldn't be read.") from exc
    if len(data) > limit:
        raise NiftiError(f"The file is larger than {limit // 1024**2} MB.")
    return decompress(data, limit)


# --- Header ------------------------------------------------------------------------------------

def _text(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("latin-1", "replace").strip()


def parse_header(raw: bytes) -> Header:
    if len(raw) < HEADER_SIZE:
        raise NiftiError("This isn't a NIfTI file: it's too short to have a header.")
    order = next((o for o in ("<", ">") if int(np.frombuffer(raw[:4], dtype=f"{o}i4")[0]) == HEADER_SIZE), None)
    if order is None:
        if any(int(np.frombuffer(raw[:4], dtype=f"{o}i4")[0]) == 540 for o in ("<", ">")):
            raise NiftiError("NIfTI-2 files aren't supported. Convert to NIfTI-1.")
        raise NiftiError("This isn't a NIfTI-1 file.")
    magic = raw[344:348]
    if magic == b"ni1\x00":
        raise NiftiError("Two-file NIfTI (.hdr and .img) isn't supported. Use a single .nii file.")
    if magic != b"n+1\x00":
        raise NiftiError("This isn't a NIfTI-1 file (no n+1 marker); Analyze 7.5 files aren't supported.")

    def arr(offset: int, dtype: str, count: int) -> np.ndarray:
        return np.frombuffer(raw, dtype=f"{order}{dtype}", count=count, offset=offset)

    dim = arr(40, "i2", 8)
    ndim = int(dim[0])
    if not 1 <= ndim <= 7:
        raise NiftiError("The header's dimensions are invalid.")
    dims = tuple(int(d) for d in dim[1:1 + ndim])
    if any(d < 1 for d in dims):
        raise NiftiError("The header's dimensions are invalid (a dimension is zero or negative).")
    datatype, bitpix = int(arr(70, "i2", 1)[0]), int(arr(72, "i2", 1)[0])
    if datatype in UNSUPPORTED:
        raise NiftiError(f"{UNSUPPORTED[datatype]} image data isn't supported.")
    if datatype not in DTYPES:
        raise NiftiError("This image uses a data type that isn't supported.")
    if np.dtype(DTYPES[datatype]).itemsize * 8 != bitpix:
        raise NiftiError("The header's data type and bits per voxel disagree, so the file is corrupt.")
    pixdim = tuple(float(v) for v in arr(76, "f4", 8))
    vox_offset = float(arr(108, "f4", 1)[0])
    slope, inter = float(arr(112, "f4", 1)[0]), float(arr(116, "f4", 1)[0])
    qform_code, sform_code = int(arr(252, "i2", 1)[0]), int(arr(254, "i2", 1)[0])
    quatern = tuple(float(v) for v in arr(256, "f4", 3))
    qoffset = tuple(float(v) for v in arr(268, "f4", 3))
    srow = arr(280, "f4", 12).reshape(3, 4).astype(np.float64)
    units = int(raw[123]) & 7
    text = [_text(raw[148:228]), _text(raw[228:252]), _text(raw[14:32]), _text(raw[328:344])]
    return Header(order, dims, datatype, bitpix, pixdim, vox_offset, slope, inter, qform_code, sform_code, quatern,
                  qoffset, srow, units, text)


def header_text(raw: bytes) -> list[str]:
    """The header's free-text fields (description, auxiliary file, database name, intent name), for
    checking for identifiers. Never returned by read()."""
    try:
        return [t for t in parse_header(raw).text if t]
    except NiftiError:
        return []


def affine(h: Header) -> np.ndarray | None:
    """Voxel indices to RAS+ millimetres, from the sform (preferred) or qform, or None."""
    if h.sform_code > 0:
        a = np.vstack([h.srow, [0, 0, 0, 1]])
        if np.isfinite(a).all() and abs(np.linalg.det(a[:3, :3])) > 1e-12:
            return a
    if h.qform_code > 0:
        b, c, d = h.quatern
        total = b * b + c * c + d * d
        if np.isfinite(total) and total <= 1.0 + 1e-5:
            a_ = math.sqrt(max(0.0, 1.0 - total))
            rot = np.array([
                [a_ * a_ + b * b - c * c - d * d, 2 * (b * c - a_ * d), 2 * (b * d + a_ * c)],
                [2 * (b * c + a_ * d), a_ * a_ + c * c - b * b - d * d, 2 * (c * d - a_ * b)],
                [2 * (b * d - a_ * c), 2 * (c * d + a_ * b), a_ * a_ + d * d - c * c - b * b]])
            qfac = -1.0 if h.pixdim[0] < 0 else 1.0
            scale = np.array([h.pixdim[1], h.pixdim[2], h.pixdim[3] * qfac])
            a = np.eye(4)
            a[:3, :3] = rot * scale
            a[:3, 3] = h.qoffset
            if np.isfinite(a).all() and abs(np.linalg.det(a[:3, :3])) > 1e-12:
                return a
    return None


def orientation(a: np.ndarray) -> tuple[list[int], list[int], float] | None:
    """For each voxel axis: the patient axis (0 x right, 1 y anterior, 2 z superior) it follows and its sign,
    and the largest tilt from those axes in degrees. None when two voxel axes follow the same patient axis."""
    axes, signs, tilt = [], [], 0.0
    for i in range(3):
        column = a[:3, i]
        norm = float(np.linalg.norm(column))
        w = int(np.argmax(np.abs(column)))
        axes.append(w)
        signs.append(1 if column[w] >= 0 else -1)
        tilt = max(tilt, math.degrees(math.acos(min(1.0, abs(column[w]) / norm))))
    return (axes, signs, tilt) if sorted(axes) == [0, 1, 2] else None


def _unit(h: Header) -> float:
    return _UNITS.get(h.units, 1.0)


# --- Reading ----------------------------------------------------------------------------------

def _data(raw: bytes, h: Header) -> np.ndarray:
    voxels = int(np.prod(h.dims, dtype=np.float64))
    if voxels > MAX_VOXELS:
        raise NiftiError(f"The image has {voxels:,} voxels; the limit is {MAX_VOXELS:,}.")
    offset = int(h.vox_offset) if math.isfinite(h.vox_offset) else -1
    offset = 352 if offset == 0 else offset
    if not 352 <= offset <= len(raw):
        raise NiftiError("The header says the image data starts outside the file, so the file is corrupt.")
    needed = voxels * (h.bitpix // 8)
    if len(raw) - offset < needed:
        raise NiftiError(f"The file is truncated: the image needs {needed:,} bytes and {len(raw) - offset:,} are present.")
    dtype = np.dtype(DTYPES[h.datatype]).newbyteorder(h.byte_order)
    return np.frombuffer(raw, dtype=dtype, count=voxels, offset=offset).reshape(h.dims, order="F")


def read(raw: bytes, volume_index: int = 0) -> Volume:
    """A NIfTI-1 volume (decompressed bytes) as a display-oriented stack of axial slices."""
    h = parse_header(raw)
    warnings: list[str] = []
    data = _data(raw, h)
    if data.ndim == 1:
        raise NiftiError("A one-dimensional image can't be viewed as a scan.")
    if data.ndim > 4:
        raise NiftiError("Images with more than four dimensions aren't supported.")
    volumes = 1
    if data.ndim == 2:
        data = data[:, :, None]
    elif data.ndim == 4:
        volumes = data.shape[3]
        if volumes > 1:
            warnings.append(f"This is a 4-D image with {volumes} volumes; volume {volume_index + 1} is used.")
        if not 0 <= volume_index < volumes:
            raise NiftiError(f"The image has {volumes} volumes.")
        data = data[:, :, :, volume_index]
    slope, inter = h.slope, h.inter
    values = np.asarray(data, dtype=np.float32)
    if math.isfinite(slope) and slope != 0 and (slope != 1 or inter != 0) and math.isfinite(inter):
        values = values * np.float32(slope) + np.float32(inter)
    bad = ~np.isfinite(values)
    if bad.any():
        warnings.append(f"{int(bad.sum()):,} voxels weren't finite numbers and were set to zero.")
        values = np.where(bad, 0.0, values)

    a = affine(h)
    unit = _unit(h)
    steps = [abs(p) * unit if p else 1.0 for p in h.pixdim[1:4]]
    if a is None:
        warnings.append("The image has no orientation, so it's shown as stored.")
        order = (2, 1, 0)
        array = np.transpose(values, order)
        spacing = [steps[1], steps[0]]
        return Volume(np.ascontiguousarray(array, dtype=np.float32), spacing, steps[2], volumes, False, warnings)
    mapped = orientation(a)
    norms = [float(np.linalg.norm(a[:3, i])) * unit for i in range(3)]
    if mapped is None:
        warnings.append("The image's axes are tilted so far that they can't be matched to the patient's axes, "
                        "so it's shown as stored.")
        array = np.transpose(values, (2, 1, 0))
        return Volume(np.ascontiguousarray(array, dtype=np.float32), [norms[1], norms[0]], norms[2], volumes, False, warnings)
    axes, signs, tilt = mapped
    if tilt > TILT_WARNING_DEGREES:
        warnings.append(f"The image is tilted {tilt:.0f} degrees from the patient's axes. It was reordered to the "
                        "nearest axes without resampling, so slices are oblique.")
    voxel_of = {w: i for i, w in enumerate(axes)}
    order = (voxel_of[2], voxel_of[1], voxel_of[0])          # slices follow z, rows y, columns x
    array = np.transpose(values, order)
    # Head first, anterior at the top, the patient's right on the image left: every axis runs from high to low.
    for out_axis, world in enumerate((2, 1, 0)):
        if signs[voxel_of[world]] > 0:
            array = np.flip(array, axis=out_axis)
    spacing = [norms[voxel_of[1]], norms[voxel_of[0]]]
    return Volume(np.ascontiguousarray(array, dtype=np.float32), spacing, norms[voxel_of[2]], volumes, True, warnings)


def read_path(path: Path, volume_index: int = 0) -> Volume:
    return read(read_file(path), volume_index)


def slice_count(raw: bytes) -> int:
    """Axial slices a volume will have, from the header alone."""
    h = parse_header(raw)
    dims = list(h.dims) + [1] * (3 - len(h.dims))
    a = affine(h)
    mapped = orientation(a) if a is not None else None
    if mapped is None:
        return dims[2]
    return dims[mapped[0].index(2)]


def header_only(path: Path) -> bytes:
    """Enough of a file to read its header: the first bytes, un-gzipped."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 16)
    except OSError as exc:
        raise NiftiError("The file couldn't be read.") from exc
    if head[:2] == b"\x1f\x8b":
        unpacker = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            return unpacker.decompress(head, 1024)
        except zlib.error as exc:
            raise NiftiError("The gzip file is corrupt.") from exc
    return head[:1024]


def training_slices(path: Path) -> list[int]:
    """Which axial slices of a volume to train on: up to 32, evenly spread over the middle 80%."""
    n = slice_count(header_only(path))
    low, high = int(n * 0.1), max(int(n * 0.1) + 1, int(math.ceil(n * 0.9)))
    return sorted({int(i) for i in np.linspace(low, high - 1, min(MAX_SLICES_PER_VOLUME, high - low))})


# --- Slices for training and the browser --------------------------------------------------------------

_cache: dict[tuple[str, int, int], tuple[np.ndarray, float, float]] = {}


def _cached(path: Path):
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        volume = read_path(path)
        low, high = np.percentile(volume.array, [1, 99])
        if len(_cache) >= 2:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (volume.array, float(low), float(high))
    return _cache[key]


def slice_image(ref: SliceRef):
    """A slice as an 8-bit greyscale PIL image, scaled with the volume's 1st-99th percentile."""
    from PIL import Image

    try:
        array, low, high = _cached(ref.path)
    except OSError as exc:
        raise ValueError("This volume couldn't be read.") from exc
    except NiftiError as exc:
        raise ValueError(str(exc)) from exc
    if not 0 <= ref.index < array.shape[0]:
        raise ValueError("This volume has no such slice.")
    scaled = np.clip((array[ref.index] - low) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def parse_ref(path: Path) -> SliceRef | None:
    """A path like volume.nii.gz#12 as a SliceRef, or None for anything else."""
    name = path.name
    if "#" not in name:
        return None
    base, _, index = name.rpartition("#")
    if not index.isdigit() or not is_nifti_name(base):
        return None
    return SliceRef(path.with_name(base), int(index))


def is_nifti_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".nii") or lowered.endswith(".nii.gz")


def guess_modality(array: np.ndarray, name: str = "") -> str:
    """CT or MR: from the file name when it says, otherwise CT if the values look like Hounsfield units."""
    words = {w for w in "".join(c if c.isalnum() else " " for c in name.lower()).split()}
    if words & {"ct", "hu"}:
        return "CT"
    if words & {"mr", "mri", "t1", "t2", "flair", "t1w", "t2w", "dwi"}:
        return "MR"
    low, high = np.percentile(array, [0.5, 99.5])
    return "CT" if low <= -500 and high >= 100 else "MR"
