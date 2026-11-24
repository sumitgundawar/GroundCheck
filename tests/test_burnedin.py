"""Burned-in text detection: the classical detector and the OCR path on synthetic
scans with and without text, the DICOM de-identification policy built on them
(redact or refuse, never pass a flagged image), and the BurnedInAnnotation tag."""

from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.imaging import burnedin, deid, dicom  # noqa: E402
from imaging_synth import POSITIONS, TEXTS, draw_text, phantom, stored, to_hu  # noqa: E402

pytestmark = pytest.mark.req("GC-REQ-022")

CT = "1.2.840.10008.5.1.4.1.1.2"


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    """Tests use the heuristic unless one installs a fake OCR; a real Tesseract must not change results."""
    monkeypatch.setenv("IMAGING_OCR", "off")
    burnedin.reset_ocr()
    yield
    burnedin.reset_ocr()


# --- Building blocks ---------------------------------------------------------------------------

def test_gray8_scales_and_rejects_flat_and_bad_input():
    assert burnedin.to_gray8(np.zeros((32, 32))) is None
    assert burnedin.to_gray8(np.full((32, 32), 7.5)) is None
    ramp = burnedin.to_gray8(np.tile(np.arange(64.0), (64, 1)))
    assert ramp.dtype == np.uint8 and ramp.min() == 0 and ramp.max() == 255
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[4:8] = 255
    assert burnedin.to_gray8(rgb).shape == (16, 16)
    assert burnedin.to_gray8(np.dstack([rgb, rgb[..., :1]])).shape == (16, 16)
    assert burnedin.to_gray8(np.full((16, 16), np.nan)) is None
    odd = np.arange(256.0).reshape(16, 16)
    odd[2, 2], odd[3, 3] = np.nan, np.inf
    assert burnedin.to_gray8(odd).dtype == np.uint8
    for bad in (np.zeros(5), np.zeros((2, 3, 5)), np.zeros((2, 2, 2, 2))):
        with pytest.raises(ValueError):
            burnedin.to_gray8(bad)


def test_window_filters_match_brute_force():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (23, 31)).astype(np.uint8)
    for k in (1, 2, 5, 9):
        for axis in (0, 1):
            got = burnedin._window(a, k, axis, np.minimum, True)
            padded = np.concatenate([a, np.take(a, [-1] * (k - 1), axis=axis)], axis=axis)
            want = np.min([np.take(padded, range(i, i + a.shape[axis]), axis=axis) for i in range(k)], axis=0)
            assert (got == want).all(), (k, axis)


def test_tophat_keeps_thin_features_and_drops_large_regions():
    g = np.zeros((60, 60), np.uint8)
    g[20:40, 20:40] = 200                       # large block
    g[5, 5:50] = 255                            # thin line
    th = burnedin.tophat(g, 9, True)
    assert th[5, 10:45].min() >= 250 and th[25:35, 25:35].max() == 0
    dark = np.full((60, 60), 200, np.uint8)
    dark[30, 5:50] = 0
    assert burnedin.tophat(dark, 9, False)[30, 10:45].min() >= 190 and burnedin.tophat(dark, 9, True)[:15].max() == 0


def brute_components(mask):
    seen = np.zeros_like(mask, bool)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys, xs = zip(*cells)
        out.append((min(ys), min(xs), max(ys), max(xs), len(cells)))
    return sorted(out)


def test_connected_components_match_a_brute_force_labeller():
    rng = np.random.default_rng(1)
    for density in (0.05, 0.2, 0.45):
        for _ in range(5):
            mask = rng.random((40, 50)) < density
            assert sorted(burnedin.components(mask)) == brute_components(mask)
    assert burnedin.components(np.zeros((10, 10), bool)) == []
    full = np.ones((6, 7), bool)
    assert burnedin.components(full) == [(0, 0, 5, 6, 42)]
    diagonal = np.eye(5, dtype=bool)
    assert burnedin.components(diagonal) == [(0, 0, 4, 4, 5)]           # 8-connected
    assert burnedin.components(np.indices((400, 400)).sum(0) % 2 == 0) is None   # a checkerboard has too many runs to label: reported as too busy


def test_a_mask_with_too_many_runs_is_reported_as_too_busy():
    noise = np.zeros((600, 600), bool)
    noise[:, ::2] = True
    assert burnedin.components(noise) is None


def test_noise_estimate_ignores_flat_areas_and_letters():
    rng = np.random.default_rng(2)
    g = np.zeros((128, 128), np.uint8)
    assert burnedin._noise(g) == 0.0
    g[:, 64:] = np.clip(100 + rng.normal(0, 4, (128, 64)), 0, 255)
    assert 2.5 < burnedin._noise(g) < 5.5
    valid = np.zeros((128, 128), bool)
    valid[:, :64] = True
    assert burnedin._noise(g, valid) == 0.0
    assert burnedin._noise(np.zeros((8, 8), np.uint8)) == 0.0


# --- Detector accuracy on synthetic scans ---------------------------------------------------------

KINDS = ["body", "gradient", "ribs", "table", "flat", "lungs"]
SIZES = ((256, 14), (384, 18), (512, 22), (512, 28))


def detect(img, **kw):
    """The heuristic on a 12-bit image, as the DICOM scan calls it."""
    return burnedin.detect_heuristic(img, full_scale=4096, **kw)


def text_cases(repeats=2):
    rng = np.random.default_rng(2026)
    i = 0
    for _ in range(repeats):
        for angle in (0, 90, 180, 270, 15, 30, 45, -30):
            for contrast in ("high", "medium", "low", "dark"):
                for pos in POSITIONS:
                    n, size = SIZES[i % len(SIZES)]
                    kind = KINDS[i % len(KINDS)]
                    text = TEXTS[i % len(TEXTS)]
                    i += 1
                    base = phantom(kind, n, rng, noise=float(rng.choice([0, 0.005, 0.01])))
                    yield {"angle": angle, "contrast": contrast, "pos": pos, "n": n, "kind": kind}, \
                        stored(draw_text(base, text, size, pos, angle, contrast, rng))


def control_cases(kinds=KINDS, per=3):
    rng = np.random.default_rng(7)
    for kind in kinds:
        for n in (256, 384, 512):
            for noise in (0.0, 0.01, 0.02)[:per]:
                yield {"kind": kind, "n": n}, stored(phantom(kind, n, rng, noise))


@pytest.fixture(scope="module")
def results():
    text = [(meta, detect(img)) for meta, img in text_cases()]
    control = [(meta, detect(img)) for meta, img in control_cases()]
    return text, control


def test_recall_on_synthetic_text_is_at_least_95_percent(results):
    """Text drawn in corners and centre, rotated (0, 90, 180, 270, 15, 30, 45, -30 degrees), bright, medium,
    low contrast and dark on a light box, over six phantoms with noise."""
    text, _ = results
    recall = np.mean([f.flagged for _, f in text])
    assert len(text) == 320 and recall >= 0.95, recall


@pytest.mark.parametrize("key,values", [("angle", (0, 90, 180, 270, 15, 30, 45, -30)), ("contrast", ("high", "medium", "low", "dark")),
                                         ("pos", POSITIONS)])
def test_every_category_of_text_is_found_at_least_85_percent(results, key, values):
    text, _ = results
    for value in values:
        hits = [f.flagged for meta, f in text if meta[key] == value]
        assert hits and np.mean(hits) >= 0.85, (key, value, np.mean(hits))


def test_plain_scans_are_rarely_flagged(results):
    """Controls: smooth phantoms (body, gradient, ribs, table, flat, lungs with vessels), with noise, no text."""
    _, control = results
    rate = np.mean([f.flagged for _, f in control])
    assert len(control) == 54 and rate <= 0.05, rate


def test_precision_and_recall_are_reported(results, capsys):
    text, control = results
    tp = sum(f.flagged for _, f in text)
    fn = len(text) - tp
    fp = sum(f.flagged for _, f in control)
    precision, recall = tp / max(1, tp + fp), tp / (tp + fn)
    with capsys.disabled():
        print(f"\nburned-in text: recall {recall:.3f} ({tp}/{len(text)}), precision {precision:.3f}, "
              f"false-positive rate {fp / len(control):.3f} ({fp}/{len(control)})")
    assert precision >= 0.95 and recall >= 0.95


def test_a_flagged_image_always_comes_with_regions_that_cover_the_text():
    rng = np.random.default_rng(4)
    for pos in POSITIONS:
        base = phantom("body", 384, rng, 0.005)
        img = draw_text(base, "SMITH^JOHN 1948", 18, pos, 0, "high", rng)
        found = detect(stored(img))
        assert found.flagged and found.regions
        inked = np.argwhere((img - base) > 0.3)
        top, left, bottom, right = inked[:, 0].min(), inked[:, 1].min(), inked[:, 0].max(), inked[:, 1].max()
        assert any(t <= top and l <= left and b > bottom and r > right for t, l, b, r in found.regions), (pos, found.regions)


def test_texture_heavy_scans_are_flagged_conservatively_never_silently_ignored():
    """Radial streak artefacts look like text to the heuristic. That is the safe direction: a flagged image is
    redacted or refused, and always has regions."""
    for meta, img in control_cases(("streaks",), per=2):
        found = detect(img)
        assert (not found.flagged) or found.regions


def test_noise_as_strong_as_text_is_flagged_because_nothing_can_be_checked():
    rng = np.random.default_rng(5)
    noise = stored(np.clip(0.3 + rng.normal(0, 0.12, (256, 256)), 0, 1))
    found = detect(noise)
    assert found.flagged and "too busy" in found.reason
    mild = stored(np.clip(0.3 + rng.normal(0, 0.02, (256, 256)), 0, 1))
    assert not detect(mild).flagged


def test_no_text_controls_with_a_single_laterality_marker_are_not_flagged():
    rng = np.random.default_rng(6)
    base = phantom("body", 384, rng, 0.005)
    marked = draw_text(base, "R", 22, "tl", 0, "high", rng)
    assert not detect(stored(marked)).flagged


def test_short_text_of_two_or_three_characters_is_a_documented_blind_spot():
    rng = np.random.default_rng(6)
    base = phantom("body", 384, rng, 0.005)
    found = detect(stored(draw_text(base, "JS", 20, "tl", 0, "high", rng)))
    assert found.flagged is False        # fewer than 4 characters isn't called text; see the module docstring


def test_very_large_and_degenerate_images():
    big = np.zeros((4200, 100), np.uint16)
    big[::7] = 500
    found = burnedin.detect_heuristic(big)
    assert found.flagged and "too large" in found.reason and found.regions == [(0, 0, 4200, 100)]
    tiny = np.zeros((8, 8))
    tiny[2] = 1
    assert burnedin.detect_heuristic(tiny).flagged is False
    assert burnedin.detect_heuristic(np.zeros((64, 64))).reason == "flat image"


def test_an_image_too_busy_to_separate_strokes_is_flagged_fail_closed():
    busy = (np.indices((256, 256)).sum(0) // 2 % 2).astype(float)            # a fine checkerboard: strokes everywhere
    found = burnedin.detect_heuristic(busy)
    assert found.flagged and "too busy" in found.reason and found.regions == [(0, 0, 256, 256)]


def test_speed_is_reasonable_for_a_512_slice():
    import time

    rng = np.random.default_rng(9)
    img = stored(phantom("body", 512, rng, 0.01))
    start = time.perf_counter()
    for _ in range(5):
        detect(img, skew=False)
    assert (time.perf_counter() - start) / 5 < 0.5


def test_merging_overlapping_boxes():
    assert burnedin._merge([(0, 0, 10, 10), (5, 5, 15, 15), (30, 30, 40, 40)]) == [(0, 0, 15, 15), (30, 30, 40, 40)]
    assert burnedin._merge([]) == []


# --- OCR path ----------------------------------------------------------------------------------------

def fake_tesseract(words, version_error=None, data_error=None):
    module = types.ModuleType("pytesseract")

    class Output:
        DICT = "dict"

    def get_tesseract_version():
        if version_error:
            raise version_error
        return "5.3.0"

    def image_to_data(image, output_type=None, config=""):
        if data_error:
            raise data_error
        return {"text": [w[0] for w in words], "conf": [w[1] for w in words], "left": [w[2] for w in words],
                "top": [w[3] for w in words], "width": [w[4] for w in words], "height": [w[5] for w in words]}

    module.Output, module.get_tesseract_version, module.image_to_data = Output, get_tesseract_version, image_to_data
    module.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
    return module


def plain():
    rng = np.random.default_rng(10)
    return stored(phantom("body", 256, rng, 0.005))


def with_text():
    rng = np.random.default_rng(11)
    return stored(draw_text(phantom("body", 384, rng, 0.005), "SMITH^JOHN 1948", 18, "tl", 0, "high", rng))


def test_ocr_is_off_when_not_installed(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    monkeypatch.setitem(sys.modules, "pytesseract", None)            # import fails
    assert burnedin.ocr_available() is False and burnedin.detect_ocr(plain()) is None
    assert burnedin.detect(plain()).flagged is False and burnedin.detect(with_text()).method == "heuristic"


def test_ocr_is_off_when_the_program_is_missing(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    fake = fake_tesseract([])
    fake.get_tesseract_version = lambda: (_ for _ in ()).throw(fake.TesseractNotFoundError("no tesseract"))
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    assert burnedin.ocr_available() is False
    assert burnedin.detect(with_text()).flagged is True            # the heuristic still works


def test_the_ocr_switch_turns_it_off_even_when_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract([("SMITH", 90, 4, 4, 40, 12)]))
    for value in ("off", "false", "0", "no", "OFF"):
        monkeypatch.setenv("IMAGING_OCR", value)
        burnedin.reset_ocr()
        assert burnedin.ocr_available() is False


def test_ocr_words_flag_an_image_the_heuristic_would_pass(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract([("JS", 95, 10, 10, 20, 10), ("Smith", 91, 30, 40, 50, 12),
                                                                      ("x", 99, 5, 5, 5, 5), ("noise", 20, 100, 100, 30, 10)]))
    found = burnedin.detect(plain())
    assert found.flagged and found.method == "ocr" and found.regions
    top, left, bottom, right = found.regions[0]
    assert top <= 40 // 2 and left <= 30 // 2 and bottom >= 52 // 2 and right >= 80 // 2     # scaled back from the 2x image


def test_ocr_finding_nothing_never_lowers_a_heuristic_flag(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract([]))
    assert burnedin.detect(with_text()).flagged is True
    assert burnedin.detect(plain()).flagged is False


def test_ocr_and_heuristic_regions_are_combined(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract([("PATIENT", 90, 400, 600, 100, 20)]))
    found = burnedin.detect(with_text())
    assert found.method == "ocr+heuristic" and len(found.regions) >= 2


def test_ocr_failing_at_run_time_falls_back_to_the_heuristic(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract([], data_error=RuntimeError("tesseract crashed")))
    assert burnedin.detect_ocr(plain()) is None
    assert burnedin.detect(with_text()).flagged is True


def test_ocr_needs_confident_words_of_three_characters(monkeypatch):
    monkeypatch.setenv("IMAGING_OCR", "auto")
    words = [("ab", 99, 5, 5, 20, 10), ("a-b", 99, 5, 30, 20, 10), ("abc", 59.9, 5, 60, 20, 10), ("   ", 99, 5, 90, 20, 10)]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract(words))
    assert burnedin.detect_ocr(plain()).flagged is False
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tesseract(words + [("abc", 60, 5, 120, 20, 10)]))
    burnedin.reset_ocr()
    assert burnedin.detect_ocr(plain()).flagged is True


# --- DICOM policy --------------------------------------------------------------------------------------

def make_dataset(image: np.ndarray, burned: str | None = None, frames: int = 1, uid: str | None = None) -> Dataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = CT, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = CT, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.51", uid or "1.2.826.0.1.3680043.8.498.52"
    ds.Modality, ds.PatientName, ds.PatientID = "CT", "Smith^John", "ID123456"
    ds.InstanceNumber, ds.PixelSpacing, ds.SliceThickness = 1, [1.0, 1.0], 5
    ds.ImagePositionPatient, ds.ImageOrientationPatient = [0, 0, 0], [1, 0, 0, 0, 1, 0]
    if burned is not None:
        ds.BurnedInAnnotation = burned
    hu = to_hu(image)
    ds.Rows, ds.Columns = hu.shape
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
    stored_values = (np.stack([hu] * frames) if frames > 1 else hu) + 1024
    ds.PixelData = stored_values.astype(np.int16).tobytes()
    return ds


def as_bytes(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


@pytest.fixture()
def clean_image():
    rng = np.random.default_rng(21)
    return phantom("body", 384, rng, 0.005)


@pytest.fixture()
def texty_image(clean_image):
    return draw_text(clean_image, "SMITH^JOHN DOB 12/03/1948", 18, "bl", 0, "high", np.random.default_rng(22))


def test_a_plain_image_passes_untouched(clean_image):
    ds = make_dataset(clean_image)
    before = bytes(ds.PixelData)
    report = deid.deidentify(ds)
    assert report["burned_in_flagged"] == 0 and report["burned_in_redacted"] == 0
    assert bytes(ds.PixelData) == before
    assert "113101" not in {c.CodeValue for c in ds.DeidentificationMethodCodeSequence}


def test_text_in_the_pixels_is_blacked_out_and_the_result_rechecked(texty_image):
    ds = make_dataset(texty_image)
    original = ds.pixel_array.copy()
    report = deid.deidentify(ds)
    assert report["burned_in_flagged"] == 1 and report["burned_in_redacted"] >= 1
    assert ds.PixelData != make_dataset(texty_image).PixelData
    after = ds.pixel_array
    assert after.max() < original.max()                                        # the bright text is gone
    assert not burnedin.scan_dataset(ds).flagged
    assert "113101" in {c.CodeValue for c in ds.DeidentificationMethodCodeSequence}
    changed = np.argwhere(after != original)
    assert changed.size and changed[:, 0].min() > clean_rows(texty_image) - 40     # only the text area changed
    assert ds.PatientIdentityRemoved == "YES" and ds.PatientName == ""


def clean_rows(image):
    return image.shape[0] - 60


def test_block_mode_refuses_instead_of_redacting(texty_image, monkeypatch):
    monkeypatch.setenv("IMAGING_BURNED_IN_ACTION", "block")
    ds = make_dataset(texty_image)
    with pytest.raises(deid.DeidError, match="Text was found"):
        deid.deidentify(ds)


@pytest.mark.parametrize("value", ["off", "none", "allow", "false", "0", "", "REDACT", "skip", "pass"])
def test_no_setting_turns_the_check_off(texty_image, monkeypatch, value):
    monkeypatch.setenv("IMAGING_BURNED_IN_ACTION", value)
    assert deid.burned_in_action() == "redact"
    ds = make_dataset(texty_image)
    deid.deidentify(ds)
    assert not burnedin.scan_dataset(ds).flagged and ds.pixel_array.max() < make_dataset(texty_image).pixel_array.max()


def test_block_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("IMAGING_BURNED_IN_ACTION", " Block ")
    assert deid.burned_in_action() == "block"


def test_the_tag_saying_yes_is_always_respected(clean_image):
    for value in ("YES", "yes", "Yes"):
        with pytest.raises(deid.DeidError, match="burned into the pixels"):
            deid.deidentify(make_dataset(clean_image, burned=value))


def test_the_tag_saying_no_is_not_trusted(texty_image):
    ds = make_dataset(texty_image, burned="NO")
    report = deid.deidentify(ds)
    assert report["burned_in_flagged"] == 1
    ds = make_dataset(texty_image, burned=None)
    assert deid.deidentify(ds)["burned_in_flagged"] == 1


def test_a_plain_image_with_tag_no_passes(clean_image):
    assert deid.deidentify(make_dataset(clean_image, burned="NO"))["burned_in_flagged"] == 0


def test_a_failed_redaction_refuses_the_image(texty_image, monkeypatch):
    monkeypatch.setattr(burnedin, "redact_dataset", lambda ds, regions: None)
    with pytest.raises(deid.DeidError, match="couldn't be removed"):
        deid.deidentify(make_dataset(texty_image))


def test_redaction_that_raises_refuses_the_image(texty_image, monkeypatch):
    def boom(ds, regions):
        raise OSError("disk")

    monkeypatch.setattr(burnedin, "redact_dataset", boom)
    with pytest.raises(deid.DeidError, match="couldn't be removed"):
        deid.deidentify(make_dataset(texty_image))


def test_a_scan_that_cannot_run_refuses_the_image(clean_image, monkeypatch):
    def boom(ds, **kw):
        raise RuntimeError("cannot decode")

    monkeypatch.setattr(burnedin, "scan_dataset", boom)
    with pytest.raises(deid.DeidError, match="couldn't be checked"):
        deid.deidentify(make_dataset(clean_image))


def test_an_image_too_busy_to_check_is_refused_not_blacked_out_whole():
    ds = make_dataset((np.indices((256, 256)).sum(0) // 2 % 2).astype(float))
    with pytest.raises(deid.DeidError, match="too complex to check"):
        deid.deidentify(ds)


def test_datasets_without_pixels_are_not_scanned():
    ds = make_dataset(np.zeros((8, 8)))
    del ds.PixelData
    assert deid.deidentify(ds)["burned_in_flagged"] == 0


def test_text_in_one_frame_of_a_multiframe_image_is_found_and_every_frame_redacted(texty_image, clean_image):
    ds = make_dataset(clean_image, frames=5)
    frames = ds.pixel_array.copy()
    frames[2] = make_dataset(texty_image).pixel_array
    ds.PixelData = frames.astype(np.int16).tobytes()
    assert burnedin.scan_dataset(ds).flagged
    deid.deidentify(ds)
    after = ds.pixel_array
    assert after.shape == (5, 384, 384) and after.max() < frames.max()
    assert not burnedin.scan_dataset(ds).flagged


def test_a_long_multiframe_image_is_sampled_not_exhausted(clean_image):
    ds = make_dataset(clean_image, frames=40)
    seen = []
    original = burnedin.detect

    def spy(frame, skew=True, full_scale=None):
        seen.append(1)
        return original(frame, skew, full_scale)

    burnedin.detect, saved = spy, original
    try:
        burnedin.scan_dataset(ds, max_frames=8)
    finally:
        burnedin.detect = saved
    assert len(seen) == 8


def test_an_rgb_secondary_capture_with_text_is_redacted_in_every_channel():
    rng = np.random.default_rng(41)
    gray = draw_text(phantom("body", 384, rng, 0.005), "SMITH^JOHN 1948", 18, "tl", 0, "high", rng)
    rgb = (np.stack([gray] * 3, axis=-1) * 255).astype(np.uint8)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 384, 384, 3
    ds.PhotometricInterpretation, ds.PlanarConfiguration = "RGB", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = rgb.tobytes()
    ds.PatientName = "X"
    found = burnedin.scan_dataset(ds)
    assert found.flagged
    burnedin.redact_dataset(ds, found.regions)
    after = ds.pixel_array
    assert after.shape == (384, 384, 3) and not burnedin.scan_dataset(ds).flagged and after[..., 0].max() < 255


@pytest.mark.parametrize("signed,bits", [(0, 8), (0, 16), (1, 16)])
def test_redaction_keeps_the_pixel_format(signed, bits, texty_image):
    ds = make_dataset(texty_image)
    values = ds.pixel_array
    dtype = {(0, 8): np.uint8, (0, 16): np.uint16, (1, 16): np.int16}[(signed, bits)]
    scaled = (values - values.min()).astype(np.float64)
    scaled = (scaled / scaled.max() * (200 if bits == 8 else 3000)).astype(dtype)
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits, bits, bits - 1, signed
    ds.RescaleIntercept = 0
    ds.PixelData = scaled.tobytes()
    found = burnedin.scan_dataset(ds)
    assert found.flagged
    burnedin.redact_dataset(ds, found.regions)
    assert ds.pixel_array.dtype == dtype and len(ds.PixelData) == scaled.nbytes


def test_import_redacts_text_and_tells_the_user(texty_image, clean_image):
    uploads = [("a.dcm", as_bytes(make_dataset(texty_image))), ("b.dcm", as_bytes(make_dataset(clean_image)))]
    uploads[1] = ("b.dcm", as_bytes(_second_slice(clean_image)))
    [series] = dicom.read(uploads)
    assert any("blacked out" in w and "1 images" in w for w in series.warnings)
    assert series.deid_report["burned_in_flagged"] == 1
    assert series.volume.shape == (2, 384, 384)
    assert series.volume[0][-60:].max() < to_hu(texty_image)[-60:].max()


def _second_slice(image):
    ds = make_dataset(image)
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.InstanceNumber, ds.ImagePositionPatient = 2, [0, 0, 5]
    return ds


def test_import_refuses_a_series_whose_text_cannot_be_removed(texty_image, monkeypatch):
    monkeypatch.setenv("IMAGING_BURNED_IN_ACTION", "block")
    with pytest.raises(dicom.DicomError, match="Text was found"):
        dicom.read([("a.dcm", as_bytes(make_dataset(texty_image)))])


def test_import_gives_a_plain_series_no_burned_in_warning(clean_image):
    [series] = dicom.read([("a.dcm", as_bytes(make_dataset(clean_image)))])
    assert series.deid_report["burned_in_flagged"] == 0 and not any("burned" in w or "blacked" in w for w in series.warnings)


def test_only_the_first_slice_of_a_series_is_searched_for_slanted_text(clean_image, monkeypatch):
    calls = []
    original = deid.burnedin.scan_dataset

    def spy(ds, max_frames=8, deep=True):
        calls.append(deep)
        return original(ds, max_frames=max_frames, deep=deep)

    monkeypatch.setattr(deid.burnedin, "scan_dataset", spy)
    slices = []
    for i in range(3):
        ds = _second_slice(clean_image)
        ds.InstanceNumber, ds.ImagePositionPatient = i + 1, [0, 0, 5 * i]
        slices.append((f"{i}.dcm", as_bytes(ds)))
    dicom.read(slices)
    assert calls == [True, False, False]
