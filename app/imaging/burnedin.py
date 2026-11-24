"""Finding text burned into the pixels of an image.

Patient names, IDs and dates are sometimes drawn into the image itself (ultrasound
and X-ray exports, screenshots, secondary captures), where tag-based
de-identification can't reach them. This module looks at the pixels.

Two detectors, used together:

- An OCR engine when one is installed (pytesseract with the Tesseract program).
  Optional: imported only when used.
- A classical text detector that always works, with nothing installed: thin
  bright or dark strokes are extracted with a morphological top-hat, joined
  into glyph-sized connected components, and grouped into lines of similar
  glyphs. Four or more glyphs in a row (at any angle), or a word whose letters touch, are called text.

An image is flagged when either detector says so. An image the heuristic can't
analyse (too busy to separate strokes) is flagged too. The detectors can
miss text (very small, one or two characters, very faint, or a font unlike any
tested) and can flag things that aren't text (a ruler, a row of regular dots),
so a result is never "this image is clean": callers either redact the flagged
regions or refuse the image, and an unflagged image is only "no text found"."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

MIN_GLYPH = 5            # px
MIN_GLYPHS_IN_LINE = 4
MAX_RUNS = 60_000
MAX_GLYPHS = 1_500
MAX_SIDE = 4096
FLOOR_CONTRAST = 12      # of 255
BUSY_FRACTION = 0.22
MAX_THRESHOLD = 200


@dataclass
class Finding:
    flagged: bool
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)   # (top, left, bottom, right), bottom/right exclusive
    glyphs: int = 0
    confidence: float = 0.0
    method: str = "heuristic"
    reason: str = ""

    def as_dict(self) -> dict:
        return {"flagged": self.flagged, "regions": len(self.regions), "glyphs": self.glyphs,
                "confidence": round(self.confidence, 2), "method": self.method, "reason": self.reason}


# --- Image preparation ----------------------------------------------------------------------------

def to_gray8(pixels, full_scale: float | None = None) -> np.ndarray | None:
    """A 2-D uint8 image scaled to the 0.01th-99.99th percentile, or None for a flat image. With the
    image type's full scale (for example 4096 for 12 bits), a narrow range is never stretched by more than 4x,
    so noise in a nearly empty image isn't blown up into strokes."""
    a = np.asarray(pixels)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3].astype(np.float32).mean(axis=-1)
    if a.ndim != 2:
        raise ValueError("expected a 2-D image")
    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a, [0.01, 99.99])
    if hi - lo < 1e-6:
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-6:
            return None
    if full_scale and hi - lo < 0.25 * full_scale:
        hi = lo + 0.25 * full_scale
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _shift(a: np.ndarray, step: int, axis: int, fill: np.ndarray) -> np.ndarray:
    """a moved so out[i] = a[i + step] along axis, with the edge replicated."""
    out = np.empty_like(a)
    n = a.shape[axis]
    step = min(step, n)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    src[axis], dst[axis] = slice(step, n), slice(0, n - step)
    out[tuple(dst)] = a[tuple(src)]
    edge = [slice(None)] * a.ndim
    edge[axis] = slice(n - 1, n)
    tail = [slice(None)] * a.ndim
    tail[axis] = slice(n - step, n)
    out[tuple(tail)] = a[tuple(edge)]
    return out


def _window(a: np.ndarray, k: int, axis: int, op, forward: bool) -> np.ndarray:
    """op (np.minimum or np.maximum) over a window of k samples starting at i
    (forward) or ending at i, using log2(k) passes."""
    if not forward:
        a = np.flip(a, axis)
    out, size = a, 1
    while size < k:
        step = min(size, k - size)
        out = op(out, _shift(out, step, axis, out))
        size += step
    return out if forward else np.flip(out, axis)


def tophat(g: np.ndarray, k: int, bright: bool = True) -> np.ndarray:
    """Bright (or dark) features narrower than k pixels: g minus its opening
    (or the closing minus g). Large regions and their edges give nothing."""
    first, second = (np.minimum, np.maximum) if bright else (np.maximum, np.minimum)
    out = g
    for axis in (0, 1):
        out = _window(out, k, axis, first, True)
    for axis in (0, 1):
        out = _window(out, k, axis, second, False)
    return (g.astype(np.int16) - out.astype(np.int16)) if bright else (out.astype(np.int16) - g.astype(np.int16))


# --- Connected components, from runs ------------------------------------------------------------

def components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]] | None:
    """8-connected components of a boolean mask as (top, left, bottom, right, area),
    bottom/right inclusive. None when the mask is too busy to label."""
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    ys, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    n = len(ys)
    if n == 0:
        return []
    if n > MAX_RUNS:
        return None
    ends = ends - 1                     # inclusive
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row_start = np.searchsorted(ys, np.arange(mask.shape[0] + 1))
    ys_l, s_l, e_l = ys.tolist(), starts.tolist(), ends.tolist()
    for y in range(mask.shape[0] - 1):
        a0, a1 = row_start[y], row_start[y + 1]
        b0, b1 = row_start[y + 1], row_start[y + 2]
        i, j = a0, b0
        while i < a1 and j < b1:
            if s_l[i] <= e_l[j] + 1 and s_l[j] <= e_l[i] + 1:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
            if e_l[i] < e_l[j]:
                i += 1
            else:
                j += 1
    stats: dict[int, list[int]] = {}
    for r in range(n):
        root = find(r)
        s = stats.get(root)
        if s is None:
            stats[root] = [ys_l[r], s_l[r], ys_l[r], e_l[r], e_l[r] - s_l[r] + 1]
        else:
            s[0] = min(s[0], ys_l[r])
            s[1] = min(s[1], s_l[r])
            s[2] = max(s[2], ys_l[r])
            s[3] = max(s[3], e_l[r])
            s[4] += e_l[r] - s_l[r] + 1
    return [tuple(v) for v in stats.values()]


# --- Glyphs and lines -----------------------------------------------------------------------------

def _runs_per_line(crop: np.ndarray, axis: int) -> float:
    """Average number of separate ink runs along each row (axis 1) or column (axis 0) that has ink."""
    m = crop if axis == 1 else crop.T
    padded = np.zeros((m.shape[0], m.shape[1] + 1), np.int8)
    padded[:, :-1] = m
    starts = (np.diff(padded, axis=1, prepend=0) == 1).sum(axis=1)
    inked = starts[m.any(axis=1)]
    return float(inked.mean()) if inked.size else 0.0


def _prominent(th: np.ndarray, mask: np.ndarray, box, threshold: float) -> bool:
    """Whether a component stands well above the detection threshold: strokes
    of text do, quantisation steps and noise sit just over it."""
    top, left, bottom, right = box
    values = th[top:bottom + 1, left:right + 1][mask[top:bottom + 1, left:right + 1]]
    return values.size > 0 and float(np.percentile(values, 75)) >= 2.0 * threshold


def _split(comps, mask: np.ndarray, th: np.ndarray, threshold: float, max_side: int):
    """Components that look like one glyph, and those that look like a word
    whose letters touch (long, with several strokes crossing each line)."""
    glyphs, words = [], []
    for top, left, bottom, right, area in comps:
        h, w = bottom - top + 1, right - left + 1
        size, small = max(h, w), min(h, w)
        if size < MIN_GLYPH or small > max_side:
            continue
        fill = area / (h * w)
        long_side = size / small
        if not _prominent(th, mask, (top, left, bottom, right), threshold):
            continue
        if long_side <= 2.2:
            if size <= max_side and fill <= (0.86 if size <= 12 else 0.66):
                glyphs.append((top, left, bottom, right, area))
        elif 2.2 < long_side <= 16 and small >= 6 and 0.18 <= fill <= 0.66 and size <= 12 * max_side:
            runs = _runs_per_line(mask[top:bottom + 1, left:right + 1], 1 if w >= h else 0)
            if runs >= 3.0:
                words.append((top, left, bottom, right, area, int(round(long_side))))
        elif small >= 2 and long_side > 2.2 and fill > 0.9 and size <= max_side:
            glyphs.append((top, left, bottom, right, area))        # a bar or dash
    return glyphs, words


def _lines(glyphs, min_line: int = MIN_GLYPHS_IN_LINE) -> list[list[int]]:
    """Groups of glyphs of similar size, near each other and in a line (at any angle)."""
    n = len(glyphs)
    if n < min_line:
        return []
    g = np.array(glyphs, dtype=np.float64)
    cy, cx = (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2
    size = np.maximum(g[:, 2] - g[:, 0] + 1, g[:, 3] - g[:, 1] + 1)
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    big = np.maximum(size[:, None], size[None, :])
    ratio = np.minimum(size[:, None], size[None, :]) / big
    near = (dist <= 2.3 * big) & (ratio >= 0.4)
    np.fill_diagonal(near, False)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(near))):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    lines = []
    for members in groups.values():
        if len(members) < min_line:
            continue
        pts = np.stack([cx[members], cy[members]], 1)
        pts = pts - pts.mean(0)
        s = np.linalg.svd(pts, compute_uv=False)
        limit = 0.15 if len(members) == 4 else 0.3
        collinear = s[0] > 0 and (s[1] / s[0]) < limit and _regular(pts, size[members])
        if _overlapping(g[members]) or _uniform(g[members]):
            continue
        sizes = size[members]
        even = sizes.std() <= 0.42 * sizes.mean()
        if even and collinear:
            lines.append(members)
    return lines


def _overlapping(members: np.ndarray) -> bool:
    """Boxes that pile on top of each other (radial streaks, nested rings) aren't letters side by side."""
    top, left, bottom, right = members[:, 0], members[:, 1], members[:, 2] + 1, members[:, 3] + 1
    ih = np.clip(np.minimum(bottom[:, None], bottom[None, :]) - np.maximum(top[:, None], top[None, :]), 0, None)
    iw = np.clip(np.minimum(right[:, None], right[None, :]) - np.maximum(left[:, None], left[None, :]), 0, None)
    area = (bottom - top) * (right - left)
    ratio = (ih * iw) / np.minimum(area[:, None], area[None, :])
    np.fill_diagonal(ratio, 0)
    return float(ratio.max(axis=1).mean()) > 0.5


def _regular(pts: np.ndarray, size: np.ndarray) -> bool:
    """Along the line, the glyphs are spaced like letters: gaps between neighbours within a few
    glyph sizes, and not wildly uneven (a space between words is about twice a letter gap)."""
    u, _, vt = np.linalg.svd(pts, full_matrices=False)
    along = np.sort(pts @ vt[0])
    gaps = np.diff(along)
    return bool(gaps.size and gaps.max() <= 2.3 * size.mean() and gaps.std() <= 0.9 * max(gaps.mean(), 1e-6))


def _uniform(members: np.ndarray) -> bool:
    """Four or more blobs with the same shape and fill (dots, ellipses, a row of identical marks) aren't letters."""
    if len(members) < 4:
        return False
    h = members[:, 2] - members[:, 0] + 1
    w = members[:, 3] - members[:, 1] + 1
    fill = members[:, 4] / (h * w)
    return bool(fill.std() < 0.04 and (w / h).std() < 0.15)


def _noise(g: np.ndarray, valid: np.ndarray | None = None, block: int = 16) -> float:
    """Pixel noise (standard deviation): the 80th percentile, over 16 x 16 blocks, of each block's
    median neighbouring-pixel difference. A few edges or letters in a block don't move its median, flat
    areas (air, padding, the fill of a turned image) count as zero, and noise in part of the image still shows."""
    height, width = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    if height == 0 or width == 0:
        return 0.0
    d = np.abs(np.diff(g[:height, :width].astype(np.int16), axis=1, append=g[:height, width - 1:width].astype(np.int16)))
    blocks = d.reshape(height // block, block, width // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    meds = np.median(blocks, axis=1)
    if valid is not None:
        ok = valid[:height, :width].reshape(height // block, block, width // block, block).all(axis=(1, 3)).ravel()
        meds = meds[ok]
    if meds.size == 0:
        return 0.0
    return float(np.percentile(meds, 80)) / 0.6745 / np.sqrt(2)


def _grow(boxes, parts, shape):
    """Each box taken over the strokes near it, so words of the same line that weren't found on their own
    (a short word, digits, punctuation) are covered too. Reach: 2.5 line heights."""
    out = []
    for top, left, bottom, right in boxes:
        reach = max(8, int(2.5 * min(bottom - top, right - left)))
        changed = True
        while changed:
            changed = False
            for t, l, b, r in parts:
                if t < bottom + reach and b + 1 > top - reach and l < right + reach and r + 1 > left - reach and not (
                        t >= top and l >= left and b < bottom and r < right):
                    new = (min(top, max(0, t - 2)), min(left, max(0, l - 2)), max(bottom, min(shape[0], b + 3)),
                           max(right, min(shape[1], r + 3)))
                    if new != (top, left, bottom, right):
                        top, left, bottom, right = new
                        changed = True
        out.append((top, left, bottom, right))
    return out


def _detect_polarity(g: np.ndarray, bright: bool, valid: np.ndarray | None = None, min_line: int = MIN_GLYPHS_IN_LINE
                     ) -> tuple[list[tuple[int, int, int, int]], int, str]:
    height, width = g.shape
    max_side = max(MIN_GLYPH + 6, int(0.12 * min(height, width)))
    best: list[tuple[int, int, int, int]] = []
    best_n = 0
    noise = _noise(g, valid)
    for k in (9, 19):
        if min(height, width) < k * 5:
            continue
        th = tophat(g, k, bright)
        threshold = max(FLOOR_CONTRAST, 7.0 * noise)
        if threshold > MAX_THRESHOLD:
            return best, best_n, "too busy to separate strokes"       # noise as strong as text: nothing can be told apart
        for _ in range(4):
            mask = th > threshold
            if valid is not None:
                mask &= valid
            if mask.mean() <= BUSY_FRACTION:
                break
            threshold *= 1.6
        else:
            return best, best_n, "too busy to separate strokes"
        comps = components(mask)
        if comps is None:
            return best, best_n, "too busy to separate strokes"
        glyphs, words = _split(comps, mask, th, threshold, max_side)
        if len(glyphs) > MAX_GLYPHS:
            return best, best_n, "too busy to separate strokes"
        found: list[tuple[int, int, int, int]] = []
        for top, left, bottom, right, _, count in words:
            pad = max(2, (min(bottom - top, right - left) + 1) // 2)
            found.append((max(0, top - pad), max(0, left - pad), min(height, bottom + 1 + pad), min(width, right + 1 + pad)))
            best_n = max(best_n, count)
        for members in _lines(glyphs, min_line):
            sel = [glyphs[i] for i in members]
            pad = max(2, int(0.6 * np.median([max(b - t, r - l) + 1 for t, l, b, r, _ in sel])))
            box = (max(0, min(s[0] for s in sel) - pad), max(0, min(s[1] for s in sel) - pad),
                   min(height, max(s[2] for s in sel) + 1 + pad), min(width, max(s[3] for s in sel) + 1 + pad))
            found.append(box)
            best_n = max(best_n, len(members))
        if found:
            parts = [(t, l, b, r) for t, l, b, r, _ in comps if max(b - t, r - l) + 1 <= max_side
                     and _prominent(th, mask, (t, l, b, r), threshold)]
            best += _grow(found, parts, g.shape)
    return best, best_n, ""


def _merge(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def _rotate(g: np.ndarray, angle: float) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """g rotated by `angle` degrees (background: its median), the offset of the new
    canvas, and a mask of pixels at least 12 px inside the original image
    (the turned frame's own edges are not features)."""
    from PIL import Image

    fill = int(np.median(g))
    img = Image.fromarray(g).rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=fill)
    inside = Image.new("L", (g.shape[1], g.shape[0]), 255).rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    valid = np.asarray(inside) > 127
    for axis in (0, 1):
        valid = _window(valid.view(np.uint8), 25, axis, np.minimum, True).astype(bool)
    valid = np.roll(valid, (12, 12), axis=(0, 1))
    return np.asarray(img), ((img.width - g.shape[1]) / 2.0, (img.height - g.shape[0]) / 2.0), valid


def _unrotate(box: tuple[int, int, int, int], angle: float, shape: tuple[int, int], offset: tuple[float, float]):
    """A box found in the image rotated by `angle` as a box in the original (the box around its corners)."""
    import math

    top, left, bottom, right = box
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
        x, y = x - offset[0] - cx, y - offset[1] - cy
        # PIL rotates counter-clockwise by `angle` in image coordinates (y down), so undo that.
        xs.append(cx + x * cos - y * sin)
        ys.append(cy + x * sin + y * cos)
    return (max(0, int(min(ys))), max(0, int(min(xs))), min(shape[0], int(max(ys)) + 1), min(shape[1], int(max(xs)) + 1))


SKEW_ANGLES = (-45, -30, -15, 15, 30, 45)
SKEW_MIN_LINE = 6          # a turned image needs a longer line, so that turning it doesn't invent text


def detect_heuristic(pixels, skew: bool = True, full_scale: float | None = None) -> Finding:
    """The classical detector. Axis-aligned and any-angle lines of separate
    glyphs are found directly; with `skew`, the image is also turned by
    +-15, 30 and 45 degrees so words whose letters touch are found when
    slanted. Never raises on a valid 2-D image."""
    g = to_gray8(pixels, full_scale)
    if g is None:
        return Finding(False, reason="flat image")
    if max(g.shape) > MAX_SIDE:
        return Finding(True, [(0, 0, g.shape[0], g.shape[1])], reason="too large to check")
    boxes: list[tuple[int, int, int, int]] = []
    glyphs, busy = 0, ""
    for bright in (True, False):
        found, n, why = _detect_polarity(g, bright)
        boxes += found
        glyphs = max(glyphs, n)
        busy = busy or why
    if not boxes and not busy and skew:
        for angle in SKEW_ANGLES:
            turned, offset, valid = _rotate(g, angle)
            for bright in (True, False):
                found, n, _ = _detect_polarity(turned, bright, valid, SKEW_MIN_LINE)
                boxes += [_unrotate(b, angle, g.shape, offset) for b in found]
                glyphs = max(glyphs, n)
            if boxes:
                break
    if boxes:
        return Finding(True, _merge(boxes), glyphs, min(1.0, glyphs / 8), "heuristic", "text-like lines of glyphs")
    if busy:
        return Finding(True, [(0, 0, g.shape[0], g.shape[1])], 0, 0.0, "heuristic", busy)
    return Finding(False, reason="no text found")


# --- OCR (optional) ---------------------------------------------------------------------------------

_ocr_state: dict[str, object] = {}


def ocr_available() -> bool:
    """Whether pytesseract and the Tesseract program can be used. Checked once."""
    if os.environ.get("IMAGING_OCR", "auto").strip().lower() in ("off", "false", "0", "no"):
        return False
    if "module" not in _ocr_state:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            _ocr_state["module"] = pytesseract
        except Exception:  # noqa: BLE001 - not installed, no binary, or broken: the heuristic is used
            _ocr_state["module"] = None
    return _ocr_state["module"] is not None


def reset_ocr() -> None:
    _ocr_state.clear()


def detect_ocr(pixels, min_conf: float = 60.0) -> Finding | None:
    """Words found by Tesseract (3+ letters or digits, confident), or None
    when OCR can't run."""
    if not ocr_available():
        return None
    g = to_gray8(pixels)
    if g is None:
        return Finding(False, method="ocr", reason="flat image")
    from PIL import Image

    pytesseract = _ocr_state["module"]
    scale = 2 if max(g.shape) <= 1024 else 1
    boxes: list[tuple[int, int, int, int]] = []
    words = 0
    try:
        for image in (g, 255 - g):
            pil = Image.fromarray(image)
            if scale > 1:
                pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.BICUBIC)
            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config="--psm 11")
            for text, conf, x, y, w, h in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]):
                text = str(text).strip()
                if sum(c.isalnum() for c in text) >= 3 and float(conf) >= min_conf:
                    words += 1
                    boxes.append((max(0, y // scale - 2), max(0, x // scale - 2),
                                  min(g.shape[0], (y + h) // scale + 3), min(g.shape[1], (x + w) // scale + 3)))
    except Exception:  # noqa: BLE001 - OCR failed at run time: fall back to the heuristic alone
        return None
    if boxes:
        return Finding(True, _merge(boxes), words, min(1.0, words / 3), "ocr", "words recognised")
    return Finding(False, method="ocr", reason="no words recognised")


def detect(pixels, skew: bool = True, full_scale: float | None = None) -> Finding:
    """Flagged when either the OCR engine (if installed) or the heuristic finds text."""
    heuristic = detect_heuristic(pixels, skew, full_scale)
    ocr = detect_ocr(pixels)
    if ocr is None or not ocr.flagged:
        return heuristic
    regions = _merge(heuristic.regions + ocr.regions) if heuristic.flagged and heuristic.reason != "too busy to separate strokes" \
        else ocr.regions
    return Finding(True, regions, max(heuristic.glyphs, ocr.glyphs), max(heuristic.confidence, ocr.confidence), "ocr+heuristic"
                   if heuristic.flagged else "ocr", ocr.reason)


# --- DICOM ----------------------------------------------------------------------------------------------

def _frames(array: np.ndarray, samples: int) -> list[np.ndarray]:
    if samples > 1:
        return [array] if array.ndim == 3 else [array[i] for i in range(array.shape[0])]
    return [array] if array.ndim == 2 else [array[i] for i in range(array.shape[0])]


def scan_dataset(ds, max_frames: int = 8, deep: bool = True) -> Finding:
    """Look for burned-in text in a pydicom dataset's pixels. Frames of a
    multi-frame image are sampled evenly (first, last and between)."""
    array = ds.pixel_array
    full_scale = float(2 ** int(getattr(ds, "BitsStored", 8) or 8))
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    frames = _frames(array, samples)
    if len(frames) > max_frames:
        picks = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, max_frames)})
        frames = [frames[i] for i in picks]
    regions: list[tuple[int, int, int, int]] = []
    glyphs, flagged, reason, method = 0, False, "no text found", "heuristic"
    for frame in frames:
        found = detect(frame, deep, full_scale)
        if found.flagged:
            flagged, reason, method = True, found.reason, found.method
            regions += found.regions
            glyphs = max(glyphs, found.glyphs)
    return Finding(flagged, _merge(regions), glyphs, min(1.0, glyphs / 8), method, reason)


def redact_dataset(ds, regions: list[tuple[int, int, int, int]]) -> None:
    """Black out regions in every frame, in place. The fill is the image's
    darkest stored value. The pixel data is rewritten uncompressed."""
    from pydicom.uid import ExplicitVRLittleEndian

    array = np.array(ds.pixel_array)
    fill = array.min()
    rows, cols = int(ds.Rows), int(ds.Columns)
    for top, left, bottom, right in regions:
        top, left = max(0, top), max(0, left)
        bottom, right = min(rows, bottom), min(cols, right)
        if array.ndim == 2 or (array.ndim == 3 and int(getattr(ds, "SamplesPerPixel", 1) or 1) > 1):
            array[top:bottom, left:right] = fill
        else:
            array[:, top:bottom, left:right] = fill
    ds.PixelData = np.ascontiguousarray(array).tobytes()
    if getattr(ds, "file_meta", None) is not None:
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds["PixelData"].is_undefined_length = False
    if hasattr(ds, "is_little_endian"):
        ds.is_little_endian, ds.is_implicit_VR = True, False
