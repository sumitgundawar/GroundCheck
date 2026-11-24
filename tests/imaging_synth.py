"""Synthetic scans for the burned-in text tests: smooth phantoms with and
without text drawn into the pixels. Values are on a 0..1 scale (0 air, about
0.26 soft tissue, 1 the brightest stored value)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXTS = ["SMITH^JOHN", "ID 9434765919", "2026-03-01 10:15", "DOB 12/03/1948", "Jane Doe", "MRN 00123456", "Patient: A. Jones"]
POSITIONS = ("tl", "tr", "bl", "br", "c")


def phantom(kind: str, n: int, rng: np.random.Generator, noise: float = 0.01) -> np.ndarray:
    y, x = np.mgrid[:n, :n].astype(float)
    img = np.zeros((n, n))
    if kind == "body":
        img[(x - n / 2) ** 2 / (n * 0.42) ** 2 + (y - n / 2) ** 2 / (n * 0.34) ** 2 < 1] = 0.27
        img[(x - n * 0.38) ** 2 + (y - n * 0.45) ** 2 < (n * 0.1) ** 2] = 0.18
        img[(x - n * 0.62) ** 2 + (y - n * 0.45) ** 2 < (n * 0.12) ** 2] = 0.32
        img[n // 5: n // 3, n // 5: n // 3] = 0.6
    elif kind == "gradient":
        img = 0.05 + 0.3 * x / n + 0.1 * y / n
    elif kind == "lungs":
        img[(x - n / 2) ** 2 + (y - n / 2) ** 2 < (n * 0.44) ** 2] = 0.27
        for cx in (n * 0.35, n * 0.65):
            img[(x - cx) ** 2 / (n * 0.14) ** 2 + (y - n / 2) ** 2 / (n * 0.25) ** 2 < 1] = 0.04
        for _ in range(18):
            cx, cy = rng.uniform(0.25, 0.75, 2) * n
            img[(x - cx) ** 2 / (n * 0.008) ** 2 + (y - cy) ** 2 / (n * 0.02) ** 2 < 1] = 0.4
    elif kind == "ribs":
        img[(x - n / 2) ** 2 + (y - n / 2) ** 2 < (n * 0.44) ** 2] = 0.2
        for a in np.linspace(0.3, 2.8, 9):
            cx, cy = n / 2 + np.cos(a) * n * 0.4, n / 2 + np.sin(a) * n * 0.4
            img[(x - cx) ** 2 / (n * 0.03) ** 2 + (y - cy) ** 2 / (n * 0.02) ** 2 < 1] = 0.55
    elif kind == "streaks":
        img[(x - n / 2) ** 2 + (y - n / 2) ** 2 < (n * 0.4) ** 2] = 0.27
        for a in np.linspace(0, np.pi, 12, endpoint=False):
            d = np.abs((x - n / 2) * np.sin(a) - (y - n / 2) * np.cos(a))
            img[d < 1.5] += 0.08
        img[(x - n * 0.5) ** 2 + (y - n * 0.5) ** 2 < (n * 0.03) ** 2] = 0.8
    elif kind == "table":
        img[(x - n / 2) ** 2 / (n * 0.35) ** 2 + (y - n * 0.45) ** 2 / (n * 0.3) ** 2 < 1] = 0.27
        img[int(n * 0.85): int(n * 0.88), int(n * 0.2): int(n * 0.8)] = 0.5
    elif kind == "flat":
        img[:] = 0.1
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 1)


def draw_text(base: np.ndarray, text: str, size: int, pos: str, angle: float, contrast: str,
              rng: np.random.Generator, margin: int = 6) -> np.ndarray:
    """The image with text drawn into it. contrast: "high" (brightest value), "medium" (0.4 above the
    background), "low" (0.22 above) or "dark" (black text on whatever is under it, over a bright box)."""
    n = base.shape[0]
    font = ImageFont.load_default(size=size)
    layer = Image.new("L", (n * 2, size * 3), 0)
    ImageDraw.Draw(layer).text((4, 4), text, fill=255, font=font)
    crop = layer.crop(layer.getbbox())
    if angle:
        crop = crop.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    w, h = crop.size
    w, h = min(w, n - 2 * margin), min(h, n - 2 * margin)
    crop = crop.crop((0, 0, w, h))
    x0 = {"tl": margin, "bl": margin, "tr": n - w - margin, "br": n - w - margin, "c": (n - w) // 2}[pos]
    y0 = {"tl": margin, "tr": margin, "bl": n - h - margin, "br": n - h - margin, "c": (n - h) // 2}[pos]
    mask = np.zeros((n, n))
    mask[y0:y0 + h, x0:x0 + w] = np.asarray(crop, float) / 255
    out = base.copy()
    local = float(np.median(base[max(0, y0 - 6): y0 + h + 6, max(0, x0 - 6): x0 + w + 6]))
    if contrast == "dark":
        box = np.zeros((n, n))
        box[max(0, y0 - 3): y0 + h + 3, max(0, x0 - 3): x0 + w + 3] = 1
        out = out * (1 - box) + 0.85 * box
        return out * (1 - mask)
    value = {"high": 1.0, "medium": min(1.0, local + 0.4), "low": min(1.0, local + 0.22)}[contrast]
    return out * (1 - mask) + value * mask


def stored(image: np.ndarray, bits: int = 12) -> np.ndarray:
    """0..1 as stored integers, and what a CT would say in HU (stored - 1024)."""
    return np.round(image * (2**bits - 1)).astype(np.int32)


def to_hu(image: np.ndarray) -> np.ndarray:
    return stored(image, 12).astype(np.float64) - 1024.0
