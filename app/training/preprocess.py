"""Turning image files into model input, the same way in training and in use."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def open_image(source: Path | bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
        image.load()
    except Exception as exc:  # noqa: BLE001 - any failure means an unreadable image
        raise ValueError("This image couldn't be read. Use PNG, JPEG, BMP, TIFF or WebP.") from exc
    image = ImageOps.exif_transpose(image)
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # 16-bit and float images (common for scans): window to the 0.5th-99.5th
        # percentile, then scale to 8 bits.
        array = np.asarray(image, dtype=np.float32)
        lo, hi = np.percentile(array, [0.5, 99.5])
        array = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    return image


def is_grayscale(image: Image.Image) -> bool:
    if image.mode in ("1", "L", "LA"):
        return True
    rgb = np.asarray(image.convert("RGB"))
    return bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())


def to_array(image: Image.Image, size: int, channels: int) -> np.ndarray:
    """A (channels, size, size) uint8 array."""
    image = image.convert("L" if channels == 1 else "RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.uint8)
    return array[None] if channels == 1 else array.transpose(2, 0, 1)


def normalise(batch, mean: list[float], std: list[float], in_channels: int):
    """uint8 tensor (N, C, H, W) to float, scaled and standardised. Grayscale
    input is repeated to three channels when the model expects three."""
    import torch

    x = batch.float().div_(255)
    if x.shape[1] == 1 and in_channels == 3:
        x = x.repeat(1, 3, 1, 1)
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    if mean_t.shape[1] != x.shape[1]:
        mean_t, std_t = mean_t.mean().view(1, 1, 1, 1), std_t.mean().view(1, 1, 1, 1)
    return (x - mean_t) / std_t
