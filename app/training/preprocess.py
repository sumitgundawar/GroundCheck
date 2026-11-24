"""Turning image files into model input, the same way in training and in use."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def open_image(source) -> Image.Image:
    from ..imaging import nifti

    ref = source if isinstance(source, nifti.SliceRef) else (nifti.parse_ref(source) if isinstance(source, Path) else None)
    if ref is not None:
        return nifti.slice_image(ref)
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


def augment(x, generator=None):
    """Label-preserving changes to a normalised float batch, drawn per image:
    a random shift, a rotation of up to 8 degrees, a crop covering 90-100% of
    the image resized back, light brightness and contrast jitter and light
    Gaussian noise. No flips, which would swap left and right."""
    import math

    import torch
    import torch.nn.functional as F

    n = x.shape[0]
    dev = x.device

    def uniform(low, high, shape):
        return low + (high - low) * torch.rand(shape, device=dev, generator=generator)

    angle = uniform(-8.0, 8.0, (n,)) * math.pi / 180
    scale = uniform(0.9, 1.0, (n,)).sqrt()
    shift = uniform(-1.0, 1.0, (n, 2)) * (1 - scale).unsqueeze(1) + uniform(-0.125, 0.125, (n, 2))
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    gain = 1 + uniform(-0.1, 0.1, (n, 1, 1, 1))
    bias = uniform(-0.1, 0.1, (n, 1, 1, 1))
    noise = torch.randn(x.shape, device=dev, generator=generator) * 0.03
    return x * gain + bias + noise


def tta_views(batch):
    """The batch as is, then shifted a few pixels two ways, for test-time averaging."""
    import torch

    shift = max(1, batch.shape[-1] // 32)
    return [batch, torch.roll(batch, (shift, shift), dims=(2, 3)), torch.roll(batch, (-shift, -shift), dims=(2, 3))]
