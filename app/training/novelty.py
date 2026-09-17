"""Recognising images unlike the ones a model was trained on.

A classifier always picks one of its classes, even for a photo of a cat or a
scan of the wrong body part, and can do so with high confidence. So each
model also records what its training images look like inside the network:
at the end of every block, the mean and spread of each feature map, summarised
per class. A new image is unfamiliar when, at any block, its statistics are
far from every class (Mahalanobis distance).

Early blocks see texture, contrast and noise, so they catch a different
scanner, modality or preprocessing; late blocks see shapes, so they catch
the wrong anatomy. Using only the last block, as many systems do, misses a
modality change: measured on real data, it flagged no MRI patches for a CT
model, while the first block flagged all of them.

Each block's cut-off is set on validation images so that, across all
blocks, about FLAG_RATE of images the model should know are flagged."""

from __future__ import annotations

import numpy as np

FLAG_RATE = 0.02


def blocks(model, architecture: str) -> list:
    """The modules whose outputs are summarised, shallow to deep."""
    if architecture == "small-cnn":
        return [model[i] for i in range(4)]
    if architecture == "resnet18":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    raise ValueError(architecture)


def forward(model, architecture: str, x):
    """Logits and per-block features (mean and standard deviation of each
    channel) in one pass."""
    import torch

    captured: list = []
    hooks = [b.register_forward_hook(lambda _m, _i, out: captured.append(out)) for b in blocks(model, architecture)]
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        for h in hooks:
            h.remove()
    # The spread of a 1x1 feature map is undefined (small images, deep blocks): use 0.
    feats = [torch.cat([c.mean(dim=(2, 3)), c.flatten(2).std(dim=2) if c.shape[2] * c.shape[3] > 1
                        else torch.zeros_like(c.mean(dim=(2, 3)))], dim=1).float().cpu().numpy()
             for c in captured]
    return logits, feats


def fit(features: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    features = np.nan_to_num(features.astype(np.float64))
    dim = features.shape[1]
    means = np.stack([features[labels == c].mean(axis=0) if (labels == c).any() else features.mean(axis=0)
                      for c in range(num_classes)])
    centred = features - means[labels]
    covariance = centred.T @ centred / max(1, len(features) - 1)
    # Shrink towards a scaled identity so the inverse is stable with few images.
    covariance = 0.9 * covariance + 0.1 * (np.trace(covariance) / dim) * np.eye(dim)
    return {"means": means.astype(np.float32), "precision": np.linalg.pinv(covariance).astype(np.float32)}


def _whitening(stats: dict) -> np.ndarray:
    """A matrix W with W Wᵀ = precision, so Mahalanobis distance becomes plain
    distance after multiplying by W. Computed once per model and kept: doing
    it per class, per image, made a scan's check the slowest part of it."""
    cached = stats.get("_whitening")
    if cached is not None:
        return cached
    precision = stats["precision"].astype(np.float64)
    precision = (precision + precision.T) / 2          # symmetric up to rounding
    values, vectors = np.linalg.eigh(precision)
    whitening = vectors * np.sqrt(np.clip(values, 0, None))
    stats["_whitening"] = whitening
    return whitening


def distances(features: np.ndarray, stats: dict) -> np.ndarray:
    """Distance from each image to its nearest class mean, measured with the
    training data's own spread (Mahalanobis)."""
    whitening = _whitening(stats)
    points = np.nan_to_num(features.astype(np.float64)) @ whitening
    centres = stats["means"].astype(np.float64) @ whitening
    # |p - c|² for every point and centre, without building the differences.
    squared = (np.square(points).sum(axis=1)[:, None] - 2 * points @ centres.T
               + np.square(centres).sum(axis=1)[None, :])
    return np.sqrt(np.maximum(squared.min(axis=1), 0))


def cutoff(validation_distances: np.ndarray, layers: int = 1) -> float:
    return float(np.quantile(validation_distances, 1 - FLAG_RATE / layers))


def fit_layers(train_feats: list[np.ndarray], labels: np.ndarray, num_classes: int,
               val_feats: list[np.ndarray]) -> tuple[list[dict], list[float]]:
    stats = [fit(f, labels, num_classes) for f in train_feats]
    cutoffs = [cutoff(distances(v, s), len(stats)) for v, s in zip(val_feats, stats)]
    return stats, cutoffs


def flagged(feats: list[np.ndarray], stats: list[dict], cutoffs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """(unfamiliar per image, worst distance-to-cutoff ratio per image)."""
    ratios = np.stack([distances(f, s) / c for f, s, c in zip(feats, stats, cutoffs)], axis=1)
    worst = ratios.max(axis=1)
    return worst > 1.0, worst


def save(path, stats: list[dict]) -> None:
    import torch
    from safetensors.torch import save_file

    tensors = {}
    for i, s in enumerate(stats):
        tensors[f"means_{i}"] = torch.from_numpy(s["means"])
        tensors[f"precision_{i}"] = torch.from_numpy(s["precision"])
    save_file(tensors, str(path))


def load(path) -> list[dict]:
    from safetensors.torch import load_file

    raw = {k: v.numpy() for k, v in load_file(str(path)).items()}
    count = len([k for k in raw if k.startswith("means_")])
    return [{"means": raw[f"means_{i}"], "precision": raw[f"precision_{i}"]} for i in range(count)]
