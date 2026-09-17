"""Recognising images unlike the ones a model was trained on.

A classifier always picks one of its classes, even for a photo of a cat or a
scan of the wrong body part, and can do so with high confidence. So each
model also records where its training images sit in its own feature space:
the mean of each class and their shared spread. A new image whose features
are far from every class mean, measured by Mahalanobis distance, is outside
what the model knows, and the model abstains.

The cut-off is set so that 1% of the validation images, which the model
should know, would be flagged."""

from __future__ import annotations

import numpy as np

FLAG_RATE = 0.01


def feature_extractor(model, architecture: str):
    import torch.nn as nn

    if architecture == "small-cnn":
        return model[:-1]  # everything before the final linear layer
    if architecture == "resnet18":
        return nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    raise ValueError(architecture)


def fit(features: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    features = features.astype(np.float64)
    dim = features.shape[1]
    means = np.stack([features[labels == c].mean(axis=0) if (labels == c).any() else features.mean(axis=0)
                      for c in range(num_classes)])
    centred = features - means[labels]
    covariance = centred.T @ centred / max(1, len(features) - 1)
    # Shrink towards a scaled identity so the inverse is stable with few images.
    covariance = 0.9 * covariance + 0.1 * (np.trace(covariance) / dim) * np.eye(dim)
    precision = np.linalg.pinv(covariance)
    return {"means": means.astype(np.float32), "precision": precision.astype(np.float32)}


def distances(features: np.ndarray, stats: dict) -> np.ndarray:
    """Distance from each image to its nearest class mean."""
    means, precision = stats["means"].astype(np.float64), stats["precision"].astype(np.float64)
    best = np.full(len(features), np.inf)
    for mean in means:
        diff = features.astype(np.float64) - mean
        best = np.minimum(best, np.einsum("ij,jk,ik->i", diff, precision, diff))
    return np.sqrt(np.maximum(best, 0))


def cutoff(validation_distances: np.ndarray) -> float:
    return float(np.quantile(validation_distances, 1 - FLAG_RATE))
