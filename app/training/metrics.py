"""Evaluating a trained classifier.

Besides accuracy, a clinical model is judged by sensitivity and specificity
for each class, how well its confidence is calibrated, and when it should
not answer. GroundCheck picks each model's confidence threshold on the
validation images: the lowest confidence at which the images it answers are
at least MODEL_TARGET_ACCURACY correct, judged by the lower end of a 95%
confidence interval (Wilson), so a small validation set can't flatter it. Below the threshold, the model
abstains and the image goes to a clinician. The test images then show
whether that holds on images the threshold was not chosen on."""

from __future__ import annotations

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, with ties given their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Area under the ROC curve (Mann-Whitney U). None if only one class is present."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: how far confidence is from accuracy, on average."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def wilson_lower(correct: np.ndarray | float, total: np.ndarray | float, z: float = 1.96):
    """Lower bound of the 95% Wilson score interval for a proportion."""
    total = np.maximum(total, 1)
    p = correct / total
    centre = p + z * z / (2 * total)
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - margin) / (1 + z * z / total)


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target: float) -> dict:
    """The lowest confidence threshold at which answered images meet the
    target accuracy, by the Wilson lower bound. If none does, the model
    abstains on everything."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    order = np.argsort(-confidence, kind="mergesort")
    conf_sorted, correct_sorted = confidence[order], correct[order]
    cumulative = np.cumsum(correct_sorted)
    counts = np.arange(1, len(order) + 1)
    # Only thresholds at the end of a run of equal confidences are valid:
    # a threshold includes every image with that confidence.
    last_of_value = np.r_[conf_sorted[1:] != conf_sorted[:-1], True]
    ok = (wilson_lower(cumulative, counts) >= target) & last_of_value
    if not ok.any():
        return {"threshold": 1.0, "coverage": 0.0, "accuracy": None, "met": False, "target": target}
    k = int(np.nonzero(ok)[0].max())
    return {"threshold": float(conf_sorted[k]), "coverage": float((k + 1) / len(order)),
            "accuracy": float(cumulative[k] / (k + 1)),
            "accuracy_lower_bound": float(wilson_lower(cumulative[k], k + 1)), "met": True, "target": target}


def evaluate(probs: np.ndarray, labels: np.ndarray, classes: list[str], threshold: float) -> dict:
    n, k = probs.shape
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predicted == labels
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)

    per_class = []
    for c in range(k):
        tp = confusion[c, c]
        fn = confusion[c].sum() - tp
        fp = confusion[:, c].sum() - tp
        tn = n - tp - fn - fp
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if precision and sensitivity else 0.0 if (tp + fn) else None)
        per_class.append({
            "name": classes[c], "support": int(tp + fn),
            "sensitivity": _num(sensitivity), "specificity": _num(specificity),
            "precision": _num(precision), "f1": _num(f1),
            "auc": _num(roc_auc(probs[:, c], labels == c)),
        })
    recalls = [p["sensitivity"] for p in per_class if p["sensitivity"] is not None]
    f1s = [p["f1"] for p in per_class if p["f1"] is not None]
    aucs = [p["auc"] for p in per_class if p["auc"] is not None]
    answered = confidence >= threshold
    return {
        "images": int(n),
        "accuracy": float(correct.mean()) if n else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "auc": (per_class[1]["auc"] if k == 2 else float(np.mean(aucs)) if aucs else None),
        "calibration_error": calibration_error(confidence, correct) if n else None,
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "abstention": {
            "threshold": threshold,
            "coverage": float(answered.mean()) if n else None,
            "answered_accuracy": float(correct[answered].mean()) if answered.any() else None,
            "abstained": int((~answered).sum()),
        },
    }


def _num(value) -> float | None:
    return None if value is None else round(float(value), 4)
