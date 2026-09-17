"""Running an imaging model over a series.

A model from the training library (app/training/library.py) is used through
its card's "imaging" settings: the modality it was trained on, the display
window, how its training images were oriented relative to standard
radiological display, and the patch size in millimetres. A series is
searched slice by slice with overlapping patches of that physical size, so
scanners with different pixel spacing see the same anatomy per patch.

Every patch is either answered or abstained, as in the model library: below
the model's confidence threshold, or unlike its training images (novelty
statistics far from its training images at any network block). Answered patches of the same class on a slice are
merged into one region. A series whose modality differs from the model's is
refused outright. When at least half of a series' patches are unfamiliar, the
model abstains on the whole series and reports no regions.

Other models can implement ImagingModel directly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from ..training import library, novelty, preprocess
from . import dicom

UNFAMILIAR_SERIES = 0.5


class AnalysisError(ValueError):
    """A model can't analyse this series. The message is safe to show."""


class ImagingModel(Protocol):
    id: str
    name: str
    modality: str | None

    def analyse(self, volume: np.ndarray, pixel_spacing: list[float], modality: str) -> dict: ...


def _merge(boxes: list[tuple[int, int, int, int, float]]) -> list[dict]:
    """Merge overlapping boxes (x0, y0, x1, y1, confidence) into regions."""
    regions: list[list] = []
    for x0, y0, x1, y1, conf in sorted(boxes, key=lambda b: -b[4]):
        for r in regions:
            if x0 <= r[2] and r[0] <= x1 and y0 <= r[3] and r[1] <= y1:
                r[0], r[1], r[2], r[3] = min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1)
                r[4] = max(r[4], conf)
                r[5] += 1
                break
        else:
            regions.append([x0, y0, x1, y1, conf, 1])
    return [{"box": r[:4], "confidence": round(r[4], 4), "patches": r[5]} for r in regions]


def _runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


@dataclass
class LibraryModel:
    id: str
    name: str
    modality: str | None
    card: dict

    @classmethod
    def load(cls, model_id: str) -> "LibraryModel":
        card = library.get_model(model_id)
        imaging = card.get("imaging") or {}
        return cls(id=model_id, name=card["name"], modality=imaging.get("modality"), card=card)

    def analyse(self, volume: np.ndarray, pixel_spacing: list[float], modality: str,
                check_modality: bool = True, progress=None) -> dict:
        import torch

        imaging = self.card.get("imaging") or {}
        if check_modality and self.modality and modality != self.modality:
            raise AnalysisError(f"{self.name} was trained on {self.modality} images; this series is {modality}.")
        model, card = library._load(self.id)
        spec = card["input"]
        stats = card.get("_novelty_stats")
        cutoffs = (card.get("novelty") or {}).get("cutoffs")
        threshold = card["threshold"]["threshold"]
        size = spec["image_size"]
        patch_mm = imaging.get("patch_mm")
        spacing = float(pixel_spacing[0]) if pixel_spacing else 1.0
        _, rows, cols = volume.shape
        patch = min(rows, cols) if not patch_mm else max(16, min(rows, cols, int(round(patch_mm / spacing))))
        stride = max(1, patch // 2)
        orientation = imaging.get("orientation", "identity")

        slices, totals = [], {"patches": 0, "unfamiliar": 0, "low_confidence": 0, "answered": 0}
        present: dict[str, list[int]] = {}
        best: dict[str, float] = {}
        for index in range(volume.shape[0]):
            grey = dicom.window(volume[index], modality, imaging.get("window"))
            boxes, arrays = [], []
            for y in range(0, rows - patch + 1, stride):
                for x in range(0, cols - patch + 1, stride):
                    piece = grey[y:y + patch, x:x + patch]
                    if orientation == "transpose":
                        piece = piece.T
                    if piece.mean() < 8:      # air or outside the body: nothing to classify
                        continue
                    image = Image.fromarray(np.ascontiguousarray(piece)).resize((size, size), Image.Resampling.BILINEAR)
                    arrays.append(preprocess.to_array(image, size, spec["channels"]))
                    boxes.append((x, y, x + patch, y + patch))
            findings: list[dict] = []
            counts = {"patches": len(boxes), "unfamiliar": 0, "low_confidence": 0, "answered": 0}
            if boxes:
                x = preprocess.normalise(torch.from_numpy(np.stack(arrays)), spec["mean"], spec["std"],
                                         spec["model_channels"])
                logits, feats = novelty.forward(model, card["architecture"], x)
                probs = torch.softmax(logits, dim=1).numpy()
                unfamiliar = novelty.flagged(feats, stats, cutoffs)[0] if stats is not None else None
                accepted: dict[str, list] = {}
                for k, box in enumerate(boxes):
                    if unfamiliar is not None and unfamiliar[k]:
                        counts["unfamiliar"] += 1
                        continue
                    confidence = float(probs[k].max())
                    if confidence < threshold:
                        counts["low_confidence"] += 1
                        continue
                    counts["answered"] += 1
                    label = card["classes"][int(probs[k].argmax())]
                    accepted.setdefault(label, []).append((*box, confidence))
                for label, label_boxes in accepted.items():
                    for region in _merge(label_boxes):
                        findings.append({"label": label, **region})
                    present.setdefault(label, []).append(index)
                    best[label] = max(best.get(label, 0), max(b[4] for b in label_boxes))
            for key in totals:
                totals[key] += counts[key]
            slices.append({"index": index, "findings": sorted(findings, key=lambda f: -f["confidence"]), **counts})
            if progress:
                progress(index + 1, volume.shape[0])

        unfamiliar_share = totals["unfamiliar"] / totals["patches"] if totals["patches"] else 1.0
        unfamiliar_series = unfamiliar_share >= UNFAMILIAR_SERIES
        if unfamiliar_series:
            # Most of the series is unlike anything the model learned from, so
            # what it says about the rest can't be trusted either: abstain on
            # the whole series rather than show a few confident labels.
            for s in slices:
                s["withheld"] = len(s["findings"])
                s["findings"] = []
            present, best = {}, {}
        summary = {
            "labels": [{"label": label, "slices": len(idx), "ranges": _runs(idx), "max_confidence": round(best[label], 4)}
                       for label, idx in sorted(present.items(), key=lambda kv: -len(kv[1]))],
            "totals": totals,
            "unfamiliar_share": round(unfamiliar_share, 4),
            "abstained_share": round((totals["unfamiliar"] + totals["low_confidence"]) / totals["patches"], 4)
            if totals["patches"] else 1.0,
            "unfamiliar_series": unfamiliar_series,
            "abstained": unfamiliar_series or totals["answered"] == 0,
            "patch_pixels": patch, "patch_mm": patch_mm, "orientation": orientation,
            "threshold": threshold,
        }
        return {"model_id": self.id, "model_name": self.name, "model_modality": self.modality,
                "summary": summary, "slices": slices}
