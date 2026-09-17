"""The model library: trained models, each a folder in MODEL_LIBRARY_DIR
holding its weights (weights.safetensors) and its model card (model.json).

A model folder is self-contained, so it can be copied to another machine's
library, or downloaded as a zip, and used there.

Using a model on an image returns every class's probability, and abstains
when the top probability is below the threshold chosen for that model on
validation images, so an uncertain prediction goes to a clinician."""

from __future__ import annotations

import io
import json
import re
import shutil
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path

from .. import config
from . import architectures, novelty, preprocess

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
_loaded: "OrderedDict[str, tuple[object, dict]]" = OrderedDict()
_lock = threading.Lock()


class LibraryError(ValueError):
    """A model can't be found or used. The message is safe to show."""


def _folder(model_id: str) -> Path:
    if not _ID.match(model_id or ""):
        raise LibraryError("No such model.")
    folder = config.MODEL_LIBRARY_DIR / model_id
    if not (folder / "model.json").is_file() or not (folder / "weights.safetensors").is_file():
        raise LibraryError("No such model.")
    return folder


def get_model(model_id: str) -> dict:
    folder = _folder(model_id)
    card = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    card["size_mb"] = round((folder / "weights.safetensors").stat().st_size / 1024**2, 1)
    return card


def _summary(card: dict) -> dict:
    result = card.get("test") or card.get("validation") or {}
    return {
        "id": card["id"], "name": card["name"], "architecture": card["architecture"],
        "architecture_label": architectures.ARCHITECTURES.get(card["architecture"], {}).get("label", card["architecture"]),
        "classes": card["classes"], "image_size": card["input"]["image_size"],
        "evaluated_on": "test" if card.get("test") else "validation",
        "accuracy": result.get("accuracy"), "balanced_accuracy": result.get("balanced_accuracy"),
        "auc": result.get("auc"), "threshold": card["threshold"]["threshold"],
        "threshold_met": card["threshold"]["met"], "target_accuracy": card["threshold"]["target"],
        "coverage": (result.get("abstention") or {}).get("coverage"),
        "answered_accuracy": (result.get("abstention") or {}).get("answered_accuracy"),
        "target_met": result.get("target_met"),
        "novelty_check": bool((card.get("novelty") or {}).get("cutoffs")),
        "imaging": card.get("imaging"),
        "dataset": card["dataset"]["name"], "images": card["dataset"]["total"],
        "created_at": card["created_at"], "created_by": card.get("created_by"),
        "device_name": card["hardware"].get("device_name"), "training_seconds": card.get("training_seconds"),
        "size_mb": card.get("size_mb"),
    }


def list_models() -> list[dict]:
    if not config.MODEL_LIBRARY_DIR.is_dir():
        return []
    cards = []
    for folder in config.MODEL_LIBRARY_DIR.iterdir():
        try:
            cards.append(_summary(get_model(folder.name)))
        except (LibraryError, OSError, ValueError, KeyError):
            continue
    return sorted(cards, key=lambda c: c["created_at"], reverse=True)


IMAGING_ORIENTATIONS = ("identity", "transpose")


def set_imaging(model_id: str, settings: dict | None) -> dict:
    """How the model is applied to CT or MR series (app/imaging/analysis.py):
    the modality it was trained on, the display window its images used, how
    its images were oriented relative to standard radiological display, and
    the width of anatomy each training image showed, in millimetres. None
    removes the settings."""
    from ..imaging import dicom

    folder = _folder(model_id)
    card = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if settings is None:
        card.pop("imaging", None)
    else:
        modality = settings.get("modality")
        if modality not in dicom.WINDOWS:
            raise LibraryError("Modality must be CT or MR.")
        window = settings.get("window") or None
        if window is not None and window not in dicom.WINDOWS[modality]:
            raise LibraryError(f"Window must be one of: {', '.join(dicom.WINDOWS[modality])}.")
        orientation = settings.get("orientation") or "identity"
        if orientation not in IMAGING_ORIENTATIONS:
            raise LibraryError("Orientation must be identity or transpose.")
        patch_mm = settings.get("patch_mm")
        if patch_mm is not None and not (10 <= float(patch_mm) <= 600):
            raise LibraryError("The width of anatomy must be between 10 and 600 mm.")
        card["imaging"] = {"modality": modality, "window": window, "orientation": orientation,
                           "patch_mm": None if patch_mm is None else float(patch_mm),
                           "note": str(settings.get("note") or "")[:500]}
    tmp = folder / "model.json.tmp"
    tmp.write_text(json.dumps(card, indent=2), encoding="utf-8")
    tmp.replace(folder / "model.json")
    with _lock:
        _loaded.pop(model_id, None)
    return card.get("imaging")


def delete_model(model_id: str) -> None:
    folder = _folder(model_id)
    with _lock:
        _loaded.pop(model_id, None)
    shutil.rmtree(folder)


def export_zip(model_id: str) -> bytes:
    folder = _folder(model_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.json", "weights.safetensors", "novelty.safetensors"):
            if (folder / name).is_file():
                archive.write(folder / name, f"{model_id}/{name}")
    return buffer.getvalue()


def _load(model_id: str):
    with _lock:
        if model_id in _loaded:
            _loaded.move_to_end(model_id)
            return _loaded[model_id]
    import torch  # noqa: F401
    from safetensors.torch import load_file

    card = get_model(model_id)
    model = architectures.build(card["architecture"], len(card["classes"]), card["input"]["model_channels"])
    model.load_state_dict(load_file(str(_folder(model_id) / "weights.safetensors")))
    model.eval()
    novelty_file = _folder(model_id) / "novelty.safetensors"
    if novelty_file.is_file() and (card.get("novelty") or {}).get("cutoffs"):
        card = {**card, "_novelty_stats": novelty.load(novelty_file)}
    with _lock:
        _loaded[model_id] = (model, card)
        while len(_loaded) > 3:
            _loaded.popitem(last=False)
    return model, card


def predict(model_id: str, image_bytes: bytes) -> dict:
    import torch

    model, card = _load(model_id)
    try:
        image = preprocess.open_image(image_bytes)
    except ValueError as exc:
        raise LibraryError(str(exc)) from exc
    spec = card["input"]
    array = preprocess.to_array(image, spec["image_size"], spec["channels"]).copy()
    x = preprocess.normalise(torch.from_numpy(array[None]), spec["mean"], spec["std"], spec["model_channels"])
    logits, feats = novelty.forward(model, card["architecture"], x)
    probs = torch.softmax(logits, dim=1)[0].tolist()
    distance = None
    if "_novelty_stats" in card:
        _, worst = novelty.flagged(feats, card["_novelty_stats"], card["novelty"]["cutoffs"])
        distance = float(worst[0])  # 1.0 is the cut-off
    ranked = sorted(({"class": c, "probability": round(p, 4)} for c, p in zip(card["classes"], probs)),
                    key=lambda r: -r["probability"])
    threshold = card["threshold"]["threshold"]
    top = ranked[0]
    unfamiliar = distance is not None and distance > 1.0
    unsure = top["probability"] < threshold
    if unfamiliar:
        reason, message = "unfamiliar", ("This image doesn't look like the images the model was trained on, so "
                                         "it won't classify it. Refer the image to a clinician.")
    elif unsure:
        reason, message = "low_confidence", (f"Not confident enough to say. The top class, {top['class']}, is at "
                                             f"{top['probability']:.0%}, below this model's {threshold:.0%} "
                                             "threshold. Refer the image to a clinician.")
    else:
        reason, message = None, f"{top['class']}, with {top['probability']:.0%} confidence."
    abstained = reason is not None
    return {
        "model_id": model_id,
        "prediction": None if abstained else top["class"],
        "abstained": abstained,
        "abstain_reason": reason,
        "confidence": top["probability"],
        "threshold": threshold,
        "novelty_score": None if distance is None else round(distance, 3),   # above 1 is unfamiliar
        "probabilities": ranked,
        "message": message,
        "intended_use": card.get("intended_use"),
    }
