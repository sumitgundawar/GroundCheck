"""Reading a folder of images for training.

Two layouts are understood:

    scans/                         scans/
      liver/  a.png b.png            train/  liver/ ...  kidney/ ...
      kidney/ c.png ...              val/    liver/ ...  kidney/ ...
                                     test/   liver/ ...  kidney/ ...

In the first, GroundCheck splits the images into training, validation and
test sets itself, keeping each class's share the same in every set. In the
second, the folders' own split is used (val and test are optional).

Folders are only read from TRAINING_DATA_DIRS, and a path can't escape them
through .. or a symbolic link."""

from __future__ import annotations

import hashlib
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .. import config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train": {"train", "training"}, "val": {"val", "valid", "validation"}, "test": {"test", "testing"}}
MAX_IMAGES = 1_000_000
MIN_IMAGES_PER_CLASS = 5


class DatasetError(ValueError):
    """The folder can't be used. The message is safe to show."""


@dataclass(frozen=True)
class ImageItem:
    path: str       # relative to the dataset folder
    label: int
    split: str      # train, val or test


def roots() -> list[Path]:
    return [r for r in config.TRAINING_DATA_DIRS if r.is_dir()]


def resolve(path: str) -> Path:
    """An existing folder inside one of the allowed roots."""
    if not path or "\x00" in path:
        raise DatasetError("Choose a folder.")
    candidate = Path(os.path.expanduser(path))
    if not candidate.is_absolute():
        candidate = config.ROOT_DIR / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DatasetError("That folder doesn't exist.") from None
    if not resolved.is_dir():
        raise DatasetError("That isn't a folder.")
    if not any(resolved == r or r in resolved.parents for r in roots()):
        raise DatasetError("Training can only read folders inside TRAINING_DATA_DIRS.")
    return resolved


def resolve_file(dataset: Path, relative: str) -> Path:
    file = (dataset / relative).resolve()
    if dataset not in file.parents or not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
        raise DatasetError("No such image in this dataset.")
    return file


def _visible_dirs(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name.lower())
    except OSError:
        return []


def _images(folder: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith(".") and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                found.append(Path(dirpath) / name)
                if len(found) > MAX_IMAGES:
                    raise DatasetError(f"This folder has more than {MAX_IMAGES:,} images.")
    return found


def _split_of(name: str) -> str | None:
    lowered = name.lower()
    return next((split for split, names in SPLIT_NAMES.items() if lowered in names), None)


def _has_images_directly(folder: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir())
    except OSError:
        return False


def browse(path: str | None) -> dict:
    """Subfolders of a folder, for choosing a dataset."""
    if not path:
        return {"path": None, "parent": None,
                "folders": [{"name": str(r), "path": str(r), "looks_like_dataset": _looks_like_dataset(r)}
                            for r in roots()]}
    folder = resolve(path)
    parent = folder.parent if any(r in folder.parents for r in roots()) else None
    return {
        "path": str(folder),
        "parent": str(parent) if parent else None,
        "folders": [{"name": d.name, "path": str(d), "looks_like_dataset": _looks_like_dataset(d)}
                    for d in _visible_dirs(folder)][:500],
    }


def _looks_like_dataset(folder: Path) -> bool:
    subdirs = _visible_dirs(folder)[:50]
    if any(_split_of(d.name) == "train" for d in subdirs):
        return True
    # Most subfolders hold images directly: class folders, not a home folder
    # that happens to contain a few picture folders.
    sample = subdirs[:10]
    with_images = sum(_has_images_directly(d) for d in sample)
    return len(sample) >= 2 and with_images >= 2 and with_images >= 0.6 * len(sample)


def read(path: str, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 7
         ) -> tuple[Path, list[str], list[ImageItem], dict]:
    """The dataset's classes, every image with its class and split, and a
    summary. Raises DatasetError when the folder can't be trained on."""
    folder = resolve(path)
    subdirs = _visible_dirs(folder)
    split_dirs = {s: d for d in subdirs if (s := _split_of(d.name))}
    warnings: list[str] = []

    if "train" in split_dirs:
        layout = "split-folders"
        per_split: dict[str, dict[str, list[Path]]] = {}
        for split, split_dir in split_dirs.items():
            per_split[split] = {d.name: _images(d) for d in _visible_dirs(split_dir)}
        classes = sorted({c for groups in per_split.values() for c in groups if groups[c]}, key=str.lower)
        train_classes = {c for c, files in per_split["train"].items() if files}
        missing = [c for c in classes if c not in train_classes]
        if missing:
            raise DatasetError(f"These classes have no training images: {', '.join(missing)}.")
        index = {c: i for i, c in enumerate(classes)}
        items = [ImageItem(str(f.relative_to(folder)), index[c], split)
                 for split, groups in per_split.items() for c, files in groups.items() if c in index for f in files]
        if "val" not in split_dirs:
            items = _carve(items, "val", val_fraction, seed)
            warnings.append("There's no val folder, so part of the training images is used for validation.")
        if "test" not in split_dirs:
            warnings.append("There's no test folder, so the model is checked on validation images only.")
    else:
        layout = "class-folders"
        groups = {d.name: _images(d) for d in subdirs}
        groups = {c: files for c, files in groups.items() if files}
        classes = sorted(groups, key=str.lower)
        index = {c: i for i, c in enumerate(classes)}
        rng = random.Random(seed)
        items = []
        for c in classes:
            files = list(groups[c])
            rng.shuffle(files)
            n_val = max(1, round(len(files) * val_fraction))
            n_test = max(1, round(len(files) * test_fraction)) if len(files) >= 10 else 0
            for i, f in enumerate(files):
                split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
                items.append(ImageItem(str(f.relative_to(folder)), index[c], split))

    if len(classes) < 2:
        raise DatasetError("Put images in at least two subfolders, one for each class (for example normal/ and abnormal/).")

    counts = {c: {"train": 0, "val": 0, "test": 0} for c in classes}
    for item in items:
        counts[classes[item.label]][item.split] += 1
    small = [c for c in classes if counts[c]["train"] < MIN_IMAGES_PER_CLASS]
    if small:
        raise DatasetError(f"Each class needs at least {MIN_IMAGES_PER_CLASS} training images. "
                           f"Too few in: {', '.join(small)}.")
    totals = [sum(v.values()) for v in counts.values()]
    if max(totals) > 10 * min(totals):
        warnings.append("Some classes have over ten times as many images as others. Training weights classes "
                        "to balance this, but more images of the rarer classes would help.")

    digest = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.path):
        digest.update(f"{item.path}\0{item.label}\0{item.split}\n".encode())
    samples = []
    for c in classes:
        first = next((i for i in items if classes[i.label] == c), None)
        if first:
            samples.append({"class": c, "file": first.path})
    summary = {
        "path": str(folder),
        "name": folder.name,
        "layout": layout,
        "classes": [{"name": c, **counts[c], "total": sum(counts[c].values())} for c in classes],
        "total": len(items),
        "splits": {s: sum(1 for i in items if i.split == s) for s in ("train", "val", "test")},
        "fingerprint": digest.hexdigest()[:16],
        "samples": samples[:24],
        "warnings": warnings,
    }
    return folder, classes, items, summary


IDENTIFIER_SAMPLE = 2000
_METADATA_KEYS = ("patient", "name", "birth", "dob", "mrn", "nhs", "accession", "institution", "physician",
                  "author", "artist", "comment", "description", "usercomment")


def _content_key(path: Path) -> bytes:
    """A fast fingerprint: size plus a hash of the first 64 KB."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        return size.to_bytes(8, "little") + hashlib.blake2b(fh.read(65536), digest_size=16).digest()


def _full_hash(path: Path) -> bytes:
    digest = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def find_duplicates(folder: Path, items: list[ImageItem]) -> dict:
    """Identical image files. A copy in both the training set and the
    validation or test set makes the model look better than it is, so those
    copies are left out of validation and test when training."""
    by_key: dict[bytes, list[ImageItem]] = defaultdict(list)
    for item in items:
        try:
            by_key[_content_key(folder / item.path)].append(item)
        except OSError:
            continue
    groups: list[list[ImageItem]] = []
    for candidates in by_key.values():
        if len(candidates) < 2:
            continue
        exact: dict[bytes, list[ImageItem]] = defaultdict(list)
        for item in candidates:
            exact[_full_hash(folder / item.path)].append(item)
        groups.extend(g for g in exact.values() if len(g) > 1)
    leaked = sorted({i.path for g in groups if any(x.split == "train" for x in g)
                     for i in g if i.split != "train"})
    conflicting = [g for g in groups if len({i.label for i in g}) > 1]
    return {
        "groups": len(groups),
        "extra_copies": sum(len(g) - 1 for g in groups),
        "leaked": leaked,
        "conflicting_labels": len(conflicting),
        "examples": [[i.path for i in g][:3] for g in (conflicting or groups)[:5]],
    }


def find_identifiers(folder: Path, items: list[ImageItem], seed: int = 7) -> dict:
    """Possible patient identifiers in file names (every file) and in image
    metadata (a sample). Text burned into the pixels isn't detected."""
    from .. import deid

    in_names: dict[str, int] = defaultdict(int)
    name_examples: list[str] = []
    files_with_names = 0
    for item in items:
        stem = Path(item.path).stem.replace("_", " ")
        result = deid.deidentify(stem)
        kinds = {k for k in result.found if k not in ("long number",)}
        for kind in kinds:
            in_names[kind] += 1
        if kinds:
            files_with_names += 1
            if len(name_examples) < 3:
                name_examples.append(item.path)

    from PIL import Image

    sample = random.Random(seed).sample(items, min(IDENTIFIER_SAMPLE, len(items)))
    metadata_files, metadata_examples = 0, []
    for item in sample:
        try:
            with Image.open(folder / item.path) as image:
                fields = {str(k).lower(): v for k, v in image.info.items() if isinstance(v, (str, bytes))}
                exif = image.getexif()
                fields.update({str(k): v for k, v in exif.items() if isinstance(v, (str, bytes))})
        except Exception:  # noqa: BLE001 - unreadable files are reported when training
            continue
        if any(any(word in key for word in _METADATA_KEYS) or (isinstance(v, str) and deid.deidentify(v).changed)
               for key, v in fields.items() if key not in ("icc_profile", "exif", "dpi", "gamma", "transparency")):
            metadata_files += 1
            if len(metadata_examples) < 3:
                metadata_examples.append(item.path)
    return {
        "file_names": files_with_names,
        "file_name_kinds": dict(in_names),
        "file_name_examples": name_examples,
        "metadata_sampled": len(sample),
        "metadata_files": metadata_files,
        "metadata_examples": metadata_examples,
    }


def inspect(folder: Path, items: list[ImageItem]) -> dict:
    duplicates = find_duplicates(folder, items)
    identifiers = find_identifiers(folder, items)
    warnings = []
    if duplicates["leaked"]:
        warnings.append(f"{len(duplicates['leaked']):,} validation or test images are exact copies of training "
                        "images. They'll be left out of validation and test, so results aren't flattered.")
    if duplicates["conflicting_labels"]:
        warnings.append(f"{duplicates['conflicting_labels']:,} identical images are filed under different "
                        "classes. Check their labels.")
    if identifiers["file_names"]:
        warnings.append(f"Some file names look like they contain patient details "
                        f"({', '.join(sorted(identifiers['file_name_kinds']))}). Rename files before sharing a model "
                        "or dataset.")
    if identifiers["metadata_files"]:
        warnings.append(f"{identifiers['metadata_files']:,} of {identifiers['metadata_sampled']:,} images checked "
                        "carry metadata that may identify a patient. Strip metadata before training on shared data.")
    return {"duplicates": {k: v for k, v in duplicates.items() if k != "leaked"} | {"leaked": len(duplicates["leaked"])},
            "identifiers": identifiers, "warnings": warnings, "_leaked_paths": duplicates["leaked"]}


def _carve(items: list[ImageItem], split: str, fraction: float, seed: int) -> list[ImageItem]:
    """Move a stratified fraction of the training images into another split."""
    rng = random.Random(seed)
    by_label: dict[int, list[ImageItem]] = {}
    for item in items:
        if item.split == "train":
            by_label.setdefault(item.label, []).append(item)
    moved: set[str] = set()
    for group in by_label.values():
        chosen = rng.sample(group, max(1, round(len(group) * fraction)))
        moved.update(i.path for i in chosen)
    return [ImageItem(i.path, i.label, split) if i.path in moved else i for i in items]
