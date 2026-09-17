"""Refit a library model's unfamiliar-image check from its training data,
without retraining the model. Use it after the check's method changes:

    python scripts/refit_novelty.py <model id> [<model id> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.training import architectures, datasets, library, novelty, worker  # noqa: E402


class _Quiet:
    def update(self, *a, **k): ...
    def check_cancel(self): ...


def refit(model_id: str) -> dict:
    import torch
    from safetensors.torch import load_file

    folder_path = config.MODEL_LIBRARY_DIR / model_id
    card = json.loads((folder_path / "model.json").read_text(encoding="utf-8"))
    seed = int(card["training"]["seed"])
    torch.manual_seed(seed)
    folder, classes, items, _ = datasets.read(card["dataset"]["path"], seed=seed)
    spec = card["input"]
    model = architectures.build(card["architecture"], len(classes), spec["model_channels"], False)
    model.load_state_dict(load_file(str(folder_path / "weights.safetensors")))
    model.eval()
    progress, device = _Quiet(), torch.device("cpu")
    split = {s: [i for i in items if i.split == s] for s in ("train", "val", "test")}
    load = lambda s: worker._load_images(folder, split[s], spec["image_size"], spec["channels"], progress, s)  # noqa: E731
    feats = lambda images: worker._features(model, card["architecture"], images, device, 256, spec["mean"],  # noqa: E731
                                            spec["std"], spec["model_channels"], progress)
    train = load("train")
    subset = torch.randperm(len(train))[:20000]
    labels = torch.tensor([i.label for i in split["train"]])[subset].numpy()
    stats, cutoffs = novelty.fit_layers(feats(train[subset]), labels, len(classes), feats(load("val")))
    novelty.save(folder_path / "novelty.safetensors", stats)
    card["novelty"] = {"method": "mahalanobis-per-block", "cutoffs": cutoffs, "flag_rate": novelty.FLAG_RATE}
    if card.get("test") and split["test"]:
        flags, _ = novelty.flagged(feats(load("test")), stats, cutoffs)
        card["test"]["novelty_flagged"] = float(flags.mean())
    (folder_path / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card["novelty"] | {"test_flagged": (card.get("test") or {}).get("novelty_flagged")}


if __name__ == "__main__":
    for model_id in sys.argv[1:]:
        library.get_model(model_id)
        print(model_id, refit(model_id))
