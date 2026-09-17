"""The training process. Started by runs.py as its own Python process, so
training never slows the app and can be stopped cleanly:

    python -m app.training.worker models/runs/<run id> [--resume]

It reads run.json, trains, and writes progress.json as it goes. After every
epoch it saves a checkpoint, so a stopped run can resume where it left off. The best
model (lowest validation loss) is evaluated, given its confidence threshold,
and saved to the model library. A file named `cancel` in the run folder
stops it at the next batch."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config
from . import architectures, datasets, metrics, novelty, preprocess

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CACHE_LIMIT_BYTES = 2 * 1024**3


class Cancelled(Exception):
    pass


class Progress:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.state: dict = {"state": "running", "phase": "starting", "message": "Starting", "history": [],
                            "started_at": _now(), "pid": os.getpid()}
        self._last_write = 0.0

    def update(self, force: bool = False, **changes) -> None:
        self.state.update(changes)
        self.state["updated_at"] = _now()
        if force or time.monotonic() - self._last_write > 0.75:
            tmp = self.run_dir / "progress.json.tmp"
            tmp.write_text(json.dumps(self.state), encoding="utf-8")
            os.replace(tmp, self.run_dir / "progress.json")
            self._last_write = time.monotonic()

    def check_cancel(self) -> None:
        if (self.run_dir / "cancel").exists():
            raise Cancelled()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_images(folder: Path, items, size: int, channels: int, progress: Progress, phase: str):
    import torch

    array = np.empty((len(items), channels, size, size), dtype=np.uint8)
    for i, item in enumerate(items):
        array[i] = preprocess.to_array(preprocess.open_image(folder / item.path), size, channels)
        if i % 200 == 0:
            progress.check_cancel()
            progress.update(phase="loading", message=f"Loading {phase} images: {i:,} of {len(items):,}",
                            loaded=i, to_load=len(items))
    return torch.from_numpy(array)


def _augment(x):
    """Label-preserving changes for scans: small shifts and brightness and
    contrast jitter. No flips, which would swap left and right."""
    import torch

    n = x.shape[0]
    shift = max(1, x.shape[-1] // 16)
    dx, dy = random.randint(-shift, shift), random.randint(-shift, shift)
    x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    gain = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    bias = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    return x * gain + bias


def _recalibrate_batch_norm(model, images, device, batch_size, mean, std, in_channels, max_batches=50):
    """Recompute batch-norm statistics from training images with the current
    weights. Running averages lag behind while weights change, which with few
    batches per epoch makes evaluation much worse than training."""
    import torch

    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # a cumulative average over the batches below
    model.train()
    order = torch.randperm(len(images))
    with torch.no_grad():
        for b in range(min(max_batches, max(1, len(images) // batch_size))):
            idx = order[b * batch_size:(b + 1) * batch_size]
            model(preprocess.normalise(images[idx].to(device), mean, std, in_channels))
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    model.eval()


def _features(model, arch, images, device, batch_size, mean, std, in_channels, progress):
    """Per-block features for novelty detection, as a list with one array per block."""
    per_block: list[list] = []
    model.eval()
    for start in range(0, len(images), batch_size):
        progress.check_cancel()
        xb = preprocess.normalise(images[start:start + batch_size].to(device), mean, std, in_channels)
        _, feats = novelty.forward(model, arch, xb)
        if not per_block:
            per_block = [[] for _ in feats]
        for i, f in enumerate(feats):
            per_block[i].append(f)
    return [np.concatenate(parts) for parts in per_block]


def _predict(model, images, labels, device, batch_size, mean, std, in_channels, progress, what):
    import torch

    model.eval()
    all_probs, losses = [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            progress.check_cancel()
            xb = preprocess.normalise(images[start:start + batch_size].to(device), mean, std, in_channels)
            logits = model(xb)
            yb = labels[start:start + batch_size].to(device)
            losses.append(torch.nn.functional.cross_entropy(logits, yb, reduction="sum").item())
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 0))
    return probs, sum(losses) / max(1, len(images))


def train(run_dir: Path, resume: bool = False) -> None:
    import torch
    from safetensors.torch import save_file

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    progress = Progress(run_dir)
    checkpoint_path = run_dir / "checkpoint.pt"
    if resume:
        (run_dir / "cancel").unlink(missing_ok=True)
    progress.update(force=True, message="Reading the dataset")
    seed = int(run.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    folder, classes, items, summary = datasets.read(run["dataset"], run["val_fraction"], run["test_fraction"], seed)
    if resume and summary["fingerprint"] != run["dataset_summary"]["fingerprint"]:
        raise RuntimeError("The images in the folder have changed since this run started, so it can't resume. "
                           "Start a new run.")
    progress.update(force=True, message="Checking for duplicate images")
    checks = datasets.inspect(folder, items)
    leaked = set(checks.pop("_leaked_paths"))
    items = [i for i in items if i.path not in leaked]
    by_split = {s: [i for i in items if i.split == s] for s in ("train", "val", "test")}
    size, arch = int(run["image_size"]), run["architecture"]
    sample = random.Random(seed).sample(by_split["train"], min(200, len(by_split["train"])))
    grayscale = all(preprocess.is_grayscale(preprocess.open_image(folder / i.path)) for i in sample)
    channels = 1 if grayscale else 3
    pretrained = bool(run.get("pretrained")) and arch == "resnet18"
    in_channels = 3 if pretrained else channels

    estimated = len(items) * channels * size * size
    if estimated > CACHE_LIMIT_BYTES:
        raise RuntimeError(f"These images need {estimated / 1024**3:.1f} GB of memory at {size}px. "
                           "Choose a smaller image size.")
    tensors, labels = {}, {}
    for split, group in by_split.items():
        tensors[split] = _load_images(folder, group, size, channels, progress, split)
        labels[split] = torch.tensor([i.label for i in group], dtype=torch.long)

    if pretrained:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        sample_t = tensors["train"][: min(5000, len(tensors["train"]))].float() / 255
        mean = sample_t.mean(dim=(0, 2, 3)).tolist()
        std = [max(v, 1e-3) for v in sample_t.std(dim=(0, 2, 3)).tolist()]

    device = torch.device(run["device"])
    progress.update(force=True, phase="preparing", message="Building the model")
    model = architectures.build(arch, len(classes), in_channels, pretrained).to(device)
    counts = torch.bincount(labels["train"], minlength=len(classes)).float()
    class_weights = (counts.sum() / (len(classes) * counts.clamp(min=1))).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(run["learning_rate"]), weight_decay=1e-4)
    epochs, batch_size = int(run["epochs"]), int(run["batch_size"])
    steps_per_epoch = math.ceil(len(tensors["train"]) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(run["learning_rate"]),
                                                    total_steps=epochs * steps_per_epoch)
    patience = max(4, epochs // 4)
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history: list[dict] = []
    started = time.monotonic()
    first_epoch = 1
    earlier_seconds = 0.0
    if resume and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best, history = saved["best"], saved["history"]
        first_epoch = saved["epoch"] + 1
        earlier_seconds = saved.get("seconds", 0.0)
        torch.set_rng_state(saved["rng"])
        progress.update(force=True, history=history, best_epoch=best["epoch"],
                        message=f"Resuming after epoch {saved['epoch']}")

    for epoch in range(first_epoch, epochs + 1):
        model.train()
        epoch_start = time.monotonic()
        order = torch.randperm(len(tensors["train"]))
        running_loss, running_correct, seen = 0.0, 0, 0
        for step in range(steps_per_epoch):
            progress.check_cancel()
            idx = order[step * batch_size:(step + 1) * batch_size]
            xb = preprocess.normalise(tensors["train"][idx].to(device), mean, std, in_channels)
            xb = _augment(xb)
            yb = labels["train"][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(idx)
            running_correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
            done = (epoch - 1) * steps_per_epoch + step + 1
            done_here = done - (first_epoch - 1) * steps_per_epoch
            elapsed = time.monotonic() - started
            progress.update(phase="training", epoch=epoch, epochs=epochs, step=step + 1, steps=steps_per_epoch,
                            percent=round(100 * done / (epochs * steps_per_epoch), 1),
                            eta_seconds=round(elapsed / done_here * (epochs * steps_per_epoch - done)),
                            train_loss=running_loss / seen, message=f"Epoch {epoch} of {epochs}")
        _recalibrate_batch_norm(model, tensors["train"], device, batch_size, mean, std, in_channels)
        val_probs, val_loss = _predict(model, tensors["val"], labels["val"], device, batch_size * 2,
                                       mean, std, in_channels, progress, "validation")
        val_acc = float((val_probs.argmax(1) == labels["val"].numpy()).mean()) if len(val_probs) else None
        history.append({"epoch": epoch, "train_loss": round(running_loss / seen, 5),
                        "train_accuracy": round(running_correct / seen, 5), "val_loss": round(val_loss, 5),
                        "val_accuracy": None if val_acc is None else round(val_acc, 5),
                        "seconds": round(time.monotonic() - epoch_start, 1)})
        improved = val_loss < best["loss"] - 1e-4
        if improved:
            best = {"loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}}
        progress.update(force=True, history=history, best_epoch=best["epoch"])
        tmp = run_dir / "checkpoint.pt.tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best": best, "history": history,
                    "rng": torch.get_rng_state(), "seconds": earlier_seconds + time.monotonic() - started}, tmp)
        os.replace(tmp, checkpoint_path)
        if epoch - best["epoch"] >= patience:
            progress.update(force=True, message=f"Stopped early: no improvement for {patience} epochs")
            break

    progress.update(force=True, phase="evaluating", message="Evaluating the best model", percent=100, eta_seconds=0)
    model.load_state_dict(best["state"])
    val_probs, _ = _predict(model, tensors["val"], labels["val"], device, batch_size * 2, mean, std, in_channels,
                            progress, "validation")
    threshold = metrics.choose_threshold(val_probs, labels["val"].numpy(), config.MODEL_TARGET_ACCURACY,
                                         config.MODEL_MIN_CONFIDENCE)
    validation = metrics.evaluate(val_probs, labels["val"].numpy(), classes, threshold["threshold"])

    progress.update(force=True, message="Learning what the training images look like")
    train_subset = torch.randperm(len(tensors["train"]))[:20000]
    stats, novelty_cutoffs = novelty.fit_layers(
        _features(model, arch, tensors["train"][train_subset], device, batch_size * 2, mean, std, in_channels, progress),
        labels["train"][train_subset].numpy(), len(classes),
        _features(model, arch, tensors["val"], device, batch_size * 2, mean, std, in_channels, progress))

    test = None
    if len(tensors["test"]):
        test_probs, _ = _predict(model, tensors["test"], labels["test"], device, batch_size * 2, mean, std,
                                 in_channels, progress, "test")
        test = metrics.evaluate(test_probs, labels["test"].numpy(), classes, threshold["threshold"])
        test_flags, _ = novelty.flagged(_features(model, arch, tensors["test"], device, batch_size * 2, mean, std,
                                                  in_channels, progress), stats, novelty_cutoffs)
        test["novelty_flagged"] = float(test_flags.mean())
        test["target_met"] = (test["abstention"]["answered_accuracy"] or 0) >= config.MODEL_TARGET_ACCURACY

    progress.update(force=True, phase="saving", message="Saving to the model library")
    model_id = run["id"]
    target = config.MODEL_LIBRARY_DIR / model_id
    target.mkdir(parents=True, exist_ok=False)
    save_file({k: v.contiguous() for k, v in best["state"].items()}, str(target / "weights.safetensors"))
    novelty.save(target / "novelty.safetensors", stats)
    card = {
        "id": model_id,
        "name": run["name"],
        "task": "image-classification",
        "architecture": arch,
        "pretrained": pretrained,
        "classes": classes,
        "input": {"image_size": size, "channels": channels, "model_channels": in_channels,
                  "mean": mean, "std": std},
        "threshold": threshold,
        "novelty": {"method": "mahalanobis-per-block", "cutoffs": novelty_cutoffs, "flag_rate": novelty.FLAG_RATE},
        "validation": validation,
        "test": test,
        "history": history,
        "best_epoch": best["epoch"],
        "dataset": {**{k: summary[k] for k in ("name", "path", "layout", "classes", "total", "splits", "fingerprint")},
                    "checks": checks},
        "training": {k: run[k] for k in ("epochs", "batch_size", "learning_rate", "image_size", "device", "seed")},
        "hardware": {"device": run["device"], "device_name": run.get("device_name"), "platform": platform.platform(),
                     "torch": torch.__version__},
        "created_at": _now(),
        "created_by": run.get("created_by"),
        "training_seconds": round(earlier_seconds + time.monotonic() - started, 1),
        "notes": run.get("notes", ""),
        "intended_use": "Research and evaluation only. Not validated for clinical use.",
    }
    (target / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    progress.update(force=True, state="completed", phase="done", message="Saved to the model library",
                    model_id=model_id, finished_at=_now(),
                    summary={"target_met": (test or {}).get("target_met"),
                             "test_accuracy": (test or validation)["accuracy"],
                             "balanced_accuracy": (test or validation)["balanced_accuracy"],
                             "coverage": (test or validation)["abstention"]["coverage"],
                             "answered_accuracy": (test or validation)["abstention"]["answered_accuracy"]})


def main(argv: list[str]) -> int:
    run_dir = Path(argv[0]).resolve()
    progress = Progress(run_dir)
    try:
        train(run_dir, resume="--resume" in argv[1:])
        return 0
    except Cancelled:
        progress.state.update(json.loads((run_dir / "progress.json").read_text()) if (run_dir / "progress.json").exists() else {})
        progress.update(force=True, state="cancelled", message="Cancelled", finished_at=_now())
        return 0
    except Exception as exc:  # noqa: BLE001 - report every failure to the dashboard
        traceback.print_exc()
        previous = json.loads((run_dir / "progress.json").read_text()) if (run_dir / "progress.json").exists() else {}
        progress.state.update(previous)
        message = str(exc) if isinstance(exc, (datasets.DatasetError, RuntimeError, ValueError)) else \
            f"Training failed: {type(exc).__name__}. See worker.log in the run folder."
        progress.update(force=True, state="failed", message=message, finished_at=_now())
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
