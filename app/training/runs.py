"""Starting, following and cancelling training runs.

Each run has a folder in TRAINING_RUNS_DIR holding its settings (run.json),
live progress written by the worker (progress.json) and the worker's log.
One run trains at a time, so runs don't compete for the same GPU."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from . import architectures, datasets, hardware
from .worker import CACHE_LIMIT_BYTES

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")
_processes: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()
ACTIVE = ("queued", "running")


class TrainingError(ValueError):
    """A run can't be started or found. The message is safe to show."""


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")[:48] or "model"


def _run_dir(run_id: str) -> Path:
    if not _ID.match(run_id or ""):
        raise TrainingError("No such training run.")
    folder = config.TRAINING_RUNS_DIR / run_id
    if not (folder / "run.json").is_file():
        raise TrainingError("No such training run.")
    return folder


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _alive(run_id: str, pid: int | None) -> bool:
    proc = _processes.get(run_id)
    if proc is not None:
        return proc.poll() is None
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def get(run_id: str) -> dict:
    folder = _run_dir(run_id)
    run = _read_json(folder / "run.json")
    progress = _read_json(folder / "progress.json") or {"state": "queued", "message": "Waiting to start"}
    if progress.get("state") in ACTIVE and not _alive(run_id, progress.get("pid") or run.get("pid")):
        # Give a just-started worker a moment to write its first progress.
        started = datetime.fromisoformat(run.get("resumed_at") or run["created_at"])
        if (datetime.now(timezone.utc) - started).total_seconds() > 20 or progress.get("pid"):
            progress.update(state="failed", message="Training stopped unexpectedly. See worker.log in the run folder.")
    result = {**run, "progress": progress}
    result["can_resume"] = can_resume(result)
    return result


def list_runs(limit: int = 50) -> list[dict]:
    if not config.TRAINING_RUNS_DIR.is_dir():
        return []
    folders = sorted((p for p in config.TRAINING_RUNS_DIR.iterdir() if (p / "run.json").is_file()),
                     key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for folder in folders:
        try:
            out.append(get(folder.name))
        except TrainingError:
            continue
    return out


def active() -> dict | None:
    return next((r for r in list_runs(10) if r["progress"].get("state") in ACTIVE), None)


def _check(name: str, value, low, high, kind=float):
    try:
        value = kind(value)
    except (TypeError, ValueError):
        raise TrainingError(f"{name} must be a number.") from None
    if not low <= value <= high:
        raise TrainingError(f"{name} must be between {low} and {high}.")
    return value


def start(settings: dict, user_name: str | None = None) -> dict:
    name = (settings.get("name") or "").strip()
    if not name:
        raise TrainingError("Give the model a name.")
    arch = settings.get("architecture") or "small-cnn"
    option = next((a for a in architectures.options()["architectures"] if a["id"] == arch), None)
    if option is None:
        raise TrainingError("Choose a model architecture.")
    if not option["available"]:
        raise TrainingError(option["unavailable_reason"])
    image_size = _check("Image size", settings.get("image_size", option["image_size"]), 16, 512, int)
    if image_size not in architectures.IMAGE_SIZES:
        raise TrainingError(f"Image size must be one of {', '.join(map(str, architectures.IMAGE_SIZES))}.")
    epochs = _check("Epochs", settings.get("epochs", 20), 1, 300, int)
    batch_size = _check("Batch size", settings.get("batch_size", option["batch_size"]), 1, 1024, int)
    learning_rate = _check("Learning rate", settings.get("learning_rate", option["learning_rate"]), 1e-6, 1.0)
    val_fraction = _check("Validation share", settings.get("val_fraction", 0.15), 0.05, 0.4)
    test_fraction = _check("Test share", settings.get("test_fraction", 0.15), 0.0, 0.4)
    device_id = settings.get("device") or ""
    devices = {d["id"]: d for d in hardware.detect()["devices"]}
    if device_id not in devices:
        raise TrainingError("Choose a device that this machine has.")

    _, classes, items, summary = datasets.read(settings.get("dataset", ""), val_fraction, test_fraction)
    needed = len(items) * 3 * image_size * image_size
    if needed > CACHE_LIMIT_BYTES:
        raise TrainingError(f"{len(items):,} images at {image_size}px need up to {needed / 1024**3:.1f} GB of "
                            "memory. Choose a smaller image size.")

    with _lock:
        if active() is not None:
            raise TrainingError("A model is already training. Wait for it to finish, or cancel it.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{_slug(name)}-{stamp}"
        folder = config.TRAINING_RUNS_DIR / run_id
        folder.mkdir(parents=True, exist_ok=False)
        run = {
            "id": run_id, "name": name[:80], "dataset": summary["path"],
            "dataset_summary": {k: v for k, v in summary.items() if k != "samples"},
            "architecture": arch, "pretrained": bool(settings.get("pretrained")) and option["pretrained_option"],
            "device": device_id, "device_name": devices[device_id]["name"],
            "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "image_size": image_size,
            "val_fraction": val_fraction, "test_fraction": test_fraction, "seed": 7,
            "notes": (settings.get("notes") or "").strip()[:2000],
            "created_at": datetime.now(timezone.utc).isoformat(), "created_by": user_name,
        }
        (folder / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        run["pid"] = _spawn(run_id, folder)
        (folder / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return get(run_id)


def _spawn(run_id: str, folder: Path, resume: bool = False) -> int:
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    log = open(folder / "worker.log", "ab")  # noqa: SIM115 - handed to the child process
    args = [sys.executable, "-m", "app.training.worker", str(folder)] + (["--resume"] if resume else [])
    proc = subprocess.Popen(args, cwd=str(config.ROOT_DIR), stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    log.close()
    _processes[run_id] = proc
    return proc.pid


def can_resume(run: dict) -> bool:
    return (run["progress"].get("state") in ("cancelled", "failed")
            and (config.TRAINING_RUNS_DIR / run["id"] / "checkpoint.pt").is_file())


def resume(run_id: str) -> dict:
    folder = _run_dir(run_id)
    with _lock:
        run = get(run_id)
        if not can_resume(run):
            raise TrainingError("Only a stopped run with at least one finished epoch can resume.")
        if active() is not None:
            raise TrainingError("A model is already training. Wait for it to finish, or cancel it.")
        (folder / "cancel").unlink(missing_ok=True)
        previous = run["progress"]
        (folder / "progress.json").write_text(json.dumps({
            "state": "queued", "message": "Resuming", "history": previous.get("history", []),
            "best_epoch": previous.get("best_epoch"), "epoch": previous.get("epoch"), "epochs": run["epochs"],
            "percent": previous.get("percent")}), encoding="utf-8")
        stored = {k: v for k, v in run.items() if k not in ("progress", "can_resume")}
        stored["resumed_at"] = datetime.now(timezone.utc).isoformat()
        stored["pid"] = _spawn(run_id, folder, resume=True)
        (folder / "run.json").write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return get(run_id)


def cancel(run_id: str) -> dict:
    folder = _run_dir(run_id)
    run = get(run_id)
    if run["progress"].get("state") not in ACTIVE:
        raise TrainingError("This run has already finished.")
    (folder / "cancel").touch()
    return run
