"""Training studio: reading image folders safely, evaluation and the
abstention threshold, the unfamiliar-image check, validating run settings,
a real (tiny) training run on the CPU, using the saved model, and the API's
permission rules."""

from __future__ import annotations

import conftest

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ["GROQ_API_KEY"] = ""
os.environ["FORCE_EXTRACTIVE"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Real training runs on whatever hardware the tests happen to have. Four
# minutes is comfortable on a laptop; a loaded CI runner can need longer, and a
# slow machine failing here is a flake rather than a defect.
TRAINING_TIMEOUT = float(os.environ.get("TRAINING_TEST_TIMEOUT", "600"))

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.training import datasets, library, metrics, novelty, runs  # noqa: E402


def _png(value: int, size: int = 32, noise: int = 20, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    array = np.clip(value + rng.integers(-noise, noise, (size, size)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, "PNG")
    return buffer.getvalue()


def _make_dataset(root: Path, per_class: int = 30, split_folders: bool = False) -> Path:
    """Two easily separated classes: dark and bright images."""
    folder = root / "scans"
    for label, value in (("dark", 40), ("bright", 210)):
        for i in range(per_class):
            split = ("train" if i < per_class * 0.7 else "val" if i < per_class * 0.85 else "test")
            target = folder / split / label if split_folders else folder / label
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{label}-{i}.png").write_bytes(_png(value, seed=i))
    return folder


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    data, library_dir, runs_dir = tmp_path / "data", tmp_path / "library", tmp_path / "runs"
    data.mkdir()
    monkeypatch.setattr(config, "TRAINING_DATA_DIRS", [data.resolve()])
    monkeypatch.setattr(config, "MODEL_LIBRARY_DIR", library_dir)
    monkeypatch.setattr(config, "TRAINING_RUNS_DIR", runs_dir)
    # The worker is a separate process, so it reads these from the environment.
    monkeypatch.setenv("TRAINING_DATA_DIRS", str(data))
    monkeypatch.setenv("MODEL_LIBRARY_DIR", str(library_dir))
    monkeypatch.setenv("TRAINING_RUNS_DIR", str(runs_dir))
    return data


# --- Reading folders ---------------------------------------------------------------

def test_class_folders_are_split_by_class(studio):
    folder = _make_dataset(studio, per_class=40)
    _, classes, items, summary = datasets.read(str(folder))
    assert classes == ["bright", "dark"] and summary["layout"] == "class-folders"
    assert summary["total"] == 80 and sum(summary["splits"].values()) == 80
    for c in summary["classes"]:
        assert (c["train"], c["val"], c["test"]) == (28, 6, 6)
    assert datasets.read(str(folder))[3]["fingerprint"] == summary["fingerprint"]  # deterministic


def test_split_folders_are_used_as_given(studio):
    folder = _make_dataset(studio, per_class=20, split_folders=True)
    _, classes, _, summary = datasets.read(str(folder))
    assert summary["layout"] == "split-folders" and summary["splits"] == {"train": 28, "val": 6, "test": 6}


def test_unusable_folders_are_explained(studio, tmp_path):
    single = studio / "one-class" / "only"
    single.mkdir(parents=True)
    for i in range(10):
        (single / f"{i}.png").write_bytes(_png(100, seed=i))
    with pytest.raises(datasets.DatasetError, match="at least two subfolders"):
        datasets.read(str(studio / "one-class"))

    few = studio / "few"
    for label in ("a", "b"):
        (few / label).mkdir(parents=True)
        for i in range(3):
            (few / label / f"{i}.png").write_bytes(_png(100, seed=i))
    with pytest.raises(datasets.DatasetError, match="at least 5 training images"):
        datasets.read(str(few))

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(datasets.DatasetError, match="TRAINING_DATA_DIRS"):
        datasets.read(str(outside))
    with pytest.raises(datasets.DatasetError, match="TRAINING_DATA_DIRS"):
        datasets.resolve(str(studio / ".." / "elsewhere"))
    (studio / "link").symlink_to(outside)
    with pytest.raises(datasets.DatasetError, match="TRAINING_DATA_DIRS"):
        datasets.resolve(str(studio / "link"))
    with pytest.raises(datasets.DatasetError, match="doesn't exist"):
        datasets.resolve(str(studio / "missing"))


def test_preview_files_stay_inside_the_dataset(studio, tmp_path):
    folder = _make_dataset(studio, per_class=10)
    assert datasets.resolve_file(folder, "dark/dark-0.png").is_file()
    (tmp_path / "secret.png").write_bytes(_png(1))
    for bad in ("../../secret.png", "/etc/passwd", "dark/../../secret.png"):
        with pytest.raises(datasets.DatasetError):
            datasets.resolve_file(folder, bad)


def test_browsing_lists_folders_and_spots_datasets(studio):
    _make_dataset(studio, per_class=10)
    (studio / "notes").mkdir()
    listing = datasets.browse(str(studio))
    found = {f["name"]: f["looks_like_dataset"] for f in listing["folders"]}
    assert found == {"notes": False, "scans": True}
    assert datasets.browse(None)["folders"][0]["path"] == str(studio.resolve())


# --- Metrics -----------------------------------------------------------------------

def test_auc_and_per_class_figures():
    assert metrics.roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])) == 0.75
    assert metrics.roc_auc(np.array([0.5, 0.5]), np.array([True, True])) is None
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    result = metrics.evaluate(probs, np.array([0, 0, 1, 1]), ["a", "b"], threshold=0.65)
    assert result["accuracy"] == 0.75 and result["confusion"] == [[2, 0], [1, 1]]
    b = result["per_class"][1]
    assert (b["sensitivity"], b["specificity"], b["precision"]) == (0.5, 1.0, 1.0)
    assert result["abstention"] == {"threshold": 0.65, "coverage": 0.75, "answered_accuracy": 1.0, "abstained": 1}


def test_threshold_needs_the_lower_bound_to_meet_the_target():
    rng = np.random.default_rng(1)
    n = 2000
    confidence = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(0, 1, n) < confidence  # more confident, more often right
    labels = np.where(correct, 0, 1)
    probs = np.stack([confidence, 1 - confidence], axis=1)
    chosen = metrics.choose_threshold(probs, labels, 0.95)
    answered = confidence >= chosen["threshold"]
    assert chosen["met"] and 0 < chosen["coverage"] < 1
    assert metrics.wilson_lower(correct[answered].sum(), answered.sum()) >= 0.95
    assert chosen["accuracy"] > chosen["accuracy_lower_bound"] >= 0.95

    floored = metrics.choose_threshold(probs, labels, 0.5, floor=0.8)
    assert floored["threshold"] >= 0.8 and floored["coverage"] <= (confidence >= 0.8).mean()

    hopeless = metrics.choose_threshold(np.full((50, 2), 0.5), np.zeros(50, dtype=int), 0.95)
    assert hopeless == {**hopeless, "met": False, "threshold": 1.0, "coverage": 0.0}


# --- Starting runs ---------------------------------------------------------------

@pytest.mark.parametrize("change, message", [
    ({"name": " "}, "name"),
    ({"architecture": "gpt"}, "architecture"),
    ({"device": "cuda:7"}, "device"),
    ({"epochs": 0}, "Epochs"),
    ({"image_size": 50}, "Image size"),
    ({"learning_rate": "fast"}, "Learning rate"),
    ({"dataset": "/nowhere"}, "doesn't exist"),
])
def test_invalid_settings_are_refused(studio, change, message):
    folder = _make_dataset(studio, per_class=10)
    settings = {"name": "Test", "dataset": str(folder), "device": "cpu", "epochs": 1, "image_size": 28, **change}
    with pytest.raises((runs.TrainingError, datasets.DatasetError), match=message):
        runs.start(settings)
    assert runs.list_runs() == []


def test_a_real_training_run_saves_a_usable_model(studio, monkeypatch):
    # Twelve validation images can't show 95% accuracy with confidence (the
    # lower bound for 12 of 12 is 76%), so ask less of this tiny dataset.
    monkeypatch.setenv("MODEL_TARGET_ACCURACY", "0.7")
    folder = _make_dataset(studio, per_class=40)
    run = runs.start({"name": "Dark or bright", "dataset": str(folder), "device": "cpu", "epochs": 5,
                      "image_size": 28, "batch_size": 16})
    with pytest.raises(runs.TrainingError, match="already training"):
        runs.start({"name": "Second", "dataset": str(folder), "device": "cpu", "epochs": 1, "image_size": 28})
    deadline = time.time() + TRAINING_TIMEOUT
    while runs.get(run["id"])["progress"].get("state") in runs.ACTIVE and time.time() < deadline:
        time.sleep(1)
    progress = runs.get(run["id"])["progress"]
    log = (config.TRAINING_RUNS_DIR / run["id"] / "worker.log").read_text()
    assert progress["state"] == "completed", (progress, log)
    assert len(progress["history"]) >= 1

    [model] = library.list_models()
    assert model["id"] == run["id"] and model["classes"] == ["bright", "dark"] and model["novelty_check"]
    card = library.get_model(model["id"])
    assert card["input"]["channels"] == 1 and card["test"]["images"] == 12
    assert card["test"]["accuracy"] == 1.0 and card["threshold"]["met"]

    bright = library.predict(model["id"], _png(210, seed=99))
    assert bright["prediction"] == "bright" and not bright["abstained"]
    noise = io.BytesIO()
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(noise, "PNG")
    assert library.predict(model["id"], noise.getvalue())["abstained"]
    with pytest.raises(library.LibraryError, match="couldn't be read"):
        library.predict(model["id"], b"not an image")
    assert library.export_zip(model["id"])[:2] == b"PK"

    library.delete_model(model["id"])
    assert library.list_models() == []
    with pytest.raises(library.LibraryError):
        library.get_model(model["id"])


def test_duplicates_and_identifiers_are_found(studio):
    folder = _make_dataset(studio, per_class=20, split_folders=True)
    # A training image copied into test, and a file name with a patient's details.
    copy = folder / "test" / "dark" / "copy-of-train.png"
    copy.write_bytes((folder / "train" / "dark" / "dark-0.png").read_bytes())
    named = folder / "train" / "bright" / "Mr John Smith 12-03-1948.png"
    named.write_bytes(_png(210, seed=500))
    _, _, items, _ = datasets.read(str(folder))
    checks = datasets.inspect(folder, items)
    assert checks["duplicates"]["leaked"] == 1 and checks.pop("_leaked_paths") == ["test/dark/copy-of-train.png"]
    assert checks["identifiers"]["file_names"] == 1
    assert set(checks["identifiers"]["file_name_kinds"]) == {"name", "date"}
    assert len(checks["warnings"]) == 2


def test_a_stopped_run_resumes_where_it_left_off(studio, monkeypatch):
    monkeypatch.setenv("MODEL_TARGET_ACCURACY", "0.7")
    folder = _make_dataset(studio, per_class=60)
    run = runs.start({"name": "Resumable", "dataset": str(folder), "device": "cpu", "epochs": 40,
                      "image_size": 64, "batch_size": 4, "learning_rate": 0.0005})
    deadline = time.time() + TRAINING_TIMEOUT
    while not runs.get(run["id"])["progress"].get("history") and time.time() < deadline:
        time.sleep(0.1)
    runs.cancel(run["id"])
    while runs.get(run["id"])["progress"].get("state") in runs.ACTIVE and time.time() < deadline:
        time.sleep(0.2)
    stopped = runs.get(run["id"])
    assert stopped["progress"]["state"] == "cancelled" and stopped["can_resume"]
    done_before = len(stopped["progress"]["history"])

    runs.resume(run["id"])
    while runs.get(run["id"])["progress"].get("state") in runs.ACTIVE and time.time() < deadline:
        time.sleep(0.5)
    finished = runs.get(run["id"])
    assert finished["progress"]["state"] == "completed", (config.TRAINING_RUNS_DIR / run["id"] / "worker.log").read_text()
    epochs = [h["epoch"] for h in finished["progress"]["history"]]
    assert epochs == list(range(1, len(epochs) + 1)) and len(epochs) > done_before
    assert not (config.TRAINING_RUNS_DIR / run["id"] / "checkpoint.pt").exists()
    with pytest.raises(runs.TrainingError, match="stopped run"):
        runs.resume(run["id"])


def test_cancelling_a_finished_run_is_refused(studio):
    with pytest.raises(runs.TrainingError, match="No such"):
        runs.cancel("../../etc")


# --- API ---------------------------------------------------------------------------

def test_api_rules(studio, monkeypatch):
    _make_dataset(studio, per_class=10)
    from app.main import app

    with TestClient(app, client=conftest.REMOTE_PEER) as remote_client:
        for method, path in (("get", "/api/training/setup"), ("get", "/api/training/runs"), ("get", "/api/models"),
                             ("get", f"/api/training/browse?path={studio}")):
            assert getattr(remote_client, method)(path).status_code == 403
            assert getattr(remote_client, method)(
                path, headers=conftest.remote_headers()).status_code == 403
    with TestClient(app) as client:
        setup = client.get("/api/training/setup").json()
        assert setup["hardware"]["devices"][-1]["id"] == "cpu" and setup["active_run"] is None
        summary = client.get(f"/api/training/dataset?path={studio / 'scans'}").json()
        assert summary["total"] == 20
        image = client.get(f"/api/training/image?dataset={studio / 'scans'}&file=dark/dark-0.png")
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert client.get(f"/api/training/image?dataset={studio / 'scans'}&file=../../x.png").status_code == 400
        assert client.get("/api/models/nope").status_code == 400
        bad = client.post("/api/training/runs", json={"name": "x", "dataset": str(studio / "scans"), "device": "tpu"})
        assert bad.status_code == 400 and "device" in bad.json()["detail"]
