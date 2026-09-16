"""Local AI: open-source models running on this machine through Ollama.

This module answers three questions and performs three actions:

- What hardware is here?  Memory, Apple silicon, NVIDIA GPUs.
- Which catalogue models fit it, and which one do we recommend?
- What does the local Ollama server have installed and loaded?

- Download a model, with streamed progress.
- Make a downloaded model the active one, or switch back to none.
- Generate a JSON response with the active model.

A local model only ever drafts. The deterministic guards decide what is
shown, exactly as with a cloud model, and any failure here falls back to the
extractive path. Nothing in this module raises into the pipeline."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import httpx

from . import config

# --------------------------------------------------------------------------
# Catalogue
#
# Download sizes come from the Ollama registry manifests (sum of layer sizes)
# and licences from the licence file each model ships with. Models whose
# licence forbids commercial use are left out, since GroundCheck targets
# clinical organisations. "quality" orders recommendations: higher is better
# for grounded, JSON-structured answers.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueModel:
    name: str            # Ollama model tag
    label: str           # human name
    parameters: str      # "3B"
    download_gb: float   # registry download size
    licence: str
    quality: int


CATALOGUE: tuple[CatalogueModel, ...] = (
    CatalogueModel("gemma3:1b", "Gemma 3 1B", "1B", 0.82, "Gemma Terms of Use", 1),
    CatalogueModel("llama3.2:1b", "Llama 3.2 1B", "1B", 1.32, "Llama 3.2 Community License", 2),
    CatalogueModel("llama3.2:3b", "Llama 3.2 3B", "3B", 2.02, "Llama 3.2 Community License", 3),
    CatalogueModel("phi4-mini", "Phi-4 mini", "3.8B", 2.49, "MIT", 4),
    CatalogueModel("gemma3:4b", "Gemma 3 4B", "4B", 3.34, "Gemma Terms of Use", 5),
    CatalogueModel("mistral:7b", "Mistral 7B", "7B", 4.37, "Apache 2.0", 6),
    CatalogueModel("llama3.1:8b", "Llama 3.1 8B", "8B", 4.92, "Llama 3.1 Community License", 7),
    CatalogueModel("qwen2.5:7b", "Qwen 2.5 7B", "7B", 4.68, "Apache 2.0", 8),
    CatalogueModel("gemma3:12b", "Gemma 3 12B", "12B", 8.15, "Gemma Terms of Use", 9),
    CatalogueModel("qwen2.5:14b", "Qwen 2.5 14B", "14B", 8.99, "Apache 2.0", 10),
)

_BY_NAME = {m.name: m for m in CATALOGUE}

# Memory a model needs while answering: its weights, plus runtime overhead and
# a context window sized for a question and a few retrieved passages.
_OVERHEAD_FACTOR = 1.2
_CONTEXT_GB = 1.0


def memory_needed_gb(model: CatalogueModel) -> float:
    return round(model.download_gb * _OVERHEAD_FACTOR + _CONTEXT_GB, 1)


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_gb: float


@dataclass(frozen=True)
class Hardware:
    system: str               # "macOS", "Linux", "Windows"
    machine: str              # "arm64", "x86_64"
    cpu: str
    cpu_cores: int
    memory_gb: float
    apple_silicon: bool
    gpus: tuple[Gpu, ...]
    accelerator: str          # "apple-silicon", "nvidia", "cpu"
    model_memory_gb: float    # memory available to hold a model


def _total_memory_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _cpu_name(system: str) -> str:
    try:
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=2)
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def parse_nvidia_smi(output: str) -> tuple[Gpu, ...]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`."""
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append(Gpu(name=parts[0], memory_gb=round(float(parts[1]) / 1024, 1)))
            except ValueError:
                continue
    return tuple(gpus)


def _nvidia_gpus() -> tuple[Gpu, ...]:
    if not shutil.which("nvidia-smi"):
        return ()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return parse_nvidia_smi(out.stdout) if out.returncode == 0 else ()


def model_memory_budget(memory_gb: float, apple_silicon: bool, gpus: tuple[Gpu, ...]) -> tuple[str, float]:
    """(accelerator, GB available for a model).

    Apple silicon shares memory between CPU and GPU, and macOS lets the GPU use
    roughly two thirds of it. On NVIDIA the model must fit the largest GPU's
    own memory. Without a GPU, a model runs on the CPU in half of system memory,
    leaving the rest for the operating system and GroundCheck."""
    if apple_silicon:
        return "apple-silicon", round(memory_gb * 0.65, 1)
    if gpus:
        return "nvidia", max(g.memory_gb for g in gpus)
    return "cpu", round(memory_gb * 0.5, 1)


def detect_hardware() -> Hardware:
    system = platform.system()
    machine = platform.machine()
    memory = _total_memory_gb()
    apple = system == "Darwin" and machine == "arm64"
    gpus = () if apple else _nvidia_gpus()
    accelerator, budget = model_memory_budget(memory, apple, gpus)
    return Hardware(
        system={"Darwin": "macOS"}.get(system, system),
        machine=machine,
        cpu=_cpu_name(system),
        cpu_cores=os.cpu_count() or 1,
        memory_gb=memory,
        apple_silicon=apple,
        gpus=gpus,
        accelerator=accelerator,
        model_memory_gb=budget,
    )


# --------------------------------------------------------------------------
# Fit and recommendation
# --------------------------------------------------------------------------

def fit(model: CatalogueModel, budget_gb: float) -> str:
    """"good" leaves headroom, "tight" fits with little room, "too-large" doesn't."""
    need = memory_needed_gb(model)
    if need <= budget_gb * 0.8:
        return "good"
    if need <= budget_gb:
        return "tight"
    return "too-large"


def recommend(budget_gb: float) -> CatalogueModel | None:
    """The highest-quality model that fits with headroom."""
    candidates = [m for m in CATALOGUE if fit(m, budget_gb) == "good"]
    return max(candidates, key=lambda m: m.quality, default=None)


# --------------------------------------------------------------------------
# Selected model, persisted so a restart keeps the choice
# --------------------------------------------------------------------------

_state_lock = threading.Lock()


def _read_state() -> dict:
    try:
        return json.loads(config.LOCAL_AI_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(state: dict) -> None:
    try:
        config.LOCAL_AI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.LOCAL_AI_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        # Read-only deployments keep the choice in memory for this process.
        pass


_selected: str | None = None
_selected_loaded = False


def selected_model() -> str | None:
    """The model chosen in the dashboard, or LOCAL_MODEL from the environment."""
    global _selected, _selected_loaded
    with _state_lock:
        if not _selected_loaded:
            _selected = _read_state().get("model") or config.LOCAL_MODEL or None
            _selected_loaded = True
        return _selected


def select_model(name: str | None) -> None:
    global _selected, _selected_loaded
    with _state_lock:
        _selected, _selected_loaded = name, True
        _write_state({"model": name})


# --------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(base_url=config.OLLAMA_HOST, timeout=timeout)


def ollama_version() -> str | None:
    """The Ollama server's version, or None when it isn't reachable."""
    try:
        with _client(2.0) as c:
            r = c.get("/api/version")
            r.raise_for_status()
            return r.json().get("version")
    except (httpx.HTTPError, ValueError):
        return None


def installed_models() -> list[dict]:
    try:
        with _client(5.0) as c:
            r = c.get("/api/tags")
            r.raise_for_status()
            return [
                {"name": m.get("name", ""), "size_gb": round(m.get("size", 0) / 1e9, 2)}
                for m in r.json().get("models", [])
            ]
    except (httpx.HTTPError, ValueError):
        return []


def loaded_models() -> list[str]:
    try:
        with _client(5.0) as c:
            r = c.get("/api/ps")
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return []


def _same_model(a: str, b: str) -> bool:
    """Ollama reports "phi4-mini:latest" for "phi4-mini"."""
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"
    return norm(a) == norm(b)


def is_installed(name: str, installed: list[dict] | None = None) -> bool:
    return any(_same_model(m["name"], name) for m in (installed if installed is not None else installed_models()))


def pull(name: str) -> Iterator[dict]:
    """Download a model, yielding progress events:
    {"status": str, "completed": int, "total": int} and finally {"status": "success"}.
    Raises OllamaError if the server is unreachable or reports an error."""
    if name not in _BY_NAME:
        raise OllamaError(f"{name} is not in the GroundCheck model catalogue")
    try:
        with _client(None) as c, c.stream("POST", "/api/pull", json={"model": name, "stream": True}) as r:
            if r.status_code != 200:
                raise OllamaError(f"Ollama returned HTTP {r.status_code} for {name}")
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise OllamaError(event["error"])
                yield {
                    "status": event.get("status", ""),
                    "completed": event.get("completed", 0),
                    "total": event.get("total", 0),
                }
    except httpx.HTTPError as exc:
        raise OllamaError(f"Can't reach Ollama at {config.OLLAMA_HOST}: {exc}") from exc


def load(name: str, keep_alive: str = "30m") -> None:
    """Load a model into memory so the first question doesn't wait for it."""
    try:
        with _client(config.LOCAL_AI_TIMEOUT_SECONDS * 4) as c:
            r = c.post("/api/generate", json={"model": name, "prompt": "", "keep_alive": keep_alive})
            r.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Couldn't load {name}: {exc}") from exc


def unload(name: str) -> None:
    try:
        with _client(10.0) as c:
            c.post("/api/generate", json={"model": name, "prompt": "", "keep_alive": 0})
    except httpx.HTTPError:
        pass


# Output length cap. A grounded answer is a handful of short claims; the cap
# stops a small model that starts repeating itself from running for minutes.
_MAX_OUTPUT_TOKENS = 600


def chat_json(system: str, user: str, temperature: float = 0.0,
              model: str | None = None, schema: dict | None = None) -> str | None:
    """Ask the active local model for a JSON reply. With a JSON schema, Ollama
    constrains the output to it, which small models need to stay on format.
    Returns the raw content, or None if no model is active, Ollama is
    unreachable, or the call fails or times out."""
    name = model or active_model()
    if not name:
        return None
    try:
        with _client(config.LOCAL_AI_TIMEOUT_SECONDS) as c:
            r = c.post("/api/chat", json={
                "model": name,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "format": schema or "json",
                "stream": False,
                "options": {"temperature": temperature, "num_predict": _MAX_OUTPUT_TOKENS},
                "keep_alive": "30m",
            })
            r.raise_for_status()
            return r.json().get("message", {}).get("content")
    except (httpx.HTTPError, ValueError):
        return None


# --------------------------------------------------------------------------
# Availability, cached briefly so every question doesn't probe the server
# --------------------------------------------------------------------------

_availability: tuple[float, str | None] = (0.0, None)
_AVAILABILITY_TTL = 10.0


def active_model() -> str | None:
    """The selected model if Ollama is reachable and has it installed."""
    global _availability
    name = selected_model()
    if not name:
        return None
    checked_at, cached = _availability
    if time.monotonic() - checked_at < _AVAILABILITY_TTL and cached == name:
        return name
    if ollama_version() and is_installed(name):
        _availability = (time.monotonic(), name)
        return name
    _availability = (time.monotonic(), None)
    return None


def forget_availability() -> None:
    global _availability
    _availability = (0.0, None)


def status() -> dict:
    """Everything the dashboard's local AI panel shows."""
    hw = detect_hardware()
    version = ollama_version()
    installed = installed_models() if version else []
    loaded = loaded_models() if version else []
    recommended = recommend(hw.model_memory_gb)
    selected = selected_model()
    return {
        "hardware": {**asdict(hw), "gpus": [asdict(g) for g in hw.gpus]},
        "ollama": {"running": version is not None, "version": version, "host": config.OLLAMA_HOST},
        "selected": selected,
        "active": selected if (selected and version and is_installed(selected, installed)) else None,
        "models": [
            {
                **asdict(m),
                "memory_needed_gb": memory_needed_gb(m),
                "fit": fit(m, hw.model_memory_gb),
                "recommended": recommended is not None and m.name == recommended.name,
                "installed": is_installed(m.name, installed),
                "loaded": any(_same_model(l, m.name) for l in loaded),
            }
            for m in CATALOGUE
        ],
    }
