"""What this machine can actually give us, and the sizes that follow from it.

Batch sizes and caches are chosen from the cores, memory and accelerator the
process really has, not from constants that suit one laptop. In a container,
the cgroup limits are what count: a 2-core, 4 GB pod on a 64-core host must
size itself as 2 cores and 4 GB, or it will be throttled or killed.

Everything here is cheap to call and cached; every value can be overridden by
an environment variable, because an operator with a busy shared host is the
better judge."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from . import config

GB = 1024 ** 3


def _cgroup_value(paths: tuple[str, ...], index: int = 0) -> float | None:
    """A number from cgroup v2 or v1, or None when there's no limit."""
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        parts = text.split()
        if not parts or parts[index] in ("max", "-1"):
            return None
        try:
            value = float(parts[index])
        except ValueError:
            continue
        # cgroup v1 reports "no limit" as a huge number.
        return None if value >= 2 ** 62 else value
    return None


@lru_cache(maxsize=1)
def cpus() -> float:
    """Cores this process may use: the container's quota, the CPU affinity
    mask, or the machine's cores."""
    quota = _cgroup_value(("/sys/fs/cgroup/cpu.max",))
    period = _cgroup_value(("/sys/fs/cgroup/cpu.max",), index=1) or 100000
    if quota:
        return max(0.5, quota / period)
    v1_quota = _cgroup_value(("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",))
    v1_period = _cgroup_value(("/sys/fs/cgroup/cpu/cpu.cfs_period_us",)) or 100000
    if v1_quota:
        return max(0.5, v1_quota / v1_period)
    try:
        return float(len(os.sched_getaffinity(0)))     # Linux: respects taskset
    except AttributeError:
        return float(os.cpu_count() or 1)


@lru_cache(maxsize=1)
def memory_bytes() -> int:
    """Memory this process may use, in bytes."""
    limit = _cgroup_value(("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    if limit:
        return int(limit)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:                                              # macOS
        import subprocess

        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                  check=True).stdout.strip())
    except Exception:  # noqa: BLE001
        return 4 * GB


def available_bytes() -> int:
    """Memory free right now, when the system will say; otherwise a quarter of
    the limit, which is the pessimistic assumption."""
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        return int(fields["MemAvailable"].strip().split()[0]) * 1024
    except (OSError, KeyError, ValueError, IndexError):
        return memory_bytes() // 4


@lru_cache(maxsize=1)
def accelerator() -> str:
    """"cuda", "mps" or "cpu" for batch work, honouring BATCH_DEVICE."""
    from . import retrieval

    return retrieval.resolve_device(config.BATCH_DEVICE)


@lru_cache(maxsize=1)
def accelerator_memory_bytes() -> int:
    device = accelerator()
    if device.startswith("cuda"):
        try:
            import torch

            return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:  # noqa: BLE001
            return 4 * GB
    if device == "mps":
        return memory_bytes()          # Apple's GPU shares system memory
    return memory_bytes()


def _override(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value.isdigit() and int(value) > 0 else None


def embed_batch_size() -> int:
    """Texts embedded in one call. Bigger batches keep a GPU busy; on a CPU
    they only add memory pressure."""
    if (value := _override("EMBED_BATCH_SIZE")) is not None:
        return value
    device = accelerator()
    if device == "cpu":
        return 64 if cpus() <= 2 else 128
    budget = accelerator_memory_bytes()
    return max(128, min(1024, int(budget / GB) * 64))


def patch_batch_size(pixels: int, channels: int = 1) -> int:
    """Image patches classified in one call, from the memory a batch needs:
    the patches themselves, their resized copies, and the network's activations
    (about eight times the input, measured on the bundled models)."""
    if (value := _override("IMAGING_BATCH_SIZE")) is not None:
        return value
    per_patch = max(1, pixels * pixels * channels * 4 * 9)
    budget = accelerator_memory_bytes() // 8 if accelerator() != "cpu" else min(available_bytes() // 4, 2 * GB)
    return int(max(32, min(2048, budget // per_patch)))


def volume_cache_size() -> int:
    """How many imaging series to keep in memory. A 512x512x200 CT is 200 MB."""
    if (value := _override("IMAGING_CACHE_SERIES")) is not None:
        return value
    return int(max(1, min(8, memory_bytes() // (2 * GB))))


def sentence_cache_size() -> int:
    """Passages whose sentence vectors are kept: about 3 KB each."""
    if (value := _override("SENTENCE_CACHE_SIZE")) is not None:
        return value
    return int(max(512, min(65536, memory_bytes() // (64 * 1024))))


def training_cache_bytes() -> int:
    """How much memory a training run may use for decoded images."""
    if (value := _override("TRAINING_CACHE_MB")) is not None:
        return value * 1024 * 1024
    return int(max(GB // 2, min(8 * GB, memory_bytes() // 3)))


def recommended_workers() -> int:
    """Web workers this machine can run: one per core, but each needs about
    1.5 GB for the model and the index."""
    if (value := _override("WEB_WORKERS")) is not None:
        return value
    by_memory = int(memory_bytes() // (1536 * 1024 * 1024))
    return max(1, min(int(cpus()), by_memory, 16))


def summary() -> dict:
    """What was detected and chosen, for the dashboard and the logs."""
    return {
        "cpus": round(cpus(), 2),
        "memory_gb": round(memory_bytes() / GB, 1),
        "accelerator": accelerator(),
        "accelerator_memory_gb": round(accelerator_memory_bytes() / GB, 1),
        "embed_batch": embed_batch_size(),
        "imaging_batch": patch_batch_size(64),
        "web_workers": recommended_workers(),
        "sentence_cache": sentence_cache_size(),
        "series_cache": volume_cache_size(),
    }
