"""Hardware available for training: NVIDIA (or ROCm) GPUs, the Apple GPU,
and the CPU. The first device listed is the recommended one."""

from __future__ import annotations

import os
import platform
import subprocess


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=2).stdout.strip() or "CPU"
        except (OSError, subprocess.SubprocessError):
            return "CPU"
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "CPU"


def _memory_gb() -> float:
    from ..local_ai import detect_hardware

    return detect_hardware().memory_gb


def detect() -> dict:
    import torch

    devices: list[dict] = []
    if torch.cuda.is_available():
        vendor = "AMD" if getattr(torch.version, "hip", None) else "NVIDIA"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({"id": f"cuda:{i}", "kind": "gpu", "vendor": vendor, "name": props.name,
                            "memory_gb": round(props.total_memory / 1024**3, 1)})
    memory = _memory_gb()
    cpu = _cpu_name()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append({"id": "mps", "kind": "gpu", "vendor": "Apple", "name": f"Apple GPU ({cpu})",
                        "memory_gb": memory, "shared_memory": True})
    devices.append({"id": "cpu", "kind": "cpu", "vendor": "", "name": cpu, "cores": os.cpu_count() or 1,
                    "memory_gb": memory})
    gpus = [d for d in devices if d["kind"] == "gpu"]
    return {
        "devices": devices,
        "recommended": devices[0]["id"],
        "gpu_count": len(gpus),
        "torch": torch.__version__,
    }


def available(device_id: str) -> bool:
    return any(d["id"] == device_id for d in detect()["devices"])
