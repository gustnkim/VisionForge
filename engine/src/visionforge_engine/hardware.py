from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3


@dataclass(slots=True)
class HardwareProfile:
    profile: str
    platform: str
    architecture: str
    cpu_count: int
    total_memory_bytes: int | None
    accelerator: str
    accelerator_name: str
    accelerator_memory_bytes: int | None
    execution_providers: list[str]
    free_disk_bytes: int | None = None
    torch_version: str | None = None
    torchvision_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _total_memory_bytes() -> int | None:
    system = platform.system().lower()
    if system == "windows":
        status = _MemoryStatusEx()
        status.length = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    if system == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _profile_name(
    system: str,
    architecture: str,
    total_memory: int | None,
    accelerator: str,
) -> str:
    if system == "darwin" and architecture in {"arm64", "aarch64"}:
        if total_memory is None or total_memory <= 20 * GIB:
            return "APPLE_M1_16_BASELINE"
        return "APPLE_HIGH_MEMORY"
    if accelerator == "cuda":
        return "CUDA_ACCELERATED"
    return "CPU_FALLBACK"


def _mps_recommended_memory(torch_module: Any, total_memory: int | None) -> int | None:
    with suppress(AttributeError, RuntimeError, TypeError, ValueError):
        recommended = int(torch_module.mps.recommended_max_memory())
        if recommended > 0:
            return recommended
    return total_memory


def probe_hardware(
    path: str | Path | None = None,
    *,
    include_accelerators: bool = True,
) -> HardwareProfile:
    system = platform.system().lower()
    architecture = platform.machine().lower()
    total_memory = _total_memory_bytes()
    providers = ["cpu"]
    accelerator = "cpu"
    accelerator_name = platform.processor() or "CPU"
    accelerator_memory: int | None = None
    torch_version: str | None = None
    torchvision_version: str | None = None

    if include_accelerators:
        try:
            import torch
            import torchvision

            torch_version = torch.__version__
            torchvision_version = torchvision.__version__
            if torch.backends.mps.is_available():
                providers.insert(0, "mps")
                accelerator = "mps"
                accelerator_name = "Apple Metal Performance Shaders"
                accelerator_memory = _mps_recommended_memory(torch, total_memory)
            elif torch.cuda.is_available():
                providers.insert(0, "cuda")
                accelerator = "cuda"
                properties = torch.cuda.get_device_properties(0)
                accelerator_name = properties.name
                accelerator_memory = int(properties.total_memory)
        except (ImportError, OSError, RuntimeError):
            pass

    free_disk = None
    if path is not None:
        with suppress(OSError):
            free_disk = shutil.disk_usage(Path(path).expanduser().resolve()).free

    return HardwareProfile(
        profile=_profile_name(system, architecture, total_memory, accelerator),
        platform=system,
        architecture=architecture,
        cpu_count=os.cpu_count() or 1,
        total_memory_bytes=total_memory,
        accelerator=accelerator,
        accelerator_name=accelerator_name,
        accelerator_memory_bytes=accelerator_memory,
        execution_providers=providers,
        free_disk_bytes=free_disk,
        torch_version=torch_version,
        torchvision_version=torchvision_version,
    )


def select_torch_device(requested: str | None = None) -> tuple[str, HardwareProfile]:
    profile = probe_hardware()
    requested = (requested or os.environ.get("VISIONFORGE_DEVICE") or "auto").lower()
    if requested == "auto":
        return profile.accelerator, profile
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported training device: {requested}")
    if requested not in profile.execution_providers:
        raise ValueError(f"requested training device is unavailable: {requested}")
    return requested, profile
