from __future__ import annotations

from visionforge_engine.hardware import HardwareProfile, probe_hardware, select_torch_device


def test_hardware_profile_reports_a_supported_provider(tmp_path) -> None:
    profile = probe_hardware(tmp_path, include_accelerators=False)

    assert profile.profile in {
        "APPLE_M1_16_BASELINE",
        "APPLE_HIGH_MEMORY",
        "CUDA_ACCELERATED",
        "CPU_FALLBACK",
    }
    assert profile.cpu_count >= 1
    assert "cpu" in profile.execution_providers
    assert profile.free_disk_bytes is not None
    assert profile.free_disk_bytes > 0


def test_auto_device_is_available(monkeypatch) -> None:
    profile = HardwareProfile(
        profile="APPLE_M1_16_BASELINE",
        platform="darwin",
        architecture="arm64",
        cpu_count=10,
        total_memory_bytes=16 * 1024**3,
        accelerator="mps",
        accelerator_name="Apple MPS",
        accelerator_memory_bytes=16 * 1024**3,
        execution_providers=["mps", "cpu"],
    )
    monkeypatch.setattr("visionforge_engine.hardware.probe_hardware", lambda: profile)

    device, profile = select_torch_device("auto")

    assert device in profile.execution_providers
