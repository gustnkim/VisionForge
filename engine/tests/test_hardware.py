from __future__ import annotations

from visionforge_engine.hardware import (
    HardwareProfile,
    _mps_recommended_memory,
    _profile_name,
    probe_hardware,
    select_torch_device,
)


def test_unknown_apple_memory_uses_the_conservative_profile() -> None:
    assert _profile_name("darwin", "arm64", None, "mps") == "APPLE_M1_16_BASELINE"


def test_mps_profile_uses_the_runtime_recommended_memory() -> None:
    class FakeMps:
        @staticmethod
        def recommended_max_memory() -> int:
            return 24 * 1024**3

    class FakeTorch:
        mps = FakeMps()

    assert _mps_recommended_memory(FakeTorch(), 32 * 1024**3) == 24 * 1024**3


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
