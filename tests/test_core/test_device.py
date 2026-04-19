"""Tests for pharmacore.core.device."""
from __future__ import annotations

from pharmacore.core.device import DeviceManager


def test_detect_device() -> None:
    dm = DeviceManager()
    result = dm.detect_device()
    assert result in ("mps", "cpu")


def test_device_info() -> None:
    dm = DeviceManager()
    info = dm.device_info()
    assert isinstance(info, dict)
    for key in ("system", "machine", "processor", "chip", "memory_gb", "cpu_cores"):
        assert key in info


def test_has_mlx() -> None:
    result = DeviceManager.has_mlx()
    assert isinstance(result, bool)
