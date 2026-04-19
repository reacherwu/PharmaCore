"""Device detection and management for Apple Silicon."""
from __future__ import annotations

import importlib
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceManager:
    """Manages compute device selection and info."""

    _device_name: str = field(default="", init=False)

    def detect_device(self) -> str:
        """Detect the best available device: 'mps' or 'cpu'."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def get_device(self) -> Any:
        """Return a torch.device for the best available backend."""
        import torch
        return torch.device(self.detect_device())

    @staticmethod
    def has_mlx() -> bool:
        """Check whether the mlx package is importable."""
        return importlib.util.find_spec("mlx") is not None

    @staticmethod
    def device_info() -> dict[str, Any]:
        """Return a dict describing the current hardware."""
        info: dict[str, Any] = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "chip": "unknown",
            "memory_gb": "unknown",
            "cpu_cores": "unknown",
        }
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info["chip"] = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
            mem_bytes = _sysctl("hw.memsize")
            if mem_bytes:
                info["memory_gb"] = round(int(mem_bytes) / (1024**3), 1)
            cores = _sysctl("hw.ncpu")
            if cores:
                info["cpu_cores"] = int(cores)
        return info


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    dm = DeviceManager()
    print(f"Device : {dm.detect_device()}")
    print(f"MLX    : {dm.has_mlx()}")
    for k, v in dm.device_info().items():
        print(f"  {k}: {v}")
