"""PharmaCore configuration via pydantic settings."""
from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class PharmaConfig(BaseSettings):
    """Global configuration for PharmaCore."""

    model_config = {"env_prefix": "PHARMACORE_"}

    device: Literal["auto", "mps", "cpu"] = "auto"
    model_dir: Path = Field(default=Path.home() / ".pharmacore" / "models")
    cache_dir: Path = Field(default=Path.home() / ".pharmacore" / "cache")
    log_level: str = "INFO"
    num_workers: int = 0

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if _is_apple_silicon():
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
        return "cpu"


@lru_cache(maxsize=1)
def get_config() -> PharmaConfig:
    """Return the singleton PharmaConfig instance."""
    return PharmaConfig()
