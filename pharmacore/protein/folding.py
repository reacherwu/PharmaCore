"""Protein structure prediction wrappers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pharmacore.core.device import DeviceManager

logger = logging.getLogger(__name__)


class StructurePredictor:
    """Predict protein 3D structure from sequence."""

    METHODS = ("esmfold", "openfold")

    def __init__(self, method: str = "esmfold", device: str = "auto") -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {self.METHODS}")
        self.method = method
        self._device_mgr = DeviceManager()
        self.device = self._device_mgr.detect_device() if device == "auto" else device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load structure prediction model."""
        if self._model is not None:
            return
        if self.method == "esmfold":
            self._load_esmfold()
        else:
            raise NotImplementedError(f"{self.method} not yet supported")

    def _load_esmfold(self) -> None:
        """Load ESMFold model."""
        try:
            import torch
            model = torch.hub.load("facebookresearch/esm:main", "esmfold_v1")
            model = model.to(self.device).eval()
            self._model = model
            logger.info("ESMFold loaded on %s", self.device)
        except Exception as e:
            logger.warning("ESMFold not available: %s. Structure prediction disabled.", e)
            self._model = None

    def predict(self, sequence: str) -> str:
        """Predict structure, returns PDB string."""
        self._load_model()
        if self._model is None:
            return self._placeholder_pdb(sequence)

        import torch
        with torch.no_grad():
            output = self._model.infer_pdb(sequence)
        return output

    def predict_to_file(self, sequence: str, output_path: str | Path) -> Path:
        """Save predicted structure to PDB file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_string = self.predict(sequence)
        output_path.write_text(pdb_string)
        logger.info("Structure saved to %s", output_path)
        return output_path

    @staticmethod
    def _placeholder_pdb(sequence: str) -> str:
        """Generate a placeholder PDB when model is unavailable."""
        lines = [
            "REMARK  PharmaCore placeholder structure",
            f"REMARK  Sequence length: {len(sequence)}",
            "REMARK  Install ESMFold for real predictions: pip install fair-esm",
        ]
        for i, aa in enumerate(sequence[:100]):
            lines.append(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{i*3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            )
        lines.append("END")
        return "\n".join(lines)
