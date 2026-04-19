"""ESM-2 protein language model embeddings with Apple Silicon acceleration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from pharmacore.core.device import DeviceManager
from pharmacore.core.types import Protein

logger = logging.getLogger(__name__)


class ESMEmbedder:
    """Generate protein embeddings using ESM-2 models with MPS acceleration."""

    MODELS = {
        "esm2_t6_8M": "esm2_t6_8M_UR50D",
        "esm2_t12_35M": "esm2_t12_35M_UR50D",
        "esm2_t30_150M": "esm2_t30_150M_UR50D",
        "esm2_t33_650M": "esm2_t33_650M_UR50D",
        "esm2_t36_3B": "esm2_t36_3B_UR50D",
        "esm2_t48_15B": "esm2_t48_15B_UR50D",
    }

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self._device_mgr = DeviceManager()
        self.device = self._device_mgr.detect_device() if device == "auto" else device
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    def _load_model(self) -> None:
        """Lazy-load ESM model on first use."""
        if self._model is not None:
            return
        try:
            import esm
            import torch
        except ImportError:
            raise ImportError(
                "ESM not installed. Install with: pip install fair-esm"
            )

        logger.info("Loading %s on %s...", self.model_name, self.device)
        loader = getattr(esm.pretrained, self.model_name, None)
        if loader is None:
            raise ValueError(f"Unknown model: {self.model_name}")

        self._model, self._alphabet = loader()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model = self._model.to(self.device).eval()
        logger.info("Model loaded successfully.")

    def embed(self, sequence: str) -> np.ndarray:
        """Generate per-residue embeddings. Returns shape (seq_len, embed_dim)."""
        import torch

        self._load_model()
        data = [("protein", sequence)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self._model(tokens, repr_layers=[self._model.num_layers])

        embeddings = results["representations"][self._model.num_layers]
        # Remove BOS/EOS tokens
        return embeddings[0, 1 : len(sequence) + 1].cpu().numpy()

    def embed_batch(self, sequences: list[str]) -> list[np.ndarray]:
        """Batch embedding for multiple sequences."""
        import torch

        self._load_model()
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self._batch_converter(data)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self._model(tokens, repr_layers=[self._model.num_layers])

        embeddings = results["representations"][self._model.num_layers]
        return [
            embeddings[i, 1 : len(seq) + 1].cpu().numpy()
            for i, seq in enumerate(sequences)
        ]

    def get_protein(self, sequence: str, name: str = "") -> Protein:
        """Create a Protein object with embeddings."""
        emb = self.embed(sequence)
        return Protein(sequence=sequence, name=name, embeddings=emb)
