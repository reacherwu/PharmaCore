"""Tests for ESM-2 protein embeddings (mocked - no model download needed)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_esm_embedder_init():
    """Test ESMEmbedder can be instantiated without loading model."""
    from pharmacore.protein.esm import ESMEmbedder
    embedder = ESMEmbedder(model_name="esm2_t33_650M_UR50D", device="cpu")
    assert embedder.model_name == "esm2_t33_650M_UR50D"
    assert embedder._model is None


@patch("pharmacore.protein.esm.ESMEmbedder._load_model")
def test_embed_returns_array(mock_load):
    """Test embed returns numpy array with correct shape."""
    from pharmacore.protein.esm import ESMEmbedder
    import torch

    embedder = ESMEmbedder(device="cpu")
    seq = "ACDEFGHIK"
    embed_dim = 1280

    # Mock the model internals
    embedder._model = MagicMock()
    embedder._model.num_layers = 33
    embedder._alphabet = MagicMock()
    embedder._batch_converter = MagicMock()

    # Mock batch converter output
    fake_tokens = torch.zeros(1, len(seq) + 2, dtype=torch.long)
    embedder._batch_converter.return_value = (None, None, fake_tokens)

    # Mock model output
    fake_repr = torch.randn(1, len(seq) + 2, embed_dim)
    embedder._model.return_value = {"representations": {33: fake_repr}}

    result = embedder.embed(seq)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(seq), embed_dim)


@patch("pharmacore.protein.esm.ESMEmbedder._load_model")
def test_get_protein(mock_load):
    """Test get_protein returns Protein with embeddings."""
    from pharmacore.protein.esm import ESMEmbedder
    from pharmacore.core.types import Protein
    import torch

    embedder = ESMEmbedder(device="cpu")
    seq = "ACDEF"
    embed_dim = 1280

    embedder._model = MagicMock()
    embedder._model.num_layers = 33
    embedder._alphabet = MagicMock()
    embedder._batch_converter = MagicMock()

    fake_tokens = torch.zeros(1, len(seq) + 2, dtype=torch.long)
    embedder._batch_converter.return_value = (None, None, fake_tokens)
    fake_repr = torch.randn(1, len(seq) + 2, embed_dim)
    embedder._model.return_value = {"representations": {33: fake_repr}}

    protein = embedder.get_protein(seq, name="test_protein")
    assert isinstance(protein, Protein)
    assert protein.sequence == seq
    assert protein.name == "test_protein"
    assert protein.embeddings is not None
    assert protein.embeddings.shape[0] == len(seq)


def test_structure_predictor_init():
    """Test StructurePredictor initialization."""
    from pharmacore.protein.folding import StructurePredictor
    sp = StructurePredictor(method="esmfold", device="cpu")
    assert sp.method == "esmfold"
    assert sp._model is None
