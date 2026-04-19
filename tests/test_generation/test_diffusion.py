"""Tests for molecular generation module."""
from __future__ import annotations

import pytest

from pharmacore.core.types import Molecule


def test_molecular_generator_init():
    """Test MolecularGenerator initialization."""
    from pharmacore.generation.diffusion import MolecularGenerator
    gen = MolecularGenerator(device="cpu", seed=42)
    assert gen.device == "cpu"


def test_generate_returns_valid_molecules():
    """Test that generated molecules have valid SMILES."""
    from pharmacore.generation.diffusion import MolecularGenerator
    from rdkit import Chem

    gen = MolecularGenerator(device="cpu", seed=42)
    molecules = gen.generate(n_molecules=5, max_attempts=200)
    assert len(molecules) > 0
    for m in molecules:
        assert m.smiles
        mol = Chem.MolFromSmiles(m.smiles)
        assert mol is not None, f"Invalid SMILES: {m.smiles}"


def test_generate_similar():
    """Test generating molecules similar to a reference."""
    from pharmacore.generation.diffusion import MolecularGenerator

    gen = MolecularGenerator(device="cpu", seed=42)
    ref = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    similar = gen.generate_similar(ref, n_molecules=3, similarity_threshold=0.1, max_attempts=500)
    assert len(similar) > 0
    for m in similar:
        assert "similarity" in m.properties


def test_molecular_filter_lipinski():
    """Test Lipinski filter."""
    from pharmacore.generation.filters import MolecularFilter

    filt = MolecularFilter(rules=["lipinski"])
    aspirin = Molecule.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    result = filt.filter([aspirin])
    assert len(result) == 1  # aspirin passes Lipinski


def test_molecular_filter_removes_violations():
    """Test that filter removes molecules violating rules."""
    from pharmacore.generation.filters import MolecularFilter

    filt = MolecularFilter(rules=["lipinski"])
    # Very large molecule likely to fail
    big_mol = Molecule.from_smiles("C" * 60)  # long alkane, MW > 500
    result = filt.filter([big_mol])
    # Long alkane has MW > 500, should fail
    assert len(result) == 0
