"""Tests for pharmacore.utils.chemistry module."""
from __future__ import annotations

import pytest

from pharmacore.utils.chemistry import (
    check_drug_likeness,
    compute_descriptors,
    compute_fingerprint,
    compute_similarity,
    generate_3d_coords,
    mol_to_smiles,
    parse_smiles,
)

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


class TestParseSmiles:
    def test_parse_smiles_valid(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        # Round-trip check
        smi = mol_to_smiles(mol)
        assert isinstance(smi, str)
        assert len(smi) > 0

    def test_parse_smiles_invalid(self) -> None:
        assert parse_smiles("not_a_smiles!!!") is None
        assert parse_smiles("") is None
        assert parse_smiles(None) is None  # type: ignore[arg-type]


class TestComputeDescriptors:
    def test_compute_descriptors(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        desc = compute_descriptors(mol)
        assert 170 < desc["molecular_weight"] < 190  # aspirin MW ~180.16
        assert isinstance(desc["logp"], float)
        assert desc["hba"] >= 0
        assert desc["hbd"] >= 0
        assert desc["tpsa"] >= 0
        assert desc["rotatable_bonds"] >= 0
        assert desc["num_rings"] >= 1
        assert desc["num_aromatic_rings"] >= 1


class TestFingerprint:
    def test_fingerprint_morgan(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        fp = compute_fingerprint(mol, fp_type="morgan")
        assert fp.shape == (2048,)
        assert fp.sum() > 0

    def test_fingerprint_maccs(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        fp = compute_fingerprint(mol, fp_type="maccs")
        assert len(fp) > 0

    def test_fingerprint_invalid_type(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        with pytest.raises(ValueError, match="Unsupported fingerprint type"):
            compute_fingerprint(mol, fp_type="bad")


class TestDrugLikeness:
    def test_drug_likeness_aspirin(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        result = check_drug_likeness(mol)
        assert result["lipinski_pass"] is True
        assert result["veber_pass"] is True
        assert len(result["violations"]) == 0


class TestSimilarity:
    def test_similarity_identical(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        sim = compute_similarity(mol, mol)
        assert sim == pytest.approx(1.0)

    def test_similarity_different(self) -> None:
        mol1 = parse_smiles(ASPIRIN_SMILES)
        mol2 = parse_smiles("c1ccccc1")  # benzene
        assert mol1 is not None and mol2 is not None
        sim = compute_similarity(mol1, mol2)
        assert 0.0 <= sim < 1.0


class TestGenerate3D:
    def test_generate_3d(self) -> None:
        mol = parse_smiles(ASPIRIN_SMILES)
        assert mol is not None
        mol_3d = generate_3d_coords(mol)
        conf = mol_3d.GetConformer()
        assert conf.GetNumAtoms() > 0
        # Verify we have actual 3D coordinates (not all zeros)
        pos = conf.GetAtomPosition(0)
        coords = (pos.x, pos.y, pos.z)
        assert any(c != 0.0 for c in coords)
