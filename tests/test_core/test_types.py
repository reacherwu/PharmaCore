"""Tests for pharmacore.core.types."""
from __future__ import annotations

from pharmacore.core.types import (
    ADMETProfile,
    DockingResult,
    Molecule,
    Protein,
)

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def test_molecule_from_smiles() -> None:
    mol = Molecule.from_smiles(ASPIRIN_SMILES, name="aspirin")
    assert mol.smiles == ASPIRIN_SMILES
    assert mol.name == "aspirin"
    assert mol.to_rdkit() is not None


def test_molecule_properties() -> None:
    mol = Molecule.from_smiles(ASPIRIN_SMILES)
    assert 170 < mol.molecular_weight < 190  # aspirin MW ~180.16
    assert -1 < mol.logp < 2  # aspirin logP ~1.2
    assert mol.num_hba >= 1
    assert mol.num_hbd >= 1


def test_protein_creation() -> None:
    prot = Protein(sequence="MKTLLILAVL", name="test_protein")
    assert prot.sequence == "MKTLLILAVL"
    assert prot.name == "test_protein"
    assert prot.structure_path is None
    assert prot.embeddings is None


def test_docking_result() -> None:
    mol = Molecule(smiles=ASPIRIN_SMILES)
    prot = Protein(sequence="MKTLLILAVL")
    dr = DockingResult(molecule=mol, protein=prot, score=-7.5, confidence=0.85)
    assert dr.score == -7.5
    assert dr.confidence == 0.85


def test_admet_profile() -> None:
    profile = ADMETProfile(
        absorption={"oral_bioavailability": 0.8},
        toxicity={"herg_inhibition": False},
    )
    assert profile.absorption["oral_bioavailability"] == 0.8
    assert profile.toxicity["herg_inhibition"] is False
    assert profile.distribution == {}
