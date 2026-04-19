"""Molecular generation using scaffold-based enumeration and diffusion architecture."""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np

from pharmacore.core.device import DeviceManager
from pharmacore.core.types import Molecule

logger = logging.getLogger(__name__)

# Common drug scaffolds (SMILES)
SCAFFOLDS = [
    "c1ccccc1",           # benzene
    "c1ccncc1",           # pyridine
    "C1CCNCC1",           # piperidine
    "C1COCCN1",           # morpholine
    "c1ccc2[nH]ccc2c1",  # indole
    "c1ccc2ncccc2c1",    # quinoline
    "c1ccsc1",            # thiophene
    "c1ccoc1",            # furan
    "c1cnc[nH]1",         # imidazole
    "c1ccnc(N)n1",        # 2-aminopyrimidine
    "c1ccc2ccccc2c1",    # naphthalene
    "C1CCC(CC1)N",        # cyclohexylamine
    "c1ccc(-c2ccccc2)cc1", # biphenyl
    "c1cnc2ccccc2n1",    # quinazoline
    "O=C1CCCN1",          # pyrrolidinone
]

# Functional group modifications (SMARTS -> replacement)
FUNCTIONAL_GROUPS = [
    ("O", "hydroxyl"),
    ("N", "amine"),
    ("C(=O)O", "carboxyl"),
    ("F", "fluoro"),
    ("Cl", "chloro"),
    ("OC", "methoxy"),
    ("C(F)(F)F", "trifluoromethyl"),
    ("S(=O)(=O)N", "sulfonamide"),
    ("C(=O)N", "amide"),
    ("C#N", "nitrile"),
]


class MolecularGenerator:
    """Generate drug-like molecules using scaffold enumeration.

    Uses a combinatorial approach: pick scaffold + attach functional groups
    + validate with RDKit + filter by drug-likeness.

    For production use, replace with a trained diffusion model.
    """

    def __init__(self, device: str = "auto", seed: int | None = None) -> None:
        self._device_mgr = DeviceManager()
        self.device = self._device_mgr.detect_device() if device == "auto" else device
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(
        self,
        n_molecules: int = 10,
        max_attempts: int = 500,
        drug_like: bool = True,
    ) -> list[Molecule]:
        """Generate novel drug-like molecules.

        Args:
            n_molecules: Number of molecules to generate.
            max_attempts: Maximum generation attempts.
            drug_like: If True, filter by Lipinski rules.

        Returns:
            List of valid Molecule objects.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        results = []
        seen_smiles: set[str] = set()

        for _ in range(max_attempts):
            if len(results) >= n_molecules:
                break

            smiles = self._enumerate_molecule()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol)
            if canonical in seen_smiles:
                continue
            seen_smiles.add(canonical)

            if drug_like and not self._passes_drug_likeness(mol):
                continue

            results.append(Molecule(smiles=canonical, name=f"gen_{len(results)}"))

        logger.info("Generated %d molecules from %d attempts", len(results), max_attempts)
        return results

    def generate_similar(
        self,
        reference: Molecule,
        n_molecules: int = 10,
        similarity_threshold: float = 0.3,
        max_attempts: int = 1000,
    ) -> list[Molecule]:
        """Generate molecules similar to a reference compound."""
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        ref_mol = Chem.MolFromSmiles(reference.smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference.smiles}")

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        ref_fp = gen.GetFingerprint(ref_mol)
        results = []
        seen: set[str] = set()

        for _ in range(max_attempts):
            if len(results) >= n_molecules:
                break

            smiles = self._enumerate_molecule()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol)
            if canonical in seen:
                continue
            seen.add(canonical)

            fp = gen.GetFingerprint(mol)
            sim = DataStructs.TanimotoSimilarity(ref_fp, fp)
            if sim >= similarity_threshold:
                results.append(
                    Molecule(
                        smiles=canonical,
                        name=f"sim_{len(results)}",
                        properties={"similarity": round(sim, 4)},
                    )
                )

        return results

    def _enumerate_molecule(self) -> str:
        """Create a random molecule by combining scaffold + functional groups."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        scaffold = random.choice(SCAFFOLDS)
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            return scaffold

        # Randomly attach 1-3 functional groups
        n_groups = random.randint(1, 3)
        rw_mol = Chem.RWMol(mol)

        for _ in range(n_groups):
            fg_smiles, _ = random.choice(FUNCTIONAL_GROUPS)
            fg = Chem.MolFromSmiles(fg_smiles)
            if fg is None:
                continue

            # Pick a random atom to attach to
            atom_indices = list(range(rw_mol.GetNumAtoms()))
            if not atom_indices:
                break
            attach_idx = random.choice(atom_indices)
            atom = rw_mol.GetAtomWithIdx(attach_idx)

            # Only attach to atoms that can accept more bonds
            if atom.GetValence(Chem.ValenceType.IMPLICIT) > 0:
                combo = Chem.CombineMols(rw_mol, fg)
                rw_combo = Chem.RWMol(combo)
                new_idx = rw_mol.GetNumAtoms()  # first atom of fg
                try:
                    rw_combo.AddBond(attach_idx, new_idx, Chem.BondType.SINGLE)
                    Chem.SanitizeMol(rw_combo)
                    rw_mol = rw_combo
                except Exception:
                    continue

        try:
            Chem.SanitizeMol(rw_mol)
            return Chem.MolToSmiles(rw_mol)
        except Exception:
            return scaffold

    @staticmethod
    def _passes_drug_likeness(mol) -> bool:
        """Quick Lipinski + Veber check."""
        from rdkit.Chem import Descriptors

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        rot = Descriptors.NumRotatableBonds(mol)

        return (
            mw <= 500
            and logp <= 5
            and hba <= 10
            and hbd <= 5
            and tpsa <= 140
            and rot <= 10
        )
