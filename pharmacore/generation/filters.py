"""Drug-likeness and PAINS filters for molecular generation."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pharmacore.core.types import Molecule

logger = logging.getLogger(__name__)

# PAINS substructure SMARTS (subset of most common)
PAINS_SMARTS = [
    "[#6]1:[#6]:[#6](:[#6]:[#6]:[#6]:1)-[#7]=[#7]-[#6]",  # azo
    "[#6]-[#16](=[#8])(=[#8])-[#8]",  # sulfonyl ester
    "[#6]=[#6]-[#6](=[#8])-[#6]=[#6]",  # michael acceptor
    "[#6]1(=[#8]):[#6]:[#6]:[#6](=[#8]):[#6]:[#6]:1",  # quinone
]

# Brenk unwanted substructures
BRENK_SMARTS = [
    "[#6](=[#8])([#17])",  # acyl halide
    "[#7+]([#8-])=O",  # nitro
    "[#16]([#17])",  # sulfenyl chloride
    "C(=O)OO",  # peroxide
    "[As]",  # arsenic
    "[Se]",  # selenium
]


class MolecularFilter:
    """Filter molecules by drug-likeness rules."""

    AVAILABLE_RULES = ("lipinski", "veber", "pains", "brenk")

    def __init__(self, rules: list[str] | None = None) -> None:
        self.rules = rules or ["lipinski", "veber"]
        for r in self.rules:
            if r not in self.AVAILABLE_RULES:
                raise ValueError(f"Unknown rule: {r}. Choose from {self.AVAILABLE_RULES}")

    def filter(self, molecules: list[Molecule]) -> list[Molecule]:
        """Apply all active rules, return passing molecules."""
        from rdkit import Chem

        passed = []
        for mol_obj in molecules:
            mol = Chem.MolFromSmiles(mol_obj.smiles)
            if mol is None:
                continue
            if all(self._apply_rule(rule, mol) for rule in self.rules):
                passed.append(mol_obj)

        logger.info(
            "Filter: %d/%d molecules passed (%s)",
            len(passed), len(molecules), ", ".join(self.rules),
        )
        return passed

    def _apply_rule(self, rule: str, mol) -> bool:
        dispatch = {
            "lipinski": self.apply_lipinski,
            "veber": self.apply_veber,
            "pains": self.apply_pains,
            "brenk": self.apply_brenk,
        }
        return dispatch[rule](mol)

    @staticmethod
    def apply_lipinski(mol) -> bool:
        from rdkit.Chem import Descriptors
        return (
            Descriptors.MolWt(mol) <= 500
            and Descriptors.MolLogP(mol) <= 5
            and Descriptors.NumHAcceptors(mol) <= 10
            and Descriptors.NumHDonors(mol) <= 5
        )

    @staticmethod
    def apply_veber(mol) -> bool:
        from rdkit.Chem import Descriptors
        return (
            Descriptors.TPSA(mol) <= 140
            and Descriptors.NumRotatableBonds(mol) <= 10
        )

    @staticmethod
    def apply_pains(mol) -> bool:
        from rdkit import Chem
        for smarts in PAINS_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return False
        return True

    @staticmethod
    def apply_brenk(mol) -> bool:
        from rdkit import Chem
        for smarts in BRENK_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return False
        return True
