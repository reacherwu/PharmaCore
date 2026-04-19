"""Toxicity screening with PAINS and Brenk filters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from rdkit import Chem

from pharmacore.core.types import Molecule

# ---------------------------------------------------------------------------
# Try to use RDKit's built-in FilterCatalog for PAINS
# ---------------------------------------------------------------------------
_HAS_FILTER_CATALOG = False
try:
    from rdkit.Chem.FilterCatalog import (
        FilterCatalog,
        FilterCatalogParams,
    )
    _HAS_FILTER_CATALOG = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Manual PAINS SMARTS (subset of the most common PAINS-A patterns)
# ---------------------------------------------------------------------------
PAINS_SMARTS: dict[str, str] = {
    "rhodanine": "O=C1NC(=S)SC1",
    "catechol": "c1cc(O)c(O)cc1",
    "quinone_A": "O=C1C=CC(=O)C=C1",
    "michael_acceptor_A": "C=CC(=O)[#6]",
    "hydroxyphenyl_hydrazone": "c1ccc(O)cc1/N=N",
    "anil_alk_A": "c1ccc(/N=C/[CH2])cc1",
    "ene_one_A": "[#6]/C=C/C(=O)[#6]",
    "imine_one_A": "[#6]C(=O)/C=N",
    "mannich_A": "[NH,NH2]C([#6])([#6])[#6]=O",
    "azo": "[#7]=[#7]",
}

# ---------------------------------------------------------------------------
# Brenk unwanted substructures (subset)
# ---------------------------------------------------------------------------
BRENK_SMARTS: dict[str, str] = {
    "aldehyde": "[CH1](=O)",
    "michael_acceptor": "C=CC(=O)",
    "epoxide": "C1OC1",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl,Br,I]",
    "acid_halide": "C(=O)[F,Cl,Br,I]",
    "phosphorane": "[PX5]",
    "peroxide": "OO",
    "isocyanate": "N=C=O",
    "isothiocyanate": "N=C=S",
    "acyl_cyanide": "C(=O)C#N",
    "sulfonium": "[S+]",
    "beta_lactam": "C1(=O)NCC1",
    "crown_ether": "C1COCCOCCOCCO1",
    "hydrazine": "[NX3][NX3]",
    "nitro": "[N+](=O)[O-]",
}


@dataclass
class ScreeningResult:
    """Result of a toxicity/filter screen."""
    pains_alerts: list[str] = field(default_factory=list)
    brenk_alerts: list[str] = field(default_factory=list)
    is_pains: bool = False
    is_brenk: bool = False
    total_alerts: int = 0


class ToxicityScreener:
    """Screen molecules for PAINS and Brenk unwanted substructures."""

    def __init__(self) -> None:
        # Try built-in FilterCatalog for PAINS
        self._pains_catalog = None
        if _HAS_FILTER_CATALOG:
            try:
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                self._pains_catalog = FilterCatalog(params)
            except Exception:
                self._pains_catalog = None

        # Compile manual SMARTS as fallback / supplement
        self._pains_pats = {
            k: Chem.MolFromSmarts(s) for k, s in PAINS_SMARTS.items()
        }
        self._brenk_pats = {
            k: Chem.MolFromSmarts(s) for k, s in BRENK_SMARTS.items()
        }

    def screen(self, molecule: Union[Molecule, str]) -> ScreeningResult:
        """Screen a molecule and return alerts."""
        if isinstance(molecule, str):
            molecule = Molecule.from_smiles(molecule)
        mol = molecule.to_rdkit()

        pains_alerts = self._check_pains(mol)
        brenk_alerts = self._check_brenk(mol)

        total = len(pains_alerts) + len(brenk_alerts)
        return ScreeningResult(
            pains_alerts=pains_alerts,
            brenk_alerts=brenk_alerts,
            is_pains=len(pains_alerts) > 0,
            is_brenk=len(brenk_alerts) > 0,
            total_alerts=total,
        )

    def _check_pains(self, mol: Chem.Mol) -> list[str]:
        alerts: list[str] = []

        # Use FilterCatalog if available
        if self._pains_catalog is not None:
            entry = self._pains_catalog.GetFirstMatch(mol)
            if entry is not None:
                # Collect all matches
                matches = self._pains_catalog.GetMatches(mol)
                for m in matches:
                    alerts.append(m.GetDescription())
                return alerts

        # Fallback: manual SMARTS
        for name, pat in self._pains_pats.items():
            if pat and mol.HasSubstructMatch(pat):
                alerts.append(name)
        return alerts

    def _check_brenk(self, mol: Chem.Mol) -> list[str]:
        alerts: list[str] = []
        for name, pat in self._brenk_pats.items():
            if pat and mol.HasSubstructMatch(pat):
                alerts.append(name)
        return alerts
