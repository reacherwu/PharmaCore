"""Core domain types for PharmaCore."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class Molecule:
    """Represents a small molecule."""

    smiles: str
    name: Optional[str] = None
    properties: dict[str, Any] = field(default_factory=dict)
    _mol: Any = field(default=None, repr=False)

    @classmethod
    def from_smiles(cls, smiles: str, name: Optional[str] = None) -> Molecule:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        obj = cls(smiles=smiles, name=name)
        obj._mol = mol
        return obj

    def to_rdkit(self) -> Any:
        if self._mol is None:
            from rdkit import Chem
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol

    @property
    def molecular_weight(self) -> float:
        from rdkit.Chem import Descriptors
        return Descriptors.MolWt(self.to_rdkit())

    @property
    def logp(self) -> float:
        from rdkit.Chem import Descriptors
        return Descriptors.MolLogP(self.to_rdkit())

    @property
    def num_hba(self) -> int:
        from rdkit.Chem import Descriptors
        return Descriptors.NumHAcceptors(self.to_rdkit())

    @property
    def num_hbd(self) -> int:
        from rdkit.Chem import Descriptors
        return Descriptors.NumHDonors(self.to_rdkit())


@dataclass
class Protein:
    """Represents a protein target."""

    sequence: str
    name: Optional[str] = None
    structure_path: Optional[Path] = None
    embeddings: Optional[np.ndarray] = None


@dataclass
class DockingResult:
    """Result of a molecular docking run."""

    molecule: Molecule
    protein: Protein
    score: float
    pose_path: Optional[Path] = None
    confidence: float = 0.0


@dataclass
class AbsorptionProfile:
    """Absorption predictions."""
    oral_bioavailability: float = 0.0
    oral_bioavailability_pass: bool = False
    caco2_permeability: float = 0.0
    pgp_substrate: bool = False


@dataclass
class DistributionProfile:
    """Distribution predictions."""
    bbb_penetration: float = 0.0
    bbb_penetration_pass: bool = False
    plasma_protein_binding: float = 0.0
    vd: float = 0.0


@dataclass
class MetabolismProfile:
    """Metabolism predictions."""
    cyp_inhibition: dict[str, bool] = field(default_factory=dict)
    metabolic_stability: float = 0.0
    labile_group_count: int = 0


@dataclass
class ExcretionProfile:
    """Excretion predictions."""
    half_life_estimate: float = 0.0
    renal_clearance: float = 0.0


@dataclass
class ToxicityProfile:
    """Toxicity predictions."""
    ames_mutagenicity: bool = False
    mutagenic_alerts: list[str] = field(default_factory=list)
    herg_inhibition: bool = False
    hepatotoxicity_risk: bool = False
    hepatotoxicity_alerts: list[str] = field(default_factory=list)


@dataclass
class ADMETProfile:
    """ADMET property predictions for a molecule."""

    absorption: Any = field(default_factory=dict)
    distribution: Any = field(default_factory=dict)
    metabolism: Any = field(default_factory=dict)
    excretion: Any = field(default_factory=dict)
    toxicity: Any = field(default_factory=dict)
    molecule_smiles: str = ""


@dataclass
class PipelineResult:
    """End-to-end pipeline output."""

    target: str
    molecules: list[Molecule] = field(default_factory=list)
    docking_results: list[DockingResult] = field(default_factory=list)
    admet_profiles: list[ADMETProfile] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
