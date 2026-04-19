"""ADMET prediction using RDKit descriptor-based heuristic models."""
from __future__ import annotations

import math
from typing import Union

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdmolops

from pharmacore.core.types import (
    ADMETProfile,
    AbsorptionProfile,
    DistributionProfile,
    ExcretionProfile,
    MetabolismProfile,
    Molecule,
    ToxicityProfile,
)

# ---------------------------------------------------------------------------
# SMARTS constants
# ---------------------------------------------------------------------------

CYP3A4_INHIBITOR_SMARTS = [
    "[#7]1~[#6]~[#6]~[#7]~[#6]~1",          # imidazole-like
    "c1ccc2[nH]ccc2c1",                       # indole
    "c1cnc2ccccc2n1",                          # quinazoline
]

CYP2D6_INHIBITOR_SMARTS = [
    "c1ccc(N)cc1",                             # aniline
    "[#7]1CCCCC1",                             # piperidine
    "c1ccc(OC)cc1",                            # methoxyphenyl
]

LABILE_GROUP_SMARTS = [
    "[CX4][OH]",           # benzylic/aliphatic hydroxyl
    "[#6][SH]",            # thiol
    "C(=O)O[#6]",         # ester
    "[NH2][#6]",           # primary amine
    "[#6]OC",             # ether (O-demethylation)
    "c1ccccc1[CH3]",      # benzylic methyl
]

MUTAGENIC_SMARTS = {
    "aromatic_nitro": "[$(c[N+](=O)[O-]),$(c[N+]([O-])=O)]",
    "aromatic_amine": "[$(c-[NH2]),$(c-[NH][CH3])]",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "nitrosamine": "[NX3][NX2]=O",
    "hydrazine": "[NX3][NX3]",
}

HEPATOTOX_SMARTS = {
    "quinone": "O=C1C=CC(=O)C=C1",
    "epoxide": "C1OC1",
    "michael_acceptor": "[#6]=CC=O",
    "acyl_halide": "C(=O)[F,Cl,Br,I]",
    "thiophene": "c1ccsc1",
    "furan": "c1ccoc1",
    "aniline": "c1ccc(N)cc1",
}


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


class ADMETPredictor:
    """Heuristic ADMET predictor using RDKit descriptors."""

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            try:
                import torch
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self._cyp3a4 = [Chem.MolFromSmarts(s) for s in CYP3A4_INHIBITOR_SMARTS]
        self._cyp2d6 = [Chem.MolFromSmarts(s) for s in CYP2D6_INHIBITOR_SMARTS]
        self._labile = [Chem.MolFromSmarts(s) for s in LABILE_GROUP_SMARTS]
        self._mutag = {k: Chem.MolFromSmarts(s) for k, s in MUTAGENIC_SMARTS.items()}
        self._hepato = {k: Chem.MolFromSmarts(s) for k, s in HEPATOTOX_SMARTS.items()}

    def predict(self, molecule: Union[Molecule, str]) -> ADMETProfile:
        """Predict ADMET profile for a molecule."""
        if isinstance(molecule, str):
            molecule = Molecule.from_smiles(molecule)
        mol = molecule.to_rdkit()

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotb = Lipinski.NumRotatableBonds(mol)

        return ADMETProfile(
            absorption=self._absorption(mol, mw, logp, tpsa, hbd, hba, rotb),
            distribution=self._distribution(mol, mw, logp, tpsa),
            metabolism=self._metabolism(mol),
            excretion=self._excretion(mol, mw, logp),
            toxicity=self._toxicity(mol, logp),
            molecule_smiles=molecule.smiles,
        )

    # ------------------------------------------------------------------
    # Absorption
    # ------------------------------------------------------------------

    def _absorption(
        self, mol: Chem.Mol, mw: float, logp: float, tpsa: float,
        hbd: int, hba: int, rotb: int,
    ) -> AbsorptionProfile:
        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])
        veber_pass = tpsa <= 140 and rotb <= 10
        lipinski_pass = lipinski_violations <= 1
        oral_pass = lipinski_pass and veber_pass
        score = _clamp(1.0 - 0.2 * lipinski_violations - (0.0 if veber_pass else 0.3))

        caco2 = -0.0103 * tpsa + 0.618

        pgp = mw > 400 and hbd > 3

        return AbsorptionProfile(
            oral_bioavailability=score,
            oral_bioavailability_pass=oral_pass,
            caco2_permeability=round(caco2, 4),
            pgp_substrate=pgp,
        )

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def _distribution(
        self, mol: Chem.Mol, mw: float, logp: float, tpsa: float,
    ) -> DistributionProfile:
        bbb_score = 0.0
        bbb_score += 1.0 if mw < 450 else 0.0
        bbb_score += 1.0 if tpsa < 90 else (0.5 if tpsa < 120 else 0.0)
        bbb_score += 1.0 if 1.0 <= logp <= 3.0 else (0.5 if 0 <= logp <= 5 else 0.0)
        bbb_score /= 3.0
        bbb_pass = bbb_score >= 0.7

        ppb = _clamp(0.5 + 0.1 * logp)

        vd = 0.05 * math.exp(0.4 * logp) if logp > 0 else 0.2

        return DistributionProfile(
            bbb_penetration=round(bbb_score, 4),
            bbb_penetration_pass=bbb_pass,
            plasma_protein_binding=round(ppb, 4),
            vd=round(vd, 4),
        )

    # ------------------------------------------------------------------
    # Metabolism
    # ------------------------------------------------------------------

    def _metabolism(self, mol: Chem.Mol) -> MetabolismProfile:
        cyp3a4 = any(mol.HasSubstructMatch(pat) for pat in self._cyp3a4 if pat)
        cyp2d6 = any(mol.HasSubstructMatch(pat) for pat in self._cyp2d6 if pat)

        labile_count = sum(
            len(mol.GetSubstructMatches(pat)) for pat in self._labile if pat
        )
        stability = _clamp(1.0 - 0.15 * labile_count)

        return MetabolismProfile(
            cyp_inhibition={"CYP3A4": cyp3a4, "CYP2D6": cyp2d6},
            metabolic_stability=round(stability, 4),
            labile_group_count=labile_count,
        )

    # ------------------------------------------------------------------
    # Excretion
    # ------------------------------------------------------------------

    def _excretion(self, mol: Chem.Mol, mw: float, logp: float) -> ExcretionProfile:
        t_half = 1.0 + 0.005 * mw + 0.3 * max(logp, 0)
        t_half = round(max(t_half, 0.5), 2)

        charge = rdmolops.GetFormalCharge(mol)
        renal_cl = max(0.1, 5.0 - 0.008 * mw + 0.5 * abs(charge))
        renal_cl = round(renal_cl, 2)

        return ExcretionProfile(
            half_life_estimate=t_half,
            renal_clearance=renal_cl,
        )

    # ------------------------------------------------------------------
    # Toxicity
    # ------------------------------------------------------------------

    def _toxicity(self, mol: Chem.Mol, logp: float) -> ToxicityProfile:
        mutag_alerts: list[str] = []
        for name, pat in self._mutag.items():
            if pat and mol.HasSubstructMatch(pat):
                mutag_alerts.append(name)
        ames = len(mutag_alerts) > 0

        basic_n = Chem.MolFromSmarts("[#7;+,$([#7;!-;!$([#7]~[#8])])]")
        has_basic_n = mol.HasSubstructMatch(basic_n) if basic_n else False
        herg = logp > 3.7 and has_basic_n

        hepato_alerts: list[str] = []
        for name, pat in self._hepato.items():
            if pat and mol.HasSubstructMatch(pat):
                hepato_alerts.append(name)
        hepato = len(hepato_alerts) > 0

        return ToxicityProfile(
            ames_mutagenicity=ames,
            mutagenic_alerts=mutag_alerts,
            herg_inhibition=herg,
            hepatotoxicity_risk=hepato,
            hepatotoxicity_alerts=hepato_alerts,
        )
