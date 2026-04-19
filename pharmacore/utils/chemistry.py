"""Chemistry utilities wrapping RDKit functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from rdkit import Chem


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit Mol object.

    Returns None if the SMILES is invalid.
    """
    from rdkit import Chem as _Chem

    if not smiles or not isinstance(smiles, str):
        return None
    mol = _Chem.MolFromSmiles(smiles)
    return mol


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> str:
    """Convert an RDKit Mol object to a SMILES string."""
    from rdkit import Chem as _Chem

    return _Chem.MolToSmiles(mol, canonical=canonical)


def compute_descriptors(mol: Chem.Mol) -> dict:
    """Compute common molecular descriptors.

    Returns a dict with: molecular_weight, logp, hba, hbd, tpsa,
    rotatable_bonds, num_rings, num_aromatic_rings.
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def compute_fingerprint(
    mol: Chem.Mol,
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute a molecular fingerprint and return it as a numpy array.

    Supported fp_type values: 'morgan', 'maccs', 'rdkit'.
    """
    import numpy as _np
    from rdkit.Chem import MACCSkeys, RDKFingerprint, rdFingerprintGenerator

    if fp_type == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    elif fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "rdkit":
        fp = RDKFingerprint(mol, fpSize=n_bits)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type!r}. Use 'morgan', 'maccs', or 'rdkit'.")

    arr = _np.zeros(len(fp), dtype=_np.int8)
    for i in fp.GetOnBits():
        arr[i] = 1
    return arr


def check_drug_likeness(mol: Chem.Mol) -> dict:
    """Check Lipinski Rule-of-5 and Veber rules.

    Returns a dict with 'lipinski_pass', 'veber_pass', and 'violations' list.
    """
    desc = compute_descriptors(mol)
    violations: list[str] = []

    # Lipinski Rule of 5
    if desc["molecular_weight"] > 500:
        violations.append("MW > 500")
    if desc["logp"] > 5:
        violations.append("logP > 5")
    if desc["hba"] > 10:
        violations.append("HBA > 10")
    if desc["hbd"] > 5:
        violations.append("HBD > 5")

    lipinski_pass = sum(
        1
        for v in violations
        if v.startswith(("MW", "logP", "HBA", "HBD"))
    ) <= 1  # classic Ro5 allows at most 1 violation

    # Veber rules
    veber_violations: list[str] = []
    if desc["tpsa"] > 140:
        veber_violations.append("TPSA > 140")
    if desc["rotatable_bonds"] > 10:
        veber_violations.append("rotatable_bonds > 10")

    veber_pass = len(veber_violations) == 0
    violations.extend(veber_violations)

    return {
        "lipinski_pass": lipinski_pass,
        "veber_pass": veber_pass,
        "violations": violations,
    }


def compute_similarity(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    metric: str = "tanimoto",
) -> float:
    """Compute similarity between two molecules using Morgan fingerprints.

    Currently supports 'tanimoto' metric.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp1 = gen.GetFingerprint(mol1)
    fp2 = gen.GetFingerprint(mol2)

    if metric == "tanimoto":
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    raise ValueError(f"Unsupported metric: {metric!r}. Use 'tanimoto'.")


def generate_3d_coords(mol: Chem.Mol) -> Chem.Mol:
    """Generate 3D coordinates using ETKDG and optimize with MMFF94.

    Returns a new Mol with 3D coordinates embedded.
    """
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem

    mol_3d = _Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol_3d, params)
    if status != 0:
        raise RuntimeError("3D embedding failed.")
    AllChem.MMFFOptimizeMolecule(mol_3d)
    return mol_3d
