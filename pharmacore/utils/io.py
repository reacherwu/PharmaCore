"""File I/O utilities for molecular data."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit import Chem


def read_sdf(path: str | Path) -> list[Chem.Mol]:
    """Read molecules from an SDF file. Skips entries that fail to parse."""
    from rdkit import Chem as _Chem

    supplier = _Chem.SDMolSupplier(str(path))
    return [mol for mol in supplier if mol is not None]


def write_sdf(mols: list[Chem.Mol], path: str | Path) -> None:
    """Write a list of molecules to an SDF file."""
    from rdkit import Chem as _Chem

    writer = _Chem.SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()


def read_smiles_file(path: str | Path) -> list[str]:
    """Read SMILES strings from a text file (one per line).

    Lines starting with '#' and blank lines are skipped.
    """
    smiles_list: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line.split()[0])  # take first token
    return smiles_list


def write_smiles_file(smiles_list: list[str], path: str | Path) -> None:
    """Write SMILES strings to a text file, one per line."""
    with open(path, "w") as fh:
        for smi in smiles_list:
            fh.write(smi + "\n")


def read_pdb(path: str | Path) -> str:
    """Extract the amino-acid sequence from a PDB file (SEQRES or ATOM records).

    Returns a one-letter amino-acid sequence string.
    """
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    residues: list[str] = []
    seen: set[tuple[str, str]] = set()  # (chain, resSeq)

    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                chain = line[21]
                res_seq = line[22:27].strip()
                res_name = line[17:20].strip()
                key = (chain, res_seq)
                if key not in seen and res_name in three_to_one:
                    seen.add(key)
                    residues.append(three_to_one[res_name])

    return "".join(residues)
