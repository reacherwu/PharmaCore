"""AutoDock Vina molecular docking integration."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pharmacore.core.types import DockingResult, Molecule, Protein

logger = logging.getLogger(__name__)


class VinaDocker:
    """Molecular docking using AutoDock Vina."""

    def __init__(
        self,
        vina_path: str | None = None,
        exhaustiveness: int = 8,
        n_poses: int = 5,
    ) -> None:
        self.vina_path = vina_path or shutil.which("vina") or shutil.which("vina_split")
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses

    @property
    def is_available(self) -> bool:
        """Check if Vina binary is accessible."""
        return self.vina_path is not None and Path(self.vina_path).exists()

    def dock(
        self,
        molecule: Molecule,
        protein_pdb: str | Path,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    ) -> list[DockingResult]:
        """Dock a molecule against a protein target.

        Args:
            molecule: Molecule to dock.
            protein_pdb: Path to protein PDB file.
            center: (x, y, z) center of the docking box.
            box_size: (sx, sy, sz) dimensions of the docking box.

        Returns:
            List of DockingResult sorted by binding affinity.
        """
        if not self.is_available:
            raise RuntimeError(
                "AutoDock Vina not found. Install with: brew install autodock-vina "
                "or download from https://vina.scripps.edu/"
            )

        protein_pdb = Path(protein_pdb)
        protein = Protein(sequence="", name=protein_pdb.stem)

        with tempfile.TemporaryDirectory(prefix="pharmacore_dock_") as tmpdir:
            tmp = Path(tmpdir)
            ligand_pdbqt = self._prepare_ligand(molecule, tmp / "ligand.pdbqt")
            receptor_pdbqt = self._prepare_receptor(protein_pdb, tmp / "receptor.pdbqt")
            out_pdbqt = tmp / "output.pdbqt"

            cmd = [
                self.vina_path,
                "--receptor", str(receptor_pdbqt),
                "--ligand", str(ligand_pdbqt),
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(box_size[0]),
                "--size_y", str(box_size[1]),
                "--size_z", str(box_size[2]),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.n_poses),
                "--out", str(out_pdbqt),
            ]

            logger.info("Running Vina: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                raise RuntimeError(f"Vina failed: {result.stderr}")

            scores = self._parse_vina_output(result.stdout)
            return [
                DockingResult(
                    molecule=molecule,
                    protein=protein,
                    score=score,
                    pose_path=out_pdbqt if i == 0 else None,
                    confidence=min(1.0, abs(score) / 12.0),
                )
                for i, (score, _) in enumerate(scores)
            ]

    def _prepare_ligand(self, molecule: Molecule, output_path: Path) -> Path:
        """Convert molecule to PDBQT format for Vina."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = molecule.to_rdkit()
        if mol is None:
            raise ValueError(f"Invalid molecule: {molecule.smiles}")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)

        # Write as PDB first, then convert to PDBQT
        pdb_path = output_path.with_suffix(".pdb")
        Chem.MolToPDBFile(mol, str(pdb_path))

        # Simple PDB -> PDBQT conversion (add charges)
        self._pdb_to_pdbqt(pdb_path, output_path)
        return output_path

    def _prepare_receptor(self, protein_pdb: Path, output_path: Path) -> Path:
        """Convert protein PDB to PDBQT."""
        self._pdb_to_pdbqt(protein_pdb, output_path)
        return output_path

    @staticmethod
    def _pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
        """Simple PDB to PDBQT conversion."""
        lines = []
        for line in Path(pdb_path).read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                # Append Gasteiger charge placeholder
                padded = f"{line:<77s}" if len(line) < 77 else line[:77]
                lines.append(f"{padded}  0.000 {'A ':>2s}")
            elif line.startswith("END"):
                lines.append(line)
        pdbqt_path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _parse_vina_output(output: str) -> list[tuple[float, int]]:
        """Parse Vina output for binding affinities.

        Returns list of (affinity_kcal_mol, mode_index).
        """
        results = []
        in_table = False
        for line in output.splitlines():
            if "-----+------------" in line:
                in_table = True
                continue
            if in_table:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mode = int(parts[0])
                        affinity = float(parts[1])
                        results.append((affinity, mode))
                    except (ValueError, IndexError):
                        if results:
                            break
        return results
