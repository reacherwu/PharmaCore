"""Docking result scoring and analysis."""
from __future__ import annotations

from pharmacore.core.types import DockingResult, Molecule


class DockingScorer:
    """Analyze and rank docking results."""

    @staticmethod
    def score_pose(result: DockingResult) -> dict:
        """Compute efficiency metrics for a docking result."""
        mol = result.molecule.to_rdkit()
        heavy_atoms = mol.GetNumHeavyAtoms() if mol else 1

        from pharmacore.utils.chemistry import compute_descriptors
        desc = compute_descriptors(mol) if mol else {}

        affinity = abs(result.score)
        return {
            "binding_affinity": result.score,
            "ligand_efficiency": -result.score / heavy_atoms,
            "lipophilic_efficiency": affinity - desc.get("logp", 0),
            "heavy_atoms": heavy_atoms,
            "confidence": result.confidence,
        }

    @staticmethod
    def rank_results(results: list[DockingResult]) -> list[DockingResult]:
        """Sort results by binding affinity (lower/more negative = better)."""
        return sorted(results, key=lambda r: r.score)

    @staticmethod
    def filter_results(
        results: list[DockingResult], threshold: float = -6.0
    ) -> list[DockingResult]:
        """Keep only results with affinity below threshold."""
        return [r for r in results if r.score <= threshold]
