"""High-level drug discovery workflow API."""
from __future__ import annotations

import logging
from pathlib import Path

from pharmacore.core.types import Molecule, PipelineResult
from pharmacore.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def discover_drugs(
    target_name: str,
    target_sequence: str = "",
    protein_pdb: str | Path | None = None,
    n_molecules: int = 20,
    docking_center: tuple[float, float, float] | None = None,
) -> PipelineResult:
    """One-call drug discovery: target -> generation -> docking -> ADMET -> ranking.

    Example:
        >>> result = discover_drugs(
        ...     target_name="EGFR kinase",
        ...     target_sequence="MRPSGTAGAALLALLAALCPAS...",
        ...     n_molecules=50,
        ... )
        >>> for mol in result.molecules[:5]:
        ...     print(mol.smiles, mol.properties.get("composite_score"))
    """
    pipeline = PipelineOrchestrator()
    return pipeline.run(
        target_name=target_name,
        target_sequence=target_sequence,
        protein_pdb=protein_pdb,
        n_molecules=n_molecules,
        docking_center=docking_center,
    )


def screen_compound(smiles: str) -> dict:
    """Quick screen a single compound through ADMET + drug-likeness.

    Returns dict with drug-likeness, ADMET profile, and overall assessment.
    """
    from pharmacore.admet.predictor import ADMETPredictor
    from pharmacore.generation.filters import MolecularFilter
    from pharmacore.utils.chemistry import check_drug_likeness, compute_descriptors, parse_smiles

    mol = parse_smiles(smiles)
    if mol is None:
        return {"valid": False, "error": "Invalid SMILES"}

    descriptors = compute_descriptors(mol)
    drug_likeness = check_drug_likeness(mol)

    predictor = ADMETPredictor()
    molecule = Molecule(smiles=smiles)
    admet = predictor.predict(molecule)

    return {
        "valid": True,
        "smiles": smiles,
        "descriptors": descriptors,
        "drug_likeness": drug_likeness,
        "admet": {
            "absorption": admet.absorption if isinstance(admet.absorption, dict) else {},
            "distribution": admet.distribution if isinstance(admet.distribution, dict) else {},
            "metabolism": admet.metabolism if isinstance(admet.metabolism, dict) else {},
            "excretion": admet.excretion if isinstance(admet.excretion, dict) else {},
            "toxicity": admet.toxicity if isinstance(admet.toxicity, dict) else {},
        },
        "overall": "promising" if drug_likeness.get("lipinski_pass") else "needs_optimization",
    }


def repurpose_drug(
    smiles: str,
    disease: str,
) -> dict:
    """Evaluate a known drug for repurposing against a new disease target.

    Args:
        smiles: SMILES of the existing drug.
        disease: Disease name to check targets for.

    Returns:
        Dict with target matches, ADMET profile, and repurposing score.
    """
    from pharmacore.target.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    targets = kg.get_targets_for_disease(disease)
    screen = screen_compound(smiles)

    return {
        "drug_smiles": smiles,
        "disease": disease,
        "known_targets": [t.name if hasattr(t, "name") else str(t) for t in targets],
        "drug_screen": screen,
        "repurposing_potential": "moderate" if targets and screen.get("overall") == "promising" else "low",
    }
