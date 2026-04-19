"""Target identification and analysis for drug discovery."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """A drug target (typically a protein)."""
    name: str
    gene: str = ""
    uniprot_id: str = ""
    organism: str = "Homo sapiens"
    sequence: str = ""
    disease_associations: list[str] = field(default_factory=list)
    druggability_score: float = 0.0
    metadata: dict = field(default_factory=dict)


# Common therapeutic target families with druggability info
TARGET_FAMILIES = {
    "kinase": {"druggability": 0.85, "keywords": ["kinase", "phosphorylat"]},
    "gpcr": {"druggability": 0.90, "keywords": ["receptor", "g-protein", "gpcr"]},
    "ion_channel": {"druggability": 0.75, "keywords": ["channel", "ion"]},
    "protease": {"druggability": 0.80, "keywords": ["protease", "peptidase", "cleavage"]},
    "nuclear_receptor": {"druggability": 0.70, "keywords": ["nuclear receptor", "transcription"]},
    "epigenetic": {"druggability": 0.65, "keywords": ["histone", "methyltransferase", "deacetylase"]},
}


class TargetAnalyzer:
    """Analyze and validate drug targets."""

    def __init__(self) -> None:
        self._targets: dict[str, Target] = {}

    def create_target(
        self,
        name: str,
        gene: str = "",
        sequence: str = "",
        disease_associations: list[str] | None = None,
        **kwargs,
    ) -> Target:
        """Create and register a new drug target."""
        target = Target(
            name=name,
            gene=gene,
            sequence=sequence,
            disease_associations=disease_associations or [],
            **kwargs,
        )
        target.druggability_score = self.assess_druggability(target)
        self._targets[name] = target
        logger.info("Target created: %s (druggability=%.2f)", name, target.druggability_score)
        return target

    def assess_druggability(self, target: Target) -> float:
        """Estimate druggability score (0-1) based on target family and properties."""
        score = 0.5  # baseline

        # Check target family
        name_lower = target.name.lower() + " " + target.gene.lower()
        for family, info in TARGET_FAMILIES.items():
            if any(kw in name_lower for kw in info["keywords"]):
                score = max(score, info["druggability"])
                target.metadata["family"] = family
                break

        # Bonus for having sequence (enables structure-based design)
        if target.sequence:
            score = min(1.0, score + 0.05)

        # Bonus for disease associations
        if len(target.disease_associations) >= 2:
            score = min(1.0, score + 0.05)

        return round(score, 3)

    def validate_sequence(self, sequence: str) -> dict:
        """Validate a protein sequence."""
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        seq_upper = sequence.upper().replace(" ", "").replace("\n", "")
        invalid = set(seq_upper) - valid_aa
        return {
            "valid": len(invalid) == 0,
            "length": len(seq_upper),
            "invalid_chars": list(invalid),
            "sequence": seq_upper,
        }

    def get_target(self, name: str) -> Target | None:
        """Retrieve a registered target."""
        return self._targets.get(name)

    def list_targets(self) -> list[Target]:
        """List all registered targets."""
        return list(self._targets.values())
