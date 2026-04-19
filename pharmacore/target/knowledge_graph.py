"""Disease-target knowledge graph for target prioritization."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiseaseNode:
    """A disease in the knowledge graph."""
    name: str
    aliases: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    pathways: list[str] = field(default_factory=list)


@dataclass
class Edge:
    """Relationship between entities."""
    source: str
    target: str
    relation: str
    confidence: float = 1.0
    evidence: str = ""


class KnowledgeGraph:
    """Simple in-memory disease-target knowledge graph.

    For production, this would connect to a graph database (Neo4j)
    or use pre-built knowledge bases (OpenTargets, DisGeNET).
    """

    # Built-in disease-target associations (curated from literature)
    BUILTIN_ASSOCIATIONS = {
        "breast_cancer": {
            "targets": ["BRCA1", "BRCA2", "HER2", "ESR1", "CDK4", "CDK6", "PIK3CA", "PARP1"],
            "pathways": ["PI3K/AKT", "MAPK/ERK", "DNA repair", "Cell cycle"],
        },
        "lung_cancer": {
            "targets": ["EGFR", "ALK", "KRAS", "ROS1", "BRAF", "MET", "PD-L1"],
            "pathways": ["EGFR signaling", "RAS/MAPK", "Immune checkpoint"],
        },
        "alzheimers": {
            "targets": ["APP", "BACE1", "MAPT", "APOE", "TREM2", "GSK3B"],
            "pathways": ["Amyloid cascade", "Tau phosphorylation", "Neuroinflammation"],
        },
        "diabetes_t2": {
            "targets": ["INSR", "GLP1R", "SGLT2", "DPP4", "PPARG", "GCK"],
            "pathways": ["Insulin signaling", "Glucose metabolism", "Incretin"],
        },
        "rheumatoid_arthritis": {
            "targets": ["TNF", "IL6", "JAK1", "JAK3", "BTK", "CD20"],
            "pathways": ["TNF signaling", "JAK/STAT", "B-cell activation"],
        },
        "covid19": {
            "targets": ["ACE2", "TMPRSS2", "3CLpro", "RdRp", "PLpro", "Spike"],
            "pathways": ["Viral entry", "Viral replication", "Host immune response"],
        },
    }

    def __init__(self) -> None:
        self._diseases: dict[str, DiseaseNode] = {}
        self._edges: list[Edge] = []
        self._load_builtin()

    def _load_builtin(self) -> None:
        """Load built-in disease-target associations."""
        for disease_id, info in self.BUILTIN_ASSOCIATIONS.items():
            node = DiseaseNode(
                name=disease_id,
                targets=info["targets"],
                pathways=info["pathways"],
            )
            self._diseases[disease_id] = node
            for target in info["targets"]:
                self._edges.append(Edge(
                    source=disease_id,
                    target=target,
                    relation="associated_with",
                    confidence=0.8,
                ))

    def get_targets_for_disease(self, disease: str) -> list[str]:
        """Get known drug targets for a disease."""
        disease_key = disease.lower().replace(" ", "_").replace("'", "")
        node = self._diseases.get(disease_key)
        if node:
            return node.targets
        # Fuzzy match
        for key, n in self._diseases.items():
            if disease.lower() in key or key in disease.lower():
                return n.targets
        return []

    def get_diseases_for_target(self, target_gene: str) -> list[str]:
        """Get diseases associated with a target gene."""
        diseases = []
        for disease_id, node in self._diseases.items():
            if target_gene.upper() in [t.upper() for t in node.targets]:
                diseases.append(disease_id)
        return diseases

    def get_pathways(self, disease: str) -> list[str]:
        """Get pathways involved in a disease."""
        disease_key = disease.lower().replace(" ", "_").replace("'", "")
        node = self._diseases.get(disease_key)
        return node.pathways if node else []

    def add_association(
        self, disease: str, target: str, confidence: float = 0.5, evidence: str = ""
    ) -> None:
        """Add a disease-target association."""
        disease_key = disease.lower().replace(" ", "_")
        if disease_key not in self._diseases:
            self._diseases[disease_key] = DiseaseNode(name=disease_key)
        self._diseases[disease_key].targets.append(target)
        self._edges.append(Edge(
            source=disease_key, target=target,
            relation="associated_with", confidence=confidence, evidence=evidence,
        ))

    def list_diseases(self) -> list[str]:
        """List all diseases in the knowledge graph."""
        return list(self._diseases.keys())
