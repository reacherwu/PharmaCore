"""End-to-end drug discovery pipeline orchestrator."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pharmacore.core.config import get_config
from pharmacore.core.types import (
    ADMETProfile,
    DockingResult,
    Molecule,
    PipelineResult,
    Protein,
)

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Stages in the drug discovery pipeline."""
    TARGET = "target"
    PROTEIN = "protein"
    GENERATION = "generation"
    DOCKING = "docking"
    ADMET = "admet"
    RANKING = "ranking"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool
    data: Any = None
    duration_seconds: float = 0.0
    error: str = ""


class PipelineOrchestrator:
    """Orchestrate end-to-end drug discovery workflows.

    Connects target analysis -> protein structure -> molecular generation
    -> docking -> ADMET prediction into a unified pipeline.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._stages: list[PipelineStage] = list(PipelineStage)
        self._results: dict[PipelineStage, StageResult] = {}
        self._callbacks: dict[str, list] = {}

    def run(
        self,
        target_name: str,
        target_sequence: str = "",
        protein_pdb: str | Path | None = None,
        n_molecules: int = 10,
        docking_center: tuple[float, float, float] | None = None,
        stages: list[str] | None = None,
    ) -> PipelineResult:
        """Run the full drug discovery pipeline.

        Args:
            target_name: Name of the drug target.
            target_sequence: Protein sequence (for embedding/folding).
            protein_pdb: Path to protein PDB file (for docking).
            n_molecules: Number of molecules to generate.
            docking_center: Center of docking box (x, y, z).
            stages: Specific stages to run (default: all).

        Returns:
            PipelineResult with all outputs.
        """
        active_stages = (
            [PipelineStage(s) for s in stages] if stages else self._stages
        )
        start = time.time()
        molecules: list[Molecule] = []
        docking_results: list[DockingResult] = []
        admet_profiles: list[ADMETProfile] = []
        protein = None
        metadata: dict[str, Any] = {"target": target_name, "stages_run": []}

        logger.info("Starting pipeline for target: %s", target_name)

        # Stage 1: Target Analysis
        if PipelineStage.TARGET in active_stages:
            result = self._run_target_stage(target_name, target_sequence)
            self._results[PipelineStage.TARGET] = result
            metadata["stages_run"].append("target")
            if result.data:
                metadata["target_info"] = result.data

        # Stage 2: Protein Embedding/Structure
        if PipelineStage.PROTEIN in active_stages and target_sequence:
            result = self._run_protein_stage(target_sequence, target_name)
            self._results[PipelineStage.PROTEIN] = result
            metadata["stages_run"].append("protein")
            if result.success and result.data:
                protein = result.data

        # Stage 3: Molecular Generation
        if PipelineStage.GENERATION in active_stages:
            result = self._run_generation_stage(n_molecules)
            self._results[PipelineStage.GENERATION] = result
            metadata["stages_run"].append("generation")
            if result.success:
                molecules = result.data or []

        # Stage 4: Docking
        if PipelineStage.DOCKING in active_stages and protein_pdb and docking_center:
            result = self._run_docking_stage(molecules, protein_pdb, docking_center)
            self._results[PipelineStage.DOCKING] = result
            metadata["stages_run"].append("docking")
            if result.success:
                docking_results = result.data or []

        # Stage 5: ADMET
        if PipelineStage.ADMET in active_stages and molecules:
            result = self._run_admet_stage(molecules)
            self._results[PipelineStage.ADMET] = result
            metadata["stages_run"].append("admet")
            if result.success:
                admet_profiles = result.data or []

        # Stage 6: Ranking
        if PipelineStage.RANKING in active_stages and molecules:
            ranked = self._rank_molecules(molecules, docking_results, admet_profiles)
            molecules = ranked
            metadata["stages_run"].append("ranking")

        elapsed = time.time() - start
        metadata["total_duration_seconds"] = round(elapsed, 2)
        logger.info("Pipeline completed in %.1fs", elapsed)

        return PipelineResult(
            target=target_name,
            molecules=molecules,
            docking_results=docking_results,
            admet_profiles=admet_profiles,
            metadata=metadata,
        )

    def _run_target_stage(self, name: str, sequence: str) -> StageResult:
        """Run target analysis."""
        t0 = time.time()
        try:
            from pharmacore.target.analyzer import TargetAnalyzer
            analyzer = TargetAnalyzer()
            target = analyzer.create_target(name=name, sequence=sequence)
            info = {
                "name": target.name,
                "druggability": target.druggability_score,
                "family": target.metadata.get("family", "unknown"),
            }
            return StageResult(PipelineStage.TARGET, True, info, time.time() - t0)
        except Exception as e:
            logger.error("Target stage failed: %s", e)
            return StageResult(PipelineStage.TARGET, False, error=str(e), duration_seconds=time.time() - t0)

    def _run_protein_stage(self, sequence: str, name: str) -> StageResult:
        """Run protein embedding."""
        t0 = time.time()
        try:
            from pharmacore.protein.esm import ESMEmbedder
            embedder = ESMEmbedder()
            protein = embedder.get_protein(sequence, name=name)
            return StageResult(PipelineStage.PROTEIN, True, protein, time.time() - t0)
        except ImportError:
            logger.warning("ESM not installed, skipping protein embedding")
            protein = Protein(sequence=sequence, name=name)
            return StageResult(PipelineStage.PROTEIN, True, protein, time.time() - t0)
        except Exception as e:
            logger.error("Protein stage failed: %s", e)
            return StageResult(PipelineStage.PROTEIN, False, error=str(e), duration_seconds=time.time() - t0)

    def _run_generation_stage(self, n_molecules: int) -> StageResult:
        """Run molecular generation."""
        t0 = time.time()
        try:
            from pharmacore.generation.diffusion import MolecularGenerator
            gen = MolecularGenerator(seed=42)
            molecules = gen.generate(n_molecules=n_molecules)
            return StageResult(PipelineStage.GENERATION, True, molecules, time.time() - t0)
        except Exception as e:
            logger.error("Generation stage failed: %s", e)
            return StageResult(PipelineStage.GENERATION, False, error=str(e), duration_seconds=time.time() - t0)

    def _run_docking_stage(
        self, molecules: list[Molecule], protein_pdb: str | Path, center: tuple
    ) -> StageResult:
        """Run molecular docking."""
        t0 = time.time()
        try:
            from pharmacore.docking.vina import VinaDocker
            docker = VinaDocker()
            if not docker.is_available:
                logger.warning("Vina not installed, skipping docking")
                return StageResult(PipelineStage.DOCKING, True, [], time.time() - t0)

            all_results = []
            for mol in molecules:
                try:
                    results = docker.dock(mol, protein_pdb, center)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning("Docking failed for %s: %s", mol.smiles, e)
            return StageResult(PipelineStage.DOCKING, True, all_results, time.time() - t0)
        except Exception as e:
            logger.error("Docking stage failed: %s", e)
            return StageResult(PipelineStage.DOCKING, False, error=str(e), duration_seconds=time.time() - t0)

    def _run_admet_stage(self, molecules: list[Molecule]) -> StageResult:
        """Run ADMET prediction."""
        t0 = time.time()
        try:
            from pharmacore.admet.predictor import ADMETPredictor
            predictor = ADMETPredictor()
            profiles = [predictor.predict(mol) for mol in molecules]
            return StageResult(PipelineStage.ADMET, True, profiles, time.time() - t0)
        except Exception as e:
            logger.error("ADMET stage failed: %s", e)
            return StageResult(PipelineStage.ADMET, False, error=str(e), duration_seconds=time.time() - t0)

    def _rank_molecules(
        self,
        molecules: list[Molecule],
        docking_results: list[DockingResult],
        admet_profiles: list[ADMETProfile],
    ) -> list[Molecule]:
        """Rank molecules by composite score."""
        scored = []
        for i, mol in enumerate(molecules):
            score = 0.0

            # ADMET contribution
            if i < len(admet_profiles):
                profile = admet_profiles[i]
                if hasattr(profile, "absorption") and isinstance(profile.absorption, dict):
                    score += profile.absorption.get("oral_bioavailability", 0) * 0.3
                if hasattr(profile, "toxicity") and isinstance(profile.toxicity, dict):
                    tox = profile.toxicity.get("overall_risk", "medium")
                    score += {"low": 0.3, "medium": 0.1, "high": 0.0}.get(tox, 0.1)

            # Docking contribution
            dock_scores = [r.score for r in docking_results if r.molecule.smiles == mol.smiles]
            if dock_scores:
                best = min(dock_scores)
                score += min(0.4, abs(best) / 30.0)

            mol.properties["composite_score"] = round(score, 4)
            scored.append(mol)

        scored.sort(key=lambda m: m.properties.get("composite_score", 0), reverse=True)
        return scored
