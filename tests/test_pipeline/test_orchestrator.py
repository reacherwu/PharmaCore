"""Tests for pipeline orchestrator."""
from __future__ import annotations

import pytest

from pharmacore.pipeline.orchestrator import PipelineOrchestrator, PipelineStage


def test_orchestrator_init():
    """Test PipelineOrchestrator initialization."""
    orch = PipelineOrchestrator()
    assert len(orch._stages) == len(PipelineStage)


def test_run_generation_only():
    """Test running only the generation stage."""
    orch = PipelineOrchestrator()
    result = orch.run(
        target_name="EGFR",
        n_molecules=5,
        stages=["target", "generation", "admet", "ranking"],
    )
    assert result.target == "EGFR"
    assert len(result.molecules) > 0
    assert "total_duration_seconds" in result.metadata


def test_run_target_stage():
    """Test target analysis stage."""
    orch = PipelineOrchestrator()
    result = orch.run(
        target_name="EGFR kinase",
        target_sequence="MRPSGTAGAALLALLAALCPASRALEEKKVC",
        stages=["target"],
    )
    assert result.target == "EGFR kinase"
    assert "target" in result.metadata["stages_run"]


def test_pipeline_result_structure():
    """Test PipelineResult has correct structure."""
    orch = PipelineOrchestrator()
    result = orch.run(
        target_name="test",
        stages=["target", "generation"],
        n_molecules=3,
    )
    assert hasattr(result, "target")
    assert hasattr(result, "molecules")
    assert hasattr(result, "docking_results")
    assert hasattr(result, "admet_profiles")
    assert hasattr(result, "metadata")
