"""Tests for molecular docking module."""
from __future__ import annotations

import pytest

from pharmacore.docking.scoring import DockingScorer
from pharmacore.core.types import DockingResult, Molecule, Protein


def _make_result(score: float) -> DockingResult:
    return DockingResult(
        molecule=Molecule(smiles="CC(=O)Oc1ccccc1C(=O)O"),
        protein=Protein(sequence="ACDEF"),
        score=score,
        confidence=min(1.0, abs(score) / 12.0),
    )


def test_vina_docker_init():
    """Test VinaDocker can be instantiated."""
    from pharmacore.docking.vina import VinaDocker
    docker = VinaDocker()
    assert docker.exhaustiveness == 8
    assert docker.n_poses == 5


def test_parse_vina_output():
    """Test parsing of Vina output."""
    from pharmacore.docking.vina import VinaDocker
    docker = VinaDocker()

    sample_output = """
Scoring function : vina
Rigid receptor: receptor.pdbqt
Ligand: ligand.pdbqt
Grid center: X 10.0 Y 20.0 Z 30.0
Grid size  : X 20.0 Y 20.0 Z 20.0

-----+------------+----------+----------
 mode |   affinity | dist from best mode
      | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -8.5          0.0          0.0
   2       -7.2          2.1          4.3
   3       -6.8          3.5          6.1
"""
    results = docker._parse_vina_output(sample_output)
    assert len(results) == 3
    assert results[0] == (-8.5, 1)
    assert results[1] == (-7.2, 2)
    assert results[2] == (-6.8, 3)


def test_scoring_rank():
    """Test ranking docking results."""
    scorer = DockingScorer()
    results = [_make_result(-5.0), _make_result(-8.5), _make_result(-6.2)]
    ranked = scorer.rank_results(results)
    assert ranked[0].score == -8.5
    assert ranked[-1].score == -5.0


def test_scoring_filter():
    """Test filtering docking results by threshold."""
    scorer = DockingScorer()
    results = [_make_result(-5.0), _make_result(-8.5), _make_result(-6.2)]
    filtered = scorer.filter_results(results, threshold=-6.0)
    assert len(filtered) == 2
    assert all(r.score <= -6.0 for r in filtered)


def test_score_pose():
    """Test scoring a single pose."""
    scorer = DockingScorer()
    result = _make_result(-7.5)
    scores = scorer.score_pose(result)
    assert "binding_affinity" in scores
    assert scores["binding_affinity"] == -7.5
    assert "ligand_efficiency" in scores
