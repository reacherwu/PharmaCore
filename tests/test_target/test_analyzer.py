"""Tests for target analysis module."""
from __future__ import annotations

import pytest

from pharmacore.target.analyzer import TargetAnalyzer, Target


def test_create_target():
    """Test creating a drug target."""
    analyzer = TargetAnalyzer()
    target = analyzer.create_target(
        name="EGFR kinase",
        gene="EGFR",
        sequence="MRPSGTAGAALLALLAALCPASRALEEKKVC",
        disease_associations=["lung_cancer", "breast_cancer"],
    )
    assert isinstance(target, Target)
    assert target.name == "EGFR kinase"
    assert target.druggability_score > 0


def test_druggability_kinase():
    """Test that kinases get high druggability scores."""
    analyzer = TargetAnalyzer()
    target = analyzer.create_target(name="CDK4 kinase", gene="CDK4")
    assert target.druggability_score >= 0.8


def test_druggability_gpcr():
    """Test GPCR druggability."""
    analyzer = TargetAnalyzer()
    target = analyzer.create_target(name="GLP1 receptor", gene="GLP1R")
    assert target.druggability_score >= 0.8


def test_validate_sequence():
    """Test protein sequence validation."""
    analyzer = TargetAnalyzer()
    result = analyzer.validate_sequence("ACDEFGHIKLMNPQRSTVWY")
    assert result["valid"] is True
    assert result["length"] == 20


def test_validate_sequence_invalid():
    """Test invalid sequence detection."""
    analyzer = TargetAnalyzer()
    result = analyzer.validate_sequence("ACDEFXZ123")
    assert result["valid"] is False
    assert len(result["invalid_chars"]) > 0


def test_list_targets():
    """Test listing registered targets."""
    analyzer = TargetAnalyzer()
    analyzer.create_target(name="target1")
    analyzer.create_target(name="target2")
    targets = analyzer.list_targets()
    assert len(targets) == 2
