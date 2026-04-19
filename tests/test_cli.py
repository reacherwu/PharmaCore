"""Tests for CLI interface."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from pharmacore.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_version(runner):
    """Test --version flag."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "pharmacore" in result.output.lower()


def test_cli_info(runner):
    """Test info command."""
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "Device:" in result.output


def test_cli_analyze(runner):
    """Test analyze command with aspirin."""
    result = runner.invoke(main, ["analyze", "CC(=O)Oc1ccccc1C(=O)O"])
    assert result.exit_code == 0
    assert "Molecule:" in result.output


def test_cli_analyze_with_flags(runner):
    """Test analyze with drug-likeness flag."""
    result = runner.invoke(main, ["analyze", "CC(=O)Oc1ccccc1C(=O)O", "-l"])
    assert result.exit_code == 0
    assert "Lipinski:" in result.output


def test_cli_generate(runner):
    """Test generate command."""
    result = runner.invoke(main, ["generate", "-n", "3", "--seed", "42"])
    assert result.exit_code == 0
    assert "Generated" in result.output
