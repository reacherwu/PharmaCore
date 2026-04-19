"""Tests for ADMET predictor and toxicity screener."""
from __future__ import annotations

import pytest

from pharmacore.admet.predictor import ADMETPredictor
from pharmacore.admet.toxicity import ToxicityScreener
from pharmacore.core.types import ADMETProfile, Molecule


@pytest.fixture
def predictor() -> ADMETPredictor:
    return ADMETPredictor(device="cpu")


@pytest.fixture
def screener() -> ToxicityScreener:
    return ToxicityScreener()


# ------------------------------------------------------------------
# ADMETPredictor tests
# ------------------------------------------------------------------

class TestADMETPredictor:

    def test_predict_aspirin(self, predictor: ADMETPredictor) -> None:
        """Aspirin should have good oral bioavailability and low toxicity."""
        profile = predictor.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

        # Good oral bioavailability (small, drug-like molecule)
        assert profile.absorption.oral_bioavailability_pass is True
        assert profile.absorption.oral_bioavailability >= 0.5

        # Not a P-gp substrate (MW ~180, low HBD)
        assert profile.absorption.pgp_substrate is False

        # Low mutagenicity risk
        assert profile.toxicity.ames_mutagenicity is False

        # No hERG risk (low logP, no basic nitrogen)
        assert profile.toxicity.herg_inhibition is False

    def test_predict_known_toxic(self, predictor: ADMETPredictor) -> None:
        """Nitrobenzene should trigger mutagenic alerts."""
        profile = predictor.predict("c1ccc([N+](=O)[O-])cc1")  # nitrobenzene

        assert profile.toxicity.ames_mutagenicity is True
        assert "aromatic_nitro" in profile.toxicity.mutagenic_alerts

    def test_admet_profile_structure(self, predictor: ADMETPredictor) -> None:
        """All 5 ADMET categories must be present."""
        profile = predictor.predict("c1ccccc1")  # benzene

        assert isinstance(profile, ADMETProfile)
        assert profile.absorption is not None
        assert profile.distribution is not None
        assert profile.metabolism is not None
        assert profile.excretion is not None
        assert profile.toxicity is not None

        # Check types
        assert isinstance(profile.absorption.oral_bioavailability, float)
        assert isinstance(profile.distribution.bbb_penetration, float)
        assert isinstance(profile.metabolism.cyp_inhibition, dict)
        assert isinstance(profile.excretion.half_life_estimate, float)
        assert isinstance(profile.toxicity.ames_mutagenicity, bool)

    def test_predict_from_molecule_object(self, predictor: ADMETPredictor) -> None:
        """Should accept a Molecule object as well as SMILES string."""
        mol = Molecule.from_smiles("CCO")  # ethanol
        profile = predictor.predict(mol)
        assert isinstance(profile, ADMETProfile)
        assert profile.molecule_smiles == "CCO"

    def test_bbb_penetration_small_lipophilic(self, predictor: ADMETPredictor) -> None:
        """Small lipophilic molecule should penetrate BBB."""
        profile = predictor.predict("c1ccccc1")  # benzene: MW ~78, logP ~1.6
        assert profile.distribution.bbb_penetration_pass is True

    def test_pgp_substrate_large_molecule(self, predictor: ADMETPredictor) -> None:
        """Large molecule with many HBDs should be flagged as P-gp substrate."""
        # Erythromycin-like: MW > 400, HBD > 3
        erythromycin = "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O"
        profile = predictor.predict(erythromycin)
        assert profile.absorption.pgp_substrate is True


# ------------------------------------------------------------------
# ToxicityScreener tests
# ------------------------------------------------------------------

class TestToxicityScreener:

    def test_clean_molecule(self, screener: ToxicityScreener) -> None:
        """Simple drug-like molecule should have few or no alerts."""
        result = screener.screen("c1ccccc1")  # benzene
        assert result.is_brenk is False

    def test_pains_catechol(self, screener: ToxicityScreener) -> None:
        """Catechol should trigger PAINS alert."""
        result = screener.screen("c1cc(O)c(O)cc1")  # catechol
        assert result.is_pains is True

    def test_brenk_nitro(self, screener: ToxicityScreener) -> None:
        """Nitro group should trigger Brenk alert."""
        result = screener.screen("c1ccc([N+](=O)[O-])cc1")  # nitrobenzene
        assert result.is_brenk is True
        assert "nitro" in result.brenk_alerts

    def test_brenk_epoxide(self, screener: ToxicityScreener) -> None:
        """Epoxide should trigger Brenk alert."""
        result = screener.screen("C1OC1CC")  # propylene oxide
        assert result.is_brenk is True
        assert "epoxide" in result.brenk_alerts

    def test_screening_result_structure(self, screener: ToxicityScreener) -> None:
        """ScreeningResult should have expected fields."""
        result = screener.screen("CCO")
        assert hasattr(result, "pains_alerts")
        assert hasattr(result, "brenk_alerts")
        assert hasattr(result, "is_pains")
        assert hasattr(result, "is_brenk")
        assert hasattr(result, "total_alerts")
        assert isinstance(result.total_alerts, int)
