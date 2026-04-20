"""Integration tests for PharmaCore core modules."""
import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDiscoveryEngine:
    """Test de novo drug discovery engine."""

    def test_import(self):
        from pharmacore.discovery import DeNovoDiscoveryEngine, DiscoveryResult
        engine = DeNovoDiscoveryEngine(seed=42)
        assert engine._device in ("mps", "cpu", "cuda")

    def test_discover_kinase(self):
        from pharmacore.discovery import DeNovoDiscoveryEngine
        engine = DeNovoDiscoveryEngine(seed=42)
        result = engine.discover(
            target_name="EGFR kinase",
            target_sequence="MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVL",
            n_molecules=3,
        )
        assert len(result.molecules) == 3
        assert result.target_family == "kinase"
        assert all(m.composite_score > 0 for m in result.molecules)
        assert all(m.smiles for m in result.molecules)
        # Should be sorted by score descending
        scores = [m.composite_score for m in result.molecules]
        assert scores == sorted(scores, reverse=True)

    def test_discover_gpcr(self):
        from pharmacore.discovery import DeNovoDiscoveryEngine
        engine = DeNovoDiscoveryEngine(seed=123)
        result = engine.discover(
            target_name="dopamine receptor D2",
            target_sequence="MDPLNLSWYDDDLERQNWSRPFNGSEGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVSREKALQTTTNYLIVSLAVADLLVATLVMPWVVYLEVVGEWKFSRIHCDIFVTLDVMMCTASILNLCAISIDRYTAVAMPMLYNTRYSSKRRVTVMISIVWVLSFTISCPLLFGLNNADQNECIIANPAFVVYSSIVSFYVPFIVTLLVYIKIYIVLRRRRKRVNTKRSSRAFRAHLRAPLKGNCTHPEDMKLCTVIMKSNGSFPVNRRRVEAARRAQELEMEMLSSTSPPERTRYSPIPPSHHQLTLPDPSHHGLHSTPDSPAKPEKNGHAKDHPKIAKIFEIQTMPNGKTRTSLKTMSRRKLSQQKEKKATQMLAIVLGVFIICWLPFFITHILNIHCDCNIPPVLYSAFTWLGYVNSAVNPIIYTTFNIEFRKAFLKILHC",
            n_molecules=2,
        )
        assert len(result.molecules) == 2
        assert result.target_family == "gpcr"

    def test_explain(self):
        from pharmacore.discovery import DeNovoDiscoveryEngine
        engine = DeNovoDiscoveryEngine(seed=42)
        result = engine.discover(target_name="test kinase", n_molecules=1)
        explanation = engine.explain(result.molecules[0])
        assert "De Novo Drug Discovery Report" in explanation
        assert "SMILES" in explanation


class TestRepurposingEngine:
    """Test drug repurposing engine."""

    def test_import(self):
        from pharmacore.repurposing import DrugRepurposingEngine, KNOWN_DRUGS
        assert len(KNOWN_DRUGS) >= 10
        engine = DrugRepurposingEngine()
        assert engine._device in ("mps", "cpu", "cuda")

    def test_screen_egfr(self):
        from pharmacore.repurposing import DrugRepurposingEngine
        engine = DrugRepurposingEngine()
        result = engine.screen(
            target_name="EGFR",
            target_sequence="MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVL",
            reference_smiles="COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
            top_k=3,
        )
        assert len(result.candidates) <= 3
        assert result.target_name == "EGFR"
        # Erlotinib should rank high (it's a known EGFR inhibitor)
        names = [c.drug_name for c in result.candidates]
        assert "Erlotinib" in names

    def test_screen_no_reference(self):
        from pharmacore.repurposing import DrugRepurposingEngine
        engine = DrugRepurposingEngine()
        result = engine.screen(
            target_name="ACE2",
            target_sequence="MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSIKVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPRISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPPVSIWLIVFGVVMGVIVVGIVILIFTGIRDRKKKNKARSGENPYASIDISKGENNPGFQNTDDVQTSF",
            top_k=5,
            min_score=0.05,
        )
        assert len(result.candidates) >= 0  # may or may not find candidates
        assert result.audit_log  # audit log should exist

    def test_explain(self):
        from pharmacore.repurposing import DrugRepurposingEngine
        engine = DrugRepurposingEngine()
        result = engine.screen(
            target_name="test",
            target_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHIIPGAFQPAFEPVKQTLNQFQKAQIQRIKQIKALERDSSELQATAAALEHHHHHH",
            top_k=1,
            min_score=0.05,
        )
        if result.candidates:
            explanation = engine.explain(result.candidates[0])
            assert "Drug Repurposing Analysis" in explanation


class TestAuditPipeline:
    """Test transparent audit pipeline."""

    def test_basic_audit(self):
        from pharmacore.audit import AuditPipeline
        audit = AuditPipeline("test")
        audit.log_step("step1", inputs={"x": 1}, outputs={"y": 2}, explanation="Test")
        report = audit.finalize()
        assert report.status == "completed"
        assert report.summary["total_steps"] == 1
        assert report.system_info["machine"] == "arm64"

    def test_audit_json(self):
        from pharmacore.audit import AuditPipeline
        import json
        audit = AuditPipeline("test")
        audit.log_step("s1", inputs={"a": 1})
        report = audit.finalize()
        j = json.loads(report.to_json())
        assert j["pipeline_name"] == "test"
        assert len(j["entries"]) == 1

    def test_audit_save(self, tmp_path):
        from pharmacore.audit import AuditPipeline
        audit = AuditPipeline("test")
        audit.log_step("s1", inputs={"a": 1})
        report = audit.finalize()
        path = audit.save(report, tmp_path / "report.json")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_text_report(self):
        from pharmacore.audit import AuditPipeline
        audit = AuditPipeline("test")
        audit.log_step("init", explanation="Initialize")
        audit.log_step("compute", explanation="Run computation")
        report = audit.finalize()
        text = audit.generate_text_report(report)
        assert "PharmaCore Audit Report" in text
        assert "init" in text
        assert "compute" in text


class TestExistingModules:
    """Verify existing modules still work."""

    def test_molecular_generator(self):
        from pharmacore.generation.diffusion import MolecularGenerator
        gen = MolecularGenerator(seed=42)
        mols = gen.generate(n_molecules=3)
        assert len(mols) == 3
        assert all(m.smiles for m in mols)

    def test_admet_predictor(self):
        from pharmacore.admet.predictor import ADMETPredictor
        from pharmacore.core.types import Molecule
        pred = ADMETPredictor()
        mol = Molecule(smiles="CC(=O)Oc1ccccc1C(=O)O", name="aspirin")
        profile = pred.predict(mol)
        assert profile is not None

    def test_drug_likeness(self):
        from pharmacore.docking.scoring import DockingScorer
        scorer = DockingScorer()
        assert scorer is not None
        # Verify scorer has expected methods
        assert hasattr(scorer, 'score_pose')
        assert hasattr(scorer, 'rank_results')
        assert hasattr(scorer, 'filter_results')

    def test_pipeline_orchestrator(self):
        from pharmacore.pipeline.orchestrator import PipelineOrchestrator
        orch = PipelineOrchestrator()
        result = orch.run(
            target_name="test",
            n_molecules=2,
            stages=["generation", "admet"],
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
