#!/usr/bin/env python3
"""PharmaCore Quick Start Example.

Demonstrates the core drug discovery workflow:
1. Analyze a known drug (Aspirin)
2. Generate new drug candidates
3. Run ADMET profiling
4. Execute the full pipeline
"""
from pharmacore.admet.predictor import ADMETPredictor, ToxicityScreener
from pharmacore.core.device import DeviceManager
from pharmacore.generation.diffusion import MolecularGenerator
from pharmacore.pipeline.orchestrator import PipelineOrchestrator
from pharmacore.utils.chemistry import (
    check_drug_likeness,
    compute_descriptors,
    compute_fingerprint,
    parse_smiles,
)


def main():
    print("=" * 60)
    print("PharmaCore — Quick Start Example")
    print("=" * 60)

    # --- System Info ---
    dm = DeviceManager()
    info = dm.device_info()
    print(f"\nDevice: {info['device']} | {info.get('chip', 'Unknown')} | {info['memory_gb']} GB")

    # --- Step 1: Analyze Aspirin ---
    print("\n--- Step 1: Analyze Aspirin ---")
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    desc = compute_descriptors(aspirin)
    print(f"  MW: {desc['molecular_weight']:.1f}")
    print(f"  LogP: {desc['logp']:.2f}")
    print(f"  HBA: {desc['hba']}, HBD: {desc['hbd']}")
    print(f"  TPSA: {desc['tpsa']:.1f}")

    dl = check_drug_likeness(aspirin)
    print(f"  Lipinski: {'PASS' if dl['lipinski_pass'] else 'FAIL'}")
    print(f"  Veber: {'PASS' if dl['veber_pass'] else 'FAIL'}")

    # --- Step 2: ADMET Profiling ---
    print("\n--- Step 2: ADMET Profiling ---")
    predictor = ADMETPredictor()
    profile = predictor.predict("CC(=O)Oc1ccccc1C(=O)O")
    print(f"  Absorption: {profile.absorption_score:.2f}")
    print(f"  BBB Penetration: {'Yes' if profile.bbb_penetration else 'No'}")
    print(f"  hERG Liability: {'Yes' if profile.herg_liability else 'No'}")
    print(f"  Overall: {profile.overall_score:.2f}")

    # --- Step 3: Toxicity Screening ---
    print("\n--- Step 3: Toxicity Screening ---")
    screener = ToxicityScreener()
    tox = screener.screen("CC(=O)Oc1ccccc1C(=O)O")
    print(f"  Clean: {tox['is_clean']}")
    print(f"  PAINS alerts: {len(tox['pains_alerts'])}")
    print(f"  Brenk alerts: {len(tox['brenk_alerts'])}")

    # --- Step 4: Generate Candidates ---
    print("\n--- Step 4: Generate Drug Candidates ---")
    gen = MolecularGenerator(seed=42)
    molecules = gen.generate(n_molecules=10, max_attempts=200)
    print(f"  Generated {len(molecules)} valid drug-like molecules:")
    for mol in molecules[:5]:
        print(f"    {mol.name}: {mol.smiles}")

    # --- Step 5: Full Pipeline ---
    print("\n--- Step 5: End-to-End Pipeline ---")
    pipeline = PipelineOrchestrator()
    result = pipeline.run(
        target_name="EGFR",
        n_molecules=20,
        stages=["target", "generation", "admet", "ranking"],
    )
    print(f"  Stages: {result.metadata.get('stages_run', [])}")
    print(f"  Candidates: {len(result.molecules)}")
    if result.molecules:
        top = result.molecules[0]
        print(f"  Top hit: {top.smiles} (score: {top.properties.get('score', 'N/A')})")

    print("\n" + "=" * 60)
    print("Pipeline complete. See docs/ for advanced usage.")
    print("=" * 60)


if __name__ == "__main__":
    main()
