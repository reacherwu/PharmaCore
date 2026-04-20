#!/usr/bin/env python3
"""PharmaCore End-to-End Demo — showcases both core capabilities.

Run: python scripts/demo.py
"""
import sys
import time

def banner(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def main():
    t_total = time.time()
    
    banner("PharmaCore — Apple Silicon AI Drug Discovery Platform")
    print("Two core capabilities:")
    print("  1. De Novo Drug Discovery (target-driven molecular generation)")
    print("  2. Drug Repurposing (find new uses for existing drugs)")
    print()

    # ── Demo 1: De Novo Discovery ──────────────────────────────────────
    banner("DEMO 1: De Novo Drug Discovery for EGFR Kinase")
    
    from pharmacore.discovery import DeNovoDiscoveryEngine
    
    engine = DeNovoDiscoveryEngine(seed=42)
    print(f"Engine initialized on: {engine._device}")
    print("Target: EGFR (Epidermal Growth Factor Receptor)")
    print("Family: Kinase — key oncology target")
    print()
    
    t0 = time.time()
    result = engine.discover(
        target_name="EGFR kinase",
        target_sequence=(
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVL"
            "GNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSN"
            "YDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVISD"
        ),
        n_molecules=5,
    )
    elapsed = time.time() - t0
    
    print(f"Generated {len(result.molecules)} drug candidates in {elapsed:.1f}s\n")
    print(f"{'Rank':<5} {'Name':<20} {'Score':<8} {'Scaffold':<15} {'SMILES'}")
    print("-" * 90)
    for i, mol in enumerate(result.molecules, 1):
        print(f"{i:<5} {mol.name:<20} {mol.composite_score:<8.3f} {mol.scaffold_name:<15} {mol.smiles[:40]}")
    
    print(f"\nTop candidate explanation:")
    print(engine.explain(result.molecules[0]))

    # ── Demo 2: Drug Repurposing ───────────────────────────────────────
    banner("DEMO 2: Drug Repurposing Screen for EGFR")
    
    from pharmacore.repurposing import DrugRepurposingEngine
    
    engine2 = DrugRepurposingEngine()
    print(f"Screening 12 FDA-approved drugs against EGFR...")
    print()
    
    t0 = time.time()
    result2 = engine2.screen(
        target_name="EGFR",
        target_sequence=(
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVL"
            "GNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSN"
            "YDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVISD"
        ),
        reference_smiles="COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",  # erlotinib
        top_k=5,
    )
    elapsed2 = time.time() - t0
    
    print(f"Found {len(result2.candidates)} candidates in {elapsed2:.1f}s\n")
    print(f"{'Rank':<5} {'Drug':<15} {'Score':<8} {'Confidence':<12} {'Original Use'}")
    print("-" * 80)
    for i, c in enumerate(result2.candidates, 1):
        print(f"{i:<5} {c.drug_name:<15} {c.composite_score:<8.3f} {c.confidence:<12} {c.original_indication}")
    
    print(f"\nTop candidate explanation:")
    print(engine2.explain(result2.candidates[0]))

    # ── Demo 3: Audited Pipeline ───────────────────────────────────────
    banner("DEMO 3: Transparent Audit Trail")
    
    from pharmacore.audit import AuditedDiscovery
    
    ad = AuditedDiscovery()
    result3 = ad.run_discovery(
        target_name="BRAF_V600E",
        target_sequence="MAALSGGGGGGAEPGQALFNGDMEPEAGAGAGAAASSAADPAIPEEVWNIKQMIKLTQEHIEALLDKFGGEHNPPSIYLEAYEEYTSKLDALQQREQQLLESLGNGTDFSVSSSASMDTVTSSSSSSLSVLPSSLSVFQ",
        n_molecules=3,
        output_dir="/tmp/pharmacore_demo",
    )
    
    print(f"Audit report saved to: {result3['json_path']}")
    print(f"Text report saved to:  {result3['text_path']}")
    print(f"\nAudit report preview:")
    print(result3['text_report'][:500])

    # ── Summary ────────────────────────────────────────────────────────
    banner("Summary")
    total = time.time() - t_total
    print(f"Total demo time: {total:.1f}s")
    print(f"Device: Apple Silicon MPS")
    print(f"Models: ESM-2 35M (sparse 50%), ChemBERTa-zinc (sparse 50%)")
    print(f"All computation 100% local — no data left this machine.")
    print()
    print("PharmaCore — making drug discovery accessible on consumer hardware.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
