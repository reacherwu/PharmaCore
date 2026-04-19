#!/usr/bin/env python3
"""PharmaCore benchmark suite.

Measures performance of core modules on Apple Silicon.
Run: python benchmarks/run_benchmarks.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_device_detection() -> dict:
    """Benchmark device detection speed."""
    from pharmacore.core.device import DeviceManager
    dm = DeviceManager()

    t0 = time.perf_counter()
    for _ in range(100):
        dm.detect_device()
    elapsed = time.perf_counter() - t0

    return {
        "name": "device_detection",
        "iterations": 100,
        "total_seconds": round(elapsed, 4),
        "per_call_ms": round(elapsed / 100 * 1000, 3),
        "device": dm.detect_device(),
    }


def benchmark_smiles_parsing() -> dict:
    """Benchmark SMILES parsing throughput."""
    from pharmacore.utils.chemistry import parse_smiles

    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # ibuprofen
        "OC(=O)C1=CC=CC=C1O",  # salicylic acid
    ] * 20  # 100 molecules

    t0 = time.perf_counter()
    for smi in test_smiles:
        parse_smiles(smi)
    elapsed = time.perf_counter() - t0

    return {
        "name": "smiles_parsing",
        "molecules": len(test_smiles),
        "total_seconds": round(elapsed, 4),
        "molecules_per_second": round(len(test_smiles) / elapsed, 1),
    }


def benchmark_descriptors() -> dict:
    """Benchmark molecular descriptor computation."""
    from pharmacore.utils.chemistry import compute_descriptors, parse_smiles

    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    n = 500

    t0 = time.perf_counter()
    for _ in range(n):
        compute_descriptors(mol)
    elapsed = time.perf_counter() - t0

    return {
        "name": "descriptor_computation",
        "iterations": n,
        "total_seconds": round(elapsed, 4),
        "per_molecule_ms": round(elapsed / n * 1000, 3),
    }


def benchmark_fingerprints() -> dict:
    """Benchmark fingerprint generation."""
    from pharmacore.utils.chemistry import compute_fingerprint, parse_smiles

    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    n = 500

    results = {}
    for fp_type in ["morgan", "maccs", "rdkit"]:
        t0 = time.perf_counter()
        for _ in range(n):
            compute_fingerprint(mol, fp_type=fp_type)
        elapsed = time.perf_counter() - t0
        results[fp_type] = {
            "iterations": n,
            "total_seconds": round(elapsed, 4),
            "per_call_ms": round(elapsed / n * 1000, 3),
        }

    return {"name": "fingerprint_generation", "results": results}


def benchmark_drug_likeness() -> dict:
    """Benchmark drug-likeness filtering."""
    from pharmacore.utils.chemistry import check_drug_likeness, parse_smiles

    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    n = 500

    t0 = time.perf_counter()
    for _ in range(n):
        check_drug_likeness(mol)
    elapsed = time.perf_counter() - t0

    return {
        "name": "drug_likeness_check",
        "iterations": n,
        "total_seconds": round(elapsed, 4),
        "per_molecule_ms": round(elapsed / n * 1000, 3),
    }


def benchmark_admet() -> dict:
    """Benchmark ADMET prediction."""
    from pharmacore.admet.predictor import ADMETPredictor

    predictor = ADMETPredictor()
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "OC(=O)C1=CC=CC=C1O",
        "C1CCCCC1N",
    ]

    t0 = time.perf_counter()
    for smi in smiles_list * 20:  # 100 predictions
        predictor.predict(smi)
    elapsed = time.perf_counter() - t0

    return {
        "name": "admet_prediction",
        "molecules": 100,
        "total_seconds": round(elapsed, 4),
        "per_molecule_ms": round(elapsed / 100 * 1000, 3),
    }


def benchmark_generation() -> dict:
    """Benchmark molecular generation."""
    from pharmacore.generation.diffusion import MolecularGenerator

    gen = MolecularGenerator(seed=42)

    t0 = time.perf_counter()
    molecules = gen.generate(n_molecules=50, max_attempts=500)
    elapsed = time.perf_counter() - t0

    return {
        "name": "molecule_generation",
        "requested": 50,
        "generated": len(molecules),
        "total_seconds": round(elapsed, 4),
        "per_molecule_ms": round(elapsed / max(len(molecules), 1) * 1000, 3),
    }


def benchmark_pipeline() -> dict:
    """Benchmark end-to-end pipeline (generation + ADMET)."""
    from pharmacore.pipeline.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()

    t0 = time.perf_counter()
    result = orch.run(
        target_name="EGFR",
        n_molecules=20,
        stages=["target", "generation", "admet", "ranking"],
    )
    elapsed = time.perf_counter() - t0

    return {
        "name": "pipeline_e2e",
        "stages": result.metadata.get("stages_run", []),
        "molecules": len(result.molecules),
        "total_seconds": round(elapsed, 4),
    }


def get_system_info() -> dict:
    """Collect system information."""
    import torch
    from pharmacore.core.device import DeviceManager

    dm = DeviceManager()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": dm.detect_device(),
        "mps_available": torch.backends.mps.is_available(),
        "mlx_available": dm.has_mlx(),
        **dm.device_info(),
    }


def main() -> None:
    """Run all benchmarks."""
    print("=" * 60)
    print("PharmaCore Benchmark Suite")
    print("=" * 60)

    sys_info = get_system_info()
    print(f"\nSystem: {sys_info.get('chip', 'Unknown')} | "
          f"{sys_info.get('memory_gb', '?')} GB | "
          f"Python {sys_info['python']} | "
          f"PyTorch {sys_info['torch']}")
    print(f"Device: {sys_info['device']} | MPS: {sys_info['mps_available']}\n")

    benchmarks = [
        benchmark_device_detection,
        benchmark_smiles_parsing,
        benchmark_descriptors,
        benchmark_fingerprints,
        benchmark_drug_likeness,
        benchmark_admet,
        benchmark_generation,
        benchmark_pipeline,
    ]

    results = {"system": sys_info, "benchmarks": []}

    for bench_fn in benchmarks:
        name = bench_fn.__name__.replace("benchmark_", "")
        print(f"Running {name}...", end=" ", flush=True)
        try:
            result = bench_fn()
            results["benchmarks"].append(result)
            # Print key metric
            if "per_molecule_ms" in result:
                print(f"{result['per_molecule_ms']} ms/molecule")
            elif "per_call_ms" in result:
                print(f"{result['per_call_ms']} ms/call")
            elif "molecules_per_second" in result:
                print(f"{result['molecules_per_second']} mol/s")
            else:
                print(f"{result.get('total_seconds', '?')}s")
        except Exception as e:
            print(f"FAILED: {e}")
            results["benchmarks"].append({"name": name, "error": str(e)})

    # Save results
    output_path = Path(__file__).parent / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
