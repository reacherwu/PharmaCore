# PharmaCore

**Apple Silicon-native AI drug discovery platform.** Fully local, no cloud APIs, transparent and auditable.

Two core capabilities:
1. **De Novo Drug Discovery** — target-driven molecular generation using sparse AI models
2. **Drug Repurposing** — find new uses for existing FDA-approved drugs

## Why PharmaCore

- **100% Local** — all computation on your machine, no data leaves your device
- **Apple Silicon Optimized** — MPS acceleration on M1/M2/M3/M4 chips
- **Sparse AI Models** — 50% parameter reduction with 97%+ quality retention
- **Transparent** — every computation step logged with full audit trail
- **Fast** — sub-20ms protein inference, sub-5ms molecular inference on M4

## Quick Start

```bash
git clone https://github.com/reacherwu/PharmaCore.git
cd PharmaCore
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### De Novo Drug Discovery

```python
from pharmacore.discovery import DeNovoDiscoveryEngine

engine = DeNovoDiscoveryEngine(seed=42)
result = engine.discover(
    target_name="EGFR kinase",
    target_sequence="MRPSGTAGAALLALLAALCPASRA...",
    n_molecules=10,
)

for mol in result.molecules:
    print(f"{mol.name}: {mol.smiles} (score={mol.composite_score:.3f})")
```

### Drug Repurposing

```python
from pharmacore.repurposing import DrugRepurposingEngine

engine = DrugRepurposingEngine()
result = engine.screen(
    target_name="EGFR",
    target_sequence="MRPSGTAGAALLALLAALCPASRA...",
    reference_smiles="COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    top_k=5,
)

for c in result.candidates:
    print(f"{c.drug_name}: score={c.composite_score:.3f} ({c.original_indication})")
```

### Audited Pipeline

```python
from pharmacore.audit import AuditedDiscovery

ad = AuditedDiscovery()
result = ad.run_discovery(
    target_name="BRAF_kinase",
    target_sequence="MAALSGGGGGG...",
    n_molecules=5,
    output_dir="output/audit",
)
# Generates JSON audit trail + human-readable report
```

## Sparse Models

PharmaCore uses magnitude-pruned sparse models for efficient inference:

| Model | Params | Sparsity | Quality Retention | Inference (M4) |
|-------|--------|----------|-------------------|----------------|
| ESM-2 8M | 7.8M | 50% | 97.5% | ~8ms |
| ESM-2 35M | 33.5M | 50% | 97.3% | ~12ms |
| ChemBERTa-zinc | 44.1M | 50% | 97.3% | ~4ms |

Models are in `models/` directory. To sparsify additional models:

```bash
python scripts/sparsify_model.py --model esm2-35m --sparsity 0.5
```

## Architecture

```
pharmacore/
├── core/           # Types, config, device management
├── discovery/      # De novo drug discovery engine (540 lines)
├── repurposing/    # Drug repurposing engine (410 lines)
├── audit/          # Transparent audit pipeline (409 lines)
├── generation/     # Molecular generation (scaffold enumeration)
├── docking/        # AutoDock Vina wrapper
├── admet/          # ADMET property prediction
├── scoring/        # Drug-likeness scoring (Lipinski/Veber/QED)
└── pipeline/       # Pipeline orchestrator
```

## Key Technologies

- **ESM-2** (Meta) — protein language model for target encoding
- **ChemBERTa** (zinc-base-v1) — molecular language model for drug encoding
- **RDKit** — cheminformatics (fingerprints, descriptors, SMILES)
- **PyTorch + MPS** — Apple Silicon GPU acceleration
- **Magnitude Pruning** — 50% unstructured sparsity

## Benchmarks (Apple M4 Mac mini, 16GB)

| Task | Time | Details |
|------|------|---------|
| De novo discovery (5 mols) | ~7s | Target-driven, AI-scored |
| Drug repurposing screen | ~18s | 12 drugs × 1 target |
| Protein embedding | ~12ms | ESM-2 35M sparse, 160aa |
| Molecular embedding | ~4ms | ChemBERTa sparse |
| Full audited pipeline | ~20s | Discovery + audit report |

## License

MIT

## Citation

```bibtex
@software{pharmacore2026,
  title={PharmaCore: Apple Silicon-Native AI Drug Discovery},
  author={Reacher Wu},
  year={2026},
  url={https://github.com/reacherwu/PharmaCore}
}
```
