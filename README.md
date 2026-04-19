<p align="center">
  <img src="docs/logo.png" alt="PharmaCore" width="200"/>
</p>

<h1 align="center">PharmaCore</h1>

<p align="center">
  <strong>Apple Silicon-Native AI Drug Discovery Platform</strong><br>
  The first fully local, end-to-end drug discovery pipeline optimized for Apple Silicon.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#license">License</a>
</p>

---

## Why PharmaCore?

Drug discovery costs **$2.6B per approved drug** and takes **10-15 years**. Current AI tools require expensive cloud GPUs, proprietary APIs, and fragmented toolchains. PharmaCore changes this:

- **100% Local** — No cloud dependencies. Your data never leaves your machine.
- **Apple Silicon Native** — Built for M-series chips with MPS/MLX acceleration. Runs on a MacBook.
- **End-to-End** — From target identification to lead optimization in a single pipeline.
- **Open Source** — Apache 2.0. No vendor lock-in.

## Features

### Core Modules

| Module | Description | Status |
|--------|-------------|--------|
| **Target Analysis** | Protein target identification, druggability scoring, binding site prediction | ✅ Ready |
| **Protein Encoding** | ESM-2 protein language model embeddings with MPS acceleration | ✅ Ready |
| **Molecular Generation** | Scaffold-based enumeration with drug-likeness filters | ✅ Ready |
| **Molecular Docking** | AutoDock Vina integration with automated scoring | ✅ Ready |
| **ADMET Prediction** | Absorption, Distribution, Metabolism, Excretion, Toxicity profiling | ✅ Ready |
| **Pipeline Orchestrator** | End-to-end workflow: target → generate → dock → filter → rank | ✅ Ready |
| **CLI Interface** | Full command-line interface for all operations | ✅ Ready |

### Apple Silicon Optimization

- **MPS Backend** — PyTorch Metal Performance Shaders for GPU-accelerated inference
- **MLX Support** — Apple's ML framework for transformer models (optional)
- **Unified Memory** — Efficient memory management for 16GB+ configurations
- **Neural Engine** — Automatic dispatch to Apple Neural Engine when available

### Drug-Likeness Filters

- Lipinski Rule of Five
- Veber oral bioavailability rules
- PAINS (Pan-Assay Interference) substructure filters
- Brenk structural alerts
- hERG cardiac toxicity liability screening
- Blood-Brain Barrier permeability prediction
- P-glycoprotein substrate prediction

## Quickstart

### Requirements

- macOS 13+ (Ventura or later) with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- 16GB unified memory recommended

### Installation

```bash
git clone https://github.com/reacherwu/PharmaCore.git
cd PharmaCore
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Quick Example

```python
from pharmacore.pipeline.orchestrator import PipelineOrchestrator

# Run end-to-end drug discovery pipeline
pipeline = PipelineOrchestrator()
result = pipeline.run(
    target_name="EGFR",
    target_sequence="MRPSGTAGAALLALLAALCPASRALEEKKVC...",
    n_molecules=100,
    stages=["target", "generation", "admet", "ranking"],
)

# Top candidates ranked by drug-likeness
for mol in result.molecules[:5]:
    print(f"{mol.name}: {mol.smiles} (score: {mol.properties['score']:.3f})")
```

### CLI Usage

```bash
# Analyze a molecule
pharmacore analyze "CC(=O)Oc1ccccc1C(=O)O" --admet --3d

# Generate drug candidates
pharmacore generate --target EGFR --n-molecules 50

# Run full pipeline
pharmacore pipeline --target EGFR --output results/

# System info
pharmacore info
```

## Architecture

```
pharmacore/
├── core/           # Configuration, device detection, type system
│   ├── config.py   # Pydantic-based settings with Apple Silicon defaults
│   ├── device.py   # MPS/MLX/CPU device management
│   └── types.py    # Molecule, Protein, DockingResult data models
├── utils/
│   └── chemistry.py  # RDKit wrappers: SMILES, descriptors, fingerprints
├── target/
│   └── analyzer.py   # Target identification and druggability scoring
├── protein/
│   └── esm.py        # ESM-2 protein embeddings with MPS acceleration
├── generation/
│   └── diffusion.py  # Molecular generation with scaffold enumeration
├── docking/
│   └── vina.py       # AutoDock Vina docking and scoring
├── admet/
│   └── predictor.py  # ADMET prediction and toxicity screening
├── pipeline/
│   └── orchestrator.py  # End-to-end pipeline orchestration
└── cli.py            # Click-based command-line interface
```

### Design Principles

1. **Lazy Loading** — Heavy dependencies (PyTorch, ESM-2) loaded only when needed
2. **Device Abstraction** — Automatic MPS → MLX → CPU fallback chain
3. **Composable Pipeline** — Each stage is independent; run any subset
4. **Type Safety** — Pydantic models throughout for validation and serialization
5. **Zero Config** — Sensible defaults; works out of the box on any Mac

## Benchmarks

Measured on Apple M4 (10-core, 16GB unified memory):

| Operation | Performance |
|-----------|-------------|
| SMILES Parsing | 1,091 molecules/sec |
| Descriptor Computation | 0.091 ms/molecule |
| Morgan Fingerprint | 0.024 ms/molecule |
| ADMET Prediction | 0.206 ms/molecule |
| Drug-Likeness Check | 0.098 ms/molecule |
| Molecule Generation | 0.15 ms/molecule |
| Full Pipeline (20 mol) | 8.9 ms total |

## Roadmap

### v0.2.0 — Model Integration
- [ ] ESM-2-650M protein embeddings with MPS acceleration
- [ ] DiffDock molecular docking with learned scoring
- [ ] Sparsified ESM-2-15B for 16GB Macs (novel contribution)

### v0.3.0 — Advanced Generation
- [ ] Diffusion-based de novo molecular generation
- [ ] Reinforcement learning for multi-objective optimization
- [ ] Retrosynthesis planning

### v0.4.0 — LLM Brain
- [ ] Sparsified Qwen2.5-32B as reasoning engine
- [ ] Natural language drug discovery queries
- [ ] Automated literature mining

### v1.0.0 — Production
- [ ] Clinical trial data integration
- [ ] Regulatory compliance reporting
- [ ] Multi-target polypharmacology

## For Investors

PharmaCore addresses a **$71B market** (AI in drug discovery, projected 2032) with a unique positioning:

- **No competition** in the Apple Silicon-native drug discovery space
- **Privacy-first** — Pharma companies increasingly demand on-premise solutions
- **Cost reduction** — A Mac Studio replaces $50K/year cloud GPU spend
- **Democratization** — Makes AI drug discovery accessible to academic labs and startups

See [INVESTOR.md](docs/INVESTOR.md) for the full investment thesis.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
make install    # Install with dev dependencies
make test       # Run test suite
make lint       # Run linters
make benchmark  # Run benchmarks
```

## Citation

```bibtex
@software{pharmacore2026,
  title={PharmaCore: Apple Silicon-Native AI Drug Discovery Platform},
  author={Reacher Wu},
  year={2026},
  url={https://github.com/reacherwu/PharmaCore},
  license={Apache-2.0}
}
```

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for Apple Silicon<br>
  <sub>Making drug discovery accessible to everyone.</sub>
</p>
