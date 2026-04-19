# PharmaCore — Master Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build the world's first Apple Silicon-native, fully local AI drug discovery platform — from target identification to lead optimization — in a single unified open-source package.

**Architecture:** Modular pipeline architecture with Apple Silicon (MLX + MPS) acceleration at every layer. Each stage (target analysis, protein folding, molecular generation, docking, ADMET) is a pluggable module with a unified Python API. A pipeline orchestrator connects stages into end-to-end workflows.

**Tech Stack:** Python 3.12, MLX (Apple), PyTorch MPS, RDKit, ESM-2, DiffDock, DeepChem, AutoDock Vina

**License:** Apache 2.0

**Target:** International investors, pharma companies, academic researchers

---

## Market Gap

1. **No Apple Silicon drug discovery platform exists** — zero competition
2. **No unified open-source end-to-end platform** — everything is fragmented (DeepChem, TorchDrug, REINVENT are separate tools)
3. **Commercial tools cost $100K-500K/year** (Schrödinger, MOE) — massive price barrier
4. **Privacy/IP concerns** — pharma companies want local execution, not cloud APIs

## Key Differentiators

1. **Apple Silicon Native** — MLX + Metal acceleration, runs on MacBook/Mac Studio
2. **Fully Local** — no cloud dependency, no API keys, complete data privacy
3. **Unified Pipeline** — target → structure → generation → docking → ADMET in one tool
4. **Production-Ready** — not research code, proper packaging, CLI, API
5. **Sparse Model Innovation** — ESM-2-15B pruned to run on 16GB (novel contribution)

---

## Project Structure

```
PharmaCore/
├── README.md
├── LICENSE                    # Apache 2.0
├── pyproject.toml
├── Makefile
├── pharmacore/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Global configuration
│   │   ├── device.py          # Apple Silicon / MPS / CPU detection
│   │   └── types.py           # Shared data types (Molecule, Protein, etc.)
│   ├── target/
│   │   ├── __init__.py
│   │   ├── analyzer.py        # Target identification & analysis
│   │   └── knowledge_graph.py # Disease-target knowledge graph
│   ├── protein/
│   │   ├── __init__.py
│   │   ├── esm.py             # ESM-2 protein embeddings
│   │   ├── folding.py         # Structure prediction (ESMFold wrapper)
│   │   └── sparse_esm.py     # Sparse ESM-2-15B (innovation)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── diffusion.py       # Diffusion-based molecular generation
│   │   ├── reinvent.py        # REINVENT-style RL generation
│   │   └── filters.py         # Drug-likeness filters (Lipinski, etc.)
│   ├── docking/
│   │   ├── __init__.py
│   │   ├── diffdock.py        # DiffDock wrapper
│   │   ├── vina.py            # AutoDock Vina wrapper
│   │   └── scoring.py         # Docking score analysis
│   ├── admet/
│   │   ├── __init__.py
│   │   ├── predictor.py       # ADMET property prediction
│   │   ├── toxicity.py        # Toxicity prediction
│   │   └── models.py          # Pre-trained ADMET models
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # End-to-end pipeline orchestration
│   │   ├── drug_discovery.py  # Full discovery workflow
│   │   └── repurposing.py     # Drug repurposing workflow
│   └── utils/
│       ├── __init__.py
│       ├── chemistry.py       # RDKit utilities
│       ├── visualization.py   # Molecular visualization
│       └── io.py              # File I/O (SDF, PDB, SMILES)
├── tests/
│   ├── __init__.py
│   ├── test_core/
│   ├── test_protein/
│   ├── test_generation/
│   ├── test_docking/
│   ├── test_admet/
│   └── test_pipeline/
├── benchmarks/
│   ├── README.md
│   ├── run_benchmarks.py
│   └── results/
├── examples/
│   ├── 01_target_analysis.py
│   ├── 02_protein_folding.py
│   ├── 03_molecule_generation.py
│   ├── 04_molecular_docking.py
│   ├── 05_admet_prediction.py
│   └── 06_full_pipeline.py
├── docs/
│   ├── getting-started.md
│   ├── architecture.md
│   ├── apple-silicon.md
│   └── benchmarks.md
├── scripts/
│   ├── setup_models.py        # Download pre-trained models
│   └── benchmark.py
├── ROADMAP.md
├── CONTRIBUTING.md
├── CITATION.cff
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Implementation Phases

### Phase 1: Foundation (Tasks 1-8)
Core infrastructure, device detection, data types, project setup.

### Phase 2: Protein Module (Tasks 9-14)
ESM-2 integration with MPS acceleration.

### Phase 3: Molecular Generation (Tasks 15-20)
Diffusion-based and RL-based molecule generation.

### Phase 4: Docking (Tasks 21-25)
DiffDock and Vina integration.

### Phase 5: ADMET (Tasks 26-30)
Property prediction with DeepChem.

### Phase 6: Pipeline Orchestration (Tasks 31-35)
End-to-end workflows connecting all modules.

### Phase 7: Benchmarks & Documentation (Tasks 36-40)
Validation, README, investor-ready documentation.

---

## Task Breakdown

### Task 1: Project Skeleton & pyproject.toml

**Objective:** Create the project structure with proper Python packaging.

**Files:**
- Create: `pyproject.toml`
- Create: `pharmacore/__init__.py`
- Create: `Makefile`
- Create: `LICENSE`

**pyproject.toml:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pharmacore"
version = "0.1.0"
description = "Apple Silicon-native AI platform for drug discovery"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
authors = [{name = "PharmaCore Contributors"}]
keywords = ["drug-discovery", "ai", "apple-silicon", "molecular-generation", "protein-folding"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
]
dependencies = [
    "torch>=2.1.0",
    "numpy>=1.24.0",
    "rdkit",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "tqdm",
]

[project.optional-dependencies]
apple = ["mlx>=0.5.0"]
protein = ["fair-esm>=2.0.0"]
docking = ["vina"]
admet = ["deepchem>=2.7.0"]
all = ["pharmacore[apple,protein,docking,admet]"]
dev = ["pytest>=7.0", "pytest-cov", "ruff", "mypy"]

[project.scripts]
pharmacore = "pharmacore.cli:main"

[tool.ruff]
target-version = "py312"
line-length = 100
```

### Task 2: Core Device Detection

**Objective:** Auto-detect Apple Silicon, MPS, and fallback to CPU.

**Files:**
- Create: `pharmacore/core/__init__.py`
- Create: `pharmacore/core/device.py`
- Test: `tests/test_core/test_device.py`

### Task 3: Core Data Types

**Objective:** Define Molecule, Protein, DockingResult, ADMETProfile dataclasses.

**Files:**
- Create: `pharmacore/core/types.py`
- Test: `tests/test_core/test_types.py`

### Task 4: Chemistry Utilities

**Objective:** RDKit wrappers for SMILES parsing, molecular descriptors, fingerprints.

**Files:**
- Create: `pharmacore/utils/chemistry.py`
- Test: `tests/test_core/test_chemistry.py`

### Task 5: ESM-2 Protein Embeddings

**Objective:** Load ESM-2-650M, generate protein embeddings with MPS acceleration.

**Files:**
- Create: `pharmacore/protein/esm.py`
- Test: `tests/test_protein/test_esm.py`

### Task 6: Molecular Generation (Diffusion)

**Objective:** Implement a lightweight diffusion-based molecular generator.

**Files:**
- Create: `pharmacore/generation/diffusion.py`
- Create: `pharmacore/generation/filters.py`
- Test: `tests/test_generation/test_diffusion.py`

### Task 7: AutoDock Vina Integration

**Objective:** Wrap AutoDock Vina for molecular docking.

**Files:**
- Create: `pharmacore/docking/vina.py`
- Test: `tests/test_docking/test_vina.py`

### Task 8: ADMET Prediction

**Objective:** DeepChem-based ADMET property prediction.

**Files:**
- Create: `pharmacore/admet/predictor.py`
- Test: `tests/test_admet/test_predictor.py`

### Task 9: Pipeline Orchestrator

**Objective:** Connect all modules into end-to-end drug discovery workflow.

**Files:**
- Create: `pharmacore/pipeline/orchestrator.py`
- Create: `pharmacore/pipeline/drug_discovery.py`
- Test: `tests/test_pipeline/test_orchestrator.py`

### Task 10: CLI Interface

**Objective:** Command-line interface for running pipelines.

**Files:**
- Create: `pharmacore/cli.py`
- Test: `tests/test_cli.py`

### Task 11: Benchmarks

**Objective:** Benchmark suite comparing PharmaCore against baselines.

**Files:**
- Create: `benchmarks/run_benchmarks.py`
- Create: `benchmarks/README.md`

### Task 12: Documentation & README

**Objective:** Investor-ready documentation with architecture diagrams, benchmarks, roadmap.

**Files:**
- Create: `README.md`
- Create: `ROADMAP.md`
- Create: `CONTRIBUTING.md`
- Create: `CITATION.cff`
- Create: `docs/getting-started.md`
- Create: `docs/architecture.md`
- Create: `docs/apple-silicon.md`

---

## Validation Criteria

Before publishing to GitHub:

1. **All tests pass:** `pytest tests/ -v` — 100% pass rate
2. **Core pipeline works:** Can run target → generation → docking → ADMET on a sample
3. **Apple Silicon acceleration verified:** MPS/MLX detected and used on M-series
4. **Benchmarks documented:** Performance numbers on standard datasets
5. **README is investor-grade:** Clear value proposition, architecture diagram, benchmarks
6. **Clean code:** `ruff check` passes, type hints throughout
7. **Examples work:** All 6 example scripts run successfully

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| ESM-2 too large for 16GB | Use ESM-2-650M (2.6GB), sparse 15B is Phase 2 |
| DiffDock complex setup | Wrap with fallback to Vina |
| DeepChem compatibility | Pin versions, test on macOS |
| MLX limited ops | Fallback to PyTorch MPS |
| Investor skepticism | Strong benchmarks + working demo |
