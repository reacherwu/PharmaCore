---
license: mit
language:
  - en
tags:
  - pharmacore
  - sparse
  - drug-discovery
  - apple-silicon
  - chemberta
  - molecular-language-model
  - cheminformatics
  - smiles
  - pruning
  - efficient-inference
library_name: transformers
pipeline_tag: feature-extraction
base_model: seyonec/ChemBERTa-zinc-base-v1
model-index:
  - name: chemberta-zinc-sparse50
    results:
      - task:
          type: feature-extraction
          name: Molecular Embedding
        metrics:
          - type: cosine_similarity
            value: 0.973
            name: Quality Retention vs Dense
---

# ChemBERTa-zinc Sparse 50% — PharmaCore

A **50% magnitude-pruned** version of [seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) optimized for efficient molecular encoding on Apple Silicon.

## Why This Model?

| Metric | Dense (Original) | Sparse (This) | Improvement |
|--------|-----------------|---------------|-------------|
| Parameters (active) | 44.1M | 22M | 50% reduction |
| Inference (M4 MPS) | 5.1ms | 4.9ms | 4% faster |
| Quality Retention | 100% | 97.3% | Minimal loss |

## Use Case

Molecular encoder in the [PharmaCore](https://github.com/reacherwu/PharmaCore) drug discovery pipeline:
- Encode SMILES strings into dense embeddings for drug-target scoring
- Molecular similarity computation for drug repurposing
- Drug-likeness assessment and ADMET property prediction
- Runs entirely on consumer Apple Silicon hardware (M1/M2/M3/M4)

## Usage

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("stephenjun8192/chemberta-zinc-sparse50")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Encode a drug molecule (Erlotinib — EGFR inhibitor)
smiles = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]

print(f"Embedding shape: {embedding.shape}")
```

## Sparsification Method

- **Technique:** Global magnitude pruning (unstructured)
- **Sparsity:** 50% of all weight parameters set to zero
- **Layers pruned:** All linear layers (attention Q/K/V/O, FFN)
- **Validation:** Cosine similarity of embeddings vs dense model ≥ 0.973
- **Training data:** Pre-trained on 100K ZINC molecules (SMILES)

## Benchmarks (Apple M4 Mac mini, 16GB)

| Task | Time |
|------|------|
| Single molecule embedding | 4.9ms |
| Batch of 12 molecules | ~45ms |
| Molecular fingerprint + embedding | ~6ms |
| Drug repurposing (full screen) | ~18s |

## Part of PharmaCore

[PharmaCore](https://github.com/reacherwu/PharmaCore) — the first AI drug discovery platform that runs entirely on a MacBook. No cloud GPUs, no API keys, no data leaves your machine.

## Citation

```bibtex
@software{pharmacore2026,
  title={PharmaCore: Apple Silicon-Native AI Drug Discovery},
  author={Stephen Wu},
  year={2026},
  url={https://github.com/reacherwu/PharmaCore}
}
```
