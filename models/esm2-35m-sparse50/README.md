---
license: mit
language:
  - en
tags:
  - pharmacore
  - sparse
  - drug-discovery
  - apple-silicon
  - protein-language-model
  - esm2
  - bioinformatics
  - computational-biology
  - pruning
  - efficient-inference
library_name: transformers
pipeline_tag: feature-extraction
base_model: facebook/esm2_t12_35M_UR50D
model-index:
  - name: esm2-35m-sparse50
    results:
      - task:
          type: feature-extraction
          name: Protein Embedding
        metrics:
          - type: cosine_similarity
            value: 0.973
            name: Quality Retention vs Dense
---

# ESM-2 35M Sparse 50% — PharmaCore

A **50% magnitude-pruned** version of [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) optimized for efficient drug discovery inference on Apple Silicon.

## Why This Model?

| Metric | Dense (Original) | Sparse (This) | Improvement |
|--------|-----------------|---------------|-------------|
| Parameters (active) | 33.5M | 16.7M | 50% reduction |
| Inference (M4 MPS) | 8.2ms | 7.8ms | 5% faster |
| Quality Retention | 100% | 97.3% | Minimal loss |

## Use Case

Primary protein encoder in the [PharmaCore](https://github.com/reacherwu/PharmaCore) drug discovery pipeline:
- Higher-capacity protein embeddings for drug-target compatibility
- De novo drug discovery and drug repurposing workflows
- Full audit trail support for regulatory transparency
- Runs entirely on consumer Apple Silicon hardware (M1/M2/M3/M4)

## Usage

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("stephenjun8192/esm2-35m-sparse50")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

# Encode a protein target (e.g., EGFR kinase domain)
sequence = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVL"
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 480]

print(f"Embedding shape: {embedding.shape}")
```

## Sparsification Method

- **Technique:** Global magnitude pruning (unstructured)
- **Sparsity:** 50% of all weight parameters set to zero
- **Layers pruned:** All linear layers (attention Q/K/V/O, FFN)
- **Validation:** Cosine similarity of embeddings vs dense model ≥ 0.973

## Benchmarks (Apple M4 Mac mini, 16GB)

| Task | Time |
|------|------|
| Single protein embedding (160aa) | 7.8ms |
| Batch of 10 proteins | ~65ms |
| De novo discovery (5 molecules) | ~7s |
| Drug repurposing (12 drugs) | ~18s |

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
