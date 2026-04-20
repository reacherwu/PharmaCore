---
license: mit
tags:
  - pharmacore
  - sparse
  - drug-discovery
  - apple-silicon
base_model: seyonec/ChemBERTa-zinc-base-v1
---

# chemberta-zinc-sparse50 (PharmaCore Sparse)

50% magnitude-pruned version of [seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) 
for efficient drug discovery on Apple Silicon.

## Key Stats
- **Sparsity:** 50%
- **Quality Retention:** 97.3%
- **Use Case:** Molecular encoding in PharmaCore drug discovery pipeline

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("stephenjun8192/chemberta-zinc-sparse50")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
```

## Part of PharmaCore

[PharmaCore](https://github.com/stephenjun8192/PharmaCore) — Apple Silicon-native AI drug discovery platform.
