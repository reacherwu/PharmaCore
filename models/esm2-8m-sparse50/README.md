---
license: mit
tags:
  - pharmacore
  - sparse
  - drug-discovery
  - apple-silicon
base_model: facebook/esm2_t6_8M_UR50D
---

# esm2-8m-sparse50 (PharmaCore Sparse)

50% magnitude-pruned version of [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) 
for efficient drug discovery on Apple Silicon.

## Key Stats
- **Sparsity:** 50%
- **Quality Retention:** 97.5%
- **Use Case:** Protein encoding in PharmaCore drug discovery pipeline

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("stephenjun8192/esm2-8m-sparse50")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
```

## Part of PharmaCore

[PharmaCore](https://github.com/stephenjun8192/PharmaCore) — Apple Silicon-native AI drug discovery platform.
