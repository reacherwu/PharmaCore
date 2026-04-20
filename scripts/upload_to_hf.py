"""Upload sparse models to HuggingFace Hub.

Usage:
    # First login:
    huggingface-cli login
    
    # Then upload:
    python scripts/upload_to_hf.py
"""
from pathlib import Path
import json

def create_model_card(model_dir: Path, model_name: str, base_model: str, 
                      sparsity: float, quality: float) -> str:
    return f"""---
license: mit
tags:
  - pharmacore
  - sparse
  - drug-discovery
  - apple-silicon
base_model: {base_model}
---

# {model_name} (PharmaCore Sparse)

50% magnitude-pruned version of [{base_model}](https://huggingface.co/{base_model}) 
for efficient drug discovery on Apple Silicon.

## Key Stats
- **Sparsity:** {sparsity:.0%}
- **Quality Retention:** {quality:.1%}
- **Use Case:** {'Protein encoding' if 'esm' in model_name.lower() else 'Molecular encoding'} in PharmaCore drug discovery pipeline

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("stephenjun8192/{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

## Part of PharmaCore

[PharmaCore](https://github.com/stephenjun8192/PharmaCore) — Apple Silicon-native AI drug discovery platform.
"""


def upload_models():
    from huggingface_hub import HfApi
    api = HfApi()
    
    models = [
        {
            "dir": "models/esm2-8m-sparse50",
            "repo": "stephenjun8192/esm2-8m-sparse50",
            "name": "esm2-8m-sparse50",
            "base": "facebook/esm2_t6_8M_UR50D",
            "sparsity": 0.50,
            "quality": 0.975,
        },
        {
            "dir": "models/esm2-35m-sparse50",
            "repo": "stephenjun8192/esm2-35m-sparse50",
            "name": "esm2-35m-sparse50",
            "base": "facebook/esm2_t12_35M_UR50D",
            "sparsity": 0.50,
            "quality": 0.973,
        },
        {
            "dir": "models/chemberta-zinc-sparse50",
            "repo": "stephenjun8192/chemberta-zinc-sparse50",
            "name": "chemberta-zinc-sparse50",
            "base": "seyonec/ChemBERTa-zinc-base-v1",
            "sparsity": 0.50,
            "quality": 0.973,
        },
    ]
    
    for m in models:
        model_dir = Path(m["dir"])
        if not model_dir.exists():
            print(f"Skipping {m['name']}: directory not found")
            continue
        
        # Write model card
        card = create_model_card(model_dir, m["name"], m["base"], m["sparsity"], m["quality"])
        (model_dir / "README.md").write_text(card)
        print(f"Created model card for {m['name']}")
        
        # Create repo and upload
        try:
            api.create_repo(m["repo"], exist_ok=True)
            api.upload_folder(
                folder_path=str(model_dir),
                repo_id=m["repo"],
                commit_message=f"Upload {m['name']} sparse model from PharmaCore",
            )
            print(f"Uploaded {m['name']} to {m['repo']}")
        except Exception as e:
            print(f"Failed to upload {m['name']}: {e}")


if __name__ == "__main__":
    upload_models()
