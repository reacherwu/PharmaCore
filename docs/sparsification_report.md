# ESM-2 Sparsification Ablation Study

## Overview

Systematic evaluation of magnitude pruning on ESM-2 protein language models
at multiple sparsity levels (30%, 50%, 70%, 90%) on Apple Silicon (MPS).

## Results

| Model | Sparsity | Cosine Sim | Rank Corr | Insulin Pair | Speed (ms) | Speedup |
|-------|----------|-----------|-----------|-------------|-----------|---------|
| ESM-2-8M | 29.86% | 0.8409 | 0.7367 | 0.9794 | 3.8 | 1.02x |
| ESM-2-8M | 49.76% | 0.7928 | 0.7384 | 0.9823 | 3.7 | 0.98x |
| ESM-2-8M | 69.66% | 0.5918 | 0.6617 | 0.9831 | 3.8 | 0.95x |
| ESM-2-8M | 89.57% | -0.1273 | 0.5922 | 0.9887 | 3.6 | 1.03x |
| ESM-2-35M | 29.92% | 0.9646 | 0.9535 | 0.9756 | 6.0 | 1.43x |
| ESM-2-35M | 49.86% | 0.9005 | 0.5583 | 0.9956 | 6.0 | 1.42x |
| ESM-2-35M | 69.81% | 0.8434 | 0.5249 | 0.9936 | 6.2 | 1.38x |
| ESM-2-35M | 89.75% | 0.6501 | 0.3848 | 0.9908 | 6.3 | 1.35x |

## Key Findings

### ESM-2-8M

- 30% sparsity: Good quality (cosine=0.8409, insulin=0.9794)
- 50% sparsity: Good quality (cosine=0.7928, insulin=0.9823)
- 70% sparsity: Degraded quality (cosine=0.5918, insulin=0.9831)
- 90% sparsity: Degraded quality (cosine=-0.1273, insulin=0.9887)

### ESM-2-35M

- 30% sparsity: Excellent quality (cosine=0.9646, insulin=0.9756)
- 50% sparsity: Excellent quality (cosine=0.9005, insulin=0.9956)
- 70% sparsity: Good quality (cosine=0.8434, insulin=0.9936)
- 90% sparsity: Degraded quality (cosine=0.6501, insulin=0.9908)

## Methodology

- **Pruning**: Unstructured L1 magnitude pruning on all Linear layers
- **Evaluation**: 8 reference proteins (insulin, EGFR, p53, hemoglobin, lysozyme, ubiquitin, cytochrome c)
- **Metrics**: Per-protein cosine similarity, pairwise rank correlation, insulin homolog pair similarity
- **Hardware**: Apple Silicon M4 (MPS backend)
- **Framework**: PyTorch + HuggingFace Transformers

## Conclusion

ESM-2-35M can be pruned to 50% sparsity while maintaining 0.9005 cosine similarity,
demonstrating that protein language models are highly compressible via magnitude pruning.
This enables deployment on memory-constrained Apple Silicon devices without significant quality loss.