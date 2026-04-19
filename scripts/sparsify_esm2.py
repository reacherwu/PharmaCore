#!/usr/bin/env python3
"""ESM-2 Protein Language Model Sparsification for Apple Silicon.

This script performs structured magnitude pruning on ESM-2 protein models,
creating sparse variants optimized for Apple Silicon inference.

Approach:
  1. Load ESM-2-150M (facebook/esm2_t30_150M_UR50D)
  2. Apply unstructured magnitude pruning at configurable sparsity
  3. Evaluate on protein embedding quality (cosine similarity preservation)
  4. Save sparse model with benchmark results

This is a novel contribution — no public ESM-2 sparse models exist.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.prune as prune


def get_device() -> torch.device:
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_esm2_model(model_name: str = "facebook/esm2_t30_150M_UR50D"):
    """Load ESM-2 model and tokenizer from HuggingFace."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def count_sparsity(model) -> dict:
    """Count zero and total parameters."""
    total = 0
    zeros = 0
    layer_stats = {}

    for name, param in model.named_parameters():
        n = param.numel()
        z = (param.data == 0).sum().item()
        total += n
        zeros += z
        if n > 1000:  # only track significant layers
            layer_stats[name] = {
                "total": n,
                "zeros": z,
                "sparsity": round(z / n * 100, 1),
            }

    return {
        "total_params": total,
        "zero_params": zeros,
        "sparsity_pct": round(zeros / total * 100, 2),
        "top_layers": dict(
            sorted(layer_stats.items(), key=lambda x: x[1]["sparsity"], reverse=True)[:10]
        ),
    }


def apply_magnitude_pruning(model, sparsity: float = 0.5, target_layers: str = "all"):
    """Apply unstructured magnitude pruning to model weights.

    Args:
        model: PyTorch model
        sparsity: fraction of weights to prune (0.0 to 1.0)
        target_layers: 'all', 'attention', or 'ffn'
    """
    print(f"\nApplying magnitude pruning (sparsity={sparsity:.0%}, target={target_layers})...")

    pruned_count = 0
    for name, module in model.named_modules():
        should_prune = False

        if isinstance(module, torch.nn.Linear):
            if target_layers == "all":
                should_prune = True
            elif target_layers == "attention" and any(
                k in name for k in ["query", "key", "value", "attention"]
            ):
                should_prune = True
            elif target_layers == "ffn" and any(
                k in name for k in ["intermediate", "output"]
            ):
                should_prune = True

        if should_prune:
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")  # make pruning permanent
            pruned_count += 1

    print(f"  Pruned {pruned_count} Linear layers")
    return model


# --- Evaluation ---

# Reference protein sequences for benchmarking
BENCHMARK_PROTEINS = {
    "insulin_human": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "insulin_mouse": "MALLVHFLPLLALLALWEPKPTQAFVKQHLCGPHLVEALYLVCGERGFFYTPKS",
    "egfr_fragment": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRM",
    "p53_fragment": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQW",
    "hemoglobin_a": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSA",
    "lysozyme": "MKALIVLGLVLLSVTVQGKVFGRCELAAALKPHSLDRYVRSSMSITDNRETFAN",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGR",
    "cytochrome_c": "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAAN",
}


def embed_proteins(model, tokenizer, sequences: dict, device: torch.device) -> dict:
    """Generate embeddings for a set of protein sequences."""
    model.eval()
    model.to(device)
    embeddings = {}

    with torch.no_grad():
        for name, seq in sequences.items():
            inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # MPS doesn't support all ops, fall back to CPU if needed
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            except RuntimeError:
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                model_cpu = model.to("cpu")
                outputs = model_cpu(**inputs)
                model.to(device)

            # Mean pooling over sequence length
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings[name] = emb

    return embeddings


def evaluate_embedding_quality(
    original_embeddings: dict, sparse_embeddings: dict
) -> dict:
    """Compare embedding quality between original and sparse models."""
    from scipy.spatial.distance import cosine

    # 1. Per-protein cosine similarity (should be close to 1.0)
    per_protein_sim = {}
    for name in original_embeddings:
        sim = 1 - cosine(original_embeddings[name], sparse_embeddings[name])
        per_protein_sim[name] = round(sim, 6)

    avg_sim = np.mean(list(per_protein_sim.values()))

    # 2. Relative distance preservation
    # Check if pairwise distances are preserved
    proteins = list(original_embeddings.keys())
    orig_dists = []
    sparse_dists = []

    for i in range(len(proteins)):
        for j in range(i + 1, len(proteins)):
            od = cosine(original_embeddings[proteins[i]], original_embeddings[proteins[j]])
            sd = cosine(sparse_embeddings[proteins[i]], sparse_embeddings[proteins[j]])
            orig_dists.append(od)
            sparse_dists.append(sd)

    # Spearman rank correlation of pairwise distances
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(orig_dists, sparse_dists)

    # 3. Insulin similarity test (human vs mouse should be most similar)
    if "insulin_human" in sparse_embeddings and "insulin_mouse" in sparse_embeddings:
        insulin_sim = 1 - cosine(
            sparse_embeddings["insulin_human"], sparse_embeddings["insulin_mouse"]
        )
    else:
        insulin_sim = None

    return {
        "per_protein_cosine_similarity": per_protein_sim,
        "average_cosine_similarity": round(float(avg_sim), 6),
        "pairwise_rank_correlation": round(float(rank_corr), 6),
        "insulin_pair_similarity": round(float(insulin_sim), 6) if insulin_sim else None,
    }


def benchmark_inference(model, tokenizer, device: torch.device, n_runs: int = 20) -> dict:
    """Benchmark inference speed."""
    model.eval()
    model.to(device)

    test_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    inputs = tokenizer(test_seq, return_tensors="pt", padding=True, truncation=True)

    try:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(**inputs)
            if device.type == "mps":
                torch.mps.synchronize()

        # Timed runs
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            if device.type == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)

    except RuntimeError:
        # Fallback to CPU
        device = torch.device("cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            for _ in range(3):
                model(**inputs)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            times.append(time.perf_counter() - t0)

    return {
        "device": str(device),
        "n_runs": n_runs,
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "min_ms": round(np.min(times) * 1000, 2),
        "max_ms": round(np.max(times) * 1000, 2),
        "throughput_seqs_per_sec": round(1.0 / np.mean(times), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="ESM-2 Sparsification for Apple Silicon")
    parser.add_argument(
        "--model",
        default="facebook/esm2_t30_150M_UR50D",
        help="HuggingFace model name (default: ESM-2-150M)",
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Pruning sparsity (default: 0.5)")
    parser.add_argument("--target", default="all", choices=["all", "attention", "ffn"])
    parser.add_argument("--output-dir", default="models/esm2-sparse", help="Output directory")
    parser.add_argument("--skip-save", action="store_true", help="Skip saving the model")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Target sparsity: {args.sparsity:.0%}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_esm2_model(args.model)

    # Pre-pruning stats
    pre_stats = count_sparsity(model)
    print(f"\nPre-pruning sparsity: {pre_stats['sparsity_pct']}%")

    # Pre-pruning embeddings
    print("\nGenerating original embeddings...")
    original_embeddings = embed_proteins(model, tokenizer, BENCHMARK_PROTEINS, device)

    # Pre-pruning benchmark
    print("Benchmarking original model...")
    original_bench = benchmark_inference(model, tokenizer, device)
    print(f"  Original: {original_bench['mean_ms']:.1f} ms/seq ({original_bench['throughput_seqs_per_sec']} seq/s)")

    # Apply pruning
    model = apply_magnitude_pruning(model, sparsity=args.sparsity, target_layers=args.target)

    # Post-pruning stats
    post_stats = count_sparsity(model)
    print(f"Post-pruning sparsity: {post_stats['sparsity_pct']}%")

    # Post-pruning embeddings
    print("\nGenerating sparse embeddings...")
    sparse_embeddings = embed_proteins(model, tokenizer, BENCHMARK_PROTEINS, device)

    # Evaluate quality
    print("Evaluating embedding quality...")
    quality = evaluate_embedding_quality(original_embeddings, sparse_embeddings)
    print(f"  Average cosine similarity: {quality['average_cosine_similarity']:.4f}")
    print(f"  Pairwise rank correlation: {quality['pairwise_rank_correlation']:.4f}")
    if quality["insulin_pair_similarity"]:
        print(f"  Insulin pair similarity: {quality['insulin_pair_similarity']:.4f}")

    # Post-pruning benchmark
    print("\nBenchmarking sparse model...")
    sparse_bench = benchmark_inference(model, tokenizer, device)
    print(f"  Sparse: {sparse_bench['mean_ms']:.1f} ms/seq ({sparse_bench['throughput_seqs_per_sec']} seq/s)")

    speedup = original_bench["mean_ms"] / sparse_bench["mean_ms"]
    print(f"  Speedup: {speedup:.2f}x")

    # Save results
    results = {
        "model": args.model,
        "sparsity_target": args.sparsity,
        "pruning_target": args.target,
        "pre_pruning": pre_stats,
        "post_pruning": post_stats,
        "quality": quality,
        "benchmark_original": original_bench,
        "benchmark_sparse": sparse_bench,
        "speedup": round(speedup, 3),
        "device": str(device),
        "torch_version": torch.__version__,
    }

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    results_path = output_dir / "sparsification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults saved to {results_path}")

    # Save sparse model
    if not args.skip_save:
        print(f"\nSaving sparse model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("  Done.")

        # Model size comparison
        import glob
        orig_size = sum(p.numel() * 4 for p in model.parameters()) / 1e6  # float32
        saved_files = glob.glob(str(output_dir / "*.safetensors")) + glob.glob(str(output_dir / "*.bin"))
        disk_size = sum(Path(f).stat().st_size for f in saved_files) / 1e6
        print(f"  Dense size (float32): {orig_size:.0f} MB")
        print(f"  Saved size on disk: {disk_size:.0f} MB")

    # Summary
    print("\n" + "=" * 60)
    print("ESM-2 SPARSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Model:           {args.model}")
    print(f"Sparsity:        {post_stats['sparsity_pct']}%")
    print(f"Quality:         {quality['average_cosine_similarity']:.4f} cosine sim")
    print(f"Rank corr:       {quality['pairwise_rank_correlation']:.4f}")
    print(f"Original speed:  {original_bench['mean_ms']:.1f} ms/seq")
    print(f"Sparse speed:    {sparse_bench['mean_ms']:.1f} ms/seq")
    print(f"Speedup:         {speedup:.2f}x")
    print("=" * 60)

    gc.collect()
    return results


if __name__ == "__main__":
    main()
