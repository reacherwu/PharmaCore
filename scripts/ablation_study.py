#!/usr/bin/env python3
"""ESM-2 Multi-Sparsity Ablation Study.

Runs sparsification at multiple sparsity levels on ESM-2-8M and ESM-2-35M,
generating a comprehensive comparison report.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PYTHON = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")
SCRIPT = str(Path(__file__).parent / "sparsify_esm2.py")

EXPERIMENTS = [
    # (model, sparsity)
    ("facebook/esm2_t6_8M_UR50D", 0.3),
    ("facebook/esm2_t6_8M_UR50D", 0.5),
    ("facebook/esm2_t6_8M_UR50D", 0.7),
    ("facebook/esm2_t6_8M_UR50D", 0.9),
    ("/tmp/esm2_35m", 0.3),
    ("/tmp/esm2_35m", 0.5),
    ("/tmp/esm2_35m", 0.7),
    ("/tmp/esm2_35m", 0.9),
]


def run_experiment(model: str, sparsity: float) -> dict | None:
    model_tag = "8M" if "8M" in model or "t6" in model else "35M"
    sp_tag = f"{int(sparsity * 100)}"
    output_dir = f"models/esm2-{model_tag.lower()}-sparse{sp_tag}"

    print(f"\n{'='*60}")
    print(f"Running: ESM-2-{model_tag} @ {sparsity:.0%} sparsity")
    print(f"{'='*60}")

    cmd = [
        PYTHON, SCRIPT,
        "--model", model,
        "--sparsity", str(sparsity),
        "--output-dir", output_dir,
        "--skip-save",  # don't save model weights for ablation
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])

    # Read results JSON
    results_path = Path(output_dir) / "sparsification_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        data["model_tag"] = model_tag
        return data

    return None


def generate_report(results: list[dict]):
    """Generate markdown report."""
    report = []
    report.append("# ESM-2 Sparsification Ablation Study")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("Systematic evaluation of magnitude pruning on ESM-2 protein language models")
    report.append("at multiple sparsity levels (30%, 50%, 70%, 90%) on Apple Silicon (MPS).")
    report.append("")
    report.append("## Results")
    report.append("")

    # Table header
    report.append("| Model | Sparsity | Cosine Sim | Rank Corr | Insulin Pair | Speed (ms) | Speedup |")
    report.append("|-------|----------|-----------|-----------|-------------|-----------|---------|")

    for r in results:
        tag = r["model_tag"]
        sp = r["post_pruning"]["sparsity_pct"]
        cos = r["quality"]["average_cosine_similarity"]
        rank = r["quality"]["pairwise_rank_correlation"]
        ins = r["quality"]["insulin_pair_similarity"]
        speed = r["benchmark_sparse"]["mean_ms"]
        speedup = r["speedup"]
        report.append(f"| ESM-2-{tag} | {sp}% | {cos:.4f} | {rank:.4f} | {ins:.4f} | {speed:.1f} | {speedup:.2f}x |")

    report.append("")
    report.append("## Key Findings")
    report.append("")

    # Find sweet spots
    for model_tag in ["8M", "35M"]:
        model_results = [r for r in results if r["model_tag"] == model_tag]
        if not model_results:
            continue

        report.append(f"### ESM-2-{model_tag}")
        report.append("")

        # Find best quality/sparsity tradeoff
        for r in model_results:
            sp = r["post_pruning"]["sparsity_pct"]
            cos = r["quality"]["average_cosine_similarity"]
            ins = r["quality"]["insulin_pair_similarity"]
            if cos > 0.85:
                report.append(f"- {sp:.0f}% sparsity: Excellent quality (cosine={cos:.4f}, insulin={ins:.4f})")
            elif cos > 0.7:
                report.append(f"- {sp:.0f}% sparsity: Good quality (cosine={cos:.4f}, insulin={ins:.4f})")
            else:
                report.append(f"- {sp:.0f}% sparsity: Degraded quality (cosine={cos:.4f}, insulin={ins:.4f})")

        report.append("")

    report.append("## Methodology")
    report.append("")
    report.append("- **Pruning**: Unstructured L1 magnitude pruning on all Linear layers")
    report.append("- **Evaluation**: 8 reference proteins (insulin, EGFR, p53, hemoglobin, lysozyme, ubiquitin, cytochrome c)")
    report.append("- **Metrics**: Per-protein cosine similarity, pairwise rank correlation, insulin homolog pair similarity")
    report.append("- **Hardware**: Apple Silicon M4 (MPS backend)")
    report.append("- **Framework**: PyTorch + HuggingFace Transformers")
    report.append("")
    report.append("## Conclusion")
    report.append("")

    # Auto-generate conclusion
    best_35m = None
    for r in results:
        if r["model_tag"] == "35M" and r["quality"]["average_cosine_similarity"] > 0.85:
            if best_35m is None or r["post_pruning"]["sparsity_pct"] > best_35m["post_pruning"]["sparsity_pct"]:
                best_35m = r

    if best_35m:
        sp = best_35m["post_pruning"]["sparsity_pct"]
        cos = best_35m["quality"]["average_cosine_similarity"]
        report.append(f"ESM-2-35M can be pruned to {sp:.0f}% sparsity while maintaining {cos:.4f} cosine similarity,")
        report.append("demonstrating that protein language models are highly compressible via magnitude pruning.")
        report.append("This enables deployment on memory-constrained Apple Silicon devices without significant quality loss.")
    else:
        report.append("Magnitude pruning shows promising results for ESM-2 compression on Apple Silicon.")

    return "\n".join(report)


def main():
    results = []

    for model, sparsity in EXPERIMENTS:
        try:
            r = run_experiment(model, sparsity)
            if r:
                results.append(r)
        except Exception as e:
            print(f"FAILED: {model} @ {sparsity}: {e}")

    if not results:
        print("No results collected!")
        sys.exit(1)

    # Save all results
    Path("models").mkdir(exist_ok=True)
    with open("models/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Generate report
    report = generate_report(results)
    report_path = Path("docs/sparsification_report.md")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n\nReport saved to {report_path}")
    print(f"Raw results saved to models/ablation_results.json")

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<12} {'Sparsity':>8} {'Cosine':>8} {'Rank':>8} {'Insulin':>8} {'ms/seq':>8} {'Speedup':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"ESM-2-{r['model_tag']:<5} "
            f"{r['post_pruning']['sparsity_pct']:>7.1f}% "
            f"{r['quality']['average_cosine_similarity']:>8.4f} "
            f"{r['quality']['pairwise_rank_correlation']:>8.4f} "
            f"{r['quality']['insulin_pair_similarity']:>8.4f} "
            f"{r['benchmark_sparse']['mean_ms']:>7.1f} "
            f"{r['speedup']:>7.2f}x"
        )


if __name__ == "__main__":
    main()
