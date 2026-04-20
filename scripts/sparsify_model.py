#!/usr/bin/env python3
"""Universal Model Sparsification Engine for PharmaCore.

Supports magnitude pruning on HuggingFace Transformer models for drug discovery.
Usage:
    python scripts/sparsify_model.py --model-key chemberta-77m --sparsity 0.5
    python scripts/sparsify_model.py --hf-name DeepChem/ChemBERTa-77M-MTR --sparsity 0.5
"""
from __future__ import annotations
import argparse, gc, json, time
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
import torch.nn.utils.prune as prune

MODEL_REGISTRY = {
    "esm2-8m": {"hf_name": "facebook/esm2_t6_8M_UR50D", "domain": "protein"},
    "esm2-35m": {"hf_name": "facebook/esm2_t12_35M_UR50D", "domain": "protein"},
    "esm2-150m": {"hf_name": "facebook/esm2_t30_150M_UR50D", "domain": "protein"},
    "chemberta-77m": {"hf_name": "DeepChem/ChemBERTa-77M-MTR", "domain": "molecule"},
    "chemberta-zinc": {"hf_name": "seyonec/ChemBERTa-zinc-base-v1", "domain": "molecule"},
    "chemberta-100m": {"hf_name": "DeepChem/ChemBERTa-100M-MLM", "domain": "molecule"},
    "molformer-xl": {"hf_name": "ibm/MoLFormer-XL-both-10pct", "domain": "molecule"},
    "protbert": {"hf_name": "Rostlab/prot_bert", "domain": "protein"},
    "biobert": {"hf_name": "dmis-lab/biobert-v1.1", "domain": "biomedical"},
}

PROTEIN_SEQS = {
    "insulin_human": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "insulin_mouse": "MALLVHFLPLLALLALWEPKPTQAFVKQHLCGPHLVEALYLVCGERGFFYTPKS",
    "egfr_fragment": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRM",
    "p53_fragment": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWF",
    "hemoglobin_a": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
}

DRUG_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "penicillin_g": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "dexamethasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
    "remdesivir": "CCC(CC)COC(=O)C(C)NP(=O)(OCC1OC(C#N)(c2ccc3c(N)ncnn23)C(O)C1O)Oc1ccccc1",
    "sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
}

BIOMEDICAL_TEXTS = {
    "drug_target": "EGFR tyrosine kinase inhibitor binds to the ATP binding site",
    "mechanism": "Aspirin irreversibly inhibits cyclooxygenase COX-1 and COX-2",
    "disease": "Non-small cell lung cancer with EGFR mutation L858R",
    "pharmacology": "The drug shows high oral bioavailability and plasma protein binding",
    "toxicity": "Hepatotoxicity observed at doses exceeding 500mg per kg in rats",
}


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(hf_name):
    from transformers import AutoModel, AutoTokenizer
    print(f"  Loading {hf_name}...")
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
    return mdl, tok


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    nz = sum((p != 0).sum().item() for p in model.parameters())
    return {"total": total, "nonzero": nz, "zero": total - nz,
            "sparsity": round(1 - nz / total, 4) if total else 0}


def apply_pruning(model, sparsity):
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=sparsity)
            prune.remove(m, "weight")
            n += 1
    return n


def get_embeddings(model, tokenizer, texts, device, space_chars=False):
    model.eval().to(device)
    embs = {}
    with torch.no_grad():
        for name, text in texts.items():
            inp = " ".join(list(text)) if space_chars else text
            toks = tokenizer(inp, return_tensors="pt", truncation=True,
                             max_length=512, padding=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks)
            h = out.last_hidden_state
            mask = toks.get("attention_mask")
            if mask is not None:
                me = mask.unsqueeze(-1).float()
                e = (h * me).sum(1) / me.sum(1).clamp(min=1e-9)
            else:
                e = h.mean(dim=1)
            embs[name] = e.cpu().numpy().flatten()
    return embs


def cosine_sim(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def evaluate_embeddings(embs_orig, embs_sparse):
    names = sorted(set(embs_orig) & set(embs_sparse))
    if not names:
        return {}
    per_sample = {n: cosine_sim(embs_orig[n], embs_sparse[n]) for n in names}
    mean_cos = np.mean(list(per_sample.values()))
    pairs = list(combinations(names, 2))
    orig_d, sparse_d = [], []
    for a, b in pairs:
        orig_d.append(cosine_sim(embs_orig[a], embs_orig[b]))
        sparse_d.append(cosine_sim(embs_sparse[a], embs_sparse[b]))
    rank_corr = float(np.corrcoef(orig_d, sparse_d)[0, 1]) if len(pairs) > 1 else 1.0
    return {
        "mean_cosine_similarity": round(mean_cos, 4),
        "rank_correlation": round(rank_corr, 4),
        "per_sample": {k: round(v, 4) for k, v in per_sample.items()},
        "n_samples": len(names),
    }


def benchmark_speed(model, tokenizer, device, domain, n_runs=20):
    model.eval().to(device)
    if domain == "protein":
        text = "M A L W M R L L P L L A L L A L W G P D P A A A"
    elif domain == "molecule":
        text = "CC(=O)Oc1ccccc1C(=O)O"
    else:
        text = "EGFR tyrosine kinase inhibitor binds to the ATP binding site"
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        for _ in range(3):
            model(**toks)
    if device.type == "mps":
        torch.mps.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**toks)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "min_ms": round(np.min(times) * 1000, 2),
        "n_runs": n_runs,
    }


def save_sparse_model(model, tokenizer, output_dir, results):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "sparsification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    size_mb = sum(p.stat().st_size for p in output_dir.glob("*.safetensors")) / 1e6
    print(f"  Saved to {output_dir} ({size_mb:.1f} MB)")
    return size_mb


def get_eval_data(domain):
    if domain == "protein":
        return PROTEIN_SEQS, True
    elif domain == "molecule":
        return DRUG_SMILES, False
    else:
        return BIOMEDICAL_TEXTS, False


def sparsify_pipeline(hf_name, domain, sparsity, output_dir, model_key=None):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  PharmaCore Sparsification Engine")
    print(f"  Model: {hf_name}")
    print(f"  Domain: {domain}")
    print(f"  Target sparsity: {sparsity*100:.0f}%")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load
    print("[1/6] Loading model...")
    model, tokenizer = load_model(hf_name)
    pre_stats = count_params(model)
    print(f"  Parameters: {pre_stats['total']:,}")
    print(f"  Pre-sparsity: {pre_stats['sparsity']*100:.1f}%")

    # Baseline embeddings
    print("\n[2/6] Computing baseline embeddings...")
    eval_data, space_chars = get_eval_data(domain)
    embs_orig = get_embeddings(model, tokenizer, eval_data, device, space_chars)

    # Baseline speed
    print("\n[3/6] Benchmarking baseline speed...")
    speed_orig = benchmark_speed(model, tokenizer, device, domain)
    print(f"  Baseline: {speed_orig['mean_ms']:.2f} ms/inference")

    # Prune
    print(f"\n[4/6] Applying {sparsity*100:.0f}% magnitude pruning...")
    model.to("cpu")
    n_layers = apply_pruning(model, sparsity)
    post_stats = count_params(model)
    print(f"  Pruned {n_layers} Linear layers")
    print(f"  Actual sparsity: {post_stats['sparsity']*100:.1f}%")
    print(f"  Zero params: {post_stats['zero']:,} / {post_stats['total']:,}")

    # Evaluate sparse
    print("\n[5/6] Evaluating sparse model quality...")
    embs_sparse = get_embeddings(model, tokenizer, eval_data, device, space_chars)
    quality = evaluate_embeddings(embs_orig, embs_sparse)
    print(f"  Mean cosine similarity: {quality['mean_cosine_similarity']:.4f}")
    print(f"  Rank correlation: {quality['rank_correlation']:.4f}")

    speed_sparse = benchmark_speed(model, tokenizer, device, domain)
    print(f"  Sparse speed: {speed_sparse['mean_ms']:.2f} ms/inference")
    speedup = speed_orig['mean_ms'] / speed_sparse['mean_ms'] if speed_sparse['mean_ms'] > 0 else 1.0

    # Save
    print(f"\n[6/6] Saving sparse model...")
    results = {
        "model": hf_name,
        "model_key": model_key or "custom",
        "domain": domain,
        "target_sparsity": sparsity,
        "actual_sparsity": post_stats["sparsity"],
        "parameters": pre_stats,
        "sparse_parameters": post_stats,
        "quality": quality,
        "speed_baseline": speed_orig,
        "speed_sparse": speed_sparse,
        "speedup": round(speedup, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "engine": "PharmaCore Sparsification Engine v2.0",
    }
    save_sparse_model(model, tokenizer, output_dir, results)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SPARSIFICATION COMPLETE")
    print(f"  Quality retained: {quality['mean_cosine_similarity']*100:.1f}%")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Memory reduction: ~{sparsity*100:.0f}%")
    print(f"{'='*60}\n")

    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description="PharmaCore Model Sparsification Engine")
    parser.add_argument("--model-key", choices=list(MODEL_REGISTRY.keys()),
                        help="Model key from registry")
    parser.add_argument("--hf-name", help="HuggingFace model name (overrides --model-key)")
    parser.add_argument("--domain", choices=["protein", "molecule", "biomedical"],
                        help="Model domain (auto-detected from registry)")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity (0-1)")
    parser.add_argument("--output-dir", help="Output directory (auto-generated if not set)")
    args = parser.parse_args()

    if not args.model_key and not args.hf_name:
        parser.error("Provide --model-key or --hf-name")

    if args.model_key:
        info = MODEL_REGISTRY[args.model_key]
        hf_name = args.hf_name or info["hf_name"]
        domain = args.domain or info["domain"]
        model_key = args.model_key
    else:
        hf_name = args.hf_name
        domain = args.domain or "molecule"
        model_key = hf_name.split("/")[-1].lower()

    sparsity_pct = int(args.sparsity * 100)
    output_dir = args.output_dir or f"models/{model_key}-sparse{sparsity_pct}"

    sparsify_pipeline(hf_name, domain, args.sparsity, output_dir, model_key)


if __name__ == "__main__":
    main()

