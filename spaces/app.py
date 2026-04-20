import gradio as gr
import sys
import os
import time

# Add parent to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing pharmacore modules
try:
    from pharmacore.discovery import DeNovoDiscoveryEngine
    from pharmacore.repurposing import DrugRepurposingEngine, KNOWN_DRUGS
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


# --- De Novo Discovery ---
def run_discovery(target_name, target_sequence, n_molecules, seed):
    if not MODULES_AVAILABLE:
        return format_discovery_demo(target_name)

    try:
        engine = DeNovoDiscoveryEngine(seed=int(seed))
        start = time.time()
        result = engine.discover(
            target_name=target_name,
            target_sequence=target_sequence if target_sequence.strip() else None,
            n_molecules=int(n_molecules),
        )
        elapsed = time.time() - start

        lines = [f"## Results for {target_name}", f"Generated {len(result.molecules)} candidates in {elapsed:.1f}s\n"]
        lines.append("| Rank | Name | Score | Scaffold | SMILES |")
        lines.append("|------|------|-------|----------|--------|")
        for i, mol in enumerate(result.molecules, 1):
            lines.append(f"| {i} | {mol.name} | {mol.composite_score:.3f} | {mol.scaffold_name} | `{mol.smiles}` |")

        # Top candidate details
        top = result.molecules[0]
        lines.append(f"\n### Top Candidate: {top.name}")
        lines.append(f"- **Scaffold:** {top.scaffold_name}")
        lines.append(f"- **QED (Drug-likeness):** {top.qed:.3f}")
        lines.append(f"- **Target Compatibility:** {top.target_score:.3f}")
        lines.append(f"- **Synthetic Accessibility:** {top.sa_score:.3f}")
        lines.append(f"- **Lipinski:** {'PASS' if top.lipinski_pass else 'FAIL'}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


def format_discovery_demo(target_name):
    """Fallback demo output when modules aren't available (for HF Space)"""
    return f"""## Results for {target_name}
Generated 5 candidates in ~8s (demo mode — full inference requires Apple Silicon)

| Rank | Name | Score | Scaffold | SMILES |
|------|------|-------|----------|--------|
| 1 | PC-{target_name[:4].upper()}-0001 | 0.849 | quinazoline | `NC(=O)c1c(O)ccc2ncc(-c3ccncc3)nc12` |
| 2 | PC-{target_name[:4].upper()}-0002 | 0.799 | quinoline | `FC(F)(F)c1ccc2cccnc2c1` |
| 3 | PC-{target_name[:4].upper()}-0003 | 0.795 | benzimidazole | `CNC(=O)c1ccc2[nH]cnc2c1` |
| 4 | PC-{target_name[:4].upper()}-0004 | 0.791 | quinoline | `c1cnc2ccc(-c3ccncc3)cc2c1` |
| 5 | PC-{target_name[:4].upper()}-0005 | 0.770 | indole | `O=C(O)c1cc2[nH]ccc2c(C(=O)O)c1C(=O)O` |

### Top Candidate: PC-{target_name[:4].upper()}-0001
- **Scaffold:** quinazoline (known kinase inhibitor scaffold)
- **QED (Drug-likeness):** 0.731
- **Target Compatibility:** 0.900
- **Synthetic Accessibility:** 1.000
- **Lipinski:** PASS

> 💡 *This is a demo preview. For real-time inference, clone the repo and run on Apple Silicon.*
"""


# --- Drug Repurposing ---
def run_repurposing(target_name, target_sequence, reference_smiles, top_k):
    if not MODULES_AVAILABLE:
        return format_repurposing_demo(target_name)

    try:
        engine = DrugRepurposingEngine()
        start = time.time()
        result = engine.screen(
            target_name=target_name,
            target_sequence=target_sequence if target_sequence.strip() else None,
            reference_smiles=reference_smiles if reference_smiles.strip() else None,
            top_k=int(top_k),
        )
        elapsed = time.time() - start

        lines = [f"## Repurposing Screen for {target_name}", f"Screened {len(KNOWN_DRUGS)} FDA-approved drugs in {elapsed:.1f}s\n"]
        lines.append("| Rank | Drug | Score | Confidence | Original Indication |")
        lines.append("|------|------|-------|------------|---------------------|")
        for i, c in enumerate(result.candidates, 1):
            lines.append(f"| {i} | {c.drug_name} | {c.composite_score:.3f} | {c.confidence} | {c.original_indication} |")

        top = result.candidates[0]
        lines.append(f"\n### Top Candidate: {top.drug_name}")
        lines.append(f"- **Original Use:** {top.original_indication}")
        lines.append(f"- **Mechanism:** {top.mechanism}")
        lines.append(f"- **Protein Compatibility:** {top.protein_score:.1%}")
        lines.append(f"- **Molecular Similarity:** {top.molecular_similarity:.1%}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


def format_repurposing_demo(target_name):
    """Fallback demo output"""
    return f"""## Repurposing Screen for {target_name}
Screened 12 FDA-approved drugs (demo mode)

| Rank | Drug | Score | Confidence | Original Indication |
|------|------|-------|------------|---------------------|
| 1 | Erlotinib | 0.699 | medium | Non-small cell lung cancer |
| 2 | Sorafenib | 0.312 | low | Renal cell carcinoma |
| 3 | Sildenafil | 0.288 | low | Erectile dysfunction |
| 4 | Celecoxib | 0.265 | low | Arthritis pain |
| 5 | Remdesivir | 0.264 | low | Ebola (repurposed for COVID-19) |

### Top Candidate: Erlotinib
- **Original Use:** Non-small cell lung cancer
- **Mechanism:** EGFR tyrosine kinase inhibitor
- **Protein Compatibility:** 14.0%
- **Molecular Similarity:** 100.0%

> ✅ Erlotinib is a known EGFR inhibitor — correctly identified as top candidate.
> 💡 *Demo preview. For real inference, run on Apple Silicon locally.*
"""


# --- Gradio Interface ---
with gr.Blocks(
    title="PharmaCore — AI Drug Discovery",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
) as demo:
    gr.Markdown("""
# 🧬 PharmaCore — AI Drug Discovery on Apple Silicon

**The first AI drug discovery platform that runs entirely on consumer hardware.**
No cloud GPUs. No API keys. No data leaves your machine.

[GitHub](https://github.com/reacherwu/PharmaCore) | [Models](https://huggingface.co/collections/stephenjun8192/pharmacore-sparse-models-69e5842a51579e4b12d42f30)
""")

    with gr.Tab("🧬 De Novo Discovery"):
        gr.Markdown("Generate novel drug candidates for a protein target using sparse AI models.")
        with gr.Row():
            with gr.Column():
                target_name_disc = gr.Textbox(label="Target Name", value="EGFR kinase", placeholder="e.g., EGFR kinase, BRAF V600E")
                target_seq_disc = gr.Textbox(label="Target Sequence (optional)", value="", placeholder="Protein amino acid sequence...", lines=3)
                n_mols = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of Molecules")
                seed = gr.Number(label="Random Seed", value=42)
                btn_disc = gr.Button("🚀 Generate Candidates", variant="primary")
            with gr.Column():
                output_disc = gr.Markdown(label="Results")
        btn_disc.click(run_discovery, inputs=[target_name_disc, target_seq_disc, n_mols, seed], outputs=output_disc)

    with gr.Tab("💊 Drug Repurposing"):
        gr.Markdown("Screen existing FDA-approved drugs for new therapeutic uses.")
        with gr.Row():
            with gr.Column():
                target_name_rep = gr.Textbox(label="Target Name", value="EGFR", placeholder="e.g., EGFR, ACE2, BRAF")
                target_seq_rep = gr.Textbox(label="Target Sequence (optional)", value="", placeholder="Protein amino acid sequence...", lines=3)
                ref_smiles = gr.Textbox(label="Reference SMILES (optional)", value="COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", placeholder="Known ligand SMILES for similarity scoring")
                top_k = gr.Slider(minimum=3, maximum=12, value=5, step=1, label="Top K Results")
                btn_rep = gr.Button("🔍 Screen Drugs", variant="primary")
            with gr.Column():
                output_rep = gr.Markdown(label="Results")
        btn_rep.click(run_repurposing, inputs=[target_name_rep, target_seq_rep, ref_smiles, top_k], outputs=output_rep)

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
## How It Works

PharmaCore uses **sparse AI models** (50% pruned) for efficient inference:

| Model | Role | Params | Speed (M4) |
|-------|------|--------|------------|
| ESM-2 35M | Protein encoding | 33.5M → 16.7M | 7.8ms |
| ChemBERTa-zinc | Molecule encoding | 44.1M → 22M | 4.9ms |

### De Novo Discovery Pipeline
1. Encode protein target with sparse ESM-2
2. Enumerate drug-like scaffolds (quinazoline, quinoline, benzimidazole, etc.)
3. Score candidates: QED + target compatibility + synthetic accessibility
4. Rank and filter by Lipinski/Veber rules

### Drug Repurposing Pipeline
1. Encode target protein and reference ligand
2. Compute protein-drug compatibility for 12 FDA-approved drugs
3. Calculate molecular fingerprint similarity
4. Rank by composite score with confidence levels

### Key Differentiators
- **100% Local** — no data leaves your machine
- **Apple Silicon MPS** — optimized for M1/M2/M3/M4
- **Transparent** — full audit trail for every computation
- **Fast** — sub-10ms protein inference, sub-5ms molecular inference
- **Open Source** — MIT licensed, all models on HuggingFace
""")

if __name__ == "__main__":
    demo.launch()
