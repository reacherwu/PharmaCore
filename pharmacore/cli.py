"""PharmaCore command-line interface."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from pharmacore import __version__


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__, prog_name="pharmacore")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """PharmaCore - Apple Silicon-native AI drug discovery platform."""
    setup_logging(verbose)


@main.command()
def info() -> None:
    """Show system and device information."""
    from pharmacore.core.device import DeviceManager

    dm = DeviceManager()
    click.echo(f"PharmaCore v{__version__}")
    click.echo(f"Device: {dm.detect_device()}")
    click.echo(f"MLX available: {dm.has_mlx()}")
    info = dm.device_info()
    for k, v in info.items():
        click.echo(f"  {k}: {v}")


@main.command()
@click.argument("smiles")
@click.option("--descriptors", "-d", is_flag=True, help="Show molecular descriptors.")
@click.option("--drug-likeness", "-l", is_flag=True, help="Check drug-likeness rules.")
@click.option("--admet", "-a", is_flag=True, help="Predict ADMET properties.")
def analyze(smiles: str, descriptors: bool, drug_likeness: bool, admet: bool) -> None:
    """Analyze a molecule given its SMILES string."""
    from pharmacore.core.types import Molecule
    from pharmacore.utils.chemistry import check_drug_likeness, compute_descriptors, parse_smiles

    mol = parse_smiles(smiles)
    if mol is None:
        click.echo(f"Error: Invalid SMILES: {smiles}", err=True)
        sys.exit(1)

    molecule = Molecule.from_smiles(smiles)
    click.echo(f"Molecule: {smiles}")

    if descriptors or not (drug_likeness or admet):
        desc = compute_descriptors(mol)
        click.echo("Descriptors:")
        for k, v in desc.items():
            click.echo(f"  {k}: {v}")

    if drug_likeness:
        dl = check_drug_likeness(mol)
        click.echo(f"Lipinski: {'PASS' if dl['lipinski_pass'] else 'FAIL'}")
        click.echo(f"Veber: {'PASS' if dl['veber_pass'] else 'FAIL'}")
        if dl["violations"]:
            click.echo(f"Violations: {', '.join(dl['violations'])}")

    if admet:
        from pharmacore.admet.predictor import ADMETPredictor
        predictor = ADMETPredictor()
        profile = predictor.predict(smiles)
        click.echo("ADMET Profile:")
        for category in ["absorption", "distribution", "metabolism", "excretion", "toxicity"]:
            val = getattr(profile, category, {})
            if isinstance(val, dict):
                click.echo(f"  {category}:")
                for k, v in val.items():
                    click.echo(f"    {k}: {v}")


@main.command()
@click.option("-n", "--num", default=10, help="Number of molecules to generate.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--output", "-o", default=None, help="Output file (JSON).")
@click.option("--similar-to", default=None, help="Generate molecules similar to this SMILES.")
def generate(num: int, seed: int | None, output: str | None, similar_to: str | None) -> None:
    """Generate drug-like molecules."""
    from pharmacore.generation.diffusion import MolecularGenerator

    gen = MolecularGenerator(seed=seed)

    if similar_to:
        from pharmacore.core.types import Molecule
        ref = Molecule.from_smiles(similar_to)
        molecules = gen.generate_similar(ref, n_molecules=num)
        click.echo(f"Generated {len(molecules)} molecules similar to {similar_to}:")
    else:
        molecules = gen.generate(n_molecules=num)
        click.echo(f"Generated {len(molecules)} drug-like molecules:")

    for m in molecules:
        props = m.properties or {}
        sim_str = f" (similarity: {props['similarity']:.3f})" if "similarity" in props else ""
        click.echo(f"  {m.smiles}{sim_str}")

    if output:
        data = [{"smiles": m.smiles, "name": m.name, "properties": m.properties} for m in molecules]
        Path(output).write_text(json.dumps(data, indent=2))
        click.echo(f"Saved to {output}")


@main.command()
@click.argument("target_name")
@click.option("--sequence", "-s", default="", help="Protein sequence.")
@click.option("--pdb", default=None, help="Protein PDB file for docking.")
@click.option("-n", "--num", default=10, help="Number of molecules to generate.")
@click.option("--output", "-o", default=None, help="Output file (JSON).")
@click.option("--stages", default=None, help="Comma-separated stages to run.")
def discover(
    target_name: str,
    sequence: str,
    pdb: str | None,
    num: int,
    output: str | None,
    stages: str | None,
) -> None:
    """Run the full drug discovery pipeline."""
    from pharmacore.pipeline.orchestrator import PipelineOrchestrator

    stage_list = stages.split(",") if stages else None
    orch = PipelineOrchestrator()
    result = orch.run(
        target_name=target_name,
        target_sequence=sequence,
        protein_pdb=pdb,
        n_molecules=num,
        stages=stage_list,
    )

    click.echo(f"Pipeline complete for target: {result.target}")
    click.echo(f"Molecules generated: {len(result.molecules)}")
    click.echo(f"Duration: {result.metadata.get('total_duration_seconds', 0):.1f}s")

    if result.molecules:
        click.echo("\nTop molecules:")
        for m in result.molecules[:5]:
            click.echo(f"  {m.smiles} (score: {m.properties.get('composite_score', 'N/A')})")

    if output:
        data = {
            "target": result.target,
            "molecules": [
                {"smiles": m.smiles, "name": m.name, "properties": m.properties}
                for m in result.molecules
            ],
            "metadata": result.metadata,
        }
        Path(output).write_text(json.dumps(data, indent=2, default=str))
        click.echo(f"\nResults saved to {output}")


@main.command()
@click.argument("disease")
def targets(disease: str) -> None:
    """Look up known drug targets for a disease."""
    from pharmacore.target.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    disease_node = kg.get_disease(disease)

    if disease_node is None:
        click.echo(f"Disease not found: {disease}")
        click.echo(f"Available: {', '.join(kg.list_diseases())}")
        return

    click.echo(f"Disease: {disease_node.name}")
    click.echo(f"Known targets: {', '.join(disease_node.targets)}")
    click.echo(f"Pathways: {', '.join(disease_node.pathways)}")


if __name__ == "__main__":
    main()
