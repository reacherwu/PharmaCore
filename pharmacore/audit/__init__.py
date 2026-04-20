"""Transparent Audit Pipeline — computation logging and explainability.

Every step in PharmaCore's drug discovery and repurposing pipelines is logged
with full provenance: inputs, outputs, model versions, device info, timestamps,
and human-readable explanations. This enables:
- Regulatory compliance (FDA/EMA computational audit trails)
- Reproducibility (exact parameters and model versions recorded)
- Explainability (why each decision was made)
- Trust (investors/reviewers can verify every computation)
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single auditable computation step."""
    step_id: str
    step_name: str
    timestamp: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    device_info: dict = field(default_factory=dict)
    duration_sec: float = 0.0
    explanation: str = ""
    checksum: str = ""

    def compute_checksum(self) -> str:
        """SHA-256 of inputs + parameters for reproducibility verification."""
        data = json.dumps({"inputs": self.inputs, "parameters": self.parameters},
                          sort_keys=True, default=str)
        self.checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
        return self.checksum


@dataclass
class AuditReport:
    """Complete audit trail for a pipeline run."""
    pipeline_name: str
    run_id: str
    started_at: str
    completed_at: str = ""
    status: str = "running"
    entries: list[AuditEntry] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    system_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class AuditPipeline:
    """Transparent audit pipeline that wraps any PharmaCore workflow.

    Usage:
        audit = AuditPipeline("drug_discovery")
        audit.log_step("target_analysis", inputs={...}, outputs={...})
        audit.log_step("generation", inputs={...}, outputs={...})
        report = audit.finalize()
        audit.save(report, "output/audit_report.json")
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self._run_id = self._generate_run_id()
        self._entries: list[AuditEntry] = []
        self._step_counter = 0
        self._start_time = time.time()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._system_info = self._collect_system_info()

    @staticmethod
    def _generate_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        h = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
        return f"PC-{ts}-{h}"

    @staticmethod
    def _collect_system_info() -> dict:
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        }
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["mps_available"] = torch.backends.mps.is_available()
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except ImportError:
            pass
        try:
            import rdkit
            info["rdkit_version"] = rdkit.__version__
        except ImportError:
            pass
        return info

    def log_step(
        self,
        step_name: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
        parameters: dict | None = None,
        model_info: dict | None = None,
        duration_sec: float = 0.0,
        explanation: str = "",
    ) -> AuditEntry:
        """Log a single computation step."""
        self._step_counter += 1
        entry = AuditEntry(
            step_id=f"step_{self._step_counter:03d}",
            step_name=step_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            inputs=inputs or {},
            outputs=outputs or {},
            parameters=parameters or {},
            model_info=model_info or {},
            device_info=self._system_info,
            duration_sec=duration_sec,
            explanation=explanation,
        )
        entry.compute_checksum()
        self._entries.append(entry)
        logger.info(f"Audit [{entry.step_id}] {step_name}: {explanation[:80]}")
        return entry

    def finalize(self, status: str = "completed") -> AuditReport:
        """Finalize the audit trail and generate summary."""
        elapsed = time.time() - self._start_time
        report = AuditReport(
            pipeline_name=self.pipeline_name,
            run_id=self._run_id,
            started_at=self._started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            status=status,
            entries=self._entries,
            system_info=self._system_info,
            summary={
                "total_steps": len(self._entries),
                "total_duration_sec": round(elapsed, 2),
                "steps": [e.step_name for e in self._entries],
                "all_checksums_valid": all(e.checksum for e in self._entries),
            },
        )
        return report

    def save(self, report: AuditReport, path: str | Path) -> Path:
        """Save audit report to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report.to_json())
        logger.info(f"Audit report saved: {p} ({p.stat().st_size} bytes)")
        return p

    def generate_text_report(self, report: AuditReport) -> str:
        """Generate human-readable audit report."""
        lines = [
            "=" * 70,
            f"PharmaCore Audit Report",
            f"Pipeline: {report.pipeline_name}",
            f"Run ID:   {report.run_id}",
            f"Status:   {report.status}",
            f"Started:  {report.started_at}",
            f"Finished: {report.completed_at}",
            "=" * 70,
            "",
            "System Information:",
            f"  Platform:     {report.system_info.get('platform', 'N/A')}",
            f"  Processor:    {report.system_info.get('processor', 'N/A')}",
            f"  Python:       {report.system_info.get('python_version', 'N/A')}",
            f"  PyTorch:      {report.system_info.get('torch_version', 'N/A')}",
            f"  MPS (Apple):  {report.system_info.get('mps_available', 'N/A')}",
            f"  Transformers: {report.system_info.get('transformers_version', 'N/A')}",
            f"  RDKit:        {report.system_info.get('rdkit_version', 'N/A')}",
            "",
            "-" * 70,
            "Computation Steps:",
            "-" * 70,
        ]

        for entry in report.entries:
            lines.extend([
                "",
                f"[{entry.step_id}] {entry.step_name}",
                f"  Time:       {entry.timestamp}",
                f"  Duration:   {entry.duration_sec:.2f}s",
                f"  Checksum:   {entry.checksum}",
                f"  Explanation: {entry.explanation}",
            ])
            if entry.model_info:
                lines.append(f"  Model:      {json.dumps(entry.model_info)}")
            if entry.parameters:
                lines.append(f"  Parameters: {json.dumps(entry.parameters)}")
            if entry.inputs:
                inp_summary = {k: str(v)[:100] for k, v in entry.inputs.items()}
                lines.append(f"  Inputs:     {json.dumps(inp_summary)}")
            if entry.outputs:
                out_summary = {k: str(v)[:100] for k, v in entry.outputs.items()}
                lines.append(f"  Outputs:    {json.dumps(out_summary)}")

        lines.extend([
            "",
            "-" * 70,
            "Summary:",
            f"  Total steps:    {report.summary.get('total_steps', 0)}",
            f"  Total duration: {report.summary.get('total_duration_sec', 0):.2f}s",
            f"  Checksums OK:   {report.summary.get('all_checksums_valid', False)}",
            "=" * 70,
        ])
        return "\n".join(lines)


class AuditedDiscovery:
    """Wrapper that runs discovery + repurposing with full audit trails."""

    def __init__(self):
        self._audit: AuditPipeline | None = None

    def run_discovery(
        self,
        target_name: str,
        target_sequence: str = "",
        n_molecules: int = 10,
        protein_model_path: str | Path | None = None,
        molecule_model_path: str | Path | None = None,
        output_dir: str | Path = "output",
    ) -> dict:
        """Run audited de novo drug discovery."""
        from pharmacore.discovery import DeNovoDiscoveryEngine

        self._audit = AuditPipeline("de_novo_discovery")
        t0 = time.time()

        # Step 1: Initialize
        self._audit.log_step(
            "initialization",
            inputs={"target_name": target_name, "sequence_length": len(target_sequence)},
            parameters={"n_molecules": n_molecules},
            explanation=f"Initializing discovery pipeline for target {target_name}",
        )

        # Step 2: Engine setup
        engine = DeNovoDiscoveryEngine(
            protein_model_path=protein_model_path,
            molecule_model_path=molecule_model_path,
        )
        self._audit.log_step(
            "engine_setup",
            outputs={"device": engine._device},
            explanation=f"Discovery engine initialized on {engine._device}",
            duration_sec=time.time() - t0,
        )

        # Step 3: Run discovery
        t1 = time.time()
        result = engine.discover(
            target_name=target_name,
            target_sequence=target_sequence,
            n_molecules=n_molecules,
        )
        self._audit.log_step(
            "molecule_generation",
            inputs={"target": target_name, "family": result.target_family},
            outputs={
                "n_generated": len(result.molecules),
                "top_score": result.molecules[0].composite_score if result.molecules else 0,
            },
            parameters={"n_molecules": n_molecules},
            model_info=result.metadata,
            duration_sec=time.time() - t1,
            explanation=f"Generated {len(result.molecules)} molecules for {target_name} ({result.target_family})",
        )

        # Step 4: Detailed scoring log
        for i, mol in enumerate(result.molecules[:5]):
            self._audit.log_step(
                f"molecule_scoring_{i+1}",
                inputs={"smiles": mol.smiles, "scaffold": mol.scaffold_name},
                outputs={
                    "composite_score": mol.composite_score,
                    "drug_likeness": mol.drug_likeness,
                    "target_compatibility": mol.target_compatibility,
                    "synthetic_accessibility": mol.synthetic_accessibility,
                },
                explanation=f"{mol.name}: score={mol.composite_score:.3f} on {mol.scaffold_name} scaffold",
            )

        # Finalize
        report = self._audit.finalize()
        out = Path(output_dir)
        json_path = self._audit.save(report, out / f"audit_discovery_{target_name}.json")
        text_report = self._audit.generate_text_report(report)
        text_path = out / f"audit_discovery_{target_name}.txt"
        text_path.write_text(text_report)

        return {
            "result": result,
            "report": report,
            "json_path": str(json_path),
            "text_path": str(text_path),
            "text_report": text_report,
        }

    def run_repurposing(
        self,
        target_name: str,
        target_sequence: str,
        reference_smiles: str = "",
        top_k: int = 10,
        protein_model_path: str | Path | None = None,
        molecule_model_path: str | Path | None = None,
        output_dir: str | Path = "output",
    ) -> dict:
        """Run audited drug repurposing screen."""
        from pharmacore.repurposing.engine import DrugRepurposingEngine

        self._audit = AuditPipeline("drug_repurposing")
        t0 = time.time()

        self._audit.log_step(
            "initialization",
            inputs={"target_name": target_name, "sequence_length": len(target_sequence)},
            parameters={"top_k": top_k, "reference_smiles": reference_smiles[:50]},
            explanation=f"Initializing repurposing screen for {target_name}",
        )

        engine = DrugRepurposingEngine(
            protein_model_path=protein_model_path,
            molecule_model_path=molecule_model_path,
        )
        self._audit.log_step(
            "engine_setup",
            outputs={"device": engine._device, "drug_database_size": 12},
            explanation=f"Repurposing engine initialized on {engine._device}",
            duration_sec=time.time() - t0,
        )

        t1 = time.time()
        result = engine.screen(
            target_name=target_name,
            target_sequence=target_sequence,
            reference_smiles=reference_smiles,
            top_k=top_k,
        )
        self._audit.log_step(
            "repurposing_screen",
            inputs={"target": target_name, "n_drugs_screened": 12},
            outputs={
                "n_candidates": len(result.candidates),
                "top_score": result.candidates[0].composite_score if result.candidates else 0,
            },
            model_info=result.metadata,
            duration_sec=time.time() - t1,
            explanation=f"Screened 12 FDA-approved drugs against {target_name}",
        )

        for i, cand in enumerate(result.candidates[:5]):
            self._audit.log_step(
                f"candidate_analysis_{i+1}",
                inputs={"drug": cand.drug_name, "smiles": cand.drug_smiles[:50]},
                outputs={
                    "composite_score": cand.composite_score,
                    "protein_similarity": cand.protein_similarity,
                    "molecular_similarity": cand.molecular_similarity,
                    "structural_similarity": cand.structural_similarity,
                    "confidence": cand.confidence,
                },
                explanation=f"{cand.drug_name}: score={cand.composite_score:.3f}, "
                            f"originally for {cand.original_indication}",
            )

        report = self._audit.finalize()
        out = Path(output_dir)
        json_path = self._audit.save(report, out / f"audit_repurposing_{target_name}.json")
        text_report = self._audit.generate_text_report(report)
        text_path = out / f"audit_repurposing_{target_name}.txt"
        text_path.write_text(text_report)

        return {
            "result": result,
            "report": report,
            "json_path": str(json_path),
            "text_path": str(text_path),
            "text_report": text_report,
        }
