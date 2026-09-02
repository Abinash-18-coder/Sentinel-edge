"""
src/vlm/vlm_engine.py
On-device Multimodal Vision-Language Model interface using Moondream2.
Performs visual scene question answering and formats structured incident audits.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.vlm.schemas import IncidentAuditReport, IncidentSeverity


class MoondreamAuditor:
    """
    Manages local Moondream2 weights, image encoding, and structured prompt querying.
    """

    MODEL_ID = "vikhyatk/moondream2"
    REVISION = "2024-08-26"  # Pinned stable revision

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[*] Initializing Moondream2 VLM on compute device: {self.device} ({self.dtype})...")
        
        # Load tokenizer and model with remote code execution enabled for Moondream
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            revision=self.REVISION,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            revision=self.REVISION,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device).eval()

        print("[+] Moondream2 VLM successfully initialized into memory.")

    def audit_incident(
        self,
        image_path: Path,
        incident_id: str,
        zone_id: str,
        track_id: int,
        entity_class: str
    ) -> IncidentAuditReport:
        """
        Analyzes an incident snapshot and outputs a validated IncidentAuditReport.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"[X] Incident snapshot not found: {image_path}")

        raw_image = Image.open(image_path).convert("RGB")

        # Encode image into visual tokens
        with torch.no_grad():
            image_embeds = self.model.encode_image(raw_image)

            # Query 1: Descriptive synthesis
            desc_prompt = (
                f"Describe the anomaly or defect involving '{entity_class}' in this scene in one precise sentence."
            )
            vlm_description = self.model.answer_question(image_embeds, desc_prompt, self.tokenizer)

            # Query 2: Severity assessment
            sev_prompt = (
                "Rate the severity of this safety breach or structural defect as strictly LOW, MEDIUM, HIGH, or CRITICAL."
            )
            raw_severity = self.model.answer_question(image_embeds, sev_prompt, self.tokenizer).strip().upper()

            # Map raw response to enum
            if "CRITICAL" in raw_severity:
                severity = IncidentSeverity.CRITICAL
            elif "HIGH" in raw_severity:
                severity = IncidentSeverity.HIGH
            elif "MEDIUM" in raw_severity:
                severity = IncidentSeverity.MEDIUM
            else:
                severity = IncidentSeverity.LOW

            # Query 3: Recommended action
            action_prompt = "What is the immediate corrective action required for this event in under 10 words?"
            recommended_action = self.model.answer_question(image_embeds, action_prompt, self.tokenizer)

        report = IncidentAuditReport(
            incident_id=incident_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            zone_id=zone_id,
            track_id=track_id,
            entity_class=entity_class,
            vlm_scene_description=vlm_description.strip(),
            severity=severity,
            recommended_action=recommended_action.strip()
        )

        return report


if __name__ == "__main__":
    # Test script with a synthetic test frame
    project_root = Path(__file__).resolve().parent.parent.parent
    test_snapshot = project_root / "data" / "calibration" / "calib_000.jpg"

    if not test_snapshot.exists():
        test_snapshot.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (640, 640), color=(128, 50, 50))
        img.save(test_snapshot)

    auditor = MoondreamAuditor()
    report = auditor.audit_incident(
        image_path=test_snapshot,
        incident_id="incident_test_001",
        zone_id="zone_alpha_danger",
        track_id=1,
        entity_class="crack"
    )

    print("\n" + "=" * 60)
    print(" VLM AUDIT REPORT GENERATED")
    print("=" * 60)
    print(report.model_dump_json(indent=2))