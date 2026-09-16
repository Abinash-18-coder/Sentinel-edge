"""
tests/test_week5_integration.py
Automated regression test verifying Re-ID appearance extraction and
grounded VLM schema validation across synthetic sample frames.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from src.perception.reid_extractor import AppearanceEmbeddingExtractor
from src.vlm.schemas import IncidentAuditReport, GroundingBoundingBox, IncidentSeverity
from src.vlm.grounding_renderer import VisualGroundingRenderer


def test_week5_end_to_end() -> None:
    print("=" * 65)
    print(" SentinelEdge v2.0: Week 5 Integration Regression Suite")
    print("=" * 65)

    # 1. Test Re-ID Appearance Feature Extraction
    print("[*] Stage 1: Testing Re-ID Appearance Feature Extraction...")
    extractor = AppearanceEmbeddingExtractor(device="cpu")
    dummy_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    boxes = np.array([[50, 50, 200, 200], [250, 100, 400, 350]], dtype=np.float32)

    embeddings = extractor.extract_crops(dummy_frame, boxes)
    assert embeddings.shape == (2, 512), f"Expected shape (2, 512), got {embeddings.shape}"
    
    # Verify L2 Unit Normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)
    print("[+] Re-ID Extractor: 512-dim L2-normalized feature verification passed.")

    # 2. Test Grounded Report Construction & Rendering
    print("[*] Stage 2: Testing Grounding Visual Renderer...")
    report = IncidentAuditReport(
        incident_id="test_incident_wk5",
        timestamp="2026-09-16T12:00:00Z",
        zone_id="zone_alpha_danger",
        track_id=101,
        entity_class="crack",
        vlm_scene_description="Longitudinal surface crack extending toward electrical junction.",
        hazard_grounding_box=GroundingBoundingBox(ymin=120, xmin=140, ymax=450, xmax=380),
        severity=IncidentSeverity.CRITICAL,
        recommended_action="Isolate sector power"
    )

    out_render = Path("data/raw/incidents/test_grounded_render.jpg")
    rendered_img = VisualGroundingRenderer.render_report_overlay(
        image_bgr=dummy_frame,
        report=report,
        output_path=out_render
    )

    assert rendered_img.shape == dummy_frame.shape, "Rendered image dimensions altered!"
    print(f"[+] Grounded Visual Renderer output verified at: {out_render}")

    if out_render.exists():
        out_render.unlink()

    print("=" * 65)
    print(" ALL WEEK 5 REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_week5_end_to_end()