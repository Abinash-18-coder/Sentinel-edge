"""
src/vlm/grounding_renderer.py
Renders high-visibility visual grounding bounding boxes, severity badges,
and corrective action annotations onto incident verification snapshots.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from src.vlm.schemas import IncidentAuditReport, IncidentSeverity


class VisualGroundingRenderer:
    """Overlays secondary hazard coordinates and severity badges onto image frames."""

    SEVERITY_COLORS = {
        IncidentSeverity.LOW: (34, 197, 94),       # Emerald Green
        IncidentSeverity.MEDIUM: (234, 179, 8),    # Amber
        IncidentSeverity.HIGH: (249, 115, 22),     # Orange
        IncidentSeverity.CRITICAL: (239, 68, 68),  # Crimson Red
    }

    @staticmethod
    def render_report_overlay(
        image_bgr: np.ndarray,
        report: IncidentAuditReport,
        output_path: Path | None = None
    ) -> np.ndarray:
        """
        Denormalizes Qwen2.5-VL [0, 1000] coordinates and renders the secondary grounding box.
        """
        annotated = image_bgr.copy()
        img_h, img_w = annotated.shape[:2]

        # Convert [0, 1000] coordinates to absolute pixel coordinates
        box = report.hazard_grounding_box
        px1 = int((box.xmin / 1000.0) * img_w)
        py1 = int((box.ymin / 1000.0) * img_h)
        px2 = int((box.xmax / 1000.0) * img_w)
        py2 = int((box.ymax / 1000.0) * img_h)

        color = VisualGroundingRenderer.SEVERITY_COLORS.get(
            report.severity, (239, 68, 68)
        )

        # 1. Draw Secondary Grounded Box (Root-cause violation)
        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2, lineType=cv2.LINE_AA)

        # 2. Draw Top HUD Banner
        hud_h = 60
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (15, 23, 42), -1)

        # Title & Severity Badge
        title_txt = f"[{report.severity.value}] {report.zone_id} | Track #{report.track_id}"
        cv2.putText(annotated, title_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Recommended Action Text
        action_txt = f"ACTION: {report.recommended_action}"
        cv2.putText(annotated, action_txt, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1, cv2.LINE_AA)

        # 3. Label tag above secondary box
        tag_txt = f"HAZARD ROOT-CAUSE: {report.entity_class}"
        cv2.rectangle(annotated, (px1, max(py1 - 20, hud_h)), (px1 + 220, max(py1, hud_h + 20)), color, -1)
        cv2.putText(annotated, tag_txt, (px1 + 5, max(py1 - 5, hud_h + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

        return annotated