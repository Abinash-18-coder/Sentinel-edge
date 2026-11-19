import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.vlm.schemas import IncidentAuditReport, IncidentSeverity


class VisualGroundingRenderer:
    SEVERITY_COLORS_BGR = {
        IncidentSeverity.LOW: (94, 197, 34),
        IncidentSeverity.MEDIUM: (8, 179, 234),
        IncidentSeverity.HIGH: (22, 115, 249),
        IncidentSeverity.CRITICAL: (68, 68, 239)
    }

    @staticmethod
    def render_report_overlay(
        image_bgr: np.ndarray,
        report: IncidentAuditReport,
        primary_bbox_xyxy: tuple[float, float, float, float] | None = None,
        output_path: Path | None = None
    ) -> np.ndarray:
        annotated = image_bgr.copy()
        img_h, img_w = annotated.shape[:2]

        color = VisualGroundingRenderer.SEVERITY_COLORS_BGR.get(
            report.severity, (68, 68, 239)
        )

        if primary_bbox_xyxy is not None:
            ux1, uy1, ux2, uy2 = [int(v) for v in primary_bbox_xyxy]
            ux1 = max(0, min(ux1, img_w - 1))
            uy1 = max(0, min(uy1, img_h - 1))
            ux2 = max(0, min(ux2, img_w - 1))
            uy2 = max(0, min(uy2, img_h - 1))
            cv2.rectangle(annotated, (ux1, uy1), (ux2, uy2), (80, 220, 80), 2, lineType=cv2.LINE_AA)

        box = report.hazard_grounding_box
        px1 = max(0, min(int((box.xmin / 1000.0) * img_w), img_w - 1))
        py1 = max(0, min(int((box.ymin / 1000.0) * img_h), img_h - 1))
        px2 = max(0, min(int((box.xmax / 1000.0) * img_w), img_w - 1))
        py2 = max(0, min(int((box.ymax / 1000.0) * img_h), img_h - 1))

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3, lineType=cv2.LINE_AA)

        hud_h = 64
        cv2.rectangle(annotated, (0, 0), (img_w, hud_h), (42, 23, 15), -1)

        title_txt = f"[{report.severity.value}] {report.zone_id} | Track #{report.track_id} ({report.audit_latency_ms:.0f}ms)"
        cv2.putText(
            annotated,
            title_txt,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

        action_txt = f"ACTION: {report.recommended_action[:65]}"
        cv2.putText(
            annotated,
            action_txt,
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 232, 226),
            1,
            cv2.LINE_AA
        )

        tag_txt = f"GROUNDED: {report.entity_class.upper()}"
        (tw, th), _ = cv2.getTextSize(tag_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_y2 = max(py1, hud_h + th + 10)
        tag_y1 = tag_y2 - th - 8
        tag_x2 = min(img_w - 1, px1 + tw + 10)
        cv2.rectangle(annotated, (px1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(
            annotated,
            tag_txt,
            (px1 + 5, tag_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

        return annotated