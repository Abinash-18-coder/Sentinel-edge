"""
src/vlm/async_auditor.py
Thread-safe asynchronous worker queue executing background Grounded VLM
auditing tasks without blocking the main 30 FPS perception stream.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np

from src.vlm.schemas import IncidentAuditReport
from src.vlm.vlm_engine import GroundedVLMAuditor
from src.vlm.grounding_renderer import VisualGroundingRenderer


class AuditTask(NamedTuple):
    incident_id: str
    zone_id: str
    track_id: int
    entity_class: str
    frame_bgr: np.ndarray
    output_json_path: Path


class AsyncIncidentAuditor:
    """Manages background thread executing grounded VLM inference and rendering."""

    def __init__(self, max_queue_size: int = 32) -> None:
        self.task_queue: queue.Queue[AuditTask | None] = queue.Queue(maxsize=max_queue_size)
        self.auditor: GroundedVLMAuditor | None = None
        self.is_running = True

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[+] Async Grounded VLM Worker Thread initialized.")

    def _worker_loop(self) -> None:
        self.auditor = GroundedVLMAuditor()

        while self.is_running:
            try:
                task = self.task_queue.get(timeout=0.5)
                if task is None:
                    break
            except queue.Empty:
                continue

            try:
                raw_snapshot_path = task.output_json_path.with_suffix(".raw.jpg")
                cv2.imwrite(str(raw_snapshot_path), task.frame_bgr)

                # Execute Grounded VLM Analysis
                report: IncidentAuditReport = self.auditor.audit_incident(
                    image_path=raw_snapshot_path,
                    incident_id=task.incident_id,
                    zone_id=task.zone_id,
                    track_id=task.track_id,
                    entity_class=task.entity_class
                )

                # Save JSON Audit Report
                with open(task.output_json_path, "w", encoding="utf-8") as f:
                    f.write(report.model_dump_json(indent=2))

                # Render Secondary Grounding Visual Overlay
                grounded_render_path = task.output_json_path.with_suffix(".grounded.jpg")
                VisualGroundingRenderer.render_report_overlay(
                    image_bgr=task.frame_bgr,
                    report=report,
                    output_path=grounded_render_path
                )

                print(f"\n[+] [GROUNDED VLM AUDIT] Incident: {task.incident_id}")
                print(f"    Severity : {report.severity.value}")
                print(f"    Action   : {report.recommended_action}")
                print(f"    Grounded : Box [ymin={report.hazard_grounding_box.ymin}, xmin={report.hazard_grounding_box.xmin}]")

            except Exception as err:
                print(f"[!] Error processing {task.incident_id}: {err}")
            finally:
                self.task_queue.task_done()

    def submit_task(
        self,
        incident_id: str,
        zone_id: str,
        track_id: int,
        entity_class: str,
        frame_bgr: np.ndarray,
        output_json_path: Path
    ) -> bool:
        task = AuditTask(
            incident_id=incident_id,
            zone_id=zone_id,
            track_id=track_id,
            entity_class=entity_class,
            frame_bgr=frame_bgr.copy(),
            output_json_path=output_json_path
        )
        try:
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            print(f"[!] Warning: Task queue full. Dropping audit for {incident_id}")
            return False

    def shutdown(self) -> None:
        self.is_running = False
        self.task_queue.put(None)
        self.worker_thread.join(timeout=3.0)