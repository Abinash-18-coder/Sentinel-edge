"""
src/vlm/async_auditor.py
Thread-safe asynchronous worker queue executing background Moondream2 VLM
auditing tasks without blocking the main 30 FPS perception stream.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np

from src.vlm.schemas import IncidentAuditReport
from src.vlm.vlm_engine import MoondreamAuditor


class AuditTask(NamedTuple):
    incident_id: str
    zone_id: str
    track_id: int
    entity_class: str
    frame_bgr: np.ndarray
    output_json_path: Path


class AsyncIncidentAuditor:
    """
    Manages a background daemon worker thread consuming anomaly snapshots
    and writing audit logs to disk.
    """

    def __init__(self, max_queue_size: int = 32) -> None:
        self.task_queue: queue.Queue[AuditTask | None] = queue.Queue(maxsize=max_queue_size)
        self.auditor: MoondreamAuditor | None = None
        self.is_running = True
        
        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[+] AsyncIncidentAuditor worker thread spawned successfully.")

    def _worker_loop(self) -> None:
        """Background thread executing VLM inference sequentially."""
        # Lazy initialization inside the worker thread
        self.auditor = MoondreamAuditor()

        while self.is_running:
            try:
                task = self.task_queue.get(timeout=0.5)
                if task is None:
                    break  # Termination signal
            except queue.Empty:
                continue

            try:
                # Save snapshot to temporary image file
                temp_img_path = task.output_json_path.with_suffix(".snapshot.jpg")
                cv2.imwrite(str(temp_img_path), task.frame_bgr)

                # Run VLM inference
                report: IncidentAuditReport = self.auditor.audit_incident(
                    image_path=temp_img_path,
                    incident_id=task.incident_id,
                    zone_id=task.zone_id,
                    track_id=task.track_id,
                    entity_class=task.entity_class
                )

                # Save JSON report
                with open(task.output_json_path, "w", encoding="utf-8") as f:
                    f.write(report.model_dump_json(indent=2))

                print(f"\n[+] [ASYNC VLM WORKER] Generated Audit Report for: {task.incident_id}")
                print(f"    Severity: {report.severity.value} | Action: {report.recommended_action}")

            except Exception as err:
                print(f"[!] [ASYNC VLM WORKER] Error processing {task.incident_id}: {err}")
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
        """Non-blocking submission of anomaly snapshots from the main stream."""
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
            print(f"[!] [ASYNC VLM WORKER] Warning: Task queue full. Dropping audit for {incident_id}")
            return False

    def shutdown(self) -> None:
        """Gracefully shuts down the background worker."""
        self.is_running = False
        self.task_queue.put(None)
        self.worker_thread.join(timeout=3.0)
        print("[+] AsyncIncidentAuditor shutdown complete.")