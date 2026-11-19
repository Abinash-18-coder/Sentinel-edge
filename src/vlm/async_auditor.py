import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import queue
import threading
import time
from typing import Callable, NamedTuple

import cv2
import numpy as np

from src.vlm.grounding_renderer import VisualGroundingRenderer
from src.vlm.schemas import IncidentAuditReport
from src.vlm.vlm_engine import GroundedVLMAuditor


class AuditTask(NamedTuple):
    incident_id: str
    zone_id: str
    track_id: int
    entity_class: str
    jpeg_bytes: bytes
    hint_bbox_xyxy: tuple[float, float, float, float] | None
    output_json_path: Path


class AsyncIncidentAuditor:
    def __init__(
        self,
        max_queue_size: int = 12,
        track_debounce_sec: float = 8.0,
        vlm_mode: str = "auto",
        on_report_callback: Callable[[IncidentAuditReport], None] | None = None
    ) -> None:
        self.task_queue: queue.Queue[AuditTask | None] = queue.Queue(maxsize=max_queue_size)
        self.track_debounce_sec = track_debounce_sec
        self.vlm_mode = vlm_mode
        self.on_report_callback = on_report_callback

        self._debounce_lock = threading.Lock()
        self._last_submitted: dict[tuple[str, int], float] = {}

        self.completed_reports: list[IncidentAuditReport] = []
        self.dropped_count: int = 0
        self.debounced_count: int = 0
        self.auditor: GroundedVLMAuditor | None = None

        self._is_running = True
        self._ready_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self._ready_event.wait(timeout=15.0)
        print("[+] AsyncIncidentAuditor worker thread online (non-blocking 30 FPS safe).")

    def _worker_loop(self) -> None:
        self.auditor = GroundedVLMAuditor(mode=self.vlm_mode)
        self._ready_event.set()

        while self._is_running or not self.task_queue.empty():
            try:
                task = self.task_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if task is None:
                self.task_queue.task_done()
                break

            try:
                arr = np.frombuffer(task.jpeg_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise ValueError("Failed to decode queued JPEG snapshot.")

                report = self.auditor.audit_frame(
                    image_bgr=frame_bgr,
                    incident_id=task.incident_id,
                    zone_id=task.zone_id,
                    track_id=task.track_id,
                    entity_class=task.entity_class,
                    hint_bbox_xyxy=task.hint_bbox_xyxy
                )

                task.output_json_path.parent.mkdir(parents=True, exist_ok=True)
                task.output_json_path.write_text(
                    report.model_dump_json(indent=2),
                    encoding="utf-8"
                )

                grounded_img_path = task.output_json_path.with_suffix(".grounded.jpg")
                VisualGroundingRenderer.render_report_overlay(
                    image_bgr=frame_bgr,
                    report=report,
                    primary_bbox_xyxy=task.hint_bbox_xyxy,
                    output_path=grounded_img_path
                )

                self.completed_reports.append(report)
                if self.on_report_callback is not None:
                    self.on_report_callback(report)

                print(
                    f"[+] [ASYNC VLM] {task.incident_id} | {report.severity.value} "
                    f"| {report.audit_backend} ({report.audit_latency_ms:.1f}ms)"
                )
            except Exception as err:
                print(f"[!] [ASYNC VLM] Error auditing {task.incident_id}: {err}")
            finally:
                gc.collect()
                self.task_queue.task_done()

    def submit_task(
        self,
        incident_id: str,
        zone_id: str,
        track_id: int,
        entity_class: str,
        frame_bgr: np.ndarray,
        output_json_path: Path,
        hint_bbox_xyxy: tuple[float, float, float, float] | None = None,
        current_time: float | None = None
    ) -> bool:
        now = time.time() if current_time is None else current_time
        key = (zone_id, track_id)

        with self._debounce_lock:
            last_ts = self._last_submitted.get(key, -1e9)
            if (now - last_ts) < self.track_debounce_sec:
                self.debounced_count += 1
                return False
            self._last_submitted[key] = now

        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return False

        task = AuditTask(
            incident_id=incident_id,
            zone_id=zone_id,
            track_id=track_id,
            entity_class=entity_class,
            jpeg_bytes=buf.tobytes(),
            hint_bbox_xyxy=hint_bbox_xyxy,
            output_json_path=output_json_path
        )

        try:
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            self.dropped_count += 1
            print(f"[!] [ASYNC VLM] Queue full. Dropped audit for {incident_id}")
            return False

    def wait_all_completed(self) -> None:
        self.task_queue.join()

    def shutdown(self, timeout: float = 10.0) -> None:
        self._is_running = False
        self.task_queue.put(None)
        self.worker_thread.join(timeout=timeout)