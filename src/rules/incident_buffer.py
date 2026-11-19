import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import json
import queue
import threading
import time
from collections import deque
from typing import Any
import cv2
import numpy as np

from src.rules.spatial_engine import AnomalyEvent


class PendingCapture:
    def __init__(
        self,
        event: AnomalyEvent,
        pre_encoded_frames: list[bytes],
        post_frames_needed: int,
        output_mp4: Path,
        output_json: Path,
        fps: int,
        resolution_wh: tuple[int, int]
    ) -> None:
        self.event = event
        self.encoded_frames: list[bytes] = list(pre_encoded_frames)
        self.post_frames_needed = post_frames_needed
        self.post_frames_collected = 0
        self.output_mp4 = output_mp4
        self.output_json = output_json
        self.fps = fps
        self.width, self.height = resolution_wh

    def append_encoded_frame(self, jpeg_bytes: bytes) -> bool:
        self.encoded_frames.append(jpeg_bytes)
        self.post_frames_collected += 1
        return self.post_frames_collected >= self.post_frames_needed


class IncidentVideoBuffer:
    def __init__(
        self,
        output_dir: Path,
        pre_incident_seconds: float = 3.0,
        post_incident_seconds: float = 3.0,
        fps: int = 30,
        resolution_wh: tuple[int, int] = (640, 480),
        jpeg_quality: int = 85,
        max_active_captures: int = 4
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        self.width, self.height = resolution_wh
        self.jpeg_quality = int(np.clip(jpeg_quality, 50, 95))
        self.max_active_captures = max(1, max_active_captures)

        self.pre_buffer_size = max(1, int(pre_incident_seconds * self.fps))
        self.post_buffer_size = max(1, int(post_incident_seconds * self.fps))

        self.ring_buffer: deque[bytes] = deque(maxlen=self.pre_buffer_size)
        self.active_captures: list[PendingCapture] = []
        self.saved_incidents: list[Path] = []
        self._sequence_counter = 0

        self._write_queue: queue.Queue[PendingCapture | None] = queue.Queue(maxsize=16)
        self._is_running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _encode_frame(self, frame_bgr: np.ndarray) -> bytes:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        ok, buf = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("[X] Failed to JPEG-encode frame for circular buffer.")
        return buf.tobytes()

    def _writer_loop(self) -> None:
        while self._is_running or not self._write_queue.empty():
            try:
                capture = self._write_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if capture is None:
                self._write_queue.task_done()
                break

            try:
                self._finalize_to_disk(capture)
            except Exception as err:
                print(f"[!] [INCIDENT WRITER] Failed saving {capture.output_mp4.name}: {err}")
            finally:
                capture.encoded_frames.clear()
                gc.collect()
                self._write_queue.task_done()

    def _finalize_to_disk(self, capture: PendingCapture) -> None:
        capture.output_mp4.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(capture.output_mp4),
            fourcc,
            float(capture.fps),
            (capture.width, capture.height)
        )

        frames_written = 0
        for jpeg_bytes in capture.encoded_frames:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != capture.width or frame.shape[0] != capture.height:
                frame = cv2.resize(frame, (capture.width, capture.height))
            writer.write(frame)
            frames_written += 1

        writer.release()

        metadata: dict[str, Any] = {
            "incident_id": capture.output_mp4.stem,
            "timestamp": capture.event.timestamp,
            "readable_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(capture.event.timestamp)
            ),
            "zone_id": capture.event.zone_id,
            "zone_name": capture.event.zone_name,
            "rule_type": capture.event.rule_type,
            "track_id": capture.event.track_id,
            "class_name": capture.event.class_name,
            "dwell_time_seconds": round(capture.event.dwell_time, 2),
            "bounding_box_xyxy": [round(c, 2) for c in capture.event.bbox_xyxy],
            "video_file": str(capture.output_mp4.resolve()),
            "total_frames": frames_written,
            "fps": capture.fps
        }

        with open(capture.output_json, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)

        self.saved_incidents.append(capture.output_mp4)
        print(f"[+] INCIDENT SAVED: {capture.output_mp4.name} ({frames_written} frames) + {capture.output_json.name}")

    def estimate_buffer_ram_mb(self) -> float:
        ring_bytes = sum(len(b) for b in self.ring_buffer)
        active_bytes = sum(
            sum(len(b) for b in cap.encoded_frames)
            for cap in self.active_captures
        )
        return (ring_bytes + active_bytes) / (1024.0 * 1024.0)

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        jpeg_bytes = self._encode_frame(frame_bgr)
        self.ring_buffer.append(jpeg_bytes)

        for capture in list(self.active_captures):
            completed = capture.append_encoded_frame(jpeg_bytes)
            if completed:
                self.active_captures.remove(capture)
                try:
                    self._write_queue.put_nowait(capture)
                except queue.Full:
                    print(f"[!] [INCIDENT WRITER] Queue full. Dropping clip {capture.output_mp4.name}")
                    capture.encoded_frames.clear()

    def trigger_incident_capture(self, event: AnomalyEvent) -> Path | None:
        if len(self.active_captures) >= self.max_active_captures:
            print(f"[!] Max concurrent captures ({self.max_active_captures}) reached. Skipping track #{event.track_id}")
            return None

        self._sequence_counter += 1
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
        incident_id = f"incident_{timestamp_str}_t{event.track_id}_{self._sequence_counter:03d}"
        mp4_path = self.output_dir / f"{incident_id}.mp4"
        json_path = self.output_dir / f"{incident_id}.json"

        capture = PendingCapture(
            event=event,
            pre_encoded_frames=list(self.ring_buffer),
            post_frames_needed=self.post_buffer_size,
            output_mp4=mp4_path,
            output_json=json_path,
            fps=self.fps,
            resolution_wh=(self.width, self.height)
        )
        self.active_captures.append(capture)
        return mp4_path

    def flush_and_shutdown(self, timeout: float = 10.0) -> None:
        self._is_running = False
        self._write_queue.put(None)
        self._writer_thread.join(timeout=timeout)