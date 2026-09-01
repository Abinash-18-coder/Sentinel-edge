"""
src/rules/incident_buffer.py
In-memory circular frame buffer capturing pre- and post-incident video footage
and writing structured event telemetry to disk.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import os
import json
import time
from collections import deque
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from src.rules.spatial_engine import AnomalyEvent


class PendingCapture:
    """Tracks state of an active post-incident capture task."""

    def __init__(
        self,
        event: AnomalyEvent,
        pre_frames: list[np.ndarray],
        post_frames_needed: int,
        output_mp4: Path,
        output_json: Path,
        fps: int,
        resolution_wh: tuple[int, int]
    ) -> None:
        self.event = event
        self.frames: list[np.ndarray] = list(pre_frames)
        self.post_frames_needed = post_frames_needed
        self.post_frames_collected = 0
        self.output_mp4 = output_mp4
        self.output_json = output_json
        self.fps = fps
        self.width, self.height = resolution_wh
        self.is_completed = False

    def append_frame(self, frame: np.ndarray) -> bool:
        """Appends ongoing frame. Returns True if post-capture is complete."""
        self.frames.append(frame.copy())
        self.post_frames_collected += 1
        if self.post_frames_collected >= self.post_frames_needed:
            self._finalize_to_disk()
            self.is_completed = True
            return True
        return False

    def _finalize_to_disk(self) -> None:
        """Encodes captured frames to MP4 and dumps JSON metadata."""
        self.output_mp4.parent.mkdir(parents=True, exist_ok=True)

        # Windows-safe OpenCV video codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.output_mp4),
            fourcc,
            float(self.fps),
            (self.width, self.height)
        )

        for f in self.frames:
            # Resize defensively to ensure dimensions match header
            if f.shape[1] != self.width or f.shape[0] != self.height:
                f = cv2.resize(f, (self.width, self.height))
            writer.write(f)

        writer.release()

        # Write Telemetry JSON
        metadata: dict[str, Any] = {
            "incident_id": self.output_mp4.stem,
            "timestamp": self.event.timestamp,
            "readable_time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.event.timestamp)
            ),
            "zone_id": self.event.zone_id,
            "zone_name": self.event.zone_name,
            "track_id": self.event.track_id,
            "class_name": self.event.class_name,
            "dwell_time_seconds": round(self.event.dwell_time, 2),
            "bounding_box_xyxy": [round(c, 2) for c in self.event.bbox_xyxy],
            "video_file": str(self.output_mp4.resolve()),
            "total_frames": len(self.frames),
            "fps": self.fps,
        }

        with open(self.output_json, "w", encoding="utf-8") as jf:
            json.dump(metadata, jf, indent=2)

        print(f"\n[+] INCIDENT CAPTURED & SAVED:")
        print(f"    Video: {self.output_mp4.name} ({len(self.frames)} frames)")
        print(f"    JSON : {self.output_json.name}")


class IncidentVideoBuffer:
    """
    Continuous circular ring buffer maintaining the last N seconds of video in RAM.
    Spawns disk-recording workers when an AnomalyEvent is triggered.
    """

    def __init__(
        self,
        output_dir: Path,
        pre_incident_seconds: float = 3.0,
        post_incident_seconds: float = 3.0,
        fps: int = 30,
        resolution_wh: tuple[int, int] = (640, 480)
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width, self.height = resolution_wh

        self.pre_buffer_size = int(pre_incident_seconds * fps)
        self.post_buffer_size = int(post_incident_seconds * fps)

        # Ring buffer in RAM
        self.ring_buffer: deque[np.ndarray] = deque(maxlen=self.pre_buffer_size)
        self.active_captures: list[PendingCapture] = []

    def push_frame(self, frame: np.ndarray) -> None:
        """Pushes current live frame to circular buffer and active writers."""
        # Store clean copy in circular RAM buffer
        self.ring_buffer.append(frame.copy())

        # Update active captures that are collecting post-incident frames
        for capture in list(self.active_captures):
            completed = capture.append_frame(frame)
            if completed:
                self.active_captures.remove(capture)

    def trigger_incident_capture(self, event: AnomalyEvent) -> None:
        """Initializes a new incident capture sequence."""
        timestamp_str = time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(event.timestamp)
        )
        incident_id = f"incident_{timestamp_str}_track_{event.track_id}"
        mp4_path = self.output_dir / f"{incident_id}.mp4"
        json_path = self.output_dir / f"{incident_id}.json"

        capture = PendingCapture(
            event=event,
            pre_frames=list(self.ring_buffer),
            post_frames_needed=self.post_buffer_size,
            output_mp4=mp4_path,
            output_json=json_path,
            fps=self.fps,
            resolution_wh=(self.width, self.height),
        )
        self.active_captures.append(capture)