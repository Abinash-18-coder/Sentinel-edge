"""
src/perception/tracker.py
High-performance multi-object tracking module wrapping ByteTrack via Supervision.
Maintains persistent track IDs, velocity histories, and bounding box smoothing.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np
import supervision as sv
from ultralytics.engine.results import Results


class TrackedEntity(NamedTuple):
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]


class ByteTrackEngine:
    """
    Encapsulates ByteTrack tracking lifecycle, trajectory tracing,
    and structured entity extraction.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        # Visual annotator to draw persistent movement tails/trails
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=frame_rate * 2,  # 2-second trace trail
            thickness=2,
        )

    def update(
        self,
        ultralytics_results: Results,
        class_name_map: dict[int, str] | None = None
    ) -> tuple[sv.Detections, list[TrackedEntity]]:
        """
        Updates the tracker with raw YOLO predictions and returns
        Supervision Detections along with strongly typed TrackedEntity instances.
        """
        # Convert raw Ultralytics results into Supervision format
        raw_detections = sv.Detections.from_ultralytics(ultralytics_results)

        # Update ByteTrack state vectors
        tracked_detections = self.tracker.update_with_detections(raw_detections)

        # If no tracked detections exist in the current frame
        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return tracked_detections, []

        entities: list[TrackedEntity] = []
        for i in range(len(tracked_detections)):
            track_id = int(tracked_detections.tracker_id[i])
            cls_id = int(tracked_detections.class_id[i])
            conf = float(tracked_detections.confidence[i])
            x1, y1, x2, y2 = tracked_detections.xyxy[i].tolist()
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            cls_name = (
                class_name_map.get(cls_id, str(cls_id))
                if class_name_map is not None
                else str(cls_id)
            )

            entities.append(
                TrackedEntity(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=(center_x, center_y),
                )
            )

        return tracked_detections, entities

    def annotate_traces(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Renders trajectory motion trails onto the frame."""
        if detections.tracker_id is None or len(detections) == 0:
            return scene
        return self.trace_annotator.annotate(scene=scene, detections=detections)