"""
src/rules/spatial_engine.py
Spatial geometry rule engine and per-track debounced finite state machine.
Handles polygon intrusions, loitering dwell timers, and tripwire crossings.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Any
import cv2
import numpy as np
import supervision as sv
import yaml


class AnomalyState(str, Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"
    COOLDOWN = "COOLDOWN"


class TrackStateRecord:
    """Tracks state lifecycle and timers for a single active entity."""

    def __init__(self, track_id: int) -> None:
        self.track_id: int = track_id
        self.state: AnomalyState = AnomalyState.IDLE
        self.entry_timestamp: float = 0.0
        self.last_seen_timestamp: float = 0.0
        self.last_triggered_timestamp: float = 0.0
        self.dwell_time: float = 0.0

    def update_presence(
        self,
        current_time: float,
        is_inside: bool,
        dwell_warn_sec: float,
        dwell_breach_sec: float,
        cooldown_sec: float
    ) -> AnomalyState:
        """Updates internal state machine and returns current state."""
        self.last_seen_timestamp = current_time

        if not is_inside:
            # If the entity leaves the zone, check if it was in cooldown
            if self.state == AnomalyState.COOLDOWN:
                if (current_time - self.last_triggered_timestamp) > cooldown_sec:
                    self.state = AnomalyState.IDLE
                    self.entry_timestamp = 0.0
                    self.dwell_time = 0.0
            else:
                self.state = AnomalyState.IDLE
                self.entry_timestamp = 0.0
                self.dwell_time = 0.0
            return self.state

        # Object is INSIDE the zone
        if self.entry_timestamp == 0.0:
            self.entry_timestamp = current_time

        self.dwell_time = current_time - self.entry_timestamp

        # Check cooldown state
        if self.state == AnomalyState.COOLDOWN:
            if (current_time - self.last_triggered_timestamp) > cooldown_sec:
                # Reset after cooldown period expires
                self.state = AnomalyState.WARNING
            return self.state

        # State transitions
        if self.dwell_time >= dwell_breach_sec:
            self.state = AnomalyState.CONFIRMED_ANOMALY
            self.last_triggered_timestamp = current_time
        elif self.dwell_time >= dwell_warn_sec:
            self.state = AnomalyState.WARNING
        else:
            self.state = AnomalyState.IDLE

        return self.state


class AnomalyEvent(NamedTuple):
    zone_id: str
    zone_name: str
    track_id: int
    class_name: str
    state: AnomalyState
    dwell_time: float
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]


class SpatialRulesEngine:
    """
    Manages polygon boundaries, tripwires, and evaluation of tracked entities.
    """

    def __init__(self, config_yaml: Path, frame_resolution_wh: tuple[int, int]) -> None:
        self.config_yaml = config_yaml
        self.frame_w, self.frame_h = frame_resolution_wh
        self.zones: list[dict[str, Any]] = []
        self.tripwires: list[dict[str, Any]] = []
        self.track_records: dict[str, dict[int, TrackStateRecord]] = {}

        self._load_configuration()

    def _load_configuration(self) -> None:
        if not self.config_yaml.exists():
            raise FileNotFoundError(f"[X] Spatial config not found: {self.config_yaml}")

        with open(self.config_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Build Polygon Zones
        for z in cfg.get("zones", []):
            # Denormalize coordinates to match frame resolution
            pts = np.array(
                [[int(x * self.frame_w), int(y * self.frame_h)] for x, y in z["polygon"]],
                dtype=np.int32,
            )
            zone_obj = sv.PolygonZone(
                polygon=pts,
                triggering_anchors=[sv.Position.BOTTOM_CENTER, sv.Position.CENTER],
            )
            zone_annotator = sv.PolygonZoneAnnotator(
                zone=zone_obj,
                color=sv.Color(r=220, g=38, b=38),
                thickness=2,
                text_scale=0.6,
                text_padding=6,
            )
            self.zones.append(
                {
                    "id": z["id"],
                    "name": z["name"],
                    "polygon_pts": pts,
                    "zone_obj": zone_obj,
                    "zone_annotator": zone_annotator,
                    "monitored_classes": set(z.get("monitored_classes", [])),
                    "dwell_warning_sec": float(z.get("dwell_warning_seconds", 1.0)),
                    "dwell_breach_sec": float(z.get("dwell_breach_seconds", 2.5)),
                    "cooldown_sec": float(z.get("cooldown_seconds", 10.0)),
                }
            )
            self.track_records[z["id"]] = {}

        # Build Tripwires (LineZones)
        for tw in cfg.get("tripwires", []):
            start = sv.Point(
                x=int(tw["start_point"][0] * self.frame_w),
                y=int(tw["start_point"][1] * self.frame_h),
            )
            end = sv.Point(
                x=int(tw["end_point"][0] * self.frame_w),
                y=int(tw["end_point"][1] * self.frame_h),
            )
            line_zone = sv.LineZone(start=start, end=end)
            line_annotator = sv.LineZoneAnnotator(
                thickness=2,
                text_scale=0.5,
                text_padding=4,
            )
            self.tripwires.append(
                {
                    "id": tw["id"],
                    "name": tw["name"],
                    "line_obj": line_zone,
                    "line_annotator": line_annotator,
                    "monitored_classes": set(tw.get("monitored_classes", [])),
                }
            )

    def evaluate(
        self,
        detections: sv.Detections,
        class_name_map: dict[int, str] | None = None
    ) -> list[AnomalyEvent]:
        """
        Evaluates active detections against all spatial zones and tripwires.
        Returns a list of newly triggered AnomalyEvents.
        """
        current_time = time.time()
        triggered_events: list[AnomalyEvent] = []

        if detections.tracker_id is None or len(detections) == 0:
            return triggered_events

        for zone in self.zones:
            zone_id = zone["id"]
            zone_obj: sv.PolygonZone = zone["zone_obj"]
            is_inside_mask = zone_obj.trigger(detections=detections)

            for i in range(len(detections)):
                track_id = int(detections.tracker_id[i])
                cls_id = int(detections.class_id[i])
                bbox = tuple(detections.xyxy[i].tolist())
                is_inside = bool(is_inside_mask[i])

                # Check if class is monitored in this zone
                if cls_id not in zone["monitored_classes"]:
                    continue

                if track_id not in self.track_records[zone_id]:
                    self.track_records[zone_id][track_id] = TrackStateRecord(track_id)

                record = self.track_records[zone_id][track_id]
                prev_state = record.state

                current_state = record.update_presence(
                    current_time=current_time,
                    is_inside=is_inside,
                    dwell_warn_sec=zone["dwell_warning_sec"],
                    dwell_breach_sec=zone["dwell_breach_sec"],
                    cooldown_sec=zone["cooldown_sec"],
                )

                # Fire event strictly on state change to CONFIRMED_ANOMALY
                if (
                    current_state == AnomalyState.CONFIRMED_ANOMALY
                    and prev_state != AnomalyState.CONFIRMED_ANOMALY
                ):
                    cls_name = (
                        class_name_map.get(cls_id, str(cls_id))
                        if class_name_map
                        else str(cls_id)
                    )
                    event = AnomalyEvent(
                        zone_id=zone_id,
                        zone_name=zone["name"],
                        track_id=track_id,
                        class_name=cls_name,
                        state=current_state,
                        dwell_time=record.dwell_time,
                        timestamp=current_time,
                        bbox_xyxy=bbox,
                    )
                    triggered_events.append(event)
                    # Transition immediately to COOLDOWN to prevent spam
                    record.state = AnomalyState.COOLDOWN

        # Evaluate Tripwires
        for tw in self.tripwires:
            line_zone: sv.LineZone = tw["line_obj"]
            crossed_in, crossed_out = line_zone.trigger(detections=detections)
            for i in range(len(detections)):
                if crossed_in[i] or crossed_out[i]:
                    track_id = int(detections.tracker_id[i])
                    cls_id = int(detections.class_id[i])
                    bbox = tuple(detections.xyxy[i].tolist())
                    cls_name = (
                        class_name_map.get(cls_id, str(cls_id))
                        if class_name_map
                        else str(cls_id)
                    )
                    event = AnomalyEvent(
                        zone_id=tw["id"],
                        zone_name=tw["name"],
                        track_id=track_id,
                        class_name=cls_name,
                        state=AnomalyState.CONFIRMED_ANOMALY,
                        dwell_time=0.0,
                        timestamp=current_time,
                        bbox_xyxy=bbox,
                    )
                    triggered_events.append(event)

        return triggered_events

    def render_overlays(self, scene: np.ndarray) -> np.ndarray:
        """Renders polygon boundaries, warning fills, and tripwire graphics onto the frame."""
        annotated_scene = scene.copy()
        for zone in self.zones:
            annotator: sv.PolygonZoneAnnotator = zone["zone_annotator"]
            annotated_scene = annotator.annotate(scene=annotated_scene)

        for tw in self.tripwires:
            annotator: sv.LineZoneAnnotator = tw["line_annotator"]
            annotated_scene = annotator.annotate(
                frame=annotated_scene,
                line_counter=tw["line_obj"]
            )

        return annotated_scene