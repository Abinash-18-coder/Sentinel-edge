import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import time
from enum import Enum
from typing import Any, NamedTuple
import numpy as np
import supervision as sv
import yaml


class AnomalyState(str, Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"
    COOLDOWN = "COOLDOWN"


class TrackStateRecord:
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
        self.last_seen_timestamp = current_time

        if not is_inside:
            self.entry_timestamp = 0.0
            self.dwell_time = 0.0
            if self.state == AnomalyState.COOLDOWN:
                if (current_time - self.last_triggered_timestamp) >= cooldown_sec:
                    self.state = AnomalyState.IDLE
            else:
                self.state = AnomalyState.IDLE
            return self.state

        if self.state == AnomalyState.COOLDOWN:
            if (current_time - self.last_triggered_timestamp) >= cooldown_sec:
                self.state = AnomalyState.IDLE
                self.entry_timestamp = current_time
                self.dwell_time = 0.0
            else:
                return self.state

        if self.entry_timestamp == 0.0:
            self.entry_timestamp = current_time

        self.dwell_time = max(0.0, current_time - self.entry_timestamp)

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
    rule_type: str
    track_id: int
    class_name: str
    state: AnomalyState
    dwell_time: float
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]


class SpatialRulesEngine:
    def __init__(
        self,
        config_yaml: Path,
        frame_resolution_wh: tuple[int, int],
        max_stale_sec: float = 60.0
    ) -> None:
        self.config_yaml = config_yaml
        self.frame_w, self.frame_h = frame_resolution_wh
        self.max_stale_sec = max_stale_sec
        self._lock = threading.Lock()

        self.zones: list[dict[str, Any]] = []
        self.tripwires: list[dict[str, Any]] = []
        self.track_records: dict[str, dict[int, TrackStateRecord]] = {}
        self.tripwire_last_triggered: dict[str, dict[int, float]] = {}

        self._load_configuration()

    def _load_configuration(self) -> None:
        if not self.config_yaml.exists():
            raise FileNotFoundError(f"[X] Spatial config not found: {self.config_yaml}")

        with open(self.config_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        new_zones: list[dict[str, Any]] = []
        new_records: dict[str, dict[int, TrackStateRecord]] = {}

        for z in cfg.get("zones", []):
            pts = np.array(
                [[int(x * self.frame_w), int(y * self.frame_h)] for x, y in z["polygon"]],
                dtype=np.int32
            )
            zone_obj = sv.PolygonZone(
                polygon=pts,
                triggering_anchors=[sv.Position.BOTTOM_CENTER, sv.Position.CENTER]
            )
            zone_annotator = sv.PolygonZoneAnnotator(
                zone=zone_obj,
                color=sv.Color(r=220, g=38, b=38),
                thickness=2,
                text_scale=0.6,
                text_padding=6
            )
            new_zones.append({
                "id": z["id"],
                "name": z["name"],
                "polygon_pts": pts,
                "zone_obj": zone_obj,
                "zone_annotator": zone_annotator,
                "monitored_classes": set(z.get("monitored_classes", [])),
                "dwell_warning_sec": float(z.get("dwell_warning_seconds", 1.0)),
                "dwell_breach_sec": float(z.get("dwell_breach_seconds", 2.5)),
                "cooldown_sec": float(z.get("cooldown_seconds", 10.0))
            })
            new_records[z["id"]] = {}

        new_tripwires: list[dict[str, Any]] = []
        new_tw_triggers: dict[str, dict[int, float]] = {}

        for tw in cfg.get("tripwires", []):
            start = sv.Point(
                x=int(tw["start_point"][0] * self.frame_w),
                y=int(tw["start_point"][1] * self.frame_h)
            )
            end = sv.Point(
                x=int(tw["end_point"][0] * self.frame_w),
                y=int(tw["end_point"][1] * self.frame_h)
            )
            line_zone = sv.LineZone(
                start=start,
                end=end,
                triggering_anchors=[sv.Position.CENTER]
            )
            line_annotator = sv.LineZoneAnnotator(
                thickness=2,
                text_scale=0.5,
                text_padding=4
            )
            new_tripwires.append({
                "id": tw["id"],
                "name": tw["name"],
                "line_obj": line_zone,
                "line_annotator": line_annotator,
                "direction": str(tw.get("direction", "bidirectional")).lower(),
                "monitored_classes": set(tw.get("monitored_classes", [])),
                "cooldown_sec": float(tw.get("cooldown_seconds", 5.0))
            })
            new_tw_triggers[tw["id"]] = {}

        with self._lock:
            self.zones = new_zones
            self.track_records = new_records
            self.tripwires = new_tripwires
            self.tripwire_last_triggered = new_tw_triggers

    def reload_configuration(self) -> None:
        self._load_configuration()

    def _purge_stale_records(self, current_time: float) -> int:
        purged = 0
        for zone_id, records in self.track_records.items():
            stale_ids = [
                tid for tid, rec in records.items()
                if (current_time - rec.last_seen_timestamp) > self.max_stale_sec
            ]
            for tid in stale_ids:
                records.pop(tid, None)
                purged += 1

        for tw_id, trigger_map in self.tripwire_last_triggered.items():
            stale_tw_ids = [
                tid for tid, ts in trigger_map.items()
                if (current_time - ts) > self.max_stale_sec
            ]
            for tid in stale_tw_ids:
                trigger_map.pop(tid, None)
                purged += 1

        return purged

    def get_track_state(self, zone_id: str, track_id: int) -> AnomalyState:
        with self._lock:
            rec = self.track_records.get(zone_id, {}).get(track_id)
            return rec.state if rec is not None else AnomalyState.IDLE

    def evaluate(
        self,
        detections: sv.Detections,
        class_name_map: dict[int, str] | None = None,
        current_time: float | None = None
    ) -> list[AnomalyEvent]:
        now = time.time() if current_time is None else current_time
        triggered_events: list[AnomalyEvent] = []

        with self._lock:
            self._purge_stale_records(now)

            if detections.tracker_id is None or len(detections) == 0:
                return triggered_events

            num_dets = len(detections)

            for zone in self.zones:
                zone_id = zone["id"]
                zone_obj: sv.PolygonZone = zone["zone_obj"]
                is_inside_mask = zone_obj.trigger(detections=detections)
                monitored = zone["monitored_classes"]

                for i in range(num_dets):
                    track_id = int(detections.tracker_id[i])
                    cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                    if monitored and cls_id not in monitored:
                        continue

                    bbox = tuple(float(v) for v in detections.xyxy[i].tolist())
                    is_inside = bool(is_inside_mask[i])

                    if track_id not in self.track_records[zone_id]:
                        self.track_records[zone_id][track_id] = TrackStateRecord(track_id)

                    record = self.track_records[zone_id][track_id]
                    prev_state = record.state

                    new_state = record.update_presence(
                        current_time=now,
                        is_inside=is_inside,
                        dwell_warn_sec=zone["dwell_warning_sec"],
                        dwell_breach_sec=zone["dwell_breach_sec"],
                        cooldown_sec=zone["cooldown_sec"]
                    )

                    if (
                        new_state == AnomalyState.CONFIRMED_ANOMALY
                        and prev_state != AnomalyState.CONFIRMED_ANOMALY
                    ):
                        cls_name = (
                            class_name_map.get(cls_id, str(cls_id))
                            if class_name_map is not None
                            else str(cls_id)
                        )
                        triggered_events.append(
                            AnomalyEvent(
                                zone_id=zone_id,
                                zone_name=zone["name"],
                                rule_type="polygon_zone",
                                track_id=track_id,
                                class_name=cls_name,
                                state=new_state,
                                dwell_time=record.dwell_time,
                                timestamp=now,
                                bbox_xyxy=bbox
                            )
                        )
                        record.state = AnomalyState.COOLDOWN

            for tw in self.tripwires:
                tw_id = tw["id"]
                line_zone: sv.LineZone = tw["line_obj"]
                crossed_in, crossed_out = line_zone.trigger(detections=detections)
                direction = tw["direction"]
                monitored = tw["monitored_classes"]
                cooldown_sec = tw["cooldown_sec"]

                for i in range(num_dets):
                    if direction == "in":
                        crossed = bool(crossed_in[i])
                    elif direction == "out":
                        crossed = bool(crossed_out[i])
                    else:
                        crossed = bool(crossed_in[i] or crossed_out[i])

                    if not crossed:
                        continue

                    track_id = int(detections.tracker_id[i])
                    cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0
                    if monitored and cls_id not in monitored:
                        continue

                    last_ts = self.tripwire_last_triggered[tw_id].get(track_id, -1e9)
                    if (now - last_ts) < cooldown_sec:
                        continue

                    self.tripwire_last_triggered[tw_id][track_id] = now
                    bbox = tuple(float(v) for v in detections.xyxy[i].tolist())
                    cls_name = (
                        class_name_map.get(cls_id, str(cls_id))
                        if class_name_map is not None
                        else str(cls_id)
                    )
                    triggered_events.append(
                        AnomalyEvent(
                            zone_id=tw_id,
                            zone_name=tw["name"],
                            rule_type="tripwire",
                            track_id=track_id,
                            class_name=cls_name,
                            state=AnomalyState.CONFIRMED_ANOMALY,
                            dwell_time=0.0,
                            timestamp=now,
                            bbox_xyxy=bbox
                        )
                    )

        return triggered_events

    def render_overlays(self, scene: np.ndarray) -> np.ndarray:
        annotated_scene = scene.copy()
        with self._lock:
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