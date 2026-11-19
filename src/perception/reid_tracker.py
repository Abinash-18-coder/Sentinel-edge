import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from typing import NamedTuple
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.results import Results

from src.perception.reid_extractor import AppearanceEmbeddingExtractor


class ReIDTrackedEntity(NamedTuple):
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    recovered: bool


class TrackAppearanceGallery:
    def __init__(self, alpha: float = 0.85, max_lost_age_sec: float = 10.0) -> None:
        self.alpha = alpha
        self.max_lost_age_sec = max_lost_age_sec
        self.gallery: dict[int, np.ndarray] = {}
        self.class_ids: dict[int, int] = {}
        self.last_seen: dict[int, float] = {}

    def update(
        self,
        persistent_id: int,
        class_id: int,
        new_embedding: np.ndarray,
        timestamp: float
    ) -> None:
        norm = float(np.linalg.norm(new_embedding))
        if norm < 1e-8:
            return
        unit_emb = new_embedding / norm

        if persistent_id in self.gallery:
            blended = self.alpha * self.gallery[persistent_id] + (1.0 - self.alpha) * unit_emb
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm > 1e-8:
                self.gallery[persistent_id] = blended / blended_norm
        else:
            self.gallery[persistent_id] = unit_emb

        self.class_ids[persistent_id] = class_id
        self.last_seen[persistent_id] = timestamp

    def touch(self, persistent_id: int, timestamp: float) -> None:
        if persistent_id in self.last_seen:
            self.last_seen[persistent_id] = timestamp

    def purge_expired(self, current_time: float) -> list[int]:
        expired = [
            pid for pid, last_t in self.last_seen.items()
            if (current_time - last_t) > self.max_lost_age_sec
        ]
        for pid in expired:
            self.gallery.pop(pid, None)
            self.class_ids.pop(pid, None)
            self.last_seen.pop(pid, None)
        return expired


class DeepReIDTracker:
    def __init__(
        self,
        frame_rate: int = 30,
        reid_match_threshold: float = 0.35,
        lost_track_buffer: int = 60,
        max_lost_age_sec: float = 10.0,
        gallery_refresh_interval: int = 5
    ) -> None:
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate
        )
        self.extractor = AppearanceEmbeddingExtractor()
        self.gallery = TrackAppearanceGallery(
            alpha=0.85,
            max_lost_age_sec=max_lost_age_sec
        )
        self.reid_match_threshold = reid_match_threshold
        self.gallery_refresh_interval = max(1, gallery_refresh_interval)
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=frame_rate * 2,
            thickness=2
        )

        self.raw_to_persistent: dict[int, int] = {}
        self.next_persistent_id: int = 1
        self.frame_counter: int = 0

    def _cleanup_aliases(self, expired_pids: list[int]) -> None:
        if not expired_pids:
            return
        expired_set = set(expired_pids)
        stale_raw_ids = [
            raw_id for raw_id, pid in self.raw_to_persistent.items()
            if pid in expired_set
        ]
        for raw_id in stale_raw_ids:
            self.raw_to_persistent.pop(raw_id, None)

    def update(
        self,
        frame_bgr: np.ndarray,
        ultralytics_results: Results,
        class_name_map: dict[int, str] | None = None,
        current_time: float | None = None
    ) -> tuple[sv.Detections, list[ReIDTrackedEntity]]:
        raw_detections = sv.Detections.from_ultralytics(ultralytics_results)
        return self.update_from_detections(
            frame_bgr=frame_bgr,
            raw_detections=raw_detections,
            class_name_map=class_name_map,
            current_time=current_time
        )

    def update_from_detections(
        self,
        frame_bgr: np.ndarray,
        raw_detections: sv.Detections,
        class_name_map: dict[int, str] | None = None,
        current_time: float | None = None
    ) -> tuple[sv.Detections, list[ReIDTrackedEntity]]:
        now = time.time() if current_time is None else current_time
        self.frame_counter += 1

        expired_pids = self.gallery.purge_expired(now)
        self._cleanup_aliases(expired_pids)

        tracked_detections = self.byte_tracker.update_with_detections(raw_detections)

        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return tracked_detections, []

        num_dets = len(tracked_detections)
        raw_ids = [int(tid) for tid in tracked_detections.tracker_id]
        cls_ids = [
            int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
            for i in range(num_dets)
        ]

        resolved_ids: list[int] = [-1] * num_dets
        recovered_flags: list[bool] = [False] * num_dets
        active_pids: set[int] = set()
        new_indices: list[int] = []
        refresh_indices: list[int] = []

        for i, raw_id in enumerate(raw_ids):
            if raw_id in self.raw_to_persistent:
                pid = self.raw_to_persistent[raw_id]
                resolved_ids[i] = pid
                active_pids.add(pid)
                self.gallery.touch(pid, now)
                if self.frame_counter % self.gallery_refresh_interval == 0:
                    refresh_indices.append(i)
            else:
                new_indices.append(i)

        if refresh_indices:
            refresh_boxes = tracked_detections.xyxy[refresh_indices]
            refresh_embs = self.extractor.extract_crops(frame_bgr, refresh_boxes)
            for idx_pos, det_i in enumerate(refresh_indices):
                self.gallery.update(
                    persistent_id=resolved_ids[det_i],
                    class_id=cls_ids[det_i],
                    new_embedding=refresh_embs[idx_pos],
                    timestamp=now
                )

        if new_indices:
            new_boxes = tracked_detections.xyxy[new_indices]
            new_embs = self.extractor.extract_crops(frame_bgr, new_boxes)

            lost_pids = [
                pid for pid in self.gallery.gallery.keys()
                if pid not in active_pids
            ]

            matched_new_positions: set[int] = set()

            if lost_pids:
                cost_matrix = np.ones((len(new_indices), len(lost_pids)), dtype=np.float32)
                for r, det_i in enumerate(new_indices):
                    det_cls = cls_ids[det_i]
                    for c, lost_pid in enumerate(lost_pids):
                        if self.gallery.class_ids.get(lost_pid, det_cls) != det_cls:
                            cost_matrix[r, c] = 1.0
                        else:
                            cos_dist = 1.0 - float(np.dot(new_embs[r], self.gallery.gallery[lost_pid]))
                            cost_matrix[r, c] = max(0.0, cos_dist)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] <= self.reid_match_threshold:
                        det_i = new_indices[r]
                        recovered_pid = lost_pids[c]
                        self.raw_to_persistent[raw_ids[det_i]] = recovered_pid
                        resolved_ids[det_i] = recovered_pid
                        recovered_flags[det_i] = True
                        active_pids.add(recovered_pid)
                        self.gallery.update(
                            persistent_id=recovered_pid,
                            class_id=cls_ids[det_i],
                            new_embedding=new_embs[r],
                            timestamp=now
                        )
                        matched_new_positions.add(r)

            for r, det_i in enumerate(new_indices):
                if r in matched_new_positions:
                    continue
                new_pid = self.next_persistent_id
                self.next_persistent_id += 1
                self.raw_to_persistent[raw_ids[det_i]] = new_pid
                resolved_ids[det_i] = new_pid
                active_pids.add(new_pid)
                self.gallery.update(
                    persistent_id=new_pid,
                    class_id=cls_ids[det_i],
                    new_embedding=new_embs[r],
                    timestamp=now
                )

        tracked_detections.tracker_id = np.array(resolved_ids, dtype=int)

        entities: list[ReIDTrackedEntity] = []
        for i in range(num_dets):
            pid = resolved_ids[i]
            cid = cls_ids[i]
            conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0
            x1, y1, x2, y2 = tracked_detections.xyxy[i].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            cls_name = class_name_map.get(cid, str(cid)) if class_name_map else str(cid)

            entities.append(
                ReIDTrackedEntity(
                    track_id=pid,
                    class_id=cid,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=(cx, cy),
                    recovered=recovered_flags[i]
                )
            )

        return tracked_detections, entities

    def annotate_traces(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if detections.tracker_id is None or len(detections) == 0:
            return scene
        return self.trace_annotator.annotate(scene=scene, detections=detections)