"""
src/perception/reid_tracker.py
Appearance-enhanced multi-object tracker combining ByteTrack with deep feature embeddings.
Maintains identity persistence across long occlusions (>5 seconds) via cosine similarity gallery matching.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

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
    embedding: np.ndarray


class TrackAppearanceGallery:
    """Stores and updates moving average appearance vectors for each track ID."""

    def __init__(self, alpha: float = 0.9, max_lost_age_sec: float = 8.0) -> None:
        self.alpha = alpha  # Momentum for moving average
        self.max_lost_age_sec = max_lost_age_sec
        self.gallery: dict[int, np.ndarray] = {}
        self.last_seen: dict[int, float] = {}

    def update(self, track_id: int, new_embedding: np.ndarray, timestamp: float) -> None:
        if track_id in self.gallery:
            # Exponential Moving Average (EMA) feature update
            updated = self.alpha * self.gallery[track_id] + (1.0 - self.alpha) * new_embedding
            self.gallery[track_id] = updated / np.linalg.norm(updated)
        else:
            self.gallery[track_id] = new_embedding / np.linalg.norm(new_embedding)

        self.last_seen[track_id] = timestamp

    def purge_expired(self, current_time: float) -> None:
        """Removes stale tracks to prevent unbounded memory growth."""
        expired = [
            tid for tid, last_t in self.last_seen.items()
            if (current_time - last_t) > self.max_lost_age_sec
        ]
        for tid in expired:
            self.gallery.pop(tid, None)
            self.last_seen.pop(tid, None)


class DeepReIDTracker:
    """
    Orchestrates ByteTrack association and handles track recovery using appearance embeddings.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        reid_match_threshold: float = 0.35,  # Max cosine distance for re-identification
    ) -> None:
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=frame_rate * 3,  # 3-second buffer
            minimum_matching_threshold=0.8,
            frame_rate=frame_rate
        )
        self.extractor = AppearanceEmbeddingExtractor()
        self.gallery = TrackAppearanceGallery(alpha=0.85, max_lost_age_sec=10.0)
        self.reid_match_threshold = reid_match_threshold
        self.trace_annotator = sv.TraceAnnotator(trace_length=frame_rate * 2, thickness=2)

    def update(
        self,
        frame_bgr: np.ndarray,
        ultralytics_results: Results,
        class_name_map: dict[int, str] | None = None
    ) -> tuple[sv.Detections, list[ReIDTrackedEntity]]:
        current_time = time.time()
        self.gallery.purge_expired(current_time)

        # 1. ByteTrack Spatial Update
        raw_detections = sv.Detections.from_ultralytics(ultralytics_results)
        tracked_detections = self.byte_tracker.update_with_detections(raw_detections)

        if tracked_detections.tracker_id is None or len(tracked_detections) == 0:
            return tracked_detections, []

        # 2. Extract Appearance Embeddings for all active detections
        embeddings = self.extractor.extract_crops(frame_bgr, tracked_detections.xyxy)

        entities: list[ReIDTrackedEntity] = []
        for i in range(len(tracked_detections)):
            tid = int(tracked_detections.tracker_id[i])
            cid = int(tracked_detections.class_id[i])
            conf = float(tracked_detections.confidence[i])
            bbox = tuple(tracked_detections.xyxy[i].tolist())
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            emb = embeddings[i]

            # 3. Update Gallery Representation
            self.gallery.update(tid, emb, current_time)

            cls_name = class_name_map.get(cid, str(cid)) if class_name_map else str(cid)
            entities.append(ReIDTrackedEntity(
                track_id=tid,
                class_id=cid,
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=bbox,
                center_xy=center,
                embedding=emb
            ))

        return tracked_detections, entities

    def match_appearance(self, query_embedding: np.ndarray) -> tuple[int | None, float]:
        """Matches a query embedding against the gallery to retrieve a historical track ID."""
        if not self.gallery.gallery:
            return None, 1.0

        best_id: int | None = None
        min_distance = 1.0

        for tid, gallery_emb in self.gallery.gallery.items():
            # Cosine distance: 1.0 - dot_product (embeddings are normalized to unit length)
            cosine_dist = 1.0 - float(np.dot(query_embedding, gallery_emb))
            if cosine_dist < min_distance:
                min_distance = cosine_dist
                best_id = tid

        if min_distance <= self.reid_match_threshold:
            return best_id, min_distance
        return None, min_distance