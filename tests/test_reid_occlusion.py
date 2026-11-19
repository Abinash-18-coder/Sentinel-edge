import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import supervision as sv

from src.perception.reid_tracker import DeepReIDTracker


def create_crack_patch() -> np.ndarray:
    rng = np.random.default_rng(7)
    patch = rng.integers(175, 205, (140, 140, 3), dtype=np.uint8)
    pts = np.array([[20, 10], [45, 40], [35, 75], [85, 105], [115, 132]], dtype=np.int32)
    cv2.polylines(patch, [pts], isClosed=False, color=(20, 20, 20), thickness=4)
    cv2.line(patch, (45, 40), (75, 32), (35, 35, 35), 2)
    return patch


def create_distinct_object_patch() -> np.ndarray:
    patch = np.zeros((140, 140, 3), dtype=np.uint8)
    for y in range(0, 140, 20):
        for x in range(0, 140, 20):
            if ((x // 20) + (y // 20)) % 2 == 0:
                patch[y:y + 20, x:x + 20] = (220, 60, 20)
            else:
                patch[y:y + 20, x:x + 20] = (30, 210, 240)
    return patch


def place_patch(frame: np.ndarray, patch: np.ndarray, x1: int, y1: int) -> list[float]:
    h, w = patch.shape[:2]
    frame[y1:y1 + h, x1:x1 + w] = patch
    return [float(x1), float(y1), float(x1 + w), float(y1 + h)]


def test_end_to_end_reid_occlusion_recovery() -> None:
    print("=" * 65)
    print(" SentinelEdge v2.0: Day 9-11 Re-ID Occlusion Recovery Test")
    print("=" * 65)

    tracker = DeepReIDTracker(
        frame_rate=30,
        reid_match_threshold=0.35,
        lost_track_buffer=30,
        max_lost_age_sec=10.0,
        gallery_refresh_interval=2
    )

    crack_patch = create_crack_patch()
    other_patch = create_distinct_object_patch()
    base_time = 1000.0
    initial_id = None

    print("[*] Phase 1: Tracking initial crack (5 frames)...")
    for f in range(5):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        box = place_patch(frame, crack_patch, 80 + f * 2, 100)
        dets = sv.Detections(
            xyxy=np.array([box], dtype=np.float32),
            confidence=np.array([0.92], dtype=np.float32),
            class_id=np.array([0], dtype=int)
        )
        _, entities = tracker.update_from_detections(
            frame_bgr=frame,
            raw_detections=dets,
            class_name_map={0: "crack"},
            current_time=base_time + f * 0.033
        )
        if entities:
            initial_id = entities[0].track_id

    assert initial_id is not None, "Initial track failed to activate!"
    print(f"[+] Phase 1 Passed: Crack assigned Persistent ID #{initial_id}")

    print("[*] Phase 2: Simulating 5-second total occlusion (45 empty frames)...")
    empty_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    for f in range(45):
        tracker.update_from_detections(
            frame_bgr=empty_frame,
            raw_detections=sv.Detections.empty(),
            class_name_map={0: "crack"},
            current_time=base_time + 1.0 + (f * 0.1)
        )
    print("[+] Phase 2 Complete: ByteTrack lost buffer exceeded; track dropped from ByteTrack.")

    print("[*] Phase 3: Reintroducing shifted crack (+ brightness jitter) + new distinct object...")
    reappear_time = base_time + 6.0
    shifted_crack = np.clip(crack_patch.astype(np.int16) + 12, 0, 255).astype(np.uint8)

    reappeared_crack_id = None
    reappeared_flag = False
    other_obj_id = None

    for f in range(3):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        box_crack = place_patch(frame, shifted_crack, 360 + f, 240)
        box_other = place_patch(frame, other_patch, 40, 280)
        dets = sv.Detections(
            xyxy=np.array([box_crack, box_other], dtype=np.float32),
            confidence=np.array([0.90, 0.88], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int)
        )
        _, entities = tracker.update_from_detections(
            frame_bgr=frame,
            raw_detections=dets,
            class_name_map={0: "crack"},
            current_time=reappear_time + f * 0.033
        )
        for ent in entities:
            if ent.center_xy[0] > 300:
                reappeared_crack_id = ent.track_id
                if ent.recovered:
                    reappeared_flag = True
            else:
                other_obj_id = ent.track_id

    print(f"[+] Reappeared Crack Track ID  : #{reappeared_crack_id} (Recovered: {reappeared_flag})")
    print(f"[+] Distinct Object Track ID   : #{other_obj_id}")

    assert reappeared_crack_id == initial_id, (
        f"Expected crack to recover ID #{initial_id}, but got #{reappeared_crack_id}"
    )
    assert reappeared_flag is True, "Expected recovered=True flag on restored track!"
    assert other_obj_id is not None and other_obj_id != initial_id, (
        f"Distinct object wrongly stole ID #{initial_id}!"
    )

    print("=" * 65)
    print(" DAY 9-11 RE-ID TRACKING & OCCLUSION RECOVERY PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_end_to_end_reid_occlusion_recovery()