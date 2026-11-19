import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import cv2
import numpy as np
import supervision as sv

from src.rules.spatial_engine import AnomalyState, SpatialRulesEngine
from src.rules.incident_buffer import IncidentVideoBuffer


def make_detection(track_id: int, cx: float, cy: float, cls_id: int = 0) -> sv.Detections:
    box = np.array([[cx - 25.0, cy - 25.0, cx + 25.0, cy + 25.0]], dtype=np.float32)
    dets = sv.Detections(
        xyxy=box,
        confidence=np.array([0.85], dtype=np.float32),
        class_id=np.array([cls_id], dtype=int)
    )
    dets.tracker_id = np.array([track_id], dtype=int)
    return dets


def test_polygon_state_machine_and_cooldown() -> None:
    print("[*] Test 1: Verifying Polygon Zone State Machine & Cooldown Debounce...")
    zones_yaml = PROJECT_ROOT / "configs" / "zones.yaml"
    engine = SpatialRulesEngine(config_yaml=zones_yaml, frame_resolution_wh=(640, 480))
    class_map = {0: "crack"}

    inside_det = make_detection(track_id=1, cx=160.0, cy=240.0, cls_id=0)
    t0 = 1000.0

    events_0 = engine.evaluate(inside_det, class_map, current_time=t0)
    assert len(events_0) == 0
    assert engine.get_track_state("zone_alpha_danger", 1) == AnomalyState.IDLE

    events_1 = engine.evaluate(inside_det, class_map, current_time=t0 + 1.2)
    assert len(events_1) == 0
    assert engine.get_track_state("zone_alpha_danger", 1) == AnomalyState.WARNING

    events_2 = engine.evaluate(inside_det, class_map, current_time=t0 + 2.6)
    assert len(events_2) == 1
    assert events_2[0].rule_type == "polygon_zone"
    assert events_2[0].class_name == "crack"
    assert engine.get_track_state("zone_alpha_danger", 1) == AnomalyState.COOLDOWN

    events_spam = engine.evaluate(inside_det, class_map, current_time=t0 + 5.0)
    assert len(events_spam) == 0
    assert engine.get_track_state("zone_alpha_danger", 1) == AnomalyState.COOLDOWN

    events_reset = engine.evaluate(inside_det, class_map, current_time=t0 + 12.7)
    assert len(events_reset) == 0
    assert engine.get_track_state("zone_alpha_danger", 1) == AnomalyState.IDLE

    events_reconfirm = engine.evaluate(inside_det, class_map, current_time=t0 + 15.3)
    assert len(events_reconfirm) == 1
    print("[+] Test 1 Passed: IDLE -> WARNING -> CONFIRMED -> COOLDOWN -> CLEAN RESET verified.")


def test_tripwire_crossing_and_debounce() -> None:
    print("[*] Test 2: Verifying Tripwire Crossing & Cooldown Debounce...")
    zones_yaml = PROJECT_ROOT / "configs" / "zones.yaml"
    engine = SpatialRulesEngine(config_yaml=zones_yaml, frame_resolution_wh=(640, 480))
    class_map = {0: "crack"}
    t0 = 2000.0

    left_side = make_detection(track_id=7, cx=380.0, cy=240.0, cls_id=0)
    right_side = make_detection(track_id=7, cx=540.0, cy=240.0, cls_id=0)

    ev1 = engine.evaluate(left_side, class_map, current_time=t0)
    ev2 = engine.evaluate(right_side, class_map, current_time=t0 + 0.1)
    assert len(ev1) == 0
    assert len(ev2) == 1
    assert ev2[0].rule_type == "tripwire"

    ev3 = engine.evaluate(left_side, class_map, current_time=t0 + 0.3)
    assert len(ev3) == 0
    print("[+] Test 2 Passed: Tripwire triggered once and suppressed immediate bounce spam.")


def test_circular_buffer_ram_and_mp4_output() -> None:
    print("[*] Test 3: Verifying Circular Ring Buffer RAM (< 15 MB) & Async MP4/JSON Output...")
    temp_dir = PROJECT_ROOT / "data" / "raw" / "incidents" / "_day12_test_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    zones_yaml = PROJECT_ROOT / "configs" / "zones.yaml"
    engine = SpatialRulesEngine(config_yaml=zones_yaml, frame_resolution_wh=(640, 480))
    buf = IncidentVideoBuffer(
        output_dir=temp_dir,
        pre_incident_seconds=1.0,
        post_incident_seconds=1.0,
        fps=15,
        resolution_wh=(640, 480)
    )

    frame_base = np.full((480, 640, 3), 160, dtype=np.uint8)
    for i in range(15):
        f = frame_base.copy()
        cv2.putText(f, f"PRE-{i}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        buf.push_frame(f)

    inside_det = make_detection(track_id=9, cx=160.0, cy=240.0, cls_id=0)
    engine.evaluate(inside_det, {0: "crack"}, current_time=3000.0)
    triggered = engine.evaluate(inside_det, {0: "crack"}, current_time=3003.0)
    assert len(triggered) == 1

    expected_mp4 = buf.trigger_incident_capture(triggered[0])
    assert expected_mp4 is not None

    for i in range(15):
        f = frame_base.copy()
        cv2.putText(f, f"POST-{i}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
        buf.push_frame(f)

    peak_ram_mb = buf.estimate_buffer_ram_mb()
    print(f"    - Estimated Ring Buffer RAM Footprint: {peak_ram_mb:.3f} MB")
    assert peak_ram_mb < 15.0

    buf.flush_and_shutdown(timeout=10.0)

    expected_json = expected_mp4.with_suffix(".json")
    assert expected_mp4.exists(), f"MP4 was not written: {expected_mp4}"
    assert expected_json.exists(), f"JSON was not written: {expected_json}"

    cap = cv2.VideoCapture(str(expected_mp4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 30, f"Expected 30 frames (15 pre + 15 post), got {frame_count}"

    meta = json.loads(expected_json.read_text(encoding="utf-8"))
    assert meta["track_id"] == 9
    assert meta["class_name"] == "crack"
    assert meta["rule_type"] == "polygon_zone"
    assert meta["total_frames"] == 30

    shutil.rmtree(temp_dir)
    print(f"[+] Test 3 Passed: 30-frame MP4 & telemetry JSON verified and cleaned up.")


if __name__ == "__main__":
    print("=" * 65)
    print(" SentinelEdge v2.0: Day 12-14 Spatial Rules & Ring Buffer Suite")
    print("=" * 65)
    test_polygon_state_machine_and_cooldown()
    test_tripwire_crossing_and_debounce()
    test_circular_buffer_ram_and_mp4_output()
    print("=" * 65)
    print(" ALL DAY 12-14 SPATIAL RULES & BUFFER TESTS PASSED!")
    print("=" * 65)