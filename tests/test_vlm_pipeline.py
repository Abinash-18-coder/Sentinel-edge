import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import time
import cv2
import numpy as np

from src.vlm.async_auditor import AsyncIncidentAuditor
from src.vlm.schemas import GroundingBoundingBox, IncidentSeverity
from src.vlm.vlm_engine import GroundedVLMAuditor


def create_crack_frame() -> np.ndarray:
    rng = np.random.default_rng(11)
    frame = rng.integers(185, 210, (480, 640, 3), dtype=np.uint8)
    pts = np.array([[220, 80], [260, 160], [245, 250], [310, 340], [350, 420]], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=(18, 18, 18), thickness=5)
    cv2.line(frame, (260, 160), (315, 195), (25, 25, 25), 3)
    return frame


def create_crack_plus_window_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    frame = rng.integers(210, 232, (480, 640, 3), dtype=np.uint8)
    pts = np.array([[260, 90], [240, 175], [205, 255], [185, 340], [160, 415]], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=(45, 45, 45), thickness=4)

    cv2.rectangle(frame, (470, 130), (540, 360), (255, 255, 255), -1)
    cv2.rectangle(frame, (470, 130), (540, 360), (140, 140, 140), 10)
    cv2.ellipse(frame, (505, 130), (35, 35), 0, 180, 360, (140, 140, 140), 10)
    return frame


def test_schema_and_json_parser() -> None:
    print("[*] Test 1: Verifying Pydantic Schema Ordering & Malformed JSON Recovery...")
    inverted = GroundingBoundingBox(ymin=800, xmin=700, ymax=200, xmax=100)
    assert inverted.ymin == 200 and inverted.ymax == 800
    assert inverted.xmin == 100 and inverted.xmax == 700

    auditor = GroundedVLMAuditor(mode="auto")

    fenced_resp = (
        "Here is the analysis:\n"
        "```json\n"
        '{"scene_description": "Deep vertical concrete crack near load wall.", '
        '"box_2d": [120, 210, 890, 610], '
        '"severity": "critical", '
        '"recommended_action": "Evacuate and brace beam."}\n'
        "```"
    )
    parsed = auditor.parse_and_validate_json(fenced_resp, [100, 100, 500, 500])
    assert parsed.severity == IncidentSeverity.CRITICAL
    assert parsed.box_2d == [120, 210, 890, 610]

    broken_resp = "WARNING: crack detected but I forgot JSON brackets!"
    recovered = auditor.parse_and_validate_json(broken_resp, [150, 150, 750, 750])
    assert recovered.box_2d == [150, 150, 750, 750]
    assert recovered.severity == IncidentSeverity.MEDIUM
    print("[+] Test 1 Passed: Schema ordering, fenced JSON, and fallback recovery verified.")


def test_crack_grounding_ignores_window() -> None:
    print("[*] Test 2: Verifying Black-Hat Grounding Locks Onto Crack (Ignores Bright Arched Window)...")
    frame = create_crack_plus_window_frame()
    auditor = GroundedVLMAuditor(mode="auto")
    report = auditor.audit_frame(
        image_bgr=frame,
        incident_id="inc_crack_vs_window",
        zone_id="zone_alpha_danger",
        track_id=1,
        entity_class="crack",
        hint_bbox_xyxy=(120.0, 70.0, 560.0, 430.0)
    )

    box = report.hazard_grounding_box
    cx_px = ((box.xmin + box.xmax) / 2.0) / 1000.0 * 640.0
    print(
        f"    - Grounded Box (norm): [{box.ymin}, {box.xmin}, {box.ymax}, {box.xmax}] "
        f"| Center X: {cx_px:.1f}px | Desc: {report.vlm_scene_description}"
    )
    assert cx_px < 360.0, f"Grounding wrongly locked onto right-side window (center X={cx_px:.1f}px)!"
    assert "length" in report.vlm_scene_description.lower()
    assert "width" in report.vlm_scene_description.lower()
    print("[+] Test 2 Passed: Grounding locked onto diagonal crack and ignored arched window.")


def test_async_worker_nonblocking_and_debounce() -> None:
    print("[*] Test 3: Verifying Async VLM Queue (Non-Blocking 30 FPS + Debounce + Grounding)...")
    temp_dir = PROJECT_ROOT / "data" / "raw" / "incidents" / "_day15_vlm_test_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    received_callbacks: list[str] = []
    async_auditor = AsyncIncidentAuditor(
        max_queue_size=8,
        track_debounce_sec=8.0,
        vlm_mode="auto",
        on_report_callback=lambda rep: received_callbacks.append(rep.incident_id)
    )

    frame = create_crack_frame()
    t0 = 5000.0

    submit_start = time.perf_counter()
    ok1 = async_auditor.submit_task(
        incident_id="inc_track_1_a",
        zone_id="zone_alpha_danger",
        track_id=1,
        entity_class="crack",
        frame_bgr=frame,
        output_json_path=temp_dir / "inc_track_1_a_vlm.json",
        hint_bbox_xyxy=(200.0, 60.0, 380.0, 440.0),
        current_time=t0
    )
    ok_dup = async_auditor.submit_task(
        incident_id="inc_track_1_spam",
        zone_id="zone_alpha_danger",
        track_id=1,
        entity_class="crack",
        frame_bgr=frame,
        output_json_path=temp_dir / "inc_track_1_spam_vlm.json",
        hint_bbox_xyxy=(200.0, 60.0, 380.0, 440.0),
        current_time=t0 + 2.0
    )
    ok2 = async_auditor.submit_task(
        incident_id="inc_track_2_a",
        zone_id="zone_alpha_danger",
        track_id=2,
        entity_class="crack",
        frame_bgr=frame,
        output_json_path=temp_dir / "inc_track_2_a_vlm.json",
        hint_bbox_xyxy=(200.0, 60.0, 380.0, 440.0),
        current_time=t0 + 2.1
    )
    submit_ms = (time.perf_counter() - submit_start) * 1000.0

    assert ok1 is True
    assert ok_dup is False
    assert ok2 is True
    assert async_auditor.debounced_count == 1
    assert submit_ms < 33.0, f"Main loop submission too slow ({submit_ms:.2f}ms > 33ms budget)!"

    async_auditor.wait_all_completed()
    async_auditor.shutdown(timeout=10.0)

    assert len(async_auditor.completed_reports) == 2
    assert len(received_callbacks) == 2

    json_file = temp_dir / "inc_track_1_a_vlm.json"
    grounded_file = temp_dir / "inc_track_1_a_vlm.grounded.jpg"
    assert json_file.exists()
    assert grounded_file.exists()

    saved_data = json.loads(json_file.read_text(encoding="utf-8"))
    assert saved_data["entity_class"] == "crack"
    assert saved_data["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert "hazard_grounding_box" in saved_data

    rendered = cv2.imread(str(grounded_file))
    assert rendered is not None and rendered.shape == frame.shape

    shutil.rmtree(temp_dir)
    print(f"[+] Test 3 Passed: Non-blocking submit took {submit_ms:.2f}ms (<33ms), 2 audits + grounded renders verified.")


if __name__ == "__main__":
    print("=" * 65)
    print(" SentinelEdge v2.1: Grounded Crack Morphology & VLM Suite")
    print("=" * 65)
    test_schema_and_json_parser()
    test_crack_grounding_ignores_window()
    test_async_worker_nonblocking_and_debounce()
    print("=" * 65)
    print(" ALL VLM AUDITING & CRACK GROUNDING TESTS PASSED!")
    print("=" * 65)