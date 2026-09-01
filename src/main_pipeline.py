"""
src/main_pipeline.py
Unified real-time inference pipeline integrating YOLO perception, ByteTrack,
spatial rules state machines, and circular incident video buffering.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
import cv2
import yaml
import supervision as sv
from ultralytics import YOLO

from src.perception.tracker import ByteTrackEngine
from src.rules.spatial_engine import SpatialRulesEngine
from src.rules.incident_buffer import IncidentVideoBuffer


def load_classes(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {0: "normal", 1: "crack", 2: "corrosion", 3: "defect"}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): str(v) for k, v in data.get("names", {}).items()}


def run_pipeline(
    weights_path: Path,
    zones_yaml: Path,
    data_yaml: Path,
    source: int | str = 0,
    conf_threshold: float = 0.35,
    resolution: tuple[int, int] = (640, 480)
) -> None:
    print("=" * 65)
    print(" Sentinel-Edge: Real-Time Spatio-Temporal Pipeline (Week 2)")
    print("=" * 65)

    # 1. Initialize Ingestion Subsystem
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[X] Cannot open video stream source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Read test frame to verify dimensions
    success, test_frame = cap.read()
    if not success or test_frame is None:
        print("[X] Failed to fetch frame from ingestion source.")
        sys.exit(1)

    actual_h, actual_w = test_frame.shape[:2]
    frame_wh = (actual_w, actual_h)
    print(f"[+] Ingestion active at resolution: {frame_wh[0]}x{frame_wh[1]} px")

    # 2. Initialize Models & Engines
    class_map = load_classes(data_yaml)
    print(f"[*] Loading perception weights: {weights_path.name}...")
    model = YOLO(str(weights_path))

    print("[*] Initializing ByteTrack Engine...")
    tracker_engine = ByteTrackEngine(frame_rate=30)

    print("[*] Initializing Spatial Rules & State Machine Engine...")
    spatial_engine = SpatialRulesEngine(config_yaml=zones_yaml, frame_resolution_wh=frame_wh)

    print("[*] Initializing Circular Incident Buffer...")
    incidents_dir = weights_path.parents[2] / "data" / "raw" / "incidents"
    incident_buffer = IncidentVideoBuffer(
        output_dir=incidents_dir,
        pre_incident_seconds=3.0,
        post_incident_seconds=3.0,
        fps=30,
        resolution_wh=frame_wh
    )

    # 3. Supervision Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)

    prev_time = time.perf_counter()
    recent_anomalies_count = 0

    print("[+] Pipeline running. Press 'q' inside video window to stop.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, 1e-5)
            prev_time = current_time

            # A. Perception Inference
            results = model(frame, verbose=False, conf=conf_threshold)[0]

            # B. Spatio-Temporal Multi-Object Tracking
            tracked_detections, entities = tracker_engine.update(
                ultralytics_results=results,
                class_name_map=class_map
            )

            # C. Evaluate Spatial Rules & Debounced State Machine
            triggered_events = spatial_engine.evaluate(
                detections=tracked_detections,
                class_name_map=class_map
            )

            # D. Handle Confirmed Anomalies
            for event in triggered_events:
                recent_anomalies_count += 1
                incident_buffer.trigger_incident_capture(event)

            # E. Push Frame to Incident Buffer
            incident_buffer.push_frame(frame)

            # F. Render Visual Annotations
            annotated_frame = frame.copy()
            # 1. Render Zone & Tripwire Overlays
            annotated_frame = spatial_engine.render_overlays(annotated_frame)
            # 2. Render Trajectory Motion Trails
            annotated_frame = tracker_engine.annotate_traces(annotated_frame, tracked_detections)

            # 3. Render Bounding Boxes & Tracking IDs
            if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
                labels = [
                    f"#{tid} {class_map.get(cid, str(cid))} {conf:.2f}"
                    for tid, cid, conf in zip(
                        tracked_detections.tracker_id,
                        tracked_detections.class_id,
                        tracked_detections.confidence
                    )
                ]
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

            # G. Render Telemetry HUD
            cv2.rectangle(annotated_frame, (10, 10), (320, 85), (20, 20, 20), -1)
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | Active Tracks: {len(entities)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            cv2.putText(
                annotated_frame,
                f"Incidents Logged: {recent_anomalies_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if recent_anomalies_count > 0 else (255, 255, 255),
                2
            )

            cv2.imshow("Sentinel-Edge: Multi-Object Tracking & Anomaly HUD", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Video capture stream released cleanly.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    
    weights = project_root / "models" / "weights" / "best.pt"
    if not weights.exists():
        weights = project_root / "yolo11n.pt"

    zones_cfg = project_root / "configs" / "zones.yaml"
    data_cfg = project_root / "configs" / "data.yaml"

    run_pipeline(
        weights_path=weights,
        zones_yaml=zones_cfg,
        data_yaml=data_cfg,
        source=0,  # 0 for webcam, or path to MP4 test video: "data/raw/test_video.mp4"
        conf_threshold=0.30
    )