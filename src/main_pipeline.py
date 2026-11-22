import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import os
import time
import cv2
import psutil
import yaml
from ultralytics import YOLO

from src.api.server import draw_crack_detection
from src.perception.camera_stream import UniversalCameraStream, parse_camera_source
from src.perception.reid_tracker import DeepReIDTracker
from src.rules.incident_buffer import IncidentVideoBuffer
from src.rules.spatial_engine import SpatialRulesEngine
from src.vlm.async_auditor import AsyncIncidentAuditor


def load_classes(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {0: "crack"}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {int(k): str(v) for k, v in data.get("names", {0: "crack"}).items()}


def resolve_best_model() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "onnx" / "best.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best_fp16.onnx",
        PROJECT_ROOT / "models" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "onnx" / "best_int8.onnx",
        PROJECT_ROOT / "yolo26n.pt"
    ]
    for path in candidates:
        if path.exists():
            return path
    return PROJECT_ROOT / "yolo26n.pt"


def run_pipeline(
    weights_path: Path,
    zones_yaml: Path,
    data_yaml: Path,
    source: int | str = 0,
    conf_threshold: float = 0.25
) -> None:
    print("=" * 65)
    print(" SentinelEdge v2.1: Desktop HUD (Universal Camera + FP32 ONNX)")
    print("=" * 65)

    camera = UniversalCameraStream(source=source, max_width=640).start()
    ok, init_frame = camera.wait_for_first_frame(timeout_sec=5.0)
    if not ok or init_frame is None:
        print(f"[X] Cannot read frames from source: {source}")
        camera.stop()
        sys.exit(1)

    actual_h, actual_w = init_frame.shape[:2]
    frame_wh = (actual_w, actual_h)
    print(f"[+] Stream online: {actual_w}x{actual_h} px | Model: {weights_path.name}")

    class_map = load_classes(data_yaml)
    model = YOLO(str(weights_path), task="detect")
    tracker = DeepReIDTracker(frame_rate=30, reid_match_threshold=0.35)
    spatial_engine = SpatialRulesEngine(config_yaml=zones_yaml, frame_resolution_wh=frame_wh)

    incidents_dir = PROJECT_ROOT / "data" / "raw" / "incidents"
    incident_buffer = IncidentVideoBuffer(
        output_dir=incidents_dir,
        pre_incident_seconds=3.0,
        post_incident_seconds=3.0,
        fps=25,
        resolution_wh=frame_wh
    )
    async_auditor = AsyncIncidentAuditor(
        max_queue_size=12,
        track_debounce_sec=6.0,
        vlm_mode="auto"
    )

    prev_time = time.perf_counter()
    anomalies_count = 0
    print("[+] Pipeline running. Press 'q' in video window to quit.\n")

    try:
        while True:
            ret, frame = camera.read_latest()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-5)
            prev_time = now

            results = model.predict(frame, imgsz=640, conf=conf_threshold, verbose=False)[0]
            tracked_detections, entities = tracker.update(
                frame_bgr=frame,
                ultralytics_results=results,
                class_name_map=class_map
            )

            triggered_events = spatial_engine.evaluate(
                detections=tracked_detections,
                class_name_map=class_map
            )

            for ev in triggered_events:
                anomalies_count += 1
                mp4_path = incident_buffer.trigger_incident_capture(ev)
                incident_id = mp4_path.stem if mp4_path is not None else f"incident_{int(ev.timestamp)}_t{ev.track_id}"
                vlm_json_path = incidents_dir / f"{incident_id}_vlm_report.json"

                async_auditor.submit_task(
                    incident_id=incident_id,
                    zone_id=ev.zone_id,
                    track_id=ev.track_id,
                    entity_class=ev.class_name,
                    frame_bgr=frame,
                    output_json_path=vlm_json_path,
                    hint_bbox_xyxy=ev.bbox_xyxy
                )

            incident_buffer.push_frame(frame)

            annotated = spatial_engine.render_overlays(frame)
            annotated = tracker.annotate_traces(annotated, tracked_detections)

            for ent in entities:
                draw_crack_detection(annotated, ent)

            ram_free_gb = psutil.virtual_memory().available / (1024 ** 3)
            cv2.rectangle(annotated, (10, 10), (370, 96), (16, 16, 20), -1)
            cv2.putText(annotated, f"FPS: {fps:.1f} | Tracks: {len(entities)} | RAM Free: {ram_free_gb:.2f}GB", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (244, 244, 240), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Incidents: {anomalies_count} | Buf: {incident_buffer.estimate_buffer_ram_mb():.1f}MB", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (44, 107, 255) if anomalies_count else (200, 204, 212), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Model: {weights_path.name}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (142, 147, 158), 1, cv2.LINE_AA)

            cv2.imshow("SentinelEdge v2.1 Desktop HUD", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop(timeout_sec=5.0)
        cv2.destroyAllWindows()
        incident_buffer.flush_and_shutdown(timeout=8.0)
        async_auditor.shutdown(timeout=8.0)
        del model
        gc.collect()
        print("[+] Desktop HUD pipeline shut down cleanly.")


if __name__ == "__main__":
    env_source = parse_camera_source(os.getenv("SENTINEL_SOURCE", "0"))
    run_pipeline(
        weights_path=resolve_best_model(),
        zones_yaml=PROJECT_ROOT / "configs" / "zones.yaml",
        data_yaml=PROJECT_ROOT / "configs" / "data.yaml",
        source=env_source,
        conf_threshold=0.25
    )