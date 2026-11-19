import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import time
import cv2
import psutil
import yaml
from ultralytics import YOLO

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
        PROJECT_ROOT / "models" / "onnx" / "best_int8.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best.onnx",
        PROJECT_ROOT / "models" / "weights" / "best.pt",
        PROJECT_ROOT / "yolo26n.pt",
        PROJECT_ROOT / "yolo11n.pt"
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
    conf_threshold: float = 0.30,
    resolution: tuple[int, int] = (640, 480)
) -> None:
    print("=" * 65)
    print(" SentinelEdge v2.0: Desktop HUD Pipeline (ONNX + Re-ID + VLM)")
    print("=" * 65)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[X] Cannot open video source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    success, test_frame = cap.read()
    if not success or test_frame is None:
        print("[X] Failed to fetch initial frame from video source.")
        cap.release()
        sys.exit(1)

    actual_h, actual_w = test_frame.shape[:2]
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
        fps=30,
        resolution_wh=frame_wh
    )
    async_auditor = AsyncIncidentAuditor(
        max_queue_size=12,
        track_debounce_sec=8.0,
        vlm_mode="auto"
    )

    prev_time = time.perf_counter()
    anomalies_count = 0
    print("[+] Pipeline running. Press 'q' in video window to quit.\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

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
                bx1, by1, bx2, by2 = map(int, ent.bbox_xyxy)
                color = (56, 189, 248) if ent.recovered else (34, 197, 94)
                suffix = " [RE-ID]" if ent.recovered else ""
                label = f"#{ent.track_id} {ent.class_name} {ent.confidence:.2f}{suffix}"
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)
                cv2.putText(
                    annotated,
                    label,
                    (bx1, max(by1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA
                )

            ram_free_gb = psutil.virtual_memory().available / (1024 ** 3)
            cv2.rectangle(annotated, (10, 10), (370, 96), (18, 18, 28), -1)
            cv2.putText(annotated, f"FPS: {fps:.1f} | Tracks: {len(entities)} | RAM Free: {ram_free_gb:.2f}GB", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (34, 197, 94), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Incidents: {anomalies_count} | Buf: {incident_buffer.estimate_buffer_ram_mb():.1f}MB", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (244, 63, 94) if anomalies_count else (226, 232, 240), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Model: {weights_path.name}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

            cv2.imshow("SentinelEdge v2.0 Desktop HUD", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        incident_buffer.flush_and_shutdown(timeout=8.0)
        async_auditor.shutdown(timeout=8.0)
        del model
        gc.collect()
        print("[+] Desktop HUD pipeline shut down cleanly.")


if __name__ == "__main__":
    run_pipeline(
        weights_path=resolve_best_model(),
        zones_yaml=PROJECT_ROOT / "configs" / "zones.yaml",
        data_yaml=PROJECT_ROOT / "configs" / "data.yaml",
        source=0,
        conf_threshold=0.30
    )