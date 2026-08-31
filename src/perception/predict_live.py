"""
src/perception/predict_live.py
Real-time camera inference test verifying bounding boxes, class labels, and FPS counters.
Compatible with Python 3.12.5.
"""

from __future__ import annotations

import time
from pathlib import Path
import cv2
import supervision as sv
from ultralytics import YOLO


def run_live_stream(model_path: Path, source: int | str = 0) -> None:
    """Runs real-time camera inference with bounding boxes and FPS telemetry."""
    print(f"[*] Initializing model from: {model_path}...")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[X] Error: Could not open video source: {source}")
        return

    # Initialize Supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_padding=6)

    prev_time = time.perf_counter()

    print("[+] Video stream initialized. Press 'q' inside the video window to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Inference
        results = model(frame, verbose=False, conf=0.30)[0]
        detections = sv.Detections.from_ultralytics(results)

        # FPS Computation
        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - prev_time, 1e-5)
        prev_time = current_time

        # Format detection labels
        labels = [
            f"{model.names[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        # Draw overlays
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Draw telemetry HUD
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Active Objects: {len(detections)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sentinel-Edge: Perception Stream Verification", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Video stream closed cleanly.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    target_weights = root / "models" / "weights" / "best.pt"
    if not target_weights.exists():
        target_weights = root / "yolo11n.pt"

    run_live_stream(target_weights, source=0)