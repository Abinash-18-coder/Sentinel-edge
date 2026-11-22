import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import time
import cv2
import numpy as np
from ultralytics import YOLO


def collect_sample_images(max_images: int = 30) -> list[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    search_dirs = [
        PROJECT_ROOT / "data" / "processed" / "valid" / "images",
        PROJECT_ROOT / "data" / "processed" / "val" / "images",
        PROJECT_ROOT / "data" / "calibration",
        PROJECT_ROOT / "data" / "processed" / "train" / "images"
    ]
    for folder in search_dirs:
        if folder.exists():
            imgs = sorted([p for p in folder.glob("*") if p.suffix.lower() in valid_exts])
            if imgs:
                return imgs[:max_images]
    return []


def benchmark_model(model_path: Path, images: list[Path], conf: float = 0.22) -> None:
    if not model_path.exists():
        print(f"[-] Skipping missing model: {model_path.name}")
        return

    model = YOLO(str(model_path), task="detect")
    latencies: list[float] = []
    detected_frames = 0
    total_boxes = 0
    confs: list[float] = []

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=640, conf=conf, verbose=False)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=640, conf=conf, verbose=False)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            detected_frames += 1
            total_boxes += len(boxes)
            confs.extend(boxes.conf.cpu().numpy().tolist())

    del model
    gc.collect()

    n = max(1, len(latencies))
    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    recall_proxy = (detected_frames / float(n)) * 100.0
    print(
        f"[+] {model_path.name:<16} | Hit Rate: {detected_frames:2d}/{n:<2d} ({recall_proxy:5.1f}%) "
        f"| Boxes: {total_boxes:3d} | Avg Conf: {mean_conf:.3f} | Avg Latency: {mean_ms:6.1f} ms"
    )


if __name__ == "__main__":
    print("=" * 78)
    print(" SentinelEdge v2.1: Crack Recall & Latency Benchmark (FP32 vs FP16 vs INT8)")
    print("=" * 78)
    samples = collect_sample_images(max_images=30)
    if not samples:
        print("[X] No sample images found in data/processed/ or data/calibration/.")
        sys.exit(1)

    print(f"[*] Benchmarking on {len(samples)} real crack images (conf=0.22)...")
    models_to_test = [
        PROJECT_ROOT / "models" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "onnx" / "best.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best_fp16.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best_int8.onnx"
    ]
    for m in models_to_test:
        benchmark_model(m, samples, conf=0.22)
    print("=" * 78)