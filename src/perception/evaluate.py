"""
src/perception/evaluate.py
Validation benchmark, confusion matrix generation, and latency profiling.
Compatible with Python 3.12.5.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def run_evaluation(weights_path: Path, data_yaml: Path, split: str = "val") -> dict[str, float]:
    """Runs evaluation on the specified split and prints quantitative metrics."""
    if not weights_path.exists():
        print(f"[X] Weights file does not exist: {weights_path}")
        sys.exit(1)

    print("=" * 65)
    print(f" Evaluating Sentinel-Edge Detector: {weights_path.name}")
    print(f" Dataset Config: {data_yaml.name} | Target Split: {split}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=True
    )

    # Extract metrics
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    inference_speed_ms = float(metrics.speed["inference"])
    fps = 1000.0 / max(inference_speed_ms, 0.001)

    print("\n" + "-" * 45)
    print(" QUANTITATIVE VALIDATION METRICS REPORT")
    print("-" * 45)
    print(f" Precision (P)         : {precision * 100:.2f}%")
    print(f" Recall (R)            : {recall * 100:.2f}%")
    print(f" mAP @ 0.50            : {map50 * 100:.2f}%")
    print(f" mAP @ 0.50:0.95       : {map50_95 * 100:.2f}%")
    print(f" Inference Latency     : {inference_speed_ms:.2f} ms ({fps:.1f} FPS)")
    print("-" * 45)

    # Ensure checkpoint is synchronized with models/weights/best.pt
    target_weights = weights_path.parents[3] / "models" / "weights" / "best.pt"
    target_weights.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.resolve() != target_weights.resolve():
        shutil.copy2(weights_path, target_weights)
        print(f"[+] Synced primary deployment weights to: {target_weights}")

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "latency_ms": inference_speed_ms
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    
    # Priority: 1. models/weights/best.pt, 2. local training run, 3. base yolo11n.pt
    primary_weights = root / "models" / "weights" / "best.pt"
    local_run_weights = root / "runs" / "train" / "local_test_run" / "weights" / "best.pt"
    fallback_weights = root / "yolo11n.pt"

    if primary_weights.exists():
        chosen_weights = primary_weights
    elif local_run_weights.exists():
        chosen_weights = local_run_weights
    else:
        chosen_weights = fallback_weights

    config_path = root / "configs" / "data.yaml"
    run_evaluation(chosen_weights, config_path, split="val")