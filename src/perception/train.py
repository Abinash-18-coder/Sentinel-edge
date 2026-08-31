"""
src/perception/train.py
Production YOLO training pipeline with metric checkpoints and artifact logging.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import os
import sys
import multiprocessing
from pathlib import Path
import torch
from ultralytics import YOLO


def execute_training(
    data_yaml: Path,
    model_variant: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    project_dir: Path = Path("runs/train"),
    run_name: str = "sentinel_detector_v1"
) -> Path:
    """Executes YOLO model training and returns the path to the best weights."""
    print("=" * 65)
    print(f" Starting Sentinel-Edge Perception Training [{model_variant}]")
    print(f" Compute Device : {'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f" Python Version : {sys.version.split()[0]}")
    print("=" * 65)

    if not data_yaml.exists():
        raise FileNotFoundError(f"[X] Dataset config not found: {data_yaml}")

    # Load baseline model
    model = YOLO(model_variant)

    # Windows safety: configure workers to prevent shared memory deadlocks
    num_workers = 2 if os.name == "nt" and torch.cuda.is_available() else 0

    # Start training loop
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=num_workers,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,        # Box loss weight
        cls=0.5,        # Classification loss weight
        dfl=1.5,        # Distribution Focal Loss weight
        plots=True,     # Generate PR curves and confusion matrices
        save=True,
        val=True
    )

    best_checkpoint = project_dir / run_name / "weights" / "best.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"[X] Training finished but weights were not found at: {best_checkpoint}")

    print(f"\n[+] Training successfully completed.")
    print(f"[+] Optimal weights saved to: {best_checkpoint.resolve()}")
    return best_checkpoint


if __name__ == "__main__":
    # Required on Windows for Python 3.12 multiprocessing safety
    multiprocessing.freeze_support()

    root_dir = Path(__file__).resolve().parent.parent.parent
    dataset_yaml = root_dir / "configs" / "data.yaml"
    runs_dir = root_dir / "runs" / "train"

    # Local execution: runs 5 epochs for pipeline verification
    execute_training(
        data_yaml=dataset_yaml,
        model_variant="yolo11n.pt",
        epochs=5,
        imgsz=640,
        batch_size=8,
        project_dir=runs_dir,
        run_name="local_test_run"
    )