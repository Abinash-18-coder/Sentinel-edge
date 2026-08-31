"""
src/perception/dataset_prep.py
Dataset ingestion, bounding box integrity validation, and YOLO splitting pipeline.
Compatible with Python 3.12.5.
"""

from __future__ import annotations

import os
import shutil
import random
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np


class BoundingBox(NamedTuple):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class DatasetValidator:
    """
    Validates image readability, ensures YOLO label coordinate normalization,
    and partitions raw datasets into train, val, and test splits.
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        split_ratio: tuple[float, float, float] = (0.7, 0.2, 0.1)
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratio
        assert round(sum(split_ratio), 2) == 1.0, "Split ratios must sum exactly to 1.0"

    def validate_annotation(self, label_path: Path) -> list[BoundingBox]:
        """Reads and validates normalized bounding box coordinates from a .txt label file."""
        valid_boxes: list[BoundingBox] = []
        if not label_path.exists():
            return valid_boxes

        with open(label_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) != 5:
                    print(f"[!] Warning: Invalid format in {label_path.name}:{line_idx} -> '{line.strip()}'")
                    continue

                try:
                    cls_id = int(tokens[0])
                    xc, yc, w, h = map(float, tokens[1:])
                except ValueError as err:
                    print(f"[!] Numerical parsing error in {label_path.name}:{line_idx} -> {err}")
                    continue

                # Mathematical bounds check: normalized coordinates must satisfy [0.0, 1.0]
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    print(f"[!] Coordinate out of bounds [0, 1] in {label_path.name}:{line_idx} -> ({xc}, {yc}, {w}, {h})")
                    continue

                valid_boxes.append(BoundingBox(cls_id, xc, yc, w, h))

        return valid_boxes

    def verify_image(self, image_path: Path) -> tuple[int, int] | None:
        """Verifies image decodability using OpenCV and returns (height, width)."""
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            print(f"[X] Corrupted or unreadable image file: {image_path.name}")
            return None
        h, w = img.shape[:2]
        return h, w

    def process_and_split(self) -> None:
        """Validates all raw image-label pairs and executes deterministic dataset splitting."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        raw_images = [
            p for p in self.raw_dir.glob("**/*")
            if p.suffix.lower() in valid_extensions
        ]

        if not raw_images:
            print(f"[*] No raw data found in {self.raw_dir}. Generating synthetic anomaly dataset for zero-error pipeline execution...")
            self._generate_synthetic_samples()
            raw_images = [
                p for p in self.raw_dir.glob("**/*")
                if p.suffix.lower() in valid_extensions
            ]

        valid_pairs: list[tuple[Path, Path, list[BoundingBox]]] = []

        print(f"[*] Scanning {len(raw_images)} raw images for corruption and annotation integrity...")
        for img_path in raw_images:
            dims = self.verify_image(img_path)
            if dims is None:
                continue

            lbl_path = img_path.with_suffix(".txt")
            boxes = self.validate_annotation(lbl_path)
            valid_pairs.append((img_path, lbl_path, boxes))

        print(f"[+] Verified {len(valid_pairs)} valid image-annotation pairs.")

        # Deterministic shuffle for reproducibility
        random.seed(42)
        random.shuffle(valid_pairs)

        total = len(valid_pairs)
        n_train = int(total * self.train_ratio)
        n_val = int(total * self.val_ratio)

        splits = {
            "train": valid_pairs[:n_train],
            "val": valid_pairs[n_train:n_train + n_val],
            "test": valid_pairs[n_train + n_val:]
        }

        for split_name, pairs in splits.items():
            img_dest_dir = self.processed_dir / split_name / "images"
            lbl_dest_dir = self.processed_dir / split_name / "labels"
            img_dest_dir.mkdir(parents=True, exist_ok=True)
            lbl_dest_dir.mkdir(parents=True, exist_ok=True)

            print(f"[*] Writing {len(pairs)} records to {split_name} split...")
            for img_path, lbl_path, boxes in pairs:
                shutil.copy2(img_path, img_dest_dir / img_path.name)
                dest_lbl_path = lbl_dest_dir / f"{img_path.stem}.txt"
                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    for box in boxes:
                        f.write(f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n")

        print("[+] Dataset pipeline processing and verification complete.")

    def _generate_synthetic_samples(self) -> None:
        """Generates synthetic visual anomaly samples to verify pipeline functionality without external data."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(30):
            canvas = np.full((640, 640, 3), fill_value=210, dtype=np.uint8)
            # Add synthetic background noise/texture
            noise = np.random.randint(0, 35, (640, 640, 3), dtype=np.uint8)
            canvas = cv2.subtract(canvas, noise)

            # Draw simulated crack defect (Class ID = 1)
            pts = np.array([
                [80 + idx * 4, 100],
                [180 + idx * 3, 260],
                [160 + idx * 5, 420],
                [240 + idx * 2, 580]
            ], np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(25, 25, 25), thickness=3)

            img_path = self.raw_dir / f"sample_{idx:03d}.jpg"
            lbl_path = self.raw_dir / f"sample_{idx:03d}.txt"

            cv2.imwrite(str(img_path), canvas)
            with open(lbl_path, "w", encoding="utf-8") as f:
                # Class 1 (crack), centered bounding box covering the polyline
                f.write("1 0.350000 0.500000 0.350000 0.800000\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    raw_directory = root / "data" / "raw"
    processed_directory = root / "data" / "processed"

    pipeline = DatasetValidator(raw_dir=raw_directory, processed_dir=processed_directory)
    pipeline.process_and_split()