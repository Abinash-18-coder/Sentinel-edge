"""
src/perception/augment_and_verify.py
Ground-truth label visualization and augmentation inspection using Supervision and OpenCV.
Compatible with Python 3.12.5.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import supervision as sv
import yaml


def load_class_mapping(yaml_path: Path) -> dict[int, str]:
    """Loads class ID-to-name mapping from configs/data.yaml."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"[X] Config file not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {int(k): str(v) for k, v in data.get("names", {}).items()}


def render_ground_truth_overlays(
    images_dir: Path,
    labels_dir: Path,
    class_map: dict[int, str],
    output_dir: Path,
    max_samples: int = 5
) -> None:
    """Renders visual bounding box overlays on images and saves them for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )[:max_samples]

    if not image_files:
        print(f"[!] No images found in {images_dir}")
        return

    # Initialize Supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)

    print(f"[*] Rendering {len(image_files)} ground-truth validation overlays to {output_dir}...")

    for img_path in image_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        xyxy_boxes: list[list[float]] = []
        class_ids: list[int] = []

        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) != 5:
                        continue
                    cls_id = int(tokens[0])
                    xc, yc, bw, bh = map(float, tokens[1:])

                    # Convert normalized xywh -> pixel xyxy coordinates
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h

                    xyxy_boxes.append([x1, y1, x2, y2])
                    class_ids.append(cls_id)

        if xyxy_boxes:
            detections = sv.Detections(
                xyxy=np.array(xyxy_boxes, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            labels = [
                f"{class_map.get(cid, str(cid))}"
                for cid in detections.class_id
            ]

            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        else:
            annotated_frame = image.copy()
            cv2.putText(
                annotated_frame,
                "NO LABELS",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        destination_file = output_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(destination_file), annotated_frame)
        print(f"[+] Saved visual verification render: {destination_file.name}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    config_file = project_root / "configs" / "data.yaml"
    train_images_dir = project_root / "data" / "processed" / "train" / "images"
    train_labels_dir = project_root / "data" / "processed" / "train" / "labels"
    inspection_output_dir = project_root / "data" / "processed" / "verification_renders"

    class_mapping = load_class_mapping(config_file)
    render_ground_truth_overlays(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        class_map=class_mapping,
        output_dir=inspection_output_dir,
        max_samples=5
    )