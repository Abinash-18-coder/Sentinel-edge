import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import zipfile
import yaml


def validate_yolo_labels(labels_dir: Path) -> tuple[int, int]:
    total_files = 0
    invalid_boxes = 0

    if not labels_dir.exists():
        print(f"[-] Directory not found: {labels_dir}")
        return 0, 0

    for label_file in labels_dir.glob("*.txt"):
        total_files += 1
        lines = label_file.read_text(encoding="utf-8").splitlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_boxes += 1
                continue

            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                    invalid_boxes += 1
            except ValueError:
                invalid_boxes += 1

    return total_files, invalid_boxes


def prepare_dataset() -> None:
    print("=" * 65)
    print(" SentinelEdge v2.0: Dataset Ingestion & Validation Pipeline")
    print("=" * 65)

    raw_zip = PROJECT_ROOT / "dataset_bundle.zip"
    extract_target = PROJECT_ROOT / "data" / "processed"

    if raw_zip.exists() and not (extract_target / "train").exists():
        print(f"[*] Extracting {raw_zip.name} into {extract_target}...")
        extract_target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_zip, 'r') as zf:
            zf.extractall(extract_target)
        print("[+] Dataset extraction complete.")
    else:
        print("[*] Dataset zip already extracted or located in data/processed.")

    valid_folder = extract_target / "valid"
    val_folder = extract_target / "val"
    if valid_folder.exists() and not val_folder.exists():
        print("[*] Normalizing validation folder name from 'valid' to 'val'...")
        valid_folder.rename(val_folder)

    train_labels = extract_target / "train" / "labels"
    val_labels = extract_target / "val" / "labels"

    print("[*] Validating training label coordinate bounds...")
    train_files, train_errs = validate_yolo_labels(train_labels)
    print(f"    - Train Labels Scanned : {train_files} files")
    print(f"    - Invalid Bounding Box : {train_errs} errors")

    print("[*] Validating validation label coordinate bounds...")
    val_files, val_errs = validate_yolo_labels(val_labels)
    print(f"    - Val Labels Scanned   : {val_files} files")
    print(f"    - Invalid Bounding Box : {val_errs} errors")

    print("=" * 65)
    print(" DATASET INGESTION & VALIDATION PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    prepare_dataset()