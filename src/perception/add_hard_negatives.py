import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import zipfile


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inject_hard_negatives(
    negatives_dir: Path,
    dataset_root: Path,
    val_ratio: float = 0.2
) -> dict[str, int]:
    negatives_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in negatives_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])
    if not images:
        return {"train_added": 0, "valid_added": 0}

    val_split_name = "valid" if (dataset_root / "valid").exists() else "val"
    train_img_dir = dataset_root / "train" / "images"
    train_lbl_dir = dataset_root / "train" / "labels"
    val_img_dir = dataset_root / val_split_name / "images"
    val_lbl_dir = dataset_root / val_split_name / "labels"

    for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    val_count = max(1, int(round(len(images) * val_ratio))) if len(images) >= 5 else 0
    val_set = set(images[:val_count])

    train_added = 0
    valid_added = 0

    for idx, src_img in enumerate(images, start=1):
        out_stem = f"hn_{idx:04d}_{src_img.stem}"
        if src_img in val_set:
            dst_img = val_img_dir / f"{out_stem}{src_img.suffix.lower()}"
            dst_lbl = val_lbl_dir / f"{out_stem}.txt"
            valid_added += 1
        else:
            dst_img = train_img_dir / f"{out_stem}{src_img.suffix.lower()}"
            dst_lbl = train_lbl_dir / f"{out_stem}.txt"
            train_added += 1

        shutil.copy2(src_img, dst_img)
        dst_lbl.write_text("", encoding="utf-8")

    return {"train_added": train_added, "valid_added": valid_added}


def bundle_dataset_for_colab(dataset_root: Path, output_zip: Path) -> Path:
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(dataset_root.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(dataset_root))
    return output_zip


if __name__ == "__main__":
    neg_dir = PROJECT_ROOT / "data" / "hard_negatives"
    ds_root = PROJECT_ROOT / "data" / "processed"
    zip_out = PROJECT_ROOT / "dataset_pro_bundle.zip"

    stats = inject_hard_negatives(neg_dir, ds_root)
    if stats["train_added"] == 0 and stats["valid_added"] == 0:
        print(f"[!] Place 20-50 photos of windows, doors, cables, and plain walls inside: {neg_dir}")
        print("    Then re-run: python src/perception/add_hard_negatives.py")
    else:
        print(f"[+] Injected hard negatives -> Train: {stats['train_added']} | Valid: {stats['valid_added']}")
        bundle_dataset_for_colab(ds_root, zip_out)
        print(f"[+] Bundled Colab training archive: {zip_out.name} (ignored by Git)")