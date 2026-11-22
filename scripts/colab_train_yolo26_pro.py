from pathlib import Path
import shutil
import yaml
from ultralytics import YOLO

TASK = "detect"
DATASET_DIR = Path("/content/dataset")
EXPORT_DIR = Path("/content/sentineledge_pro_weights")


def build_data_yaml(ds_dir: Path) -> Path:
    val_name = "valid" if (ds_dir / "valid").exists() else "val"
    cfg = {
        "path": str(ds_dir),
        "train": "train/images",
        "val": f"{val_name}/images",
        "names": {0: "crack"}
    }
    yaml_path = ds_dir / "data_pro.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml = build_data_yaml(DATASET_DIR)

    base_weights = "yolo26s-seg.pt" if TASK == "segment" else "yolo26s.pt"
    print(f"[*] Training {base_weights} (task={TASK}) with crack-optimized augmentations...")

    model = YOLO(base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=640,
        batch=16,
        patience=20,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.45,
        degrees=15.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=15,
        project="/content/runs",
        name=f"sentineledge_yolo26s_{TASK}"
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    out_pt_name = "best_seg.pt" if TASK == "segment" else "best.pt"
    out_onnx_name = "best_seg.onnx" if TASK == "segment" else "best.onnx"

    shutil.copy2(best_pt, EXPORT_DIR / out_pt_name)

    trained = YOLO(str(best_pt))
    val_metrics = trained.val(data=str(data_yaml), imgsz=640)
    print(f"[+] Validation mAP50: {val_metrics.box.map50:.4f} | mAP50-95: {val_metrics.box.map:.4f}")

    onnx_exported = Path(trained.export(format="onnx", imgsz=640, simplify=True, dynamic=False))
    shutil.copy2(onnx_exported, EXPORT_DIR / out_onnx_name)
    print(f"[+] Pro weights ready in {EXPORT_DIR}: {out_pt_name}, {out_onnx_name}")


if __name__ == "__main__":
    main()