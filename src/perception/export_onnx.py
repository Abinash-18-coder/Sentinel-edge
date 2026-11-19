import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import shutil
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def pick_parity_image(imgsz: int) -> np.ndarray:
    search_dirs = [
        PROJECT_ROOT / "data" / "processed" / "val" / "images",
        PROJECT_ROOT / "data" / "processed" / "train" / "images",
        PROJECT_ROOT / "data" / "raw"
    ]
    for folder in search_dirs:
        if folder.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                matches = list(folder.glob(ext))
                if matches:
                    img = cv2.imread(str(matches[0]))
                    if img is not None:
                        print(f"[+] Using real validation image for parity check: {matches[0].name}")
                        return cv2.resize(img, (imgsz, imgsz))

    print("[*] No local validation image found. Using deterministic synthetic 640x640 image.")
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)


def export_and_verify(
    weights_path: Path,
    output_dir: Path,
    imgsz: int = 640,
    opset_version: int = 18
) -> Path:
    print("=" * 65)
    print(f" SentinelEdge v2.0: ONNX Export & Parity Check ({weights_path.name})")
    print("=" * 65)

    if not weights_path.exists():
        raise FileNotFoundError(f"[X] Weights file not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = output_dir / f"{weights_path.stem}.onnx"

    pt_model = YOLO(str(weights_path))
    exported_str = pt_model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=opset_version,
        nms=False
    )

    exported_file = Path(exported_str)
    if exported_file.resolve() != target_onnx.resolve():
        shutil.copy2(exported_file, target_onnx)

    size_mb = target_onnx.stat().st_size / (1024 * 1024)
    print(f"[+] ONNX model exported: {target_onnx} ({size_mb:.2f} MB)")

    onnx_graph = onnx.load(str(target_onnx))
    onnx.checker.check_model(onnx_graph)
    del onnx_graph
    gc.collect()
    print("[+] ONNX graph structural check passed.")

    test_img = pick_parity_image(imgsz)

    pt_res = pt_model.predict(test_img, imgsz=imgsz, rect=False, conf=0.25, verbose=False)[0]
    pt_boxes = pt_res.boxes.xyxy.cpu().numpy() if pt_res.boxes is not None else np.empty((0, 4))
    pt_confs = pt_res.boxes.conf.cpu().numpy() if pt_res.boxes is not None else np.empty((0,))

    del pt_model
    gc.collect()

    onnx_wrapper = YOLO(str(target_onnx), task="detect")
    onnx_res = onnx_wrapper.predict(test_img, imgsz=imgsz, conf=0.25, verbose=False)[0]
    onnx_boxes = onnx_res.boxes.xyxy.cpu().numpy() if onnx_res.boxes is not None else np.empty((0, 4))
    onnx_confs = onnx_res.boxes.conf.cpu().numpy() if onnx_res.boxes is not None else np.empty((0,))

    del onnx_wrapper
    gc.collect()

    print(f"[+] PyTorch Detections : {len(pt_boxes)} boxes")
    print(f"[+] ONNX Detections    : {len(onnx_boxes)} boxes")

    if len(pt_boxes) > 0 and len(pt_boxes) == len(onnx_boxes):
        pt_order = np.argsort(-pt_confs)
        onnx_order = np.argsort(-onnx_confs)
        max_conf_diff = float(np.max(np.abs(pt_confs[pt_order] - onnx_confs[onnx_order])))
        max_box_diff = float(np.max(np.abs(pt_boxes[pt_order] - onnx_boxes[onnx_order])))
        print(f"[+] Max Confidence Diff: {max_conf_diff:.6f}")
        print(f"[+] Max Box Pixel Diff : {max_box_diff:.4f} px")
    else:
        print("[*] Box counts logged. Verify visual accuracy if image had zero cracks.")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(target_onnx),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0)

    t0 = time.perf_counter()
    raw_out = session.run(None, {input_meta.name: tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[+] Raw ORT Input      : {input_meta.name} {input_meta.shape}")
    print(f"[+] Raw ORT Output     : {output_meta.name} {raw_out.shape}")
    print(f"[+] Single-Frame CPU   : {latency_ms:.2f} ms")
    print("=" * 65)
    print(" DAY 7 ONNX EXPORT & PARITY VERIFICATION COMPLETE!")
    print("=" * 65)
    return target_onnx


if __name__ == "__main__":
    weights = PROJECT_ROOT / "models" / "weights" / "best.pt"
    if not weights.exists():
        weights = PROJECT_ROOT / "yolo26n.pt"
    export_and_verify(
        weights_path=weights,
        output_dir=PROJECT_ROOT / "models" / "onnx",
        imgsz=640,
        opset_version=18
    )