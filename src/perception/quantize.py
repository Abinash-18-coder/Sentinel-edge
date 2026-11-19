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
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class CrackCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        calibration_dir: Path,
        input_name: str,
        imgsz: int = 640,
        max_samples: int = 200
    ) -> None:
        self.calibration_dir = calibration_dir
        self.input_name = input_name
        self.imgsz = imgsz
        self.max_samples = max_samples
        self.image_paths = self._collect_calibration_images()
        self.enum_data = iter(self.image_paths)
        print(f"[+] Calibration reader ready with {len(self.image_paths)} frames (streaming 1 at a time).")

    def _collect_calibration_images(self) -> list[Path]:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        existing = [p for p in self.calibration_dir.glob("*") if p.suffix.lower() in valid_exts]
        if len(existing) >= 20:
            return existing[:self.max_samples]

        source_dirs = [
            PROJECT_ROOT / "data" / "processed" / "train" / "images",
            PROJECT_ROOT / "data" / "processed" / "val" / "images",
            PROJECT_ROOT / "data" / "processed" / "valid" / "images"
        ]
        copied = list(existing)
        for src_dir in source_dirs:
            if not src_dir.exists():
                continue
            for img_file in src_dir.glob("*"):
                if img_file.suffix.lower() not in valid_exts:
                    continue
                if len(copied) >= self.max_samples:
                    break
                dest = self.calibration_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    copied.append(dest)

        if not copied:
            print("[!] No real images found in data/processed/. Creating 30 fallback calibration frames...")
            rng = np.random.default_rng(42)
            for i in range(30):
                fake = rng.integers(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
                out_path = self.calibration_dir / f"calib_{i:03d}.jpg"
                cv2.imwrite(str(out_path), fake)
                copied.append(out_path)

        return copied[:self.max_samples]

    def _preprocess_single(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            next_path = next(self.enum_data)
            return {self.input_name: self._preprocess_single(next_path)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self.enum_data = iter(self.image_paths)


def verify_quantized_model(model_path: Path, imgsz: int = 640) -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(5):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    del session
    gc.collect()

    mean_ms = float(np.mean(times))
    fps = 1000.0 / max(mean_ms, 1e-4)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[+] Verified {model_path.name:<18} | Size: {size_mb:5.2f} MB | Mean: {mean_ms:6.2f} ms | Est: {fps:5.1f} FPS")


def run_quantization(
    input_onnx: Path,
    weights_pt: Path,
    output_dir: Path,
    calibration_dir: Path,
    imgsz: int = 640
) -> tuple[Path, Path]:
    print("=" * 65)
    print(f" SentinelEdge v2.0: FP16 & Static INT8 Quantization ({input_onnx.name})")
    print("=" * 65)

    if not input_onnx.exists():
        raise FileNotFoundError(f"[X] Base ONNX model not found: {input_onnx}. Run export_onnx.py first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fp16_out = output_dir / f"{input_onnx.stem}_fp16.onnx"
    prep_out = output_dir / f"{input_onnx.stem}_prep_tmp.onnx"
    int8_out = output_dir / f"{input_onnx.stem}_int8.onnx"

    print("[*] Stage 1: Exporting native FP16 ONNX model via Ultralytics 8.4+...")
    if weights_pt.exists():
        pt_model = YOLO(str(weights_pt))
        exported_fp16_str = pt_model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            quantize=16,
            nms=False
        )
        exported_fp16 = Path(exported_fp16_str)
        if exported_fp16.resolve() != fp16_out.resolve():
            shutil.move(str(exported_fp16), str(fp16_out))
        del pt_model
        gc.collect()
        verify_quantized_model(fp16_out, imgsz)
    else:
        print(f"[!] PyTorch weights not found at {weights_pt}. Skipping FP16 export.")

    print("[*] Stage 2: Pre-processing ONNX graph (skipping SymbolicShapeInference for YOLO26 TopK)...")
    try:
        quant_pre_process(
            input_model_path=str(input_onnx),
            output_model_path=str(prep_out),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True
        )
    except Exception as err:
        print(f"[!] quant_pre_process fallback triggered ({err}). Using standard ONNX shape inference...")
        onnx.shape_inference.infer_shapes_path(str(input_onnx), str(prep_out))

    prep_model = onnx.load(str(prep_out))
    input_name = prep_model.graph.input[0].name
    del prep_model
    gc.collect()

    print("[*] Stage 3: Running Static INT8 Calibration (Conv/MatMul/Gemm optimized for Intel CPU)...")
    reader = CrackCalibrationDataReader(
        calibration_dir=calibration_dir,
        input_name=input_name,
        imgsz=imgsz,
        max_samples=200
    )

    quantize_static(
        model_input=str(prep_out),
        model_output=str(int8_out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True
        }
    )

    if prep_out.exists():
        prep_out.unlink()
    gc.collect()

    print("[*] Stage 4: Comparing FP32 vs INT8 CPU Performance...")
    verify_quantized_model(input_onnx, imgsz)
    verify_quantized_model(int8_out, imgsz)

    print("=" * 65)
    print(" DAY 8 QUANTIZATION PIPELINE COMPLETE!")
    print("=" * 65)
    return fp16_out, int8_out


if __name__ == "__main__":
    base_onnx = PROJECT_ROOT / "models" / "onnx" / "best.onnx"
    if not base_onnx.exists():
        base_onnx = PROJECT_ROOT / "models" / "onnx" / "yolo26n.onnx"

    pt_weights = PROJECT_ROOT / "models" / "weights" / "best.pt"
    if not pt_weights.exists():
        pt_weights = PROJECT_ROOT / "yolo26n.pt"

    run_quantization(
        input_onnx=base_onnx,
        weights_pt=pt_weights,
        output_dir=PROJECT_ROOT / "models" / "onnx",
        calibration_dir=PROJECT_ROOT / "data" / "calibration",
        imgsz=640
    )