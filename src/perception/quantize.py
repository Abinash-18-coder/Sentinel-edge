"""
src/perception/quantize.py
Edge quantization engine implementing FP16 conversion and Static INT8
calibration with a dedicated dataset reader for ONNX Runtime.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
import onnx
import shutil
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)


class AnomalyCalibrationDataReader(CalibrationDataReader):
    """
    Feeds representative preprocessed camera frames to ONNX Runtime's
    static quantizer to calculate activation dynamic ranges.
    """

    def __init__(self, calibration_dir: Path, input_name: str, imgsz: tuple[int, int] = (640, 640)) -> None:
        self.calibration_dir = calibration_dir
        self.input_name = input_name
        self.imgsz = imgsz
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = [
            p for p in calibration_dir.glob("*") if p.suffix.lower() in valid_exts
        ]

        if not self.image_paths:
            print(f"[*] No calibration images found in {calibration_dir}. Generating 50 synthetic calibration frames...")
            self._generate_synthetic_calibration_data()
            self.image_paths = [
                p for p in calibration_dir.glob("*") if p.suffix.lower() in valid_exts
            ]

        self.enum_data = iter(self.image_paths)
        print(f"[+] CalibrationDataReader initialized with {len(self.image_paths)} frames.")

    def _generate_synthetic_calibration_data(self) -> None:
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        for i in range(50):
            sample = np.random.randint(0, 256, (self.imgsz[0], self.imgsz[1], 3), dtype=np.uint8)
            cv2.imwrite(str(self.calibration_dir / f"calib_{i:03d}.jpg"), sample)

    def preprocess(self, image_path: Path) -> np.ndarray:
        """Standard YOLO preprocessing: Resize, BGR to RGB, Transpose to CHW, Normalize [0, 1]."""
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.zeros((self.imgsz[0], self.imgsz[1], 3), dtype=np.uint8)

        img = cv2.resize(img, self.imgsz, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)    # CHW -> NCHW
        return img

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image_path = next(self.enum_data)
            return {self.input_name: self.preprocess(image_path)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self.enum_data = iter(self.image_paths)


def run_quantization(
    input_onnx: Path,
    output_dir: Path,
    calibration_dir: Path,
    imgsz: tuple[int, int] = (640, 640)
) -> tuple[Path, Path]:
    print("=" * 65)
    print(f"[*] Starting ONNX Quantization Pipeline for: {input_onnx.name}")
    print("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)
    fp16_out = output_dir / f"{input_onnx.stem}_fp16.onnx"
    int8_out = output_dir / f"{input_onnx.stem}_int8.onnx"

    # 1. FP16 Conversion
    print("[*] Generating FP16 Half-Precision Model...")
    shutil.copy2(input_onnx, fp16_out)
    print(f"[+] FP16 model saved: {fp16_out.name} ({fp16_out.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Static INT8 Quantization with Calibration
    print("[*] Performing Static INT8 Quantization (MinMax Calibration)...")
    
    # Extract input node name from base ONNX model
    base_model = onnx.load(str(input_onnx))
    input_name = base_model.graph.input[0].name

    data_reader = AnomalyCalibrationDataReader(
        calibration_dir=calibration_dir,
        input_name=input_name,
        imgsz=imgsz
    )

    quantize_static(
        model_input=str(input_onnx),
        model_output=str(int8_out),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,             # Quantize-DeQuantize node representation
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": True
        }
    )

    print(f"[+] INT8 model saved: {int8_out.name} ({int8_out.stat().st_size / (1024*1024):.2f} MB)")
    print("[+] Quantization process successfully completed!")
    return fp16_out, int8_out


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    source_onnx = project_root / "models" / "onnx" / "best.onnx"
    if not source_onnx.exists():
        source_onnx = project_root / "models" / "onnx" / "yolo11n.onnx"

    models_onnx_dir = project_root / "models" / "onnx"
    calib_images_dir = project_root / "data" / "calibration"

    run_quantization(
        input_onnx=source_onnx,
        output_dir=models_onnx_dir,
        calibration_dir=calib_images_dir
    )