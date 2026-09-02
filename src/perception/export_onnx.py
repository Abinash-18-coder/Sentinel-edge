"""
src/perception/export_onnx.py
Exports PyTorch YOLO weights to ONNX graph format and performs numerical
output parity validation across execution backends.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def export_and_validate_onnx(
    weights_path: Path,
    output_dir: Path,
    imgsz: tuple[int, int] = (640, 640),
    opset_version: int = 17,
    tolerance: float = 1e-3
) -> Path:
    """
    Exports a PyTorch YOLO model to ONNX, checks model graph validity,
    and runs a numerical parity check against the PyTorch reference.
    """
    print("=" * 65)
    print(f"[*] Starting ONNX Model Export: {weights_path.name}")
    print(f"[*] Target Opset Version: {opset_version} | Input Resolution: {imgsz}")
    print("=" * 65)

    if not weights_path.exists():
        raise FileNotFoundError(f"[X] Weights file not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_onnx_path = output_dir / f"{weights_path.stem}.onnx"

    # 1. Initialize PyTorch YOLO model
    model = YOLO(str(weights_path))

    # 2. Perform ONNX export via Ultralytics engine
    exported_path_str = model.export(
        format="onnx",
        imgsz=list(imgsz),
        dynamic=False,        # Static shapes yield highest edge execution speed
        simplify=True,        # Runs onnx-simplifier to fold constant nodes
        opset=opset_version,
    )

    exported_file = Path(exported_path_str)
    if exported_file.resolve() != target_onnx_path.resolve():
        shutil.copy2(exported_file, target_onnx_path)

    print(f"[+] ONNX model successfully generated at: {target_onnx_path}")

    # 3. Validate ONNX Graph Structure
    print("[*] Validating ONNX graph topology...")
    onnx_model = onnx.load(str(target_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX topology verification passed (Graph is structurally sound).")

    # 4. Numerical Parity Verification
    print("[*] Executing numerical parity test (PyTorch vs ONNX Runtime)...")
    
    # Generate deterministic dummy tensor (Batch=1, Channels=3, Height=640, Width=640)
    np.random.seed(42)
    dummy_input_np = np.random.uniform(0.0, 1.0, size=(1, 3, imgsz[0], imgsz[1])).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    # PyTorch Inference
    torch_model = model.model.eval()
    with torch.no_grad():
        torch_outputs = torch_model(dummy_input_torch)
        if isinstance(torch_outputs, (list, tuple)):
            torch_out = torch_outputs[0].cpu().numpy()
        else:
            torch_out = torch_outputs.cpu().numpy()

    # ONNX Runtime Inference
    ort_session = ort.InferenceSession(
        str(target_onnx_path),
        providers=["CPUExecutionProvider"]
    )
    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: dummy_input_np})
    onnx_out = ort_outputs[0]

    # Calculate error metrics
    max_abs_error = float(np.max(np.abs(torch_out - onnx_out)))
    mean_abs_error = float(np.mean(np.abs(torch_out - onnx_out)))

    print(f"[+] Output Shape (PyTorch) : {torch_out.shape}")
    print(f"[+] Output Shape (ONNX)    : {onnx_out.shape}")
    print(f"[+] Max Absolute Error     : {max_abs_error:.6e}")
    print(f"[+] Mean Absolute Error    : {mean_abs_error:.6e}")

    if max_abs_error < tolerance:
        print("[+] SUCCESS: ONNX export satisfies numerical parity tolerance!")
    else:
        print(f"[!] WARNING: Max absolute error ({max_abs_error:.6e}) exceeds tolerance ({tolerance})")

    return target_onnx_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    weights = project_root / "models" / "weights" / "best.pt"
    if not weights.exists():
        weights = project_root / "yolo11n.pt"

    onnx_out_dir = project_root / "models" / "onnx"
    export_and_validate_onnx(
        weights_path=weights,
        output_dir=onnx_out_dir,
        imgsz=(640, 640),
        opset_version=17
    )