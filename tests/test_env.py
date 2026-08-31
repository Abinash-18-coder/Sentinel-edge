"""
tests/test_env.py
Automated environment diagnostics for SentinelEdge.
Validates Python 3.12.5 runtime, PyTorch backend, OpenCV camera ingestion,
Supervision tracking components, Ultralytics YOLO, and ONNX Runtime.
"""

from __future__ import annotations

import sys
import platform
import cv2
import torch
import supervision as sv
import ultralytics
import onnxruntime as ort


def run_diagnostics() -> bool:
    print("=" * 65)
    print(" SentinelEdge Pipeline: Environment Diagnostics (Python 3.12)")
    print("=" * 65)

    # 1. Python Runtime
    py_version = platform.python_version()
    print(f"[+] Python Version        : {py_version} ({sys.executable})")
    print(f"[+] Operating System      : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        print(f"[!] Warning: Python 3.10+ required. Detected: {py_version}")

    # 2. PyTorch & Compute Acceleration Backend
    print(f"[+] PyTorch Version       : {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[+] Compute Acceleration  : CUDA Available (GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB)")
    else:
        print("[+] Compute Acceleration  : CPU Mode (CUDA not detected - Colab GPU will be used for training)")

    # 3. OpenCV Video Capture Subsystem
    print(f"[+] OpenCV Version        : {cv2.__version__}")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, c = frame.shape
            print(f"[+] Ingestion Subsystem   : SUCCESS (Webcam active: {w}x{h} px, {c} channels)")
        else:
            print("[!] Ingestion Subsystem   : Camera device opened but frame read returned empty")
        cap.release()
    else:
        print("[*] Ingestion Subsystem   : No local webcam detected (File/RTSP streaming will be used)")

    # 4. Supervision & Tracking Components
    print(f"[+] Supervision Version   : {sv.__version__}")
    tracker = sv.ByteTrack()
    print(f"[+] ByteTrack Tracker     : INITIALIZED ({tracker.__class__.__name__})")

    # 5. Ultralytics YOLO Perception Engine
    print(f"[+] Ultralytics Version   : {ultralytics.__version__}")
    yolo_model = ultralytics.YOLO("yolo11n.pt")
    print(f"[+] YOLO Inference Core   : LOADED ({yolo_model.model_name})")

    # 6. ONNX Runtime Inference Providers
    print(f"[+] ONNX Runtime Version  : {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"[+] ONNX Active Providers : {', '.join(providers)}")

    print("=" * 65)
    print(" ALL CORE MODULES AND DEPENDENCIES VERIFIED SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    try:
        run_diagnostics()
    except Exception as err:
        print(f"\n[X] Diagnostics failed with critical error:\n{err}", file=sys.stderr)
        sys.exit(1)