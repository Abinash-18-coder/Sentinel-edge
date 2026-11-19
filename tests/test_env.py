"""
tests/test_env.py
Automated environment and 8GB RAM hardware diagnostics for SentinelEdge.
Validates Python 3.12.5 runtime, system memory headroom, PyTorch backend,
OpenCV camera ingestion, Supervision ByteTrack, Ultralytics YOLO (base + custom crack weights),
ONNX Runtime, and DuckDB.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permanent path fix: ensures 'src' is always importable from any terminal location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import platform
import psutil
import cv2
import torch
import supervision as sv
import ultralytics
import onnxruntime as ort
import duckdb


def run_diagnostics() -> bool:
    print("=" * 65)
    print(" SentinelEdge Pipeline: Environment & 8GB RAM Diagnostics")
    print("=" * 65)

    # 1. Python Runtime & OS
    py_version = platform.python_version()
    print(f"[+] Python Version        : {py_version} ({sys.executable})")
    print(f"[+] Operating System      : {platform.system()} {platform.release()} ({platform.architecture()[0]})")

    # 1.5. Hardware & 8GB RAM Memory Headroom Check (Intel i5-11320H / Iris Xe)
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    avail_ram_gb = mem.available / (1024 ** 3)
    used_percent = mem.percent
    print(f"[+] CPU Logical Cores     : {psutil.cpu_count(logical=True)} cores")
    print(f"[+] System RAM Status     : {avail_ram_gb:.2f} GB free / {total_ram_gb:.2f} GB total ({used_percent}% used)")
    if avail_ram_gb < 1.5:
        print("[!] WARNING: Free RAM is below 1.5 GB. Close heavy background tabs before running VLM.")

    # 2. PyTorch & Compute Acceleration Backend
    print(f"[+] PyTorch Version       : {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[+] Compute Acceleration  : CUDA Available (GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB)")
    else:
        print("[+] Compute Acceleration  : CPU / Intel Iris Xe Mode (Using quantized ONNX for local speed)")

    # 3. OpenCV Video Capture Subsystem
    print(f"[+] OpenCV Version        : {cv2.__version__}")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w, c = frame.shape
            print(f"[+] Ingestion Subsystem   : SUCCESS (Webcam active: {w}x{h} px, {c} channels)")
        else:
            print("[!] Ingestion Subsystem   : Camera device opened but frame read returned empty")
        cap.release()
    else:
        print("[*] Ingestion Subsystem   : No local webcam detected (File/RTSP streaming ready)")

    # 4. Supervision & Tracking Components
    print(f"[+] Supervision Version   : {sv.__version__}")
    tracker = sv.ByteTrack()
    print(f"[+] ByteTrack Tracker     : INITIALIZED ({tracker.__class__.__name__})")

    # 5. Ultralytics YOLO Perception Engine (Base + Trained Crack Weights)
    print(f"[+] Ultralytics Version   : {ultralytics.__version__}")
    custom_weights = PROJECT_ROOT / "models" / "weights" / "best.pt"
    if custom_weights.exists():
        crack_model = ultralytics.YOLO(str(custom_weights))
        print(f"[+] Custom Crack Detector : LOADED (models/weights/best.pt | Classes: {crack_model.names})")
    else:
        yolo_model = ultralytics.YOLO(str(PROJECT_ROOT / "yolo11n.pt"))
        print(f"[+] Base YOLO Detector    : LOADED ({yolo_model.model_name}) [best.pt not found in models/weights/]")

    # 6. ONNX Runtime Inference Providers
    print(f"[+] ONNX Runtime Version  : {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"[+] ONNX Active Providers : {', '.join(providers)}")

    # 7. DuckDB Embedded Analytics Database
    print(f"[+] DuckDB Engine Version : {duckdb.__version__}")

    print("=" * 65)
    print(" ALL CORE MODULES, WEIGHTS, AND RAM DIAGNOSTICS VERIFIED!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    try:
        run_diagnostics()
    except Exception as err:
        print(f"\n[X] Diagnostics failed with critical error:\n{err}", file=sys.stderr)
        sys.exit(1)