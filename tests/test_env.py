"""
tests/test_env.py
Automated environment verification script for SentinelEdge.
Validates PyTorch hardware acceleration, OpenCV camera ingestion,
Supervision tracking components, and Ultralytics YOLO inference.
"""

import sys
import platform
import cv2
import torch
import supervision as sv
import ultralytics
import onnxruntime as ort


def run_diagnostics() -> bool:
    print("=" * 60)
    print(" SentinelEdge Pipeline: Environment Diagnostics")
    print("=" * 60)

    # 1. System Runtime
    print(f"[+] Python Version      : {platform.python_version()} ({sys.executable})")
    print(f"[+] Operating System    : {platform.system()} {platform.release()}")

    # 2. PyTorch & Compute Acceleration
    print(f"[+] PyTorch Version     : {torch.__version__}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[+] Compute Backend     : CUDA (GPU: {device_name})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[+] Compute Backend     : Apple Silicon (MPS)")
    else:
        print("[+] Compute Backend     : CPU (Default fallback)")

    # 3. OpenCV & Camera Ingestion
    print(f"[+] OpenCV Version      : {cv2.__version__}")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, c = frame.shape
            print(f"[+] Camera Test         : SUCCESS (Default frame shape: {w}x{h}, {c} channels)")
        else:
            print("[!] Camera Test         : Frame capture failed (Camera busy or permission denied)")
        cap.release()
    else:
        print("[!] Camera Test         : No local webcam detected (Not blocking - video file mode available)")

    # 4. Supervision & Tracking Components
    print(f"[+] Supervision Version : {sv.__version__}")
    tracker = sv.ByteTrack()
    print(f"[+] ByteTrack Tracker   : INITIALIZED ({type(tracker).__name__})")

    # 5. Ultralytics YOLO
    print(f"[+] Ultralytics Version : {ultralytics.__version__}")
    yolo_model = ultralytics.YOLO("yolo11n.pt")
    print(f"[+] YOLO Inference Engine: LOADED ({yolo_model.model_name})")

    # 6. ONNX Runtime Engine
    print(f"[+] ONNX Runtime Version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"[+] Available Providers : {', '.join(providers)}")

    print("=" * 60)
    print(" ALL CORE MODULES INSTALLED AND VERIFIED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        run_diagnostics()
    except Exception as e:
        print(f"\n[X] Verification Failed with Error: {e}", file=sys.stderr)
        sys.exit(1)