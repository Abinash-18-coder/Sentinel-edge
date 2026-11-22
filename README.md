# SentinelEdge v2.1 — Real-Time Edge Vision & Grounded Multimodal Auditing Platform

![Python](https://img.shields.io/badge/Python-3.12.5-blue)
![YOLO26](https://img.shields.io/badge/Perception-Ultralytics%20YOLO26-0284c7)
![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%201.32%20FP32-10b981)
![DuckDB](https://img.shields.io/badge/Analytics-Embedded%20DuckDB%20WAL-f59e0b)
![FastAPI](https://img.shields.io/badge/Dashboard-FastAPI%20%2B%20WebSockets%20%2B%20Canvas-6366f1)
![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ed)

**SentinelEdge v2.1** is an end-to-end edge AI structural crack detection, appearance Re-ID tracking, and grounded multimodal auditing system engineered to run smoothly on constrained **8 GB RAM / CPU-only edge hardware** (Intel i5-11320H, Intel Iris Xe) at **$0 infrastructure cost**.

Built as the visual intelligence **Brain & Analyst** for autonomous structural inspection pipelines—including live phone camera scanning, ESP32-CAM modules, and untethered wall-climbing inspection robots via wireless RTSP streams.

---

## System Architecture ("The Brain & The Analyst")

```text
[ Laptop Webcam / Phone Wi-Fi MJPEG / ESP32-CAM / Robot RTSP / MP4 ]
                        |
                        v
+-------------------------------------------------------------------+
| UNIVERSAL INGESTION (src/perception/camera_stream.py)             |
| - Threaded latest-frame grabber (zero Wi-Fi/RTSP buffer lag)      |
| - Forced TCP RTSP transport + automatic stream reconnection       |
| - Aspect-safe 640px downscaling + live browser hot-swap           |
+-------------------------------------------------------------------+
                        |
                        v
+-------------------------------------------------------------------+
| THE BRAIN: Real-Time Edge Perception & Spatial Logic (CPU)        |
| - Ultralytics YOLO26 -> FP32 ONNX Runtime 1.32 (best.onnx)        |
| - ByteTrack (supervision 0.25.1) + 576-dim MobileNetV3 Re-ID      |
| - Full-Frame Polygon Inspection Zones, Tripwires & State Machine  |
|   (IDLE -> WARNING -> CONFIRMED_ANOMALY -> COOLDOWN)              |
+-------------------------------------------------------------------+
              |                                    |
     (Live Stream + WS)                   (Confirmed Breach)
              v                                    v
+-----------------------------+    +--------------------------------+
| LIVE OPERATIONS DASHBOARD   |    | THE ANALYST & TELEMETRY        |
| - FastAPI + Binary WS       |    | - JPEG Circular Ring Buffer    |
| - HTML5 Canvas (Zero-Leak)  |    |   (3s Pre + 3s Post MP4 Clip)  |
| - Live Camera/Model/Conf UI |    | - Non-Blocking Async VLM Queue |
| - Real-Time Latency Chart   |    |   (Black-Hat Crack Morph /     |
| - 8 GB RAM Headroom Monitor |    |    Cloud Qwen2.5-VL-3B)        |
| - One-Click CSV / PDF Export|    | - Crack Length & Width Est.    |
| - Gaussian Dwell Heatmap UI |    | - Embedded DuckDB (WAL, 256MB) |
+-----------------------------+    +--------------------------------+
```

---

## Verified Benchmarks (Intel i5-11320H, 8 GB RAM, CPU-Only)

| Model Artifact | Hit Rate | Avg Confidence | Avg Latency | Status |
| :--- | :--- | :--- | :--- | :--- |
| **`best.onnx` (FP32)** | `100.0%` | **`0.833`** | **`47.1 ms`** | **Default (Fastest + Sharpest)** |
| **`best_fp16.onnx`** | `100.0%` | `0.843` | `54.6 ms` | Live-selectable in Dashboard |
| **`best.pt` (PyTorch)** | `100.0%` | `0.829` | `74.6 ms` | Training / Reference Artifact |
| **`best_int8.onnx`** | `100.0%` | `0.476` | `55.4 ms` | Live-selectable in Dashboard |

* **Edge Crack Morphology Audit:** CLAHE + Black-Hat filtering (`cv2.MORPH_BLACKHAT`) + `cv2.distanceTransform` width/length estimation (ignores bright windows and architectural frames).
* **Core Runtime Stack:** Python `3.12.5`, PyTorch `2.13.0+cpu`, OpenCV `4.11.0`, Ultralytics `8.4.215`, ONNX Runtime `1.32.0`, Supervision `0.25.1`, DuckDB `2.0.0`.

---

## Quick Start

### 1. On-The-Go Execution (Windows PowerShell — All Cameras Supported Live)

```powershell
cd C:\Users\Abinash\Desktop\PROJECT\Sentinel-Edge
.\venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
Remove-Item Env:\SENTINEL_TEST_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\SENTINEL_SOURCE -ErrorAction SilentlyContinue
python src/api/server.py
```

Open **`http://127.0.0.1:8000`** in Chrome or Edge. Use the top dashboard toolbar to hot-swap between **Laptop Webcam (`0`)**, **Phone Wi-Fi (`http://<PHONE_IP>:8080/video`)**, **USB/DroidCam (`1`)**, **ESP32-CAM (`http://<IP>:81/stream`)**, **Robot RTSP (`rtsp://<IP>:8554/cam`)**, or local **`.mp4`** videos, switch active ONNX/PyTorch models, and tune the confidence slider live without restarting Python. Press **`Ctrl + C` once** in PowerShell to checkpoint DuckDB and exit cleanly.

### 2. Direct Camera, Cloud VLM & Millimeter Calibration Overrides (Optional)

```powershell
# Phone Live Camera (Android IP Webcam app on same Wi-Fi or Phone Hotspot)
$env:SENTINEL_SOURCE = "[http://192.168.43.1:8080/video](http://192.168.43.1:8080/video)"

# Wall-Climbing Inspection Robot (Wireless RTSP or ESP32-CAM)
$env:SENTINEL_SOURCE = "rtsp://192.168.1.50:8554/cam"

# Optional: Connect Free Colab T4 Qwen2.5-VL-3B Endpoint (0 MB local RAM)
$env:VLM_API_URL = "https://<your-tunnel>[.trycloudflare.com/v1/chat/completions](https://.trycloudflare.com/v1/chat/completions)"

# Optional: Calibrate Real-World Millimeter Crack Width Reporting
$env:SENTINEL_PIXELS_PER_MM = "5.0"

python src/api/server.py
```

### 3. Low-RAM Standalone Desktop HUD (No Browser)

```powershell
python src/main_pipeline.py
```

### 4. Docker Container Deployment (WSL2-Safe, 1.5 GB RAM / 2 CPU Cap)

```powershell
$env:SENTINEL_TEST_MODE = "1"
docker compose up -d --build
```

### 5. Automated Verification & Accuracy Benchmark

```powershell
$env:PYTHONPATH = "."
pytest tests/ -v
python src/perception/compare_accuracy.py
```

---

## Project Structure

```text
Sentinel-Edge/
├── configs/
│   ├── data.yaml                  # Class mapping ({0: 'crack'})
│   └── zones.yaml                 # Full-frame inspection zone & tripwire config
├── models/
│   ├── onnx/                      # best.onnx (FP32 default), best_fp16.onnx, best_int8.onnx
│   └── weights/                   # best.pt
├── src/
│   ├── api/                       # FastAPI WebSocket server + CSV/PDF exporter
│   ├── data/                      # 256MB-capped DuckDB WAL DB + Gaussian heatmap engine
│   ├── perception/                # UniversalCameraStream, Re-ID tracker, ONNX export & benchmark
│   ├── rules/                     # Spatial state machine + JPEG circular MP4 incident buffer
│   ├── vlm/                       # Async auditor, Black-Hat crack morphology, grounding renderer
│   └── main_pipeline.py           # Low-RAM OpenCV Desktop HUD
├── tests/                         # Full automated pytest & E2E regression suite
├── ui/
│   └── index.html                 # Zero-leak HTML5 Canvas operations dashboard
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```