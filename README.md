# SentinelEdge v2.0 — Real-Time Edge Vision & Grounded Multimodal Auditing Platform

![Python](https://img.shields.io/badge/Python-3.12.5-blue)
![YOLO26](https://img.shields.io/badge/Perception-Ultralytics%20YOLO26-0284c7)
![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%201.32%20INT8-10b981)
![DuckDB](https://img.shields.io/badge/Analytics-Embedded%20DuckDB%20WAL-f59e0b)
![FastAPI](https://img.shields.io/badge/Dashboard-FastAPI%20%2B%20WebSockets%20%2B%20Canvas-6366f1)
![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ed)

**SentinelEdge v2.0** is an end-to-end edge AI structural crack detection, appearance Re-ID tracking, and grounded multimodal auditing system engineered to run smoothly on constrained **8 GB RAM / CPU-only edge hardware** (Intel i5-11320H, Intel Iris Xe) at **$0 infrastructure cost**.

Built as the visual intelligence **Brain & Analyst** for autonomous structural inspection pipelines—including untethered wall-climbing inspection robots via wireless RTSP streams.

---

## System Architecture ("The Brain & The Analyst")

```text
  [ Webcam / MP4 / Wall-Climbing Robot RTSP Feed ]
                        │
                        ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ THE BRAIN: Real-Time Edge Perception & Spatial Logic (CPU)      │
 │ • Ultralytics YOLO26n -> Static INT8 ONNX Runtime 1.32 (4-core) │
 │ • ByteTrack + 576-dim Deterministic MobileNetV3 Re-ID Recovery  │
 │ • Polygon Danger Zones, Directional Tripwires & State Machine   │
 │   (IDLE -> WARNING -> CONFIRMED_ANOMALY -> COOLDOWN)            │
 └──────────────┬────────────────────────────────┬─────────────────┘
                │ (30 FPS Stream)                │ (Confirmed Breach)
                ▼                                ▼
 ┌──────────────────────────────┐  ┌───────────────────────────────┐
 │ LIVE OPERATIONS DASHBOARD    │  │ THE ANALYST & TELEMETRY       │
 │ • FastAPI + Binary WebSockets│  │ • JPEG Circular Ring Buffer   │
 │ • HTML5 Canvas (Zero-Leak)   │  │   (3s Pre + 3s Post MP4 Clip) │
 │ • Live Latency Sparkline     │  │ • Non-Blocking Async VLM Queue│
 │ • 8 GB RAM Headroom Monitor  │  │   (Qwen2.5-VL-3B / Edge CV)   │
 │ • One-Click CSV / PDF Export │  │ • Strict Pydantic JSON Schema │
 │ • Gaussian Dwell Heatmap UI  │  │ • Embedded DuckDB (WAL, 256MB)│
 └──────────────────────────────┘  └───────────────────────────────┘