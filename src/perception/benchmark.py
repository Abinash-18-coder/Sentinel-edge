"""
src/perception/benchmark.py
Comparative latency, throughput (FPS), and memory profiler across PyTorch and ONNX runtimes.
Generates an interview-ready Markdown benchmark table.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import time
import os
import psutil
from pathlib import Path
from typing import NamedTuple
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO


class BenchmarkResult(NamedTuple):
    model_name: str
    engine: str
    precision: str
    file_size_mb: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float
    ram_usage_mb: float


def benchmark_pytorch(weights_path: Path, imgsz: tuple[int, int] = (640, 640), iterations: int = 200) -> BenchmarkResult:
    print(f"[*] Benchmarking PyTorch Model ({weights_path.name})...")
    model = YOLO(str(weights_path)).model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1], dtype=torch.float32)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    # Warmup
    for _ in range(30):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies: list[float] = []
    process = psutil.Process(os.getpid())

    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies_np = np.array(latencies)
    mean_lat = float(np.mean(latencies_np))
    
    return BenchmarkResult(
        model_name=weights_path.name,
        engine="PyTorch (TorchScript)",
        precision="FP32",
        file_size_mb=round(weights_path.stat().st_size / (1024 * 1024), 2),
        mean_latency_ms=round(mean_lat, 2),
        p95_latency_ms=round(float(np.percentile(latencies_np, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies_np, 99)), 2),
        fps=round(1000.0 / max(mean_lat, 1e-4), 1),
        ram_usage_mb=round(process.memory_info().rss / (1024 * 1024), 2)
    )


def benchmark_onnx(onnx_path: Path, precision: str, imgsz: tuple[int, int] = (640, 640), iterations: int = 200) -> BenchmarkResult:
    print(f"[*] Benchmarking ONNX Model ({onnx_path.name} | {precision})...")
    
    # Configure low-latency execution session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    input_dtype = np.float32
    dummy_input = np.random.randn(1, 3, imgsz[0], imgsz[1]).astype(input_dtype)

    # Warmup
    for _ in range(30):
        _ = session.run(None, {input_name: dummy_input})

    latencies: list[float] = []
    process = psutil.Process(os.getpid())

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies_np = np.array(latencies)
    mean_lat = float(np.mean(latencies_np))

    return BenchmarkResult(
        model_name=onnx_path.name,
        engine="ONNX Runtime (CPU)",
        precision=precision,
        file_size_mb=round(onnx_path.stat().st_size / (1024 * 1024), 2),
        mean_latency_ms=round(mean_lat, 2),
        p95_latency_ms=round(float(np.percentile(latencies_np, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies_np, 99)), 2),
        fps=round(1000.0 / max(mean_lat, 1e-4), 1),
        ram_usage_mb=round(process.memory_info().rss / (1024 * 1024), 2)
    )


def print_markdown_table(results: list[BenchmarkResult]) -> None:
    print("\n" + "=" * 90)
    print(" SENTINEL-EDGE PERFORMANCE BENCHMARK SUITE")
    print("=" * 90)
    
    header = "| Model Checkpoint | Engine | Precision | Size (MB) | Mean (ms) | p95 (ms) | p99 (ms) | FPS | Memory (MB) |"
    divider = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(divider)
    for r in results:
        print(f"| `{r.model_name}` | {r.engine} | **{r.precision}** | {r.file_size_mb} MB | {r.mean_latency_ms} ms | {r.p95_latency_ms} ms | {r.p99_latency_ms} ms | **{r.fps}** | {r.ram_usage_mb} MB |")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    pt_file = root / "models" / "weights" / "best.pt"
    if not pt_file.exists():
        pt_file = root / "yolo11n.pt"

    onnx_fp32 = root / "models" / "onnx" / f"{pt_file.stem}.onnx"
    onnx_fp16 = root / "models" / "onnx" / f"{pt_file.stem}_fp16.onnx"
    onnx_int8 = root / "models" / "onnx" / f"{pt_file.stem}_int8.onnx"

    benchmark_runs: list[BenchmarkResult] = []

    if pt_file.exists():
        benchmark_runs.append(benchmark_pytorch(pt_file))
    if onnx_fp32.exists():
        benchmark_runs.append(benchmark_onnx(onnx_fp32, "FP32"))
    if onnx_fp16.exists():
        benchmark_runs.append(benchmark_onnx(onnx_fp16, "FP16"))
    if onnx_int8.exists():
        benchmark_runs.append(benchmark_onnx(onnx_int8, "INT8 (Static)"))

    print_markdown_table(benchmark_runs)