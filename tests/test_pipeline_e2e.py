import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import shutil
import time
import cv2
import numpy as np
import onnxruntime as ort
import psutil
import supervision as sv

from src.api.exporter import IncidentExporter
from src.data.analytics_db import SentinelAnalyticsDB
from src.data.heatmap_engine import SpatialHeatmapEngine
from src.perception.reid_tracker import DeepReIDTracker
from src.rules.incident_buffer import IncidentVideoBuffer
from src.rules.spatial_engine import SpatialRulesEngine
from src.vlm.async_auditor import AsyncIncidentAuditor


def create_synthetic_crack_frame(cx: int, cy: int) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(2026)
    frame = rng.integers(180, 205, (480, 640, 3), dtype=np.uint8)
    x1, y1 = max(10, cx - 55), max(10, cy - 80)
    x2, y2 = min(630, cx + 55), min(470, cy + 80)

    pts = np.array([
        [x1 + 20, y1 + 10],
        [cx - 8, y1 + 55],
        [cx + 12, cy + 15],
        [x2 - 22, y2 - 12]
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=False, color=(15, 15, 15), thickness=5)
    cv2.line(frame, (cx - 8, y1 + 55), (cx + 22, y1 + 42), (30, 30, 30), 2)
    return frame, [float(x1), float(y1), float(x2), float(y2)]


def test_stage_1_onnx_runtime_artifact() -> None:
    print("[*] Stage 1: Verifying Quantized ONNX Edge Runtime & Latency...")
    candidates = [
        PROJECT_ROOT / "models" / "onnx" / "best_int8.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best.onnx"
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        print("[!] Skipping ONNX latency check (no local .onnx weights in models/onnx/).")
        return

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

    session.run(None, {input_name: dummy})
    t0 = time.perf_counter()
    out = session.run(None, {input_name: dummy})[0]
    ms = (time.perf_counter() - t0) * 1000.0

    assert out.size > 0
    print(f"[+] Stage 1 Passed: {model_path.name} inferred cleanly in {ms:.2f} ms (output shape {out.shape}).")


def test_stage_2_full_vision_to_analytics_pipeline() -> None:
    print("[*] Stage 2: Running Full E2E Pipeline (Re-ID -> Rules -> Buffer -> VLM -> DuckDB -> Heatmap -> Exports)...")
    tmp_root = PROJECT_ROOT / "data" / "raw" / "incidents" / "_e2e_v2_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    incidents_dir = tmp_root / "incidents"
    exports_dir = tmp_root / "exports"
    db_path = tmp_root / "e2e_analytics.duckdb"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    zones_yaml = PROJECT_ROOT / "configs" / "zones.yaml"
    class_map = {0: "crack"}

    tracker = DeepReIDTracker(frame_rate=15, reid_match_threshold=0.35)
    spatial_engine = SpatialRulesEngine(config_yaml=zones_yaml, frame_resolution_wh=(640, 480))
    incident_buffer = IncidentVideoBuffer(
        output_dir=incidents_dir,
        pre_incident_seconds=1.0,
        post_incident_seconds=1.0,
        fps=15,
        resolution_wh=(640, 480)
    )
    analytics_db = SentinelAnalyticsDB(db_path=db_path, memory_limit="256MB", threads=2, track_batch_size=10)
    heatmap_engine = SpatialHeatmapEngine(resolution_wh=(640, 480))

    async_auditor = AsyncIncidentAuditor(
        max_queue_size=8,
        track_debounce_sec=8.0,
        vlm_mode="auto",
        on_report_callback=lambda rep: analytics_db.log_vlm_audit(rep)
    )

    sim_start = 10000.0
    triggered_incident_ids: list[str] = []

    for frame_idx in range(45):
        sim_time = sim_start + (frame_idx * 0.12)
        cx = 165 + (frame_idx % 4)
        cy = 250 + (frame_idx % 3)
        frame, box = create_synthetic_crack_frame(cx, cy)

        raw_dets = sv.Detections(
            xyxy=np.array([box], dtype=np.float32),
            confidence=np.array([0.89], dtype=np.float32),
            class_id=np.array([0], dtype=int)
        )

        tracked_dets, entities = tracker.update_from_detections(
            frame_bgr=frame,
            raw_detections=raw_dets,
            class_name_map=class_map,
            current_time=sim_time
        )

        for ent in entities:
            heatmap_engine.add_point(ent.center_xy[0], ent.center_xy[1], 640, 480)
            analytics_db.buffer_track_point(
                track_id=ent.track_id,
                class_name=ent.class_name,
                cx=ent.center_xy[0],
                cy=ent.center_xy[1],
                frame_w=640,
                frame_h=480,
                confidence=ent.confidence,
                camera_id="e2e_cam",
                timestamp=sim_time
            )

        events = spatial_engine.evaluate(
            detections=tracked_dets,
            class_name_map=class_map,
            current_time=sim_time
        )

        for ev in events:
            mp4_path = incident_buffer.trigger_incident_capture(ev)
            assert mp4_path is not None
            inc_id = mp4_path.stem
            triggered_incident_ids.append(inc_id)

            analytics_db.log_zone_breach(inc_id, ev)
            vlm_json_path = incidents_dir / f"{inc_id}_vlm_report.json"
            submitted = async_auditor.submit_task(
                incident_id=inc_id,
                zone_id=ev.zone_id,
                track_id=ev.track_id,
                entity_class=ev.class_name,
                frame_bgr=frame,
                output_json_path=vlm_json_path,
                hint_bbox_xyxy=ev.bbox_xyxy,
                current_time=sim_time
            )
            assert submitted is True

        incident_buffer.push_frame(frame)

    incident_buffer.flush_and_shutdown(timeout=10.0)
    async_auditor.wait_all_completed()
    async_auditor.shutdown(timeout=10.0)

    assert len(triggered_incident_ids) == 1, f"Expected 1 debounced breach, got {len(triggered_incident_ids)}"
    inc_id = triggered_incident_ids[0]

    mp4_file = incidents_dir / f"{inc_id}.mp4"
    meta_file = incidents_dir / f"{inc_id}.json"
    vlm_file = incidents_dir / f"{inc_id}_vlm_report.json"
    grounded_file = incidents_dir / f"{inc_id}_vlm_report.grounded.jpg"

    assert mp4_file.exists() and meta_file.exists()
    assert vlm_file.exists() and grounded_file.exists()

    summary = analytics_db.get_summary_metrics()
    assert summary["total_breaches"] == 1
    assert summary["total_vlm_audits"] == 1
    assert summary["total_track_points"] >= 40
    analytics_db.close()

    heatmap_jpg = heatmap_engine.render_jpeg()
    assert heatmap_jpg[:2] == b"\xff\xd8"

    exporter = IncidentExporter(incidents_dir)
    csv_out = exporter.export_csv(exports_dir / "e2e_audit.csv")
    pdf_out = exporter.export_pdf(exports_dir / "e2e_audit.pdf")
    assert csv_out.exists() and inc_id in csv_out.read_text(encoding="utf-8")
    assert pdf_out.exists() and pdf_out.read_bytes().startswith(b"%PDF")

    shutil.rmtree(tmp_root)
    print(f"[+] Stage 2 Passed: Full E2E breach ({inc_id}) verified across all 7 subsystems and cleaned up.")


def test_stage_3_fastapi_full_contract() -> None:
    print("[*] Stage 3: Verifying Live FastAPI Server Contract (REST + WebSockets + Analytics)...")
    os.environ["SENTINEL_TEST_MODE"] = "1"
    from fastapi.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as client:
        time.sleep(0.3)

        assert client.get("/").status_code == 200
        assert client.get("/favicon.ico").status_code == 204

        health = client.get("/api/health").json()
        assert health["status"] == "online"

        summary = client.get("/api/analytics/summary").json()
        assert "total_track_points" in summary

        heatmap = client.get("/api/analytics/heatmap")
        assert heatmap.status_code == 200 and heatmap.content[:2] == b"\xff\xd8"

        with client.websocket_connect("/ws/stream") as ws:
            jpg = ws.receive_bytes()
            tel = json.loads(ws.receive_text())
            assert jpg[:2] == b"\xff\xd8"
            assert "fps" in tel and "ram_free_gb" in tel

    tmp_db = PROJECT_ROOT / "data" / "analytics" / "test_analytics_tmp.duckdb"
    for candidate in (tmp_db, Path(f"{tmp_db}.wal")):
        if candidate.exists():
            candidate.unlink()

    mem = psutil.virtual_memory()
    print(
        f"[+] Stage 3 Passed: FastAPI contract verified | "
        f"System RAM Free: {mem.available / (1024 ** 3):.2f} GB / {mem.total / (1024 ** 3):.2f} GB"
    )


if __name__ == "__main__":
    print("=" * 65)
    print(" SentinelEdge v2.0: Master End-to-End Regression Suite")
    print("=" * 65)
    test_stage_1_onnx_runtime_artifact()
    test_stage_2_full_vision_to_analytics_pipeline()
    test_stage_3_fastapi_full_contract()
    print("=" * 65)
    print(" ALL SENTINEL-EDGE v2.0 MASTER E2E REGRESSION TESTS PASSED!")
    print("=" * 65)