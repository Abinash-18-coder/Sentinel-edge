import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import gc
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
import psutil
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.exporter import IncidentExporter
from src.data.analytics_db import SentinelAnalyticsDB
from src.data.heatmap_engine import SpatialHeatmapEngine
from src.vlm.schemas import IncidentAuditReport


class PipelineState:
    def __init__(self) -> None:
        self.is_running: bool = False
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.active_tracks: int = 0
        self.total_anomalies: int = 0
        self.buffer_ram_mb: float = 0.0
        self.model_name: str = "initializing"
        self.source_label: str = "webcam:0"
        self.latest_jpeg: bytes | None = None
        self.latest_raw_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.connected_stream_clients: list[WebSocket] = []
        self.connected_incident_clients: list[WebSocket] = []
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.analytics_db: SentinelAnalyticsDB | None = None
        self.heatmap_engine: SpatialHeatmapEngine | None = None


state = PipelineState()
INCIDENTS_DIR = PROJECT_ROOT / "data" / "raw" / "incidents"
ANALYTICS_DIR = PROJECT_ROOT / "data" / "analytics"
UI_DIR = PROJECT_ROOT / "ui"


def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {0: "crack"}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {int(k): str(v) for k, v in data.get("names", {0: "crack"}).items()}


def resolve_model_path() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "onnx" / "best_int8.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best.onnx",
        PROJECT_ROOT / "models" / "weights" / "best.pt",
        PROJECT_ROOT / "yolo26n.pt",
        PROJECT_ROOT / "yolo11n.pt"
    ]
    for path in candidates:
        if path.exists():
            return path
    return PROJECT_ROOT / "yolo26n.pt"


def resolve_video_source() -> int | str:
    raw = os.getenv("SENTINEL_SOURCE", "0").strip()
    if raw.isdigit():
        return int(raw)
    return raw


def get_system_ram_stats() -> tuple[float, float]:
    mem = psutil.virtual_memory()
    return round(float(mem.percent), 1), round(mem.available / (1024 ** 3), 2)


async def broadcast_incident_message(payload: dict[str, Any]) -> None:
    text = json.dumps(payload)
    for client in list(state.connected_incident_clients):
        try:
            await client.send_text(text)
        except Exception:
            if client in state.connected_incident_clients:
                state.connected_incident_clients.remove(client)


def schedule_incident_broadcast(payload: dict[str, Any]) -> None:
    loop = state.event_loop
    if loop is None or loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast_incident_message(payload), loop)
    except RuntimeError:
        pass


class VisionPipelineWorker:
    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        state.is_running = True
        target = self._run_test_loop if self.test_mode else self._run_live_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        state.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=8.0)

    def _encode_jpeg(self, frame_bgr: np.ndarray, quality: int = 78) -> bytes | None:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else None

    def _run_test_loop(self) -> None:
        state.model_name = "test-synthetic-mode"
        state.source_label = "synthetic:640x480"
        frame_idx = 0

        while state.is_running:
            t0 = time.perf_counter()
            frame = np.full((480, 640, 3), 24, dtype=np.uint8)
            cv2.putText(
                frame,
                f"SENTINEL-EDGE v2.0 TEST STREAM | FRAME {frame_idx}",
                (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (56, 189, 248),
                2,
                cv2.LINE_AA
            )
            if state.heatmap_engine is not None and frame_idx % 5 == 0:
                cx = 160.0 + float((frame_idx * 7) % 180)
                cy = 200.0 + float((frame_idx * 5) % 120)
                state.heatmap_engine.add_point(cx, cy, 640, 480)
                if state.analytics_db is not None:
                    state.analytics_db.buffer_track_point(
                        track_id=1,
                        class_name="crack",
                        cx=cx,
                        cy=cy,
                        frame_w=640,
                        frame_h=480,
                        confidence=0.88,
                        camera_id="synthetic_0"
                    )

            jpeg = self._encode_jpeg(frame, quality=75)
            if jpeg is not None:
                with state.frame_lock:
                    state.latest_jpeg = jpeg
                    state.latest_raw_frame = frame
            state.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            state.fps = 30.0
            frame_idx += 1
            time.sleep(0.033)

    def _on_vlm_report(self, report: IncidentAuditReport) -> None:
        if state.analytics_db is not None:
            try:
                state.analytics_db.log_vlm_audit(report)
            except Exception as err:
                print(f"[!] [DUCKDB] Failed logging VLM audit {report.incident_id}: {err}")

        grounded_name = f"{report.incident_id}_vlm_report.grounded.jpg"
        schedule_incident_broadcast({
            "type": "vlm_update",
            "incident_id": report.incident_id,
            "severity": report.severity.value,
            "vlm_summary": report.vlm_scene_description,
            "action": report.recommended_action,
            "audit_backend": report.audit_backend,
            "audit_latency_ms": report.audit_latency_ms,
            "grounded_image": grounded_name
        })

    def _run_live_loop(self) -> None:
        from ultralytics import YOLO
        from src.perception.reid_tracker import DeepReIDTracker
        from src.rules.incident_buffer import IncidentVideoBuffer
        from src.rules.spatial_engine import SpatialRulesEngine
        from src.vlm.async_auditor import AsyncIncidentAuditor

        chosen_weights = resolve_model_path()
        state.model_name = chosen_weights.name
        print(f"[*] [VISION WORKER] Loading perception engine: {chosen_weights}")

        model = YOLO(str(chosen_weights), task="detect")
        class_map = load_class_names(PROJECT_ROOT / "configs" / "data.yaml")
        zones_cfg = PROJECT_ROOT / "configs" / "zones.yaml"

        source = resolve_video_source()
        state.source_label = f"cam:{source}" if isinstance(source, int) else Path(source).name
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            fallback_vid = PROJECT_ROOT / "data" / "raw" / "test_feed.mp4"
            if fallback_vid.exists():
                print(f"[!] Primary source '{source}' unavailable. Switching to {fallback_vid.name}")
                cap = cv2.VideoCapture(str(fallback_vid))
                state.source_label = fallback_vid.name
            else:
                print(f"[!] Primary source '{source}' unavailable. Entering standby frame mode.")
                state.source_label = "standby"

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, init_frame = cap.read()
        else:
            ret, init_frame = False, None

        if not ret or init_frame is None:
            init_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = init_frame.shape[:2]
        frame_wh = (w, h)

        tracker = DeepReIDTracker(frame_rate=30, reid_match_threshold=0.35)
        spatial_engine = SpatialRulesEngine(config_yaml=zones_cfg, frame_resolution_wh=frame_wh)
        incident_buffer = IncidentVideoBuffer(
            output_dir=INCIDENTS_DIR,
            pre_incident_seconds=3.0,
            post_incident_seconds=3.0,
            fps=30,
            resolution_wh=frame_wh
        )
        async_auditor = AsyncIncidentAuditor(
            max_queue_size=12,
            track_debounce_sec=8.0,
            vlm_mode="auto",
            on_report_callback=self._on_vlm_report
        )

        prev_time = time.perf_counter()
        frame_counter = 0
        print(f"[+] [VISION WORKER] Streaming live at {w}x{h} using {state.model_name}")

        try:
            while state.is_running:
                loop_start = time.perf_counter()

                if cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if not isinstance(source, int):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                else:
                    frame = np.full((480, 640, 3), 20, dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "NO CAMERA DETECTED - STANDBY MODE",
                        (110, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (148, 163, 184),
                        2,
                        cv2.LINE_AA
                    )
                    time.sleep(0.05)

                frame_counter += 1
                results = model.predict(frame, imgsz=640, conf=0.30, verbose=False)[0]
                tracked_detections, entities = tracker.update(
                    frame_bgr=frame,
                    ultralytics_results=results,
                    class_name_map=class_map
                )
                state.active_tracks = len(entities)

                if entities and frame_counter % 3 == 0:
                    fh, fw = frame.shape[:2]
                    for ent in entities:
                        cx, cy = ent.center_xy
                        if state.heatmap_engine is not None:
                            state.heatmap_engine.add_point(cx, cy, fw, fh)
                        if state.analytics_db is not None:
                            state.analytics_db.buffer_track_point(
                                track_id=ent.track_id,
                                class_name=ent.class_name,
                                cx=cx,
                                cy=cy,
                                frame_w=fw,
                                frame_h=fh,
                                confidence=ent.confidence,
                                camera_id=state.source_label
                            )

                events = spatial_engine.evaluate(
                    detections=tracked_detections,
                    class_name_map=class_map
                )

                for ev in events:
                    state.total_anomalies += 1
                    mp4_path = incident_buffer.trigger_incident_capture(ev)
                    if mp4_path is not None:
                        incident_id = mp4_path.stem
                    else:
                        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ev.timestamp))
                        incident_id = f"incident_{ts_str}_t{ev.track_id}"

                    if state.analytics_db is not None:
                        try:
                            state.analytics_db.log_zone_breach(incident_id, ev)
                        except Exception as err:
                            print(f"[!] [DUCKDB] Failed logging breach {incident_id}: {err}")

                    vlm_json_path = INCIDENTS_DIR / f"{incident_id}_vlm_report.json"
                    async_auditor.submit_task(
                        incident_id=incident_id,
                        zone_id=ev.zone_id,
                        track_id=ev.track_id,
                        entity_class=ev.class_name,
                        frame_bgr=frame,
                        output_json_path=vlm_json_path,
                        hint_bbox_xyxy=ev.bbox_xyxy
                    )

                    schedule_incident_broadcast({
                        "type": "incident_created",
                        "incident_id": incident_id,
                        "zone_name": ev.zone_name,
                        "rule_type": ev.rule_type,
                        "track_id": ev.track_id,
                        "class_name": ev.class_name,
                        "dwell_time_seconds": round(ev.dwell_time, 2),
                        "readable_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.timestamp)),
                        "severity": "PENDING",
                        "vlm_summary": "Running asynchronous VLM audit...",
                        "action": "Awaiting audit"
                    })

                incident_buffer.push_frame(frame)
                state.buffer_ram_mb = round(incident_buffer.estimate_buffer_ram_mb(), 2)

                annotated = spatial_engine.render_overlays(frame)
                annotated = tracker.annotate_traces(annotated, tracked_detections)

                for ent in entities:
                    bx1, by1, bx2, by2 = map(int, ent.bbox_xyxy)
                    box_color = (56, 189, 248) if ent.recovered else (34, 197, 94)
                    suffix = " [RE-ID]" if ent.recovered else ""
                    label = f"#{ent.track_id} {ent.class_name} {ent.confidence:.2f}{suffix}"
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)
                    cv2.putText(
                        annotated,
                        label,
                        (bx1, max(by1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                        cv2.LINE_AA
                    )

                jpeg = self._encode_jpeg(annotated, quality=78)
                if jpeg is not None:
                    with state.frame_lock:
                        state.latest_jpeg = jpeg
                        state.latest_raw_frame = frame

                now = time.perf_counter()
                state.latency_ms = round((now - loop_start) * 1000.0, 2)
                state.fps = round(1.0 / max(now - prev_time, 1e-5), 1)
                prev_time = now

                time.sleep(0.001)
        finally:
            if cap.isOpened():
                cap.release()
            incident_buffer.flush_and_shutdown(timeout=8.0)
            async_auditor.shutdown(timeout=8.0)
            del model
            gc.collect()
            print("[+] [VISION WORKER] Cleanly stopped camera, buffer, and VLM threads.")


worker: VisionPipelineWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker
    state.event_loop = asyncio.get_running_loop()
    test_mode = os.getenv("SENTINEL_TEST_MODE", "0") == "1"

    db_file = (
        ANALYTICS_DIR / "test_analytics_tmp.duckdb"
        if test_mode
        else ANALYTICS_DIR / "sentinel_analytics.duckdb"
    )
    state.analytics_db = SentinelAnalyticsDB(db_path=db_file, memory_limit="256MB", threads=2)
    state.heatmap_engine = SpatialHeatmapEngine(resolution_wh=(640, 480))

    if not test_mode:
        hist_pts = state.analytics_db.fetch_recent_track_points(max_points=3000)
        if hist_pts:
            state.heatmap_engine.add_points_batch(hist_pts)
            print(f"[+] Hydrated spatial heatmap with {len(hist_pts)} historical DuckDB points.")

    worker = VisionPipelineWorker(test_mode=test_mode)
    worker.start()
    yield
    if worker is not None:
        worker.stop()
    if state.analytics_db is not None:
        state.analytics_db.close()
        state.analytics_db = None


app = FastAPI(title="SentinelEdge v2.0 Operations Server", lifespan=lifespan)

UI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    index_file = UI_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI file not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    ram_percent, ram_free_gb = get_system_ram_stats()
    return {
        "status": "online" if state.is_running else "stopped",
        "model_name": state.model_name,
        "source": state.source_label,
        "fps": state.fps,
        "latency_ms": state.latency_ms,
        "active_tracks": state.active_tracks,
        "total_anomalies": state.total_anomalies,
        "buffer_ram_mb": state.buffer_ram_mb,
        "ram_percent": ram_percent,
        "ram_free_gb": ram_free_gb
    }


@app.get("/api/incidents")
async def list_incidents(limit: int = 30) -> dict[str, Any]:
    exporter = IncidentExporter(INCIDENTS_DIR)
    records = exporter.load_all_records()
    return {"count": len(records), "incidents": records[:max(1, min(limit, 100))]}


@app.get("/api/analytics/summary")
async def analytics_summary() -> dict[str, Any]:
    if state.analytics_db is None:
        raise HTTPException(status_code=503, detail="Analytics database not initialized")
    return state.analytics_db.get_summary_metrics()


@app.get("/api/analytics/heatmap")
async def analytics_heatmap(overlay_live: bool = True) -> Response:
    if state.heatmap_engine is None:
        raise HTTPException(status_code=503, detail="Heatmap engine not initialized")
    bg_frame = None
    if overlay_live:
        with state.frame_lock:
            if state.latest_raw_frame is not None:
                bg_frame = state.latest_raw_frame.copy()
    jpeg_bytes = state.heatmap_engine.render_jpeg(background_bgr=bg_frame, quality=85)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.post("/api/analytics/heatmap/reset")
async def analytics_heatmap_reset() -> dict[str, str]:
    if state.heatmap_engine is not None:
        state.heatmap_engine.reset()
    return {"status": "heatmap_cleared"}


@app.get("/api/snapshots/{filename}")
async def get_grounded_snapshot(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Invalid snapshot filename")
    target = INCIDENTS_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(path=str(target), media_type="image/jpeg")


@app.get("/api/export/csv")
async def export_csv_endpoint() -> FileResponse:
    exporter = IncidentExporter(INCIDENTS_DIR)
    csv_file = INCIDENTS_DIR / "incident_audit_log.csv"
    exporter.export_csv(csv_file)
    return FileResponse(path=str(csv_file), filename="incident_audit_log.csv", media_type="text/csv")


@app.get("/api/export/pdf")
async def export_pdf_endpoint() -> FileResponse:
    exporter = IncidentExporter(INCIDENTS_DIR)
    pdf_file = INCIDENTS_DIR / "incident_audit_report.pdf"
    exporter.export_pdf(pdf_file)
    return FileResponse(path=str(pdf_file), filename="incident_audit_report.pdf", media_type="application/pdf")


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state.connected_stream_clients.append(websocket)
    try:
        while state.is_running:
            with state.frame_lock:
                frame_bytes = state.latest_jpeg

            if frame_bytes is not None:
                await websocket.send_bytes(frame_bytes)
                ram_percent, ram_free_gb = get_system_ram_stats()
                telemetry = {
                    "fps": state.fps,
                    "latency_ms": state.latency_ms,
                    "active_tracks": state.active_tracks,
                    "total_anomalies": state.total_anomalies,
                    "buffer_ram_mb": state.buffer_ram_mb,
                    "ram_percent": ram_percent,
                    "ram_free_gb": ram_free_gb,
                    "model_name": state.model_name,
                    "source": state.source_label
                }
                await websocket.send_text(json.dumps(telemetry))

            await asyncio.sleep(0.033)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if websocket in state.connected_stream_clients:
            state.connected_stream_clients.remove(websocket)


@app.websocket("/ws/incidents")
async def websocket_incidents_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state.connected_incident_clients.append(websocket)
    try:
        while state.is_running:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if websocket in state.connected_incident_clients:
            state.connected_incident_clients.remove(websocket)


class SentinelUvicornServer(uvicorn.Server):
    def handle_exit(self, sig: int, frame: Any) -> None:
        state.is_running = False
        super().handle_exit(sig, frame)


if __name__ == "__main__":
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        timeout_graceful_shutdown=3
    )
    server = SentinelUvicornServer(config=config)
    try:
        server.run()
    except (KeyboardInterrupt, SystemExit):
        print("[+] SentinelEdge server stopped cleanly.")