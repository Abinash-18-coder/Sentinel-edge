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
from pydantic import BaseModel, Field

from src.api.exporter import IncidentExporter
from src.data.analytics_db import SentinelAnalyticsDB
from src.data.heatmap_engine import SpatialHeatmapEngine
from src.perception.camera_stream import (
    UniversalCameraStream,
    format_source_label,
    parse_camera_source
)
from src.vlm.schemas import IncidentAuditReport


class SourceSwitchRequest(BaseModel):
    source: str = Field(min_length=1, max_length=300)


class InferenceConfigRequest(BaseModel):
    conf_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    model_name: str | None = Field(default=None, min_length=3, max_length=80)


class PipelineState:
    def __init__(self) -> None:
        self.is_running: bool = False
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.active_tracks: int = 0
        self.total_anomalies: int = 0
        self.buffer_ram_mb: float = 0.0
        self.model_name: str = "initializing"
        self.source_raw: str = "0"
        self.source_label: str = "cam:0"
        self.camera_status: str = "initializing"
        self.conf_threshold: float = 0.25
        self.requested_source: str | None = None
        self.requested_model: str | None = None
        self.control_lock = threading.Lock()
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


def draw_crack_detection(annotated: np.ndarray, ent: Any) -> None:
    img_h, img_w = annotated.shape[:2]
    bx1, by1, bx2, by2 = map(int, ent.bbox_xyxy)
    bx1 = max(0, min(bx1, img_w - 2))
    by1 = max(0, min(by1, img_h - 2))
    bx2 = max(bx1 + 2, min(bx2, img_w - 1))
    by2 = max(by1 + 2, min(by2, img_h - 1))

    box_color = (16, 16, 20)
    accent_color = (248, 189, 56) if ent.recovered else (44, 107, 255)

    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (242, 242, 245), 4, lineType=cv2.LINE_AA)
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2, lineType=cv2.LINE_AA)

    corner = max(8, min(18, (bx2 - bx1) // 4, (by2 - by1) // 4))
    cv2.line(annotated, (bx1, by1), (bx1 + corner, by1), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx1, by1), (bx1, by1 + corner), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx2, by1), (bx2 - corner, by1), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx2, by1), (bx2, by1 + corner), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx1, by2), (bx1 + corner, by2), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx1, by2), (bx1, by2 - corner), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx2, by2), (bx2 - corner, by2), accent_color, 3, lineType=cv2.LINE_AA)
    cv2.line(annotated, (bx2, by2), (bx2, by2 - corner), accent_color, 3, lineType=cv2.LINE_AA)

    suffix = " RE-ID" if ent.recovered else ""
    label = f"#{ent.track_id} {ent.class_name.upper()} {ent.confidence:.2f}{suffix}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    pad_x = 8
    pad_y = 5
    pill_w = tw + pad_x * 2 + 4
    pill_h = th + pad_y * 2 + baseline

    px1 = max(0, min(bx1, img_w - pill_w))
    px2 = min(img_w - 1, px1 + pill_w)
    if by1 >= pill_h + 2:
        py2 = by1 - 2
        py1 = py2 - pill_h
    else:
        py1 = min(img_h - pill_h - 1, by1 + 2)
        py2 = py1 + pill_h

    cv2.rectangle(annotated, (px1, py1), (px2, py2), box_color, -1)
    cv2.rectangle(annotated, (px1, py1), (px1 + 3, py2), accent_color, -1)
    cv2.putText(
        annotated,
        label,
        (px1 + pad_x + 3, py2 - pad_y - baseline + 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        return {0: "crack"}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {int(k): str(v) for k, v in data.get("names", {0: "crack"}).items()}


def list_available_models() -> dict[str, Path]:
    ordered_candidates = [
        PROJECT_ROOT / "models" / "onnx" / "best.onnx",
        PROJECT_ROOT / "models" / "onnx" / "best_fp16.onnx",
        PROJECT_ROOT / "models" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "onnx" / "best_int8.onnx",
        PROJECT_ROOT / "yolo26n.pt"
    ]
    found: dict[str, Path] = {}
    for p in ordered_candidates:
        if p.exists():
            found[p.name] = p
    return found


def resolve_model_path(preferred_name: str | None = None) -> Path:
    available = list_available_models()
    env_choice = (preferred_name or os.getenv("SENTINEL_MODEL", "")).strip()
    if env_choice and env_choice in available:
        return available[env_choice]
    if available:
        return next(iter(available.values()))
    return PROJECT_ROOT / "yolo26n.pt"


def resolve_video_source() -> int | str:
    raw = os.getenv("SENTINEL_SOURCE", "0").strip()
    return parse_camera_source(raw if raw else "0")


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

    def _encode_jpeg(self, frame_bgr: np.ndarray, quality: int = 80) -> bytes | None:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return encoded.tobytes() if ok else None

    def _run_test_loop(self) -> None:
        state.model_name = "test-synthetic-mode"
        state.source_raw = "synthetic"
        state.source_label = "synthetic:640x480"
        state.camera_status = "connected"
        frame_idx = 0

        while state.is_running:
            with state.control_lock:
                if state.requested_source is not None:
                    parsed = parse_camera_source(state.requested_source)
                    state.source_raw = str(parsed)
                    state.source_label = format_source_label(parsed)
                    state.requested_source = None
                if state.requested_model is not None:
                    state.model_name = state.requested_model
                    state.requested_model = None

            t0 = time.perf_counter()
            frame = np.full((480, 640, 3), 16, dtype=np.uint8)
            cv2.putText(
                frame,
                f"SENTINEL-EDGE v2.1 TEST | SRC: {state.source_label}",
                (28, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (44, 107, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"MODEL: {state.model_name} | CONF: {state.conf_threshold:.2f} | FRAME {frame_idx}",
                (28, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (158, 163, 175),
                1,
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

        initial_source = resolve_video_source()
        state.source_raw = str(initial_source)
        state.source_label = format_source_label(initial_source)

        camera = UniversalCameraStream(source=initial_source, max_width=640).start()
        ok, init_frame = camera.wait_for_first_frame(timeout_sec=4.0)
        if not ok or init_frame is None:
            init_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = init_frame.shape[:2]
        frame_wh = (w, h)

        tracker = DeepReIDTracker(frame_rate=30, reid_match_threshold=0.35)
        spatial_engine = SpatialRulesEngine(config_yaml=zones_cfg, frame_resolution_wh=frame_wh)
        incident_buffer = IncidentVideoBuffer(
            output_dir=INCIDENTS_DIR,
            pre_incident_seconds=3.0,
            post_incident_seconds=3.0,
            fps=25,
            resolution_wh=frame_wh
        )
        async_auditor = AsyncIncidentAuditor(
            max_queue_size=12,
            track_debounce_sec=6.0,
            vlm_mode="auto",
            on_report_callback=self._on_vlm_report
        )

        prev_time = time.perf_counter()
        frame_counter = 0
        print(f"[+] [VISION WORKER] Streaming live at {w}x{h} using {state.model_name}")

        try:
            while state.is_running:
                with state.control_lock:
                    req_src = state.requested_source
                    state.requested_source = None
                    req_model = state.requested_model
                    state.requested_model = None
                    active_conf = state.conf_threshold

                if req_src is not None:
                    parsed_src = parse_camera_source(req_src)
                    camera.switch_source(parsed_src)
                    state.source_raw = str(parsed_src)
                    state.source_label = format_source_label(parsed_src)

                if req_model is not None:
                    available = list_available_models()
                    if req_model in available and req_model != state.model_name:
                        print(f"[*] [VISION WORKER] Switching active model to {req_model}...")
                        try:
                            new_model = YOLO(str(available[req_model]), task="detect")
                            del model
                            gc.collect()
                            model = new_model
                            state.model_name = req_model
                            print(f"[+] [VISION WORKER] Active model switched to {req_model}")
                        except Exception as err:
                            print(f"[!] [VISION WORKER] Failed switching model to {req_model}: {err}")

                loop_start = time.perf_counter()
                cam_info = camera.get_source_info()
                state.camera_status = str(cam_info["status"])

                ret, frame = camera.read_latest()
                if not ret or frame is None:
                    standby = np.full((480, 640, 3), 14, dtype=np.uint8)
                    cv2.putText(
                        standby,
                        f"CAMERA STATUS: {state.camera_status.upper()}",
                        (145, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (44, 107, 255),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        standby,
                        f"SOURCE: {state.source_label}",
                        (145, 262),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (158, 163, 175),
                        1,
                        cv2.LINE_AA
                    )
                    jpeg = self._encode_jpeg(standby, quality=75)
                    if jpeg is not None:
                        with state.frame_lock:
                            state.latest_jpeg = jpeg
                    time.sleep(0.08)
                    continue

                fh, fw = frame.shape[:2]
                if (fw, fh) != frame_wh:
                    frame_wh = (fw, fh)
                    spatial_engine = SpatialRulesEngine(config_yaml=zones_cfg, frame_resolution_wh=frame_wh)
                    incident_buffer.flush_and_shutdown(timeout=4.0)
                    incident_buffer = IncidentVideoBuffer(
                        output_dir=INCIDENTS_DIR,
                        pre_incident_seconds=3.0,
                        post_incident_seconds=3.0,
                        fps=25,
                        resolution_wh=frame_wh
                    )
                    print(f"[+] [VISION WORKER] Adapted spatial zones & buffer to {fw}x{fh}")

                frame_counter += 1
                results = model.predict(frame, imgsz=640, conf=active_conf, verbose=False)[0]
                tracked_detections, entities = tracker.update(
                    frame_bgr=frame,
                    ultralytics_results=results,
                    class_name_map=class_map
                )
                state.active_tracks = len(entities)

                if entities and frame_counter % 3 == 0:
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
                        "vlm_summary": "Running crack morphology & VLM audit...",
                        "action": "Awaiting audit"
                    })

                incident_buffer.push_frame(frame)
                state.buffer_ram_mb = round(incident_buffer.estimate_buffer_ram_mb(), 2)

                annotated = spatial_engine.render_overlays(frame)
                annotated = tracker.annotate_traces(annotated, tracked_detections)

                for ent in entities:
                    draw_crack_detection(annotated, ent)

                jpeg = self._encode_jpeg(annotated, quality=80)
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
            camera.stop(timeout_sec=5.0)
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


app = FastAPI(title="SentinelEdge v2.1 Operations Server", lifespan=lifespan)

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
        "camera_status": state.camera_status,
        "conf_threshold": state.conf_threshold,
        "fps": state.fps,
        "latency_ms": state.latency_ms,
        "active_tracks": state.active_tracks,
        "total_anomalies": state.total_anomalies,
        "buffer_ram_mb": state.buffer_ram_mb,
        "ram_percent": ram_percent,
        "ram_free_gb": ram_free_gb
    }


@app.get("/api/config")
async def get_runtime_config() -> dict[str, Any]:
    return {
        "source": state.source_raw,
        "source_label": state.source_label,
        "camera_status": state.camera_status,
        "model_name": state.model_name,
        "available_models": list(list_available_models().keys()),
        "conf_threshold": state.conf_threshold
    }


@app.post("/api/camera/source")
async def switch_camera_source(req: SourceSwitchRequest) -> dict[str, Any]:
    cleaned = req.source.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Camera source cannot be empty")
    with state.control_lock:
        state.requested_source = cleaned
    return {
        "status": "switching",
        "requested_source": cleaned,
        "label": format_source_label(parse_camera_source(cleaned))
    }


@app.post("/api/config/inference")
async def update_inference_config(req: InferenceConfigRequest) -> dict[str, Any]:
    available = list_available_models()
    with state.control_lock:
        if req.conf_threshold is not None:
            state.conf_threshold = round(float(req.conf_threshold), 2)
        if req.model_name is not None:
            if os.getenv("SENTINEL_TEST_MODE", "0") != "1" and req.model_name not in available:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{req.model_name}' not found. Available: {list(available.keys())}"
                )
            state.requested_model = req.model_name

    return {
        "status": "updated",
        "conf_threshold": state.conf_threshold,
        "requested_model": req.model_name or state.model_name
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
                    "source": state.source_label,
                    "camera_status": state.camera_status,
                    "conf_threshold": state.conf_threshold
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