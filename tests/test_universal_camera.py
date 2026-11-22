import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import shutil
import time
import cv2
import numpy as np

from src.perception.camera_stream import (
    UniversalCameraStream,
    classify_source_type,
    parse_camera_source
)


def test_source_parsing_and_classification() -> None:
    assert parse_camera_source("0") == 0
    assert classify_source_type(0) == "usb_webcam"
    assert classify_source_type("http://192.168.1.15:8080/video") == "http_mjpeg"
    assert classify_source_type("http://192.168.1.50:81/stream") == "http_mjpeg"
    assert classify_source_type("rtsp://192.168.1.50:8554/cam") == "rtsp_robot"
    assert classify_source_type("data/raw/wall_scan.mp4") == "video_file"


def test_universal_camera_resize_and_hot_swap() -> None:
    tmp_dir = PROJECT_ROOT / "data" / "raw" / "incidents" / "_cam_test_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    vid_a = tmp_dir / "hd_clip_a.mp4"
    vid_b = tmp_dir / "sd_clip_b.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wa = cv2.VideoWriter(str(vid_a), fourcc, 20.0, (1280, 720))
    wb = cv2.VideoWriter(str(vid_b), fourcc, 20.0, (640, 480))
    for _ in range(15):
        wa.write(np.full((720, 1280, 3), 180, dtype=np.uint8))
        wb.write(np.full((480, 640, 3), 90, dtype=np.uint8))
    wa.release()
    wb.release()

    cam = UniversalCameraStream(source=str(vid_a), max_width=640).start()
    try:
        ok, frame_a = cam.wait_for_first_frame(timeout_sec=3.0)
        assert ok is True and frame_a is not None
        assert frame_a.shape[1] == 640 and frame_a.shape[0] == 360

        cam.switch_source(str(vid_b))
        switched = False
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            ok_b, frame_b = cam.read_latest()
            if ok_b and frame_b is not None and frame_b.shape[:2] == (480, 640):
                switched = True
                break
            time.sleep(0.05)
        assert switched is True, "UniversalCameraStream failed to hot-swap video sources!"
    finally:
        cam.stop(timeout_sec=4.0)
        shutil.rmtree(tmp_dir)


def test_fastapi_live_control_endpoints() -> None:
    os.environ["SENTINEL_TEST_MODE"] = "1"
    from fastapi.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as client:
        time.sleep(0.2)

        r_cfg = client.get("/api/config")
        assert r_cfg.status_code == 200
        cfg = r_cfg.json()
        assert "conf_threshold" in cfg and "available_models" in cfg

        r_src = client.post(
            "/api/camera/source",
            json={"source": "rtsp://192.168.1.50:8554/cam"}
        )
        assert r_src.status_code == 200
        assert r_src.json()["status"] == "switching"

        r_inf = client.post(
            "/api/config/inference",
            json={"conf_threshold": 0.20, "model_name": "best.onnx"}
        )
        assert r_inf.status_code == 200
        assert r_inf.json()["conf_threshold"] == 0.20

        time.sleep(0.15)
        health = client.get("/api/health").json()
        assert health["conf_threshold"] == 0.20
        assert health["model_name"] == "best.onnx"

    tmp_db = PROJECT_ROOT / "data" / "analytics" / "test_analytics_tmp.duckdb"
    for candidate in (tmp_db, Path(f"{tmp_db}.wal")):
        if candidate.exists():
            candidate.unlink()