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

from src.data.analytics_db import SentinelAnalyticsDB
from src.data.heatmap_engine import SpatialHeatmapEngine
from src.rules.spatial_engine import AnomalyEvent, AnomalyState
from src.vlm.schemas import GroundingBoundingBox, IncidentAuditReport, IncidentSeverity


def test_duckdb_persistence_and_sql() -> None:
    print("[*] Test 1: Verifying Embedded DuckDB WAL Persistence & SQL Aggregations...")
    tmp_dir = PROJECT_ROOT / "data" / "analytics" / "_day22_test_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    db_file = tmp_dir / "unit_test.duckdb"
    db = SentinelAnalyticsDB(db_path=db_file, memory_limit="256MB", threads=2, track_batch_size=10)

    for i in range(25):
        db.buffer_track_point(
            track_id=(i % 3) + 1,
            class_name="crack",
            cx=180.0 + (i * 4.0),
            cy=220.0 + (i * 2.0),
            frame_w=640,
            frame_h=480,
            confidence=0.86
        )

    ev = AnomalyEvent(
        zone_id="zone_alpha_danger",
        zone_name="Primary Structural Inspection Zone",
        rule_type="polygon_zone",
        track_id=1,
        class_name="crack",
        state=AnomalyState.CONFIRMED_ANOMALY,
        dwell_time=3.4,
        timestamp=time.time(),
        bbox_xyxy=(140.0, 180.0, 260.0, 320.0)
    )
    db.log_zone_breach("inc_duckdb_001", ev)

    report = IncidentAuditReport(
        incident_id="inc_duckdb_001",
        timestamp="2026-11-19T13:30:00Z",
        zone_id="zone_alpha_danger",
        track_id=1,
        entity_class="crack",
        vlm_scene_description="Vertical fissure detected in inspection zone.",
        hazard_grounding_box=GroundingBoundingBox(ymin=300, xmin=220, ymax=700, xmax=420),
        severity=IncidentSeverity.CRITICAL,
        recommended_action="Halt scan and brace section.",
        audit_backend="edge-cv-fallback",
        audit_latency_ms=14.5
    )
    db.log_vlm_audit(report)

    summary = db.get_summary_metrics()
    assert summary["total_track_points"] == 25
    assert summary["unique_tracks"] == 3
    assert summary["total_breaches"] == 1
    assert summary["total_vlm_audits"] == 1
    assert summary["severity_breakdown"]["CRITICAL"] == 1
    assert summary["avg_dwell_seconds"] == 3.4
    db.close()

    db_reopened = SentinelAnalyticsDB(db_path=db_file, memory_limit="256MB", threads=2)
    persisted = db_reopened.get_summary_metrics()
    assert persisted["total_track_points"] == 25
    assert persisted["total_breaches"] == 1
    pts = db_reopened.fetch_recent_track_points(max_points=100)
    assert len(pts) == 25
    db_reopened.close()

    shutil.rmtree(tmp_dir)
    print("[+] Test 1 Passed: DuckDB inserts, SQL aggregations, and WAL restart persistence verified.")


def test_heatmap_engine_rendering() -> None:
    print("[*] Test 2: Verifying Gaussian Spatial Heatmap Engine & JPEG Encoding...")
    engine = SpatialHeatmapEngine(resolution_wh=(640, 480))
    pts = [(200.0 + (i % 5) * 3.0, 240.0 + (i % 4) * 3.0, 640, 480) for i in range(60)]
    engine.add_points_batch(pts)

    bg = np.full((480, 640, 3), 120, dtype=np.uint8)
    jpeg = engine.render_jpeg(background_bgr=bg)
    assert len(jpeg) > 2000
    assert jpeg[:2] == b"\xff\xd8"

    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == (480, 640, 3)

    engine.reset()
    assert engine.total_samples == 0
    print(f"[+] Test 2 Passed: Heatmap rendered cleanly ({len(jpeg)} bytes JPEG) and reset verified.")


def test_fastapi_analytics_endpoints() -> None:
    print("[*] Test 3: Verifying FastAPI /api/analytics/summary & /api/analytics/heatmap Routes...")
    os.environ["SENTINEL_TEST_MODE"] = "1"
    from fastapi.testclient import TestClient
    from src.api.server import app

    with TestClient(app) as client:
        time.sleep(0.35)

        r_sum = client.get("/api/analytics/summary")
        assert r_sum.status_code == 200
        data = r_sum.json()
        assert data["total_track_points"] >= 1

        r_map = client.get("/api/analytics/heatmap?overlay_live=true")
        assert r_map.status_code == 200
        assert r_map.content[:2] == b"\xff\xd8"

        r_rst = client.post("/api/analytics/heatmap/reset")
        assert r_rst.status_code == 200
        assert r_rst.json()["status"] == "heatmap_cleared"

    tmp_db = PROJECT_ROOT / "data" / "analytics" / "test_analytics_tmp.duckdb"
    for candidate in (tmp_db, Path(f"{tmp_db}.wal")):
        if candidate.exists():
            candidate.unlink()

    print("[+] Test 3 Passed: FastAPI DuckDB summary, heatmap JPEG, and reset endpoints verified.")


if __name__ == "__main__":
    print("=" * 65)
    print(" SentinelEdge v2.0: Day 22-24 DuckDB & Spatial Heatmap Suite")
    print("=" * 65)
    test_duckdb_persistence_and_sql()
    test_heatmap_engine_rendering()
    test_fastapi_analytics_endpoints()
    print("=" * 65)
    print(" ALL DAY 22-24 DUCKDB & HEATMAP TESTS PASSED!")
    print("=" * 65)