import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
import time

os.environ["SENTINEL_TEST_MODE"] = "1"

from fastapi.testclient import TestClient
from src.api.server import app


def test_dashboard_and_websockets() -> None:
    print("=" * 65)
    print(" SentinelEdge v2.0: Day 19-21 FastAPI & WebSocket Test Suite")
    print("=" * 65)

    with TestClient(app) as client:
        time.sleep(0.15)

        print("[*] Test 1: Checking HTML5 Canvas Dashboard (GET /)...")
        resp_home = client.get("/")
        assert resp_home.status_code == 200
        assert 'id="videoCanvas"' in resp_home.text
        assert 'id="latencyChart"' in resp_home.text
        print("[+] Test 1 Passed: Dashboard HTML served cleanly.")

        print("[*] Test 2: Checking Health & RAM Telemetry (GET /api/health)...")
        resp_health = client.get("/api/health")
        assert resp_health.status_code == 200
        health = resp_health.json()
        assert health["status"] == "online"
        assert "ram_free_gb" in health and "fps" in health
        print(f"[+] Test 2 Passed: Online | Model: {health['model_name']} | Free RAM: {health['ram_free_gb']} GB")

        print("[*] Test 3: Checking Incident Feed & CSV/PDF Export Endpoints...")
        resp_inc = client.get("/api/incidents")
        assert resp_inc.status_code == 200
        assert "incidents" in resp_inc.json()

        resp_csv = client.get("/api/export/csv")
        assert resp_csv.status_code == 200
        assert "incident_id" in resp_csv.text

        resp_pdf = client.get("/api/export/pdf")
        assert resp_pdf.status_code == 200
        assert resp_pdf.content.startswith(b"%PDF")
        print("[+] Test 3 Passed: /api/incidents, CSV export, and PDF export verified.")

        print("[*] Test 4: Checking Binary JPEG + Telemetry WebSocket (/ws/stream)...")
        with client.websocket_connect("/ws/stream") as ws:
            frame_bytes = ws.receive_bytes()
            assert len(frame_bytes) > 500
            assert frame_bytes[:2] == b"\xff\xd8"

            telemetry_txt = ws.receive_text()
            telemetry = json.loads(telemetry_txt)
            assert "fps" in telemetry and "latency_ms" in telemetry and "ram_percent" in telemetry
        print(f"[+] Test 4 Passed: Received JPEG ({len(frame_bytes)} bytes) + telemetry JSON.")

        print("[*] Test 5: Checking Incident Notification WebSocket (/ws/incidents)...")
        with client.websocket_connect("/ws/incidents") as ws_inc:
            assert ws_inc is not None
        print("[+] Test 5 Passed: /ws/incidents connected and closed cleanly.")

    print("=" * 65)
    print(" ALL DAY 19-21 FASTAPI & CANVAS DASHBOARD TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_dashboard_and_websockets()