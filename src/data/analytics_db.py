import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import time
from typing import Any

import duckdb

from src.rules.spatial_engine import AnomalyEvent
from src.vlm.schemas import IncidentAuditReport


class SentinelAnalyticsDB:
    def __init__(
        self,
        db_path: Path,
        memory_limit: str = "256MB",
        threads: int = 2,
        track_batch_size: int = 40
    ) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.track_batch_size = max(1, track_batch_size)
        self._lock = threading.Lock()
        self._point_buffer: list[tuple[Any, ...]] = []
        self._closed = False

        self.conn = duckdb.connect(str(self.db_path))
        self.conn.execute(f"SET memory_limit='{memory_limit}';")
        self.conn.execute(f"SET threads={int(threads)};")
        self._init_schema()
        print(f"[+] DuckDB Analytics online (WAL mode | Cap: {memory_limit}): {self.db_path.name}")

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS track_points (
                    ts DOUBLE,
                    camera_id VARCHAR,
                    track_id INTEGER,
                    class_name VARCHAR,
                    cx DOUBLE,
                    cy DOUBLE,
                    frame_w INTEGER,
                    frame_h INTEGER,
                    confidence DOUBLE
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS zone_breaches (
                    incident_id VARCHAR PRIMARY KEY,
                    ts DOUBLE,
                    readable_time VARCHAR,
                    zone_id VARCHAR,
                    zone_name VARCHAR,
                    rule_type VARCHAR,
                    track_id INTEGER,
                    class_name VARCHAR,
                    dwell_time_sec DOUBLE,
                    bbox_x1 DOUBLE,
                    bbox_y1 DOUBLE,
                    bbox_x2 DOUBLE,
                    bbox_y2 DOUBLE
                );
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS vlm_audits (
                    incident_id VARCHAR PRIMARY KEY,
                    ts VARCHAR,
                    zone_id VARCHAR,
                    track_id INTEGER,
                    entity_class VARCHAR,
                    severity VARCHAR,
                    scene_description VARCHAR,
                    recommended_action VARCHAR,
                    audit_backend VARCHAR,
                    audit_latency_ms DOUBLE,
                    ymin INTEGER,
                    xmin INTEGER,
                    ymax INTEGER,
                    xmax INTEGER
                );
            """)

    def buffer_track_point(
        self,
        track_id: int,
        class_name: str,
        cx: float,
        cy: float,
        frame_w: int,
        frame_h: int,
        confidence: float,
        camera_id: str = "cam_0",
        timestamp: float | None = None
    ) -> None:
        if self._closed:
            return
        now = time.time() if timestamp is None else timestamp
        row = (
            float(now),
            camera_id,
            int(track_id),
            str(class_name),
            float(cx),
            float(cy),
            int(frame_w),
            int(frame_h),
            float(confidence)
        )
        with self._lock:
            self._point_buffer.append(row)
            if len(self._point_buffer) >= self.track_batch_size:
                self._flush_points_unlocked()

    def _flush_points_unlocked(self) -> None:
        if not self._point_buffer or self._closed:
            return
        self.conn.executemany(
            """
            INSERT INTO track_points (
                ts, camera_id, track_id, class_name, cx, cy, frame_w, frame_h, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            self._point_buffer
        )
        self._point_buffer.clear()

    def flush_points(self) -> None:
        with self._lock:
            self._flush_points_unlocked()

    def log_zone_breach(self, incident_id: str, event: AnomalyEvent) -> None:
        if self._closed:
            return
        readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
        x1, y1, x2, y2 = event.bbox_xyxy
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO zone_breaches (
                    incident_id, ts, readable_time, zone_id, zone_name, rule_type,
                    track_id, class_name, dwell_time_sec, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    incident_id,
                    float(event.timestamp),
                    readable,
                    event.zone_id,
                    event.zone_name,
                    event.rule_type,
                    int(event.track_id),
                    event.class_name,
                    round(float(event.dwell_time), 2),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            )

    def log_vlm_audit(self, report: IncidentAuditReport) -> None:
        if self._closed:
            return
        box = report.hazard_grounding_box
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO vlm_audits (
                    incident_id, ts, zone_id, track_id, entity_class, severity,
                    scene_description, recommended_action, audit_backend, audit_latency_ms,
                    ymin, xmin, ymax, xmax
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    report.incident_id,
                    report.timestamp,
                    report.zone_id,
                    int(report.track_id),
                    report.entity_class,
                    report.severity.value,
                    report.vlm_scene_description,
                    report.recommended_action,
                    report.audit_backend,
                    float(report.audit_latency_ms),
                    int(box.ymin),
                    int(box.xmin),
                    int(box.ymax),
                    int(box.xmax)
                ]
            )

    def fetch_recent_track_points(
        self,
        max_points: int = 5000,
        since_ts: float | None = None
    ) -> list[tuple[float, float, int, int]]:
        with self._lock:
            self._flush_points_unlocked()
            if since_ts is not None:
                rows = self.conn.execute(
                    """
                    SELECT cx, cy, frame_w, frame_h
                    FROM track_points
                    WHERE ts >= ?
                    ORDER BY ts DESC
                    LIMIT ?;
                    """,
                    [float(since_ts), int(max_points)]
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT cx, cy, frame_w, frame_h
                    FROM track_points
                    ORDER BY ts DESC
                    LIMIT ?;
                    """,
                    [int(max_points)]
                ).fetchall()
        return [(float(r[0]), float(r[1]), int(r[2]), int(r[3])) for r in rows]

    def get_summary_metrics(self) -> dict[str, Any]:
        with self._lock:
            self._flush_points_unlocked()

            total_points = int(self.conn.execute("SELECT COUNT(*) FROM track_points;").fetchone()[0])
            unique_tracks = int(self.conn.execute("SELECT COUNT(DISTINCT track_id) FROM track_points;").fetchone()[0])

            breach_row = self.conn.execute(
                "SELECT COUNT(*), COALESCE(AVG(dwell_time_sec), 0.0) FROM zone_breaches;"
            ).fetchone()
            total_breaches = int(breach_row[0])
            avg_dwell = round(float(breach_row[1]), 2)

            vlm_row = self.conn.execute(
                "SELECT COUNT(*), COALESCE(AVG(audit_latency_ms), 0.0) FROM vlm_audits;"
            ).fetchone()
            total_audits = int(vlm_row[0])
            avg_vlm_ms = round(float(vlm_row[1]), 2)

            sev_rows = self.conn.execute(
                "SELECT severity, COUNT(*) FROM vlm_audits GROUP BY severity;"
            ).fetchall()
            severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            for sev, cnt in sev_rows:
                severity_counts[str(sev)] = int(cnt)

            zone_rows = self.conn.execute(
                """
                SELECT zone_name, COUNT(*) AS breach_count, ROUND(AVG(dwell_time_sec), 2) AS avg_dwell
                FROM zone_breaches
                GROUP BY zone_name
                ORDER BY breach_count DESC
                LIMIT 5;
                """
            ).fetchall()
            top_zones = [
                {"zone_name": str(r[0]), "breaches": int(r[1]), "avg_dwell_sec": float(r[2])}
                for r in zone_rows
            ]

        return {
            "total_track_points": total_points,
            "unique_tracks": unique_tracks,
            "total_breaches": total_breaches,
            "avg_dwell_seconds": avg_dwell,
            "total_vlm_audits": total_audits,
            "avg_vlm_latency_ms": avg_vlm_ms,
            "severity_breakdown": severity_counts,
            "top_zones": top_zones
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_points_unlocked()
            try:
                self.conn.execute("CHECKPOINT;")
            except Exception:
                pass
            self.conn.close()
            self._closed = True
            print(f"[+] DuckDB Analytics checkpointed and closed cleanly: {self.db_path.name}")