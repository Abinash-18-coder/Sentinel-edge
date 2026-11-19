import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class IncidentExporter:
    def __init__(self, incidents_dir: Path) -> None:
        self.incidents_dir = incidents_dir
        self.incidents_dir.mkdir(parents=True, exist_ok=True)

    def load_all_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        json_files = sorted(self.incidents_dir.glob("*.json"), reverse=True)

        for jf in json_files:
            if jf.name.endswith("_vlm_report.json"):
                continue

            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                vlm_path = self.incidents_dir / f"{jf.stem}_vlm_report.json"
                grounded_img = self.incidents_dir / f"{jf.stem}_vlm_report.grounded.jpg"

                if vlm_path.exists():
                    vlm_data = json.loads(vlm_path.read_text(encoding="utf-8"))
                    data["vlm_summary"] = vlm_data.get("vlm_scene_description", "N/A")
                    data["severity"] = vlm_data.get("severity", "MEDIUM")
                    data["action"] = vlm_data.get("recommended_action", "Inspect area")
                    data["audit_backend"] = vlm_data.get("audit_backend", "unknown")
                    data["audit_latency_ms"] = vlm_data.get("audit_latency_ms", 0.0)
                else:
                    data["vlm_summary"] = "Awaiting VLM audit"
                    data["severity"] = "PENDING"
                    data["action"] = "Standard verification"
                    data["audit_backend"] = "pending"
                    data["audit_latency_ms"] = 0.0

                data["grounded_image"] = grounded_img.name if grounded_img.exists() else None
                records.append(data)
            except Exception as err:
                print(f"[!] Skipping corrupted incident JSON {jf.name}: {err}")

        return records

    def export_csv(self, output_file: Path) -> Path:
        records = self.load_all_records()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "incident_id",
            "readable_time",
            "zone_name",
            "rule_type",
            "track_id",
            "class_name",
            "dwell_time_seconds",
            "severity",
            "audit_backend",
            "audit_latency_ms",
            "action",
            "vlm_summary"
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        return output_file

    def export_pdf(self, output_file: Path) -> Path:
        records = self.load_all_records()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_file),
            pagesize=letter,
            rightMargin=32,
            leftMargin=32,
            topMargin=32,
            bottomMargin=32
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=17,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12
        )
        cell_style = ParagraphStyle(
            "CellContent",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1e293b")
        )

        elements: list[Any] = [
            Paragraph("SENTINEL-EDGE v2.0: STRUCTURAL INCIDENT AUDIT REPORT", title_style),
            Paragraph(
                f"Automated multi-stage edge vision & grounded VLM telemetry export. Total recorded incidents: {len(records)}",
                subtitle_style
            ),
            Spacer(1, 8)
        ]

        table_data: list[list[Any]] = [
            ["Incident ID", "Timestamp", "Zone / Rule", "Class", "Severity", "Recommended Action"]
        ]

        if not records:
            table_data.append([
                Paragraph("No incidents recorded yet.", cell_style),
                "", "", "", "", ""
            ])
        else:
            for r in records[:50]:
                zone_label = f"{r.get('zone_name', 'N/A')} ({r.get('rule_type', 'zone')})"
                table_data.append([
                    Paragraph(str(r.get("incident_id", ""))[:22], cell_style),
                    Paragraph(str(r.get("readable_time", "")), cell_style),
                    Paragraph(zone_label, cell_style),
                    Paragraph(f"#{r.get('track_id', '-')} {r.get('class_name', '')}", cell_style),
                    Paragraph(str(r.get("severity", "PENDING")), cell_style),
                    Paragraph(str(r.get("action", "")), cell_style)
                ])

        table = Table(table_data, colWidths=[95, 82, 115, 65, 58, 133])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        elements.append(table)
        doc.build(elements)
        return output_file