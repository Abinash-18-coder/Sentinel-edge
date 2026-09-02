"""
src/vlm/schemas.py
Pydantic data models enforcing strict schema contracts on VLM incident reports.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class IncidentSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentAuditReport(BaseModel):
    """Structured report produced by the Multimodal Vision-Language Model."""
    incident_id: str = Field(description="Unique identifier of the incident")
    timestamp: str = Field(description="ISO 8601 formatted timestamp")
    zone_id: str = Field(description="Spatial zone ID where breach occurred")
    track_id: int = Field(description="Tracking identifier of the entity")
    entity_class: str = Field(description="Detected object class (e.g. worker, crack, intrusion)")
    vlm_scene_description: str = Field(description="Natural-language synthesis of the visual evidence")
    severity: IncidentSeverity = Field(description="Risk assessment score: LOW, MEDIUM, HIGH, CRITICAL")
    recommended_action: str = Field(description="Recommended corrective or defensive action")