"""
src/vlm/schemas.py
Pydantic data models enforcing strict schema validation and spatial visual
grounding coordinates on VLM incident reports.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class IncidentSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GroundingBoundingBox(BaseModel):
    """Normalized bounding coordinates [ymin, xmin, ymax, xmax] in [0, 1000] scale."""
    ymin: int = Field(ge=0, le=1000, description="Top-most boundary coordinate")
    xmin: int = Field(ge=0, le=1000, description="Left-most boundary coordinate")
    ymax: int = Field(ge=0, le=1000, description="Bottom-most boundary coordinate")
    xmax: int = Field(ge=0, le=1000, description="Right-most boundary coordinate")


class IncidentAuditReport(BaseModel):
    """Validated structured incident report produced by Qwen2.5-VL / SmolVLM."""
    incident_id: str = Field(description="Unique identifier of the incident")
    timestamp: str = Field(description="ISO-8601 formatted timestamp")
    zone_id: str = Field(description="Spatial zone ID where breach occurred")
    track_id: int = Field(description="Re-ID tracking identifier of the entity")
    entity_class: str = Field(description="Detected object class (worker, crack, etc.)")
    vlm_scene_description: str = Field(description="Natural-language description of the visual scene")
    hazard_grounding_box: GroundingBoundingBox = Field(
        description="Localized bounding box of the specific hazard or violation"
    )
    severity: IncidentSeverity = Field(description="Risk assessment: LOW, MEDIUM, HIGH, CRITICAL")
    recommended_action: str = Field(description="Recommended action in under 10 words")