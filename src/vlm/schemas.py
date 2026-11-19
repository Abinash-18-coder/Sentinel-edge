import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class IncidentSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GroundingBoundingBox(BaseModel):
    ymin: int = Field(ge=0, le=1000)
    xmin: int = Field(ge=0, le=1000)
    ymax: int = Field(ge=0, le=1000)
    xmax: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def ensure_ordered_coordinates(self) -> "GroundingBoundingBox":
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymax == self.ymin:
            self.ymax = min(1000, self.ymin + 10)
        if self.xmax == self.xmin:
            self.xmax = min(1000, self.xmin + 10)
        return self


class RawVLMOutput(BaseModel):
    scene_description: str = Field(min_length=3, max_length=300)
    box_2d: list[int] = Field(min_length=4, max_length=4)
    severity: IncidentSeverity
    recommended_action: str = Field(min_length=3, max_length=120)


class IncidentAuditReport(BaseModel):
    incident_id: str
    timestamp: str
    zone_id: str
    track_id: int
    entity_class: str
    vlm_scene_description: str
    hazard_grounding_box: GroundingBoundingBox
    severity: IncidentSeverity
    recommended_action: str
    audit_backend: str
    audit_latency_ms: float = Field(ge=0.0)