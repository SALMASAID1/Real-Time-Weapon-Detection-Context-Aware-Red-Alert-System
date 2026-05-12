"""
src/api/schemas/detection.py
==============================
Pydantic schemas for the WebSocket message contract and REST API responses.
These are the strict data contracts between the Python backend and the React frontend.
Any field change here MUST be reflected in the React useDetectionStream hook.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id:   int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox:       BoundingBox
    is_weapon:  bool   # True for weapon classes (0-3, 6); False for hand/person


class ThreatEvent(BaseModel):
    """
    Schema for a single scored detection event.
    Embedded inside WebSocketFrame for HIGH/LOW threat detections.
    Also the shape of records returned by GET /api/threats.
    """
    event_id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:       datetime
    camera_id:       str
    detection:       Detection
    proximity_iou:   float = Field(..., ge=0.0, le=1.0)
    persistence:     float = Field(..., ge=0.0, le=1.0)
    composite_score: float = Field(..., ge=0.0, le=1.0)
    threat_level:    str   # "NONE" | "LOW" | "HIGH"
    acknowledged:    bool = False


class WebSocketFrame(BaseModel):
    """
    The JSON payload pushed over the WebSocket for every processed frame.
    React's useDetectionStream hook parses this schema on every message event.
    """
    frame_id:      int
    camera_id:     str
    timestamp:     float           # Unix timestamp (seconds)
    frame_b64:     str             # Base64 JPEG of annotated frame
    detections:    List[Detection]
    threat_events: List[ThreatEvent]
    threat_level:  str             # Highest threat level this frame
    fps:           float
    gradcam_b64:   Optional[str] = None  # Base64 JPEG of Grad-CAM overlay (HIGH only)


class SystemSettings(BaseModel):
    """
    Runtime configuration schema.
    GET /api/settings → SystemSettings
    PUT /api/settings ← SystemSettings
    """
    iou_threshold:      float = Field(0.45, ge=0.0, le=1.0,
                                      description="Hand-Weapon IoU threshold")
    conf_threshold:     float = Field(0.35, ge=0.0, le=1.0,
                                      description="Minimum detection confidence")
    alert_threshold:    float = Field(0.70, ge=0.0, le=1.0,
                                      description="Composite score threshold for Red Alert")
    inference_every_n:  int   = Field(3,    ge=1, le=10,
                                      description="Run any inference every N frames")
    sahi_every_n:       int   = Field(5,    ge=1, le=10,
                                      description="Run full SAHI pass every N frames (relative to inference frames)")
    persist_max_frames: int   = Field(10,   ge=1, le=60,
                                      description="Frame count for persistence score saturation")
    telegram_enabled:   bool  = True
    audio_enabled:      bool  = True
    gradcam_on_high:    bool  = True   # Only generate Grad-CAM on HIGH alerts (saves latency)
