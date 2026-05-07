"""
src/api/routers/settings.py
=============================
Runtime settings API — consumed by React ThresholdPanel (Settings page).

Endpoints
---------
GET  /api/settings      — Current system configuration
PUT  /api/settings      — Update configuration (applied live, no restart)

Live-apply pattern
------------------
Settings are stored in a module-level singleton (SystemSettings instance).
The InferenceEngine, ThreatScorer, and AlertDispatcher all hold a reference
to this same object. When PUT /api/settings is called, the singleton's fields
are mutated in-place — the change propagates to all running components on the
next frame cycle without any restart.

This is safe because all these components run in the same asyncio event loop
and GIL — there is no concurrent write from multiple threads to the settings.

Validated fields (see src/api/schemas/detection.py :: SystemSettings)
----------------------------------------------------------------------
  iou_threshold       : Hand-Weapon IoU to consider "held" (0.0 – 1.0)
  conf_threshold      : Minimum detection confidence to log (0.0 – 1.0)
  alert_threshold     : Composite score for Red Alert escalation (0.0 – 1.0)
  sahi_every_n        : SAHI cadence (1 = every frame, 3 = every third)
  persist_max_frames  : Frames for persistence saturation
  telegram_enabled    : Toggle Telegram channel
  audio_enabled       : Toggle audio cue
  gradcam_on_high     : Toggle Grad-CAM generation (performance impact)
"""

from fastapi import APIRouter
from src.api.schemas.detection import SystemSettings

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Module-level settings singleton — shared reference with InferenceEngine
_current_settings = SystemSettings()


@router.get("", response_model=SystemSettings)
async def get_settings():
    """Return current runtime settings. React reads these on Settings page mount."""
    return _current_settings


@router.put("", response_model=SystemSettings)
async def update_settings(new_settings: SystemSettings):
    """
    Apply new settings. All fields are replaced atomically.
    Validated by Pydantic before assignment — invalid ranges are rejected
    with HTTP 422 before reaching the inference engine.
    """
    global _current_settings
    _current_settings = new_settings
    return _current_settings
