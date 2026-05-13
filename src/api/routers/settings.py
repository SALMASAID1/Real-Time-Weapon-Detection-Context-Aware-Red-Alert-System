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

import logging
from fastapi import APIRouter, Request
from src.api.schemas.detection import SystemSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Module-level settings singleton — shared reference with InferenceEngine
_current_settings = SystemSettings()


def get_current_settings() -> SystemSettings:
    """Public accessor so main.py can read the initial settings at startup."""
    return _current_settings


@router.get("", response_model=SystemSettings)
async def get_settings():
    """Return current runtime settings. React reads these on Settings page mount."""
    return _current_settings


@router.put("", response_model=SystemSettings)
async def update_settings(new_settings: SystemSettings, request: Request):
    """
    Apply new settings. All fields are replaced atomically.
    Validated by Pydantic before assignment — invalid ranges are rejected
    with HTTP 422 before reaching the inference engine.

    The engine's settings dict is updated in-place so changes propagate
    to the running inference loop on the next frame cycle.
    """
    global _current_settings
    _current_settings = new_settings

    # Propagate to the running InferenceEngine if available
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        engine.settings["conf_threshold"] = new_settings.conf_threshold
        engine.settings["iou_threshold"] = new_settings.iou_threshold
        engine.settings["sahi_every_n"] = new_settings.sahi_every_n
        engine.settings["inference_every_n"] = new_settings.inference_every_n
        engine.settings["gradcam_on_high"] = new_settings.gradcam_on_high

        # Propagate alert_threshold to ThreatScorer so it takes effect live
        if hasattr(engine, 'threat_scorer'):
            engine.threat_scorer.thresholds['high'] = new_settings.alert_threshold

        logger.info(
            f"Settings propagated to engine: "
            f"conf={new_settings.conf_threshold}, "
            f"iou={new_settings.iou_threshold}, "
            f"sahi_n={new_settings.sahi_every_n}, "
            f"inference_n={new_settings.inference_every_n}, "
            f"gradcam={new_settings.gradcam_on_high}, "
            f"alert_thresh={new_settings.alert_threshold}"
        )

    return _current_settings
