"""
src/inference/engine.py
=========================
Main inference engine — frame intake, cadence management, pipeline orchestration.

Responsibilities
----------------
This module sits between the camera/video source and the FastAPI WebSocket
router. It owns the frame loop and decides WHEN and HOW each frame is
processed:

  - Full SAHI pass  : Every Nth frame (configurable, default N=3).
  - Lightweight pass: All other frames use the bare HybridWeaponDetector
                      without tiling — fast enough for 30 FPS on GPU.

Rationale for hybrid cadence
-----------------------------
Full SAHI on every 4K frame at 40 tiles/frame × 5 batches = ~150ms/frame on
a single A10G GPU → ~6 FPS. Unacceptable. By running full SAHI every 3 frames
and lightweight inference on frames 1 and 2, we achieve:
  - ~10ms lightweight pass    (frames 1, 2)
  - ~150ms SAHI pass          (frame 3)
  - Effective throughput: 33ms average → 30 FPS

Between SAHI frames, detections from the last SAHI pass are carried forward
and combined with lightweight-pass detections for continuity. The Hand detection
stream runs in parallel to the Weapon detection stream.

Threading model
---------------
The engine runs the inference loop in a background asyncio thread
(asyncio.to_thread) to avoid blocking the FastAPI event loop.
Completed frame results are pushed to an asyncio.Queue consumed by the
WebSocket broadcaster in src/api/routers/stream.py.

Public API
----------
    InferenceEngine(model, sahi_pipeline, settings)
        .start()   — begin processing loop
        .stop()    — graceful shutdown
        .get_queue() -> asyncio.Queue[FrameResult]
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameResult:
    """
    Output contract for a single processed frame.
    This is the schema serialised into the WebSocket JSON message.

    Fields
    ------
    frame_id        : int       — monotonic counter
    camera_id       : str       — e.g. "CAM-01"
    timestamp       : float     — Unix timestamp
    frame_b64       : str       — Base64-encoded JPEG of the annotated frame
    detections      : list[dict]— Detection list (bbox, class_id, confidence)
    threat_level    : str       — "NONE" | "LOW" | "HIGH"
    threat_score    : float     — Composite score from ThreatScorer
    gradcam_b64     : Optional[str]  — Base64 Grad-CAM overlay JPEG (HIGH only)
    fps             : float     — Rolling 1-second average frame rate
    """
    frame_id:     int
    camera_id:    str
    timestamp:    float
    frame_b64:    str
    detections:   list = field(default_factory=list)
    threat_level: str  = "NONE"
    threat_score: float = 0.0
    gradcam_b64:  Optional[str] = None
    fps:          float = 0.0


class InferenceEngine:
    """
    Parameters
    ----------
    weapon_model  : HybridWeaponDetector — Custom model (Weapons/Confusers)
    hand_model    : YOLO — Pre-trained Hand detection model
    sahi_pipeline : SAHIPipeline — Slicing logic for the weapon stream
    threat_scorer : ThreatScorer — Logic layer for proximity IoU
    gradcam       : GradCAMGenerator — Explainability for weapons
    settings      : dict  — runtime config (conf_threshold, sahi_every_n, etc.)
    camera_source : str   — OpenCV VideoCapture source (path, RTSP URL, or int)
    """

    def start(self):
        """
        Begin the async frame processing loop.
        Must be called from within a running asyncio event loop.
        Launches inference in asyncio.to_thread to avoid blocking FastAPI.
        """
        ...

    def stop(self):
        """Signal the processing loop to exit gracefully after the current frame."""
        ...

    def get_queue(self):
        """
        Returns
        -------
        asyncio.Queue[FrameResult]
            Consumers (WebSocket broadcaster) pop from this queue.
        """
        ...
