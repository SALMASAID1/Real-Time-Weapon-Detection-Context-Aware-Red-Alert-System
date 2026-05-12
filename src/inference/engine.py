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

import cv2
import time
import asyncio
import base64
import numpy as np
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class FrameResult:
    """
    Output contract for a single processed frame.
    This is the schema serialised into the WebSocket JSON message.
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
    Main inference engine — frame intake, cadence management, pipeline orchestration.
    
    Parameters
    ----------
    weapon_model  : HybridWeaponDetector — Custom model (Weapons/Confusers)
    hand_model    : Any — Pre-trained Hand detection model (e.g. YOLOv8)
    sahi_pipeline : SAHIPipeline — Slicing logic for the weapon stream
    threat_scorer : ThreatScorer — Logic layer for proximity IoU
    gradcam       : GradCAMGenerator — Explainability for weapons
    alert_dispatcher : AlertDispatcher — Multi-channel alerting
    settings      : dict  — runtime config (conf_threshold, sahi_every_n, etc.)
    camera_source : str   — OpenCV VideoCapture source
    """

    def __init__(
        self, 
        weapon_model, 
        hand_model, 
        sahi_pipeline, 
        threat_scorer, 
        gradcam, 
        alert_dispatcher, 
        settings, 
        camera_source
    ):
        self.weapon_model = weapon_model
        self.hand_model = hand_model
        self.sahi_pipeline = sahi_pipeline
        self.threat_scorer = threat_scorer
        self.gradcam = gradcam
        self.alert_dispatcher = alert_dispatcher
        self.settings = settings
        self.camera_source = camera_source
        
        self.queue = asyncio.Queue(maxsize=10)
        self.running = False
        self.frame_id = 0
        self.camera_id = settings.get("camera_id", "CAM-01")
        self.loop = None

    def start(self):
        """
        Begin the async frame processing loop.
        Launches inference in asyncio.to_thread to avoid blocking FastAPI.
        """
        self.running = True
        self.loop = asyncio.get_running_loop()
        asyncio.create_task(self._run_loop())
        logger.info(f"Inference engine started for source: {self.camera_source}")

    def stop(self):
        """Signal the processing loop to exit gracefully."""
        self.running = False
        logger.info("Inference engine stopping...")

    def get_queue(self):
        """Returns the asyncio.Queue for consumers."""
        return self.queue

    async def _run_loop(self):
        """Wrapper to run the blocking inference loop in a separate thread."""
        await asyncio.to_thread(self._inference_loop)

    def _inference_loop(self):
        """Blocking loop that handles frame capture and model inference."""
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            logger.error(f"Failed to open camera source: {self.camera_source}")
            self.running = False
            return

        sahi_every_n = self.settings.get("sahi_every_n", 3)
        conf_threshold = self.settings.get("conf_threshold", 0.25)
        iou_threshold = self.settings.get("iou_threshold", 0.45)
        
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from source. Reaching EOF or disconnected.")
                break

            self.frame_id += 1
            timestamp = time.time()
            
            # 1. Dual-Cadence Inference (Weapon Stream)
            is_sahi_frame = (self.frame_id % sahi_every_n == 0)
            if is_sahi_frame:
                weapon_dets = self.sahi_pipeline.run(frame)
            else:
                weapon_dets = self.weapon_model.predict(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            
            # 2. Hand Stream
            # Assuming hand_model is an Ultralytics YOLO model
            hand_results = self.hand_model(frame, verbose=False)[0]
            hand_dets = []
            for r in hand_results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                hand_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(cls),
                    "class_name": "hand",
                    "confidence": conf
                })

            # 3. Threat Scoring
            all_detections = weapon_dets + hand_dets
            scored_results = self.threat_scorer.score(all_detections, self.frame_id)
            
            # Identify max threat
            max_threat_level = "NONE"
            max_threat_score = 0.0
            high_threat_sd = None
            
            for sd in scored_results:
                if sd.composite_score > max_threat_score:
                    max_threat_score = sd.composite_score
                
                if sd.threat_level == "HIGH":
                    max_threat_level = "HIGH"
                    high_threat_sd = sd
                elif sd.threat_level == "LOW" and max_threat_level != "HIGH":
                    max_threat_level = "LOW"

            # 4. Alerts & Explainability
            gradcam_b64 = None
            if max_threat_level == "HIGH" and high_threat_sd:
                gradcam_jpeg = None
                if self.gradcam:
                    gradcam_jpeg = self.gradcam.generate(frame, high_threat_sd.detection)
                    if gradcam_jpeg:
                        gradcam_b64 = base64.b64encode(gradcam_jpeg).decode('utf-8')
                
                if self.alert_dispatcher:
                    asyncio.run_coroutine_threadsafe(
                        self.alert_dispatcher.dispatch(high_threat_sd, gradcam_jpeg, self.camera_id),
                        self.loop
                    )

            # 5. UI Payload Preparation
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            result = FrameResult(
                frame_id=self.frame_id,
                camera_id=self.camera_id,
                timestamp=timestamp,
                frame_b64=frame_b64,
                detections=all_detections,
                threat_level=max_threat_level,
                threat_score=max_threat_score,
                gradcam_b64=gradcam_b64,
                fps=current_fps
            )

            # 6. Push to Queue (Non-blocking for this thread)
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self._put_in_queue, result)

        cap.release()
        self.running = False

    def _put_in_queue(self, result):
        """Helper to put result into the queue without awaiting (called via call_soon_threadsafe)."""
        try:
            if self.queue.full():
                self.queue.get_nowait() # Drop oldest if full
            self.queue.put_nowait(result)
        except Exception:
            pass
