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
import threading
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class _CameraThread:
    """Continuously reads frames in a background thread."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.grabbed = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if not grabbed:
                # Create a mock placeholder frame if camera fails
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA OFFLINE", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                grabbed = True
                time.sleep(0.1) # throttle loop when camera is failing
                
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

@dataclass
class FrameResult:
    """
    Output contract for a single processed frame.
    This is the schema serialised into the WebSocket JSON message.

    Fields match the React useDetectionStream hook expectations:
      frame_b64      — base64 JPEG of the raw frame
      detections     — all detections (weapons + hands)
      threat_events  — scored weapon events (for ThreatLogTable)
      threat_level   — highest threat level this frame ("NONE"/"LOW"/"HIGH")
      gradcam_b64    — base64 Grad-CAM JPEG overlay (only on HIGH)
    """
    frame_id:      int
    camera_id:     str
    timestamp:     float
    frame_b64:     str
    detections:    list = field(default_factory=list)
    threat_events: list = field(default_factory=list)
    threat_level:  str  = "NONE"
    threat_score:  float = 0.0
    gradcam_b64:   Optional[str] = None
    fps:           float = 0.0


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
        cap = _CameraThread(self.camera_source)
        time.sleep(0.5) # Give camera thread a moment to start
        if not cap.read()[0]:
            logger.error(f"Failed to open camera source: {self.camera_source}")
            self.running = False
            cap.stop()
            return

        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0.0

        last_all_detections = []
        last_scored_results = []
        last_max_threat_level = "NONE"
        last_max_threat_score = 0.0
        last_high_threat_sd = None

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.frame_id += 1
            timestamp = time.time()
            
            # Read settings on each frame so live-apply works
            inference_every_n = self.settings.get("inference_every_n", 3)
            sahi_every_n = self.settings.get("sahi_every_n", 5)
            conf_threshold = self.settings.get("conf_threshold", 0.25)
            iou_threshold = self.settings.get("iou_threshold", 0.45)
            gradcam_on_high = self.settings.get("gradcam_on_high", True)

            h, w = frame.shape[:2]

            if self.frame_id % inference_every_n != 0:
                # Frame skip: reuse previous detections
                all_detections = last_all_detections
                scored_results = last_scored_results
                max_threat_level = last_max_threat_level
                max_threat_score = last_max_threat_score
                high_threat_sd = last_high_threat_sd
            else:
                # 1. Dual-Cadence Inference (Weapon Stream)
                try:
                    # Note: sahi_every_n is relative to inference frames
                    is_sahi_frame = (self.frame_id % (inference_every_n * sahi_every_n) == 0)
                    if is_sahi_frame:
                        weapon_dets = self.sahi_pipeline.run(frame)
                    else:
                        weapon_dets = self.weapon_model.predict(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                    
                    # Normalize weapon coordinates to pixel space BEFORE threat scoring
                    for det in weapon_dets:
                        bbox = det.get("bbox", [0,0,0,0])
                        if all(0 <= v <= 1.0 for v in bbox) and w > 1 and h > 1:
                            det["bbox"] = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
                except Exception as e:
                    logger.error(f"Weapon inference failed: {e}")
                    weapon_dets = []

                # 2. Hand Stream
                try:
                    hand_results = self.hand_model(frame, verbose=False)[0]
                    hand_dets = []
                    for r in hand_results.boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = r
                        hand_dets.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_id": int(cls),
                            "class_name": "hand",
                            "confidence": conf,
                            "is_weapon": False
                        })
                except Exception as e:
                    logger.error(f"Hand inference failed: {e}")
                    hand_dets = []

                # 3. Threat Scoring
                all_detections = weapon_dets + hand_dets
                try:
                    scored_results = self.threat_scorer.score(all_detections, self.frame_id)
                except Exception as e:
                    logger.error(f"Threat scoring failed: {e}")
                    scored_results = []

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
                
                # Cache for next skipped frames
                last_all_detections = all_detections
                last_scored_results = scored_results
                last_max_threat_level = max_threat_level
                last_max_threat_score = max_threat_score
                last_high_threat_sd = high_threat_sd

            # Build threat_events list for the frontend (use current frame_id and timestamp)
            threat_events = []
            for sd in scored_results:
                threat_events.append({
                    "event_id": f"{self.frame_id}_{sd.detection.get('class_id', 0)}",
                    "timestamp": timestamp,
                    "camera_id": self.camera_id,
                    "detection": sd.detection,
                    "proximity_iou": sd.proximity_iou,
                    "persistence": sd.persistence,
                    "composite_score": sd.composite_score,
                    "threat_level": sd.threat_level,
                    "acknowledged": False,
                })

            # 4. Alerts & Explainability
            gradcam_b64 = None
            if max_threat_level == "HIGH" and high_threat_sd:
                gradcam_jpeg = None
                if self.gradcam and gradcam_on_high:
                    try:
                        gradcam_jpeg = self.gradcam.generate(frame, high_threat_sd.detection)
                        if gradcam_jpeg:
                            gradcam_b64 = base64.b64encode(gradcam_jpeg).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Grad-CAM overlay failed: {e}")
                
                if self.alert_dispatcher:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self.alert_dispatcher.dispatch(high_threat_sd, gradcam_jpeg, self.camera_id),
                            self.loop
                        )
                    except Exception as e:
                        logger.error(f"Alert dispatch failed: {e}")

            # 5. UI Payload Preparation
            h, w = frame.shape[:2]
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Convert detections to the format expected by the React frontend
            ui_detections = self._format_detections_for_ui(all_detections, w, h)

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
                detections=ui_detections,
                threat_events=threat_events,
                threat_level=max_threat_level,
                threat_score=max_threat_score,
                gradcam_b64=gradcam_b64,
                fps=current_fps
            )

            # 6. Push to Queue (Non-blocking for this thread)
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self._put_in_queue, result)

        cap.stop()
        self.running = False

    @staticmethod
    def _format_detections_for_ui(detections: list, frame_w: int, frame_h: int) -> list:
        """
        Normalize detection dicts for the React VideoCanvas component.

        Ensures:
          - bbox is an object {x1, y1, x2, y2} in pixel coordinates
          - is_weapon field is present
          - class_name is present
        
        Handles two input formats:
          - Weapon detections from decode_predictions: bbox [x1,y1,x2,y2] normalized (0-1)
          - Hand detections from Ultralytics YOLO: bbox [x1,y1,x2,y2] in pixels
        """
        formatted = []
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            
            # Determine if bbox values are normalized (all between 0 and 1)
            is_normalized = all(0 <= v <= 1.0 for v in bbox)
            
            if is_normalized and (frame_w > 1 and frame_h > 1):
                # Convert from normalized [0,1] to pixel coordinates
                px1 = bbox[0] * frame_w
                py1 = bbox[1] * frame_h
                px2 = bbox[2] * frame_w
                py2 = bbox[3] * frame_h
            else:
                px1, py1, px2, py2 = bbox
            
            # Determine is_weapon based on class_id (0=Weapon) or existing field
            class_id = det.get("class_id", -1)
            is_weapon = det.get("is_weapon", class_id == 0)
            
            formatted.append({
                "bbox": {"x1": px1, "y1": py1, "x2": px2, "y2": py2},
                "class_id": class_id,
                "class_name": det.get("class_name", f"class_{class_id}"),
                "confidence": det.get("confidence", 0.0),
                "is_weapon": is_weapon,
            })
        return formatted

    def _put_in_queue(self, result):
        """Helper to put result into the queue without awaiting (called via call_soon_threadsafe)."""
        try:
            if self.queue.full():
                self.queue.get_nowait() # Drop oldest if full
            self.queue.put_nowait(result)
        except Exception:
            pass

