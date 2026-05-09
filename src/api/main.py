"""
src/api/main.py
=================
FastAPI application entry point.

Architecture summary
--------------------
                  ┌─────────────────────────────┐
  Camera/RTSP ──▶│   InferenceEngine (thread)   │
                  │   SAHIPipeline               │
                  │   ThreatScorer               │
                  │   GradCAMGenerator           │
                  └─────────┬───────────────────┘
                            │ asyncio.Queue[FrameResult]
                            ▼
                  ┌─────────────────────────────┐
                  │   WebSocket Broadcaster      │──▶ React Client 1
                  │   (background asyncio task)  │──▶ React Client 2
                  └─────────────────────────────┘
                            │ RED ALERT
                            ▼
                  ┌─────────────────────────────┐
                  │   AlertDispatcher            │──▶ Telegram Bot
                  │                              │──▶ Audio (pygame)
                  │                              │──▶ JSON Event Log
                  └─────────────────────────────┘

REST API surface (all prefixed /api/):
  /api/threats         — ThreatEvent CRUD (src/api/routers/threats.py)
  /api/settings        — Runtime config (src/api/routers/settings.py)

WebSocket:
  /ws/stream/{camera_id}  — Live frame stream (src/api/routers/stream.py)

CORS
----
Allowed origins in development: http://localhost:5173 (Vite dev server)
In production: replace with the deployed frontend domain.

Start the server
----------------
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (load from .env via python-dotenv)
---------------------------------------------------------
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  CAMERA_SOURCE          — OpenCV source (int for webcam, path or RTSP URL)
  MODEL_WEIGHTS_PATH     — Path to models/weights/best.pt
  LOG_PATH               — Path to threat event log file
"""

import os
import json
import asyncio
import logging
import dataclasses
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from ultralytics import YOLO

from src.api.routers import stream, threats, settings as settings_router
from src.api.routers.stream import manager as ws_manager
from src.models.hybrid_model import HybridWeaponDetector
from src.inference.sahi_pipeline import SAHIPipeline
from src.threat_logic.threat_scorer import ThreatScorer
from src.threat_logic.iou_calculator import IoUCalculator
from src.threat_logic.alert_dispatcher import AlertDispatcher
from src.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

async def broadcast_worker(queue: asyncio.Queue):
    """
    Background task that consumes FrameResult objects from the InferenceEngine
    and broadcasts them to all connected WebSocket clients.
    """
    logger.info("Broadcaster worker started.")
    try:
        while True:
            result = await queue.get()
            message = json.dumps(dataclasses.asdict(result))
            await ws_manager.broadcast(result.camera_id, message)
            queue.task_done()
    except asyncio.CancelledError:
        logger.info("Broadcaster worker stopping...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise models, pipelines, and start the inference engine.
    Shutdown: gracefully stop the engine and background tasks.
    """
    # --- Startup ---
    load_dotenv()
    
    # 1. Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weapon_model = HybridWeaponDetector(
        backbone_variant=os.getenv("BACKBONE_VARIANT", "yolo11m.pt"),
        nc=3,
        device=device
    )
    
    weights_path = os.getenv("MODEL_WEIGHTS_PATH", "models/weights/best.pt")
    if os.path.exists(weights_path):
        try:
            weapon_model.load_state_dict(torch.load(weights_path, map_location=device))
            logger.info(f"Loaded weapon model weights from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load weapon weights: {e}")
            
    hand_model = YOLO(os.getenv("HAND_MODEL_PATH", "yolov8n.pt"))
    
    # 2. Logic & Pipelines
    sahi_pipeline = SAHIPipeline(weapon_model)
    iou_calc = IoUCalculator(mode="giou")
    threat_scorer = ThreatScorer(iou_calc)
    
    # Grad-CAM is optional (placeholder for Milestone 3/4 integration)
    gradcam = None 
    
    alert_dispatcher = AlertDispatcher(
        telegram_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        audio_path=os.getenv("AUDIO_ALERT_PATH", "assets/alert.wav"),
        log_path=os.getenv("THREAT_LOG_PATH", "data/logs/threats.jsonl")
    )
    
    # 3. Engine Orchestration
    settings = {
        "sahi_every_n": int(os.getenv("SAHI_EVERY_N", 3)),
        "conf_threshold": float(os.getenv("CONF_THRESHOLD", 0.25)),
        "camera_id": os.getenv("CAMERA_ID", "CAM-01")
    }
    
    camera_source = os.getenv("CAMERA_SOURCE", "0")
    if camera_source.isdigit():
        camera_source = int(camera_source)
        
    engine = InferenceEngine(
        weapon_model=weapon_model,
        hand_model=hand_model,
        sahi_pipeline=sahi_pipeline,
        threat_scorer=threat_scorer,
        gradcam=gradcam,
        alert_dispatcher=alert_dispatcher,
        settings=settings,
        camera_source=camera_source
    )
    
    app.state.engine = engine
    engine.start()
    
    # 4. Background Broadcaster
    broadcaster_task = asyncio.create_task(broadcast_worker(engine.get_queue()))
    app.state.broadcaster_task = broadcaster_task
    
    yield
    
    # --- Shutdown ---
    engine.stop()
    broadcaster_task.cancel()
    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="AWD&TA — Weapon Detection API",
    description="Real-Time Weapon Detection & Context-Aware Red Alert System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(stream.router)
app.include_router(threats.router)
app.include_router(settings_router.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
