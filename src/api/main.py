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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.routers import stream, threats, settings as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise model, SAHI pipeline, inference engine, start background
             broadcaster task.
    Shutdown: signal inference engine to stop, await task cancellation.
    """
    # --- Startup ---
    # 1. Load environment variables
    # 2. Instantiate HybridWeaponDetector (load weights)
    # 3. Instantiate SAHIPipeline, ThreatScorer, GradCAMGenerator, AlertDispatcher
    # 4. Instantiate InferenceEngine and call .start()
    # 5. Launch background broadcaster asyncio task
    yield
    # --- Shutdown ---
    # 6. engine.stop() — graceful frame loop exit
    # 7. Cancel broadcaster task


app = FastAPI(
    title="AWD&TA — Weapon Detection API",
    description="Real-Time Weapon Detection & Context-Aware Red Alert System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server and future production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
    """Health check — used by React to confirm backend is reachable on load."""
    return {"status": "ok", "version": "1.0.0"}
