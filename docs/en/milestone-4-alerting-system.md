# Milestone 4: Multimodal Alerting System & Real-Time Dashboard

This document details the implementation of Milestone 4, which focuses on building the real-time response infrastructure: multi-channel alerting, the inference orchestration engine, the FastAPI backend, and the React surveillance dashboard.

---

## Step 1: Alert Dispatcher — Multi-Channel Notification Engine
**File:** `src/threat_logic/alert_dispatcher.py`

When the Threat Scorer escalates a detection to "HIGH", the AlertDispatcher fires notifications through three independent channels simultaneously.

**Key Actions:**
1. **Telegram Bot (Async):** Uses `python-telegram-bot` v20 async API. Sends a formatted Markdown message with camera ID, timestamp, class name, confidence, composite score, and proximity IoU. If a Grad-CAM JPEG is available, it is sent as a photo attachment. Dispatched as a fire-and-forget `asyncio.create_task()` so the inference loop is never blocked waiting for the Telegram API.
2. **Audio Alert (Pygame):** Plays a pre-loaded WAV file (`data/audio/alert.wav`) via `pygame.mixer.Sound`. Pygame is initialized once at startup — not per-alert — to avoid latency spikes.
3. **Event Log (JSONL):** Appends a structured JSON record to `data/logs/threats.jsonl`. Fields: `event_id`, `timestamp`, `camera_id`, `class_name`, `confidence`, `composite_score`, `bbox`, `dispatched`, `acknowledged`.
4. **Cooldown Deduplication:** Each detection is assigned a spatial key based on its grid cell position. If the same key fires within 30 seconds, Telegram and audio are skipped (preventing notification spam), but the event is still logged.

---

## Step 2: Inference Engine — Frame Orchestration
**File:** `src/inference/engine.py`

The InferenceEngine sits between the camera source and the WebSocket broadcaster. It owns the frame loop and manages all pipeline components.

**Key Actions:**
1. **Dual-Cadence Strategy:** Full SAHI tiled inference runs every Nth frame (default N=3). Lightweight single-pass inference handles all other frames. This achieves ~30 FPS effective throughput vs ~6 FPS with SAHI on every frame.
2. **Two-Stream Pipeline:** The Weapon stream (Hybrid YOLO-Swin) and Hand stream (YOLOv8) run in parallel on each frame. Their detections are merged before threat scoring.
3. **Threat Scoring Integration:** All detections are passed to the `ThreatScorer`, which computes composite scores and identifies the highest-threat detection.
4. **Alert Triggering:** On HIGH threat, the engine dispatches alerts via `AlertDispatcher` and optionally generates a Grad-CAM heatmap overlay.
5. **Queue-Based Broadcasting:** Completed `FrameResult` objects (base64-encoded JPEG + detections + threat level + Grad-CAM) are pushed to an `asyncio.Queue`. A background task in `main.py` consumes this queue and fans out to all connected WebSocket clients.

---

## Step 3: FastAPI Backend — API Surface
**File:** `src/api/main.py` and routers in `src/api/routers/`

The backend provides both WebSocket streaming and REST endpoints.

**Key Actions:**
1. **Lifespan Management:** All models, pipelines, and the inference engine are initialized during FastAPI startup via `@asynccontextmanager`. On shutdown, the engine is stopped and background tasks are cancelled gracefully.
2. **WebSocket Streaming** (`/ws/stream/{camera_id}`): A `ConnectionManager` maintains a set of active connections per camera. The broadcaster task sends serialized `FrameResult` JSON to all connected clients.
3. **Threats REST API** (`/api/threats`): Full CRUD over the JSONL event log — paginated listing with filters (camera, class, level, timestamp, acknowledged), single-event retrieval, acknowledgement, and deletion.
4. **Settings REST API** (`/api/settings`): GET/PUT for runtime configuration. PUT propagates changes (thresholds, SAHI cadence) directly to the running InferenceEngine's settings dict.
5. **Pydantic Schemas** (`src/api/schemas/detection.py`): Strict data contracts — `Detection`, `ThreatEvent`, `WebSocketFrame`, `SystemSettings` — ensuring type safety between Python backend and React frontend.

---

## Step 4: React Dashboard — Surveillance UI
**Directory:** `ui/src/`

A React (Vite) single-page application providing three views for security operators.

**Key Actions:**
1. **LiveMonitor** (`pages/LiveMonitor.jsx`): The primary surveillance page. Owns the WebSocket connection via `useWebSocket` hook. Renders the live video feed on a `VideoCanvas`, displays real-time detections in a side panel, and shows a `RedAlertBanner` overlay on HIGH threats.
2. **ThreatHistory** (`pages/ThreatHistory.jsx`): Historical event log fetched via `GET /api/threats`. Supports pagination, filtering by threat level and weapon class, and a refresh button.
3. **Settings** (`pages/Settings.jsx`): Runtime configuration via `ThresholdPanel` — sliders for confidence threshold, IoU threshold, SAHI cadence, etc. Submits via `PUT /api/settings`.
4. **WebSocket Hook** (`hooks/useWebSocket.js`): Manages the full WebSocket lifecycle with exponential backoff reconnection (1s → 2s → 4s → ... → 30s cap). Parses incoming JSON frames and exposes `lastMessage` and `isConnected` state.
5. **Design System** (`index.css`): 15KB dark-theme design system with CSS variables, glassmorphism cards, and responsive layouts.

---

## Pipeline Scripts
*   `src/api/main.py`: Application entry point and lifespan management.
*   `src/api/routers/stream.py`: WebSocket endpoint and ConnectionManager.
*   `src/api/routers/threats.py`: Threat event CRUD REST API.
*   `src/api/routers/settings.py`: Runtime settings GET/PUT with engine propagation.
*   `src/threat_logic/alert_dispatcher.py`: Multi-channel notification dispatch.
*   `src/inference/engine.py`: Frame loop, dual-cadence inference, queue broadcasting.
