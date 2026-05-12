"""
src/api/routers/stream.py
===========================
WebSocket endpoint for real-time frame streaming to React clients.

Endpoint: WS /ws/stream/{camera_id}

Lifecycle
---------
1. Client connects (React app mounts VideoCanvas or LiveMonitor page).
2. Server registers the connection in a broadcast set.
3. InferenceEngine pushes FrameResult objects into an asyncio.Queue.
4. A background broadcaster task consumes the queue and fans out to all
   connected WebSocket clients as JSON (serialised WebSocketFrame schema).
5. On client disconnect (WebSocket close or React unmount), the connection
   is removed from the broadcast set gracefully.

Multiple clients
----------------
All connected clients receive the same broadcast. This supports multiple
operator workstations monitoring the same camera feed without opening
additional inference threads. The inference engine runs once; the broadcaster
fans out to N clients.

Frame serialisation
-------------------
The FrameResult.frame_b64 and FrameResult.gradcam_b64 fields are already
base64-encoded by the InferenceEngine. The WebSocket sends pure JSON text
frames (not binary frames) to maximise compatibility with browser WebSocket
clients — no binary ArrayBuffer parsing needed in React.

Error handling
--------------
If a client's send() raises WebSocketDisconnect, it is silently removed
from the broadcast set. The inference engine continues unaffected.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """
    Manages the set of active WebSocket connections.
    Thread-safe for the asyncio event loop (single-threaded).
    """

    def __init__(self):
        self.active: dict[str, set] = {}  # camera_id → set of WebSocket

    async def connect(self, camera_id: str, ws: WebSocket):
        await ws.accept()
        self.active.setdefault(camera_id, set()).add(ws)

    def disconnect(self, camera_id: str, ws: WebSocket):
        self.active.get(camera_id, set()).discard(ws)

    async def broadcast(self, camera_id: str, message: str):
        """Send JSON message to all clients watching this camera."""
        connections = self.active.get(camera_id, set())
        if not connections:
            return
        dead = set()
        for ws in connections:
            try:
                await ws.send_text(message)
            except (WebSocketDisconnect, Exception):
                dead.add(ws)
        if dead:
            self.active.get(camera_id, set()).difference_update(dead)


manager = ConnectionManager()


@router.websocket("/ws/stream/{camera_id}")
async def stream_endpoint(ws: WebSocket, camera_id: str):
    """
    WebSocket endpoint consumed by React useWebSocket hook.

    The React client connects here. After connection, it receives
    serialised WebSocketFrame JSON messages at the inference engine's cadence.

    The client sends no messages upstream (receive-only stream).
    Future extension: client can send JSON control messages to adjust settings
    without the REST /api/settings endpoint.
    """
    await manager.connect(camera_id, ws)
    try:
        # Keep the connection alive; the broadcaster pushes frames
        while True:
            # Await client disconnect signal (we don't expect inbound messages)
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(camera_id, ws)
