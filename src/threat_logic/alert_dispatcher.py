"""
src/threat_logic/alert_dispatcher.py
=======================================
Multi-channel alert dispatch on Red Alert escalation.

Channels
--------
1. Telegram Bot (async)
   - Sends a formatted message including: timestamp, camera ID, class name,
     confidence, composite score, and the Grad-CAM overlay JPEG as a photo.
   - Uses python-telegram-bot v20 async API.
   - Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
   - Fire-and-forget: dispatched in a non-blocking asyncio task so the
     inference loop is not delayed waiting for the Telegram API response.

2. Audio Cue (pygame)
   - Plays a pre-loaded WAV file via pygame.mixer.
   - pygame is initialised once at startup (not per-alert) to avoid latency.
   - The audio cue runs in the calling thread — it is non-blocking because
     pygame.mixer plays asynchronously.

3. Event Log (persistent)
   - Appends a structured JSON record to a rotating log file (or SQLite DB).
   - This is the data source for the React ThreatHistory page via
     GET /api/threats.
   - Fields logged:  event_id, timestamp, camera_id, class_name, confidence,
                     composite_score, bbox, gradcam_image_path, acknowledged.

Design: idempotent dispatch
----------------------------
Each Red Alert event is assigned a stable event_id (UUID). If the same
weapon persists across multiple frames and re-triggers the HIGH threshold,
the dispatcher checks: was this event_id already dispatched within the last
COOLDOWN_SECONDS (default 30)?  If yes, skip Telegram and audio (prevent
notification spam) but still update the log record.

Public API
----------
    AlertDispatcher(telegram_token, chat_id, audio_path, log_path, cooldown)
        .dispatch(scored_detection, gradcam_jpeg: bytes, camera_id: str)
            -> str  (event_id)
        .acknowledge(event_id: str) -> bool
"""


class AlertDispatcher:
    """
    Parameters
    ----------
    telegram_token : str   — Bot token from BotFather (loaded from .env)
    chat_id        : str   — Telegram chat or group ID to receive alerts
    audio_path     : str   — Path to alert WAV file
    log_path       : str   — Path to JSON log file (or SQLite DB path)
    cooldown_seconds: int  — Minimum seconds between repeated alerts for the
                            same persistent detection event (default 30)
    """

    async def dispatch(self, scored_detection, gradcam_jpeg: bytes, camera_id: str) -> str:
        """
        Dispatch all channels for a HIGH threat event.

        Parameters
        ----------
        scored_detection : ScoredDetection  — from ThreatScorer
        gradcam_jpeg     : bytes            — JPEG bytes of the Grad-CAM overlay
        camera_id        : str              — Source camera identifier

        Returns
        -------
        str — Stable event_id (UUID4) for this alert event.
        """
        ...

    async def acknowledge(self, event_id: str) -> bool:
        """
        Mark an alert as acknowledged by an operator.
        Called by the React frontend via DELETE /api/threats/{id} or a dedicated
        PATCH endpoint. Updates the log record.

        Returns
        -------
        bool — True if event_id was found and updated, False otherwise.
        """
        ...
