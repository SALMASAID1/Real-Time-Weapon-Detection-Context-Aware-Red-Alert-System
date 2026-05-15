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


import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional

import telegram
import pygame
from src.threat_logic.threat_scorer import ScoredDetection

logger = logging.getLogger(__name__)

class AlertDispatcher:
    """
    Multi-channel alert dispatch on Red Alert escalation.
    
    Parameters
    ----------
    telegram_token : str   — Bot token from BotFather (loaded from .env)
    chat_id        : str   — Telegram chat or group ID to receive alerts
    audio_path     : str   — Path to alert WAV file
    log_path       : str   — Path to JSON log file
    cooldown_seconds: int  — Minimum seconds between repeated alerts for the
                            same persistent detection event (default 30)
    """

    def __init__(self, telegram_token: str, chat_id: str, audio_path: str, log_path: str, cooldown_seconds: int = 30):
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.audio_path = audio_path
        self.log_path = log_path
        self.cooldown_seconds = cooldown_seconds
        
        # Initialize Telegram Bot
        self.bot = telegram.Bot(token=telegram_token) if telegram_token else None
        
        # Initialize Pygame Mixer
        try:
            pygame.mixer.init()
            if os.path.exists(audio_path):
                self.alert_sound = pygame.mixer.Sound(audio_path)
            else:
                logger.warning(f"Audio file not found: {audio_path}")
                self.alert_sound = None
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            self.alert_sound = None
            
        # Cooldown registry: event_key -> {time, id}
        self.dispatch_history = {}

    async def dispatch(self, scored_detection: ScoredDetection, gradcam_jpeg: Optional[bytes], camera_id: str, event_id: str) -> str:
        """
        Dispatch all channels for a HIGH threat event.

        Parameters
        ----------
        scored_detection : ScoredDetection  — from ThreatScorer
        gradcam_jpeg     : bytes            — JPEG bytes of the Grad-CAM overlay
        camera_id        : str              — Source camera identifier

        Returns
        -------
        str — Stable event_id for this alert event.
        """
        det = scored_detection.detection
        x1, y1, x2, y2 = det['bbox']
        # Create a spatial key to group persistent detections in the same area
        gx, gy = int((x1+x2)/200), int((y1+y2)/200)
        event_key = f"{det.get('class_id', 0)}_{gx}_{gy}"
        
        now = datetime.now()
        
        # Check cooldown
        if event_key in self.dispatch_history:
            last_time = self.dispatch_history[event_key]['time']
            if (now - last_time).total_seconds() < self.cooldown_seconds:
                # Still log the event but skip Telegram/Audio
                current_event_id = self.dispatch_history[event_key]['id']
                self._log_event(current_event_id, scored_detection, camera_id, dispatched=False)
                return current_event_id
        
        # Update history
        self.dispatch_history[event_key] = {'time': now, 'id': event_id}
        
        # 1. Audio Alert (offloaded to thread so pygame doesn't block the event loop)
        if self.alert_sound:
            await asyncio.to_thread(self.alert_sound.play, 1)
            
        # 2. Telegram Alert (Fire-and-forget task)
        if self.bot and self.chat_id:
            asyncio.create_task(self._send_telegram_alert(scored_detection, gradcam_jpeg, camera_id))
            
        # 3. Event Log
        self._log_event(event_id, scored_detection, camera_id, dispatched=True)
        
        return event_id

    async def _send_telegram_alert(self, scored_detection, gradcam_jpeg, camera_id):
        try:
            det = scored_detection.detection
            message = (
                f"🚨 *RED ALERT: HIGH THREAT DETECTED* 🚨\n\n"
                f"📍 *Camera:* {camera_id}\n"
                f"🕒 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔍 *Object:* {det.get('class_name', 'Weapon')}\n"
                f"📈 *Confidence:* {det['confidence']:.2f}\n"
                f"🎯 *Threat Score:* {scored_detection.composite_score:.2f}\n"
                f"📏 *Proximity IoU:* {scored_detection.proximity_iou:.2f}"
            )
            
            if gradcam_jpeg:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=gradcam_jpeg,
                    caption=message,
                    parse_mode='Markdown'
                )
            else:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def _log_event(self, event_id, scored_detection, camera_id, dispatched):
        det = scored_detection.detection
        log_entry = {
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "camera_id": camera_id,
            "class_name": det.get('class_name', 'Weapon'),
            "confidence": det['confidence'],
            "composite_score": scored_detection.composite_score,
            "bbox": det['bbox'],
            "dispatched": dispatched,
            "acknowledged": False
        }
        
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Simple append to JSON Lines file
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    async def acknowledge(self, event_id: str) -> bool:
        """
        Mark an alert as acknowledged.
        Currently a stub that returns True if the log exists.
        """
        return os.path.exists(self.log_path)
