import asyncio
import os
import sys

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.threat_logic.alert_dispatcher import AlertDispatcher
from src.threat_logic.threat_scorer import ScoredDetection

async def main():
    print("Testing AlertDispatcher...")
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except ImportError:
        print("python-dotenv not installed. Skipping .env file load.")
    
    # Configuration
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio", "alert.wav"))
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "logs", "threats.jsonl"))
    
    if not telegram_token or not chat_id:
        print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set in your environment.")
        print("Please set them to test the Telegram integration.")
    else:
        print(f"Using Telegram Chat ID: {chat_id}")
        
    # Create dummy audio file if it doesn't exist to avoid error
    if not os.path.exists(audio_path):
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        print(f"Audio file {audio_path} not found. Please place a valid .wav file there to test audio.")

    # Initialize Dispatcher
    dispatcher = AlertDispatcher(
        telegram_token=telegram_token,
        chat_id=chat_id,
        audio_path=audio_path,
        log_path=log_path,
        cooldown_seconds=5 # Short cooldown for testing
    )
    
    # Create a dummy high-threat detection
    dummy_detection = {
        "bbox": [100, 100, 200, 200],
        "class_id": 1,
        "class_name": "Assault Rifle",
        "confidence": 0.95
    }
    
    scored_det = ScoredDetection(
        detection=dummy_detection,
        confidence=0.95,
        proximity_iou=0.85,
        persistence=1.0,
        composite_score=0.98,
        threat_level="HIGH"
    )
    
    print("\n[Test 1] Dispatching Initial High Threat Alert...")
    event_id = await dispatcher.dispatch(
        scored_detection=scored_det,
        gradcam_jpeg=None, # Passing None for testing without actual image
        camera_id="TEST-CAM-01"
    )
    print(f"Dispatched successfully. Event ID: {event_id}")
    
    print("\n[Test 2] Testing Cooldown Logic...")
    print("Dispatching same threat again immediately (should hit cooldown)...")
    event_id2 = await dispatcher.dispatch(
        scored_detection=scored_det,
        gradcam_jpeg=None,
        camera_id="TEST-CAM-01"
    )
    if event_id == event_id2:
        print("Cooldown logic worked! Same event ID returned.")
    else:
        print("Cooldown logic failed.")
        
    print("\n[Test 3] Check Logs...")
    if os.path.exists(log_path):
        print(f"Log file found at: {log_path}")
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"Last log entry: {lines[-1].strip()}")
    else:
        print("Log file was not created.")
        
    print("\nTesting completed. Check your Telegram for the message if tokens were provided.")
    
    # Wait for the audio to finish playing
    import pygame
    print("Waiting for audio to finish playing...")
    while pygame.mixer.get_busy():
        await asyncio.sleep(0.1)
    
    # Wait a moment to allow the background Telegram task to finish sending
    print("Waiting 3 seconds for Telegram message to send...")
    await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(main())
