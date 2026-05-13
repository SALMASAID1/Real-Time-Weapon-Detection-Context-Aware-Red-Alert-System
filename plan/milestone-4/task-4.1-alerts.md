# Task 4.1: Multi-Channel Alert System (Telegram & Audio)

## Description
Develop the alert dispatching layer in `src/threat_logic/alert_dispatcher.py` to notify security personnel when a "High" threat is detected.

## Details
 1. [x] **Telegram Integration**: Use `python-telegram-bot` to send asynchronous alerts containing:
   - A JPEG snapshot of the detection with bounding boxes.
   - The timestamp and threat score.
   - Camera ID and location metadata.
 2. [x] **Local Audio Alerts**: Implement a local alarm trigger using the `pygame.mixer` to play a 1500Hz alert sound on the server machine.
 3. [x] **Dispatcher Logic**: Ensure the dispatcher handles cooldowns (throttling) to prevent alert fatigue during sustained detections.

## Learning Resources
- [Python Telegram Bot Documentation](https://python-telegram-bot.org/)
- [Playing Sounds with Pygame Mixer](https://www.pygame.org/docs/ref/mixer.html)
- [Building an Async Alerting System in Python](https://towardsdatascience.com/building-a-real-time-alert-system-with-python-and-telegram-4d2b2f6b4b9b)
