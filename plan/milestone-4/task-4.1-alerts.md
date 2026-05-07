# Task 4.1: Telegram & Pygame Alert Triggers

## Description
Implement the multi-channel notification system in `utils/alerts.py`.

## Details
1. **Telegram Notifications**: 
   - Set up a Telegram Bot.
   - Send text alerts and captured frames of the threat to a specified chat/group.
2. **Audio Alerts (Pygame)**:
   - Use `pygame.mixer` to play a siren or alarm sound locally when a Red Alert is triggered.
3. **Throttling**: Implement logic to prevent spamming alerts for the same event.

## Learning Resources
- [How to create a Telegram Bot and send messages](https://core.telegram.org/bots/tutorial)
- [Python Telegram Bot Documentation](https://python-telegram-bot.org/)
- [Pygame Mixer: Playing Sound in Python](https://www.pygame.org/docs/ref/mixer.html)
