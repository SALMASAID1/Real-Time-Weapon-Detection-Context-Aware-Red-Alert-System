# Task 4.2: System Integration & Final Polish

## Description
Assemble all components into a seamless real-time detection pipeline.

## Details
 1. [x] **InferenceEngine Orchestration**: Implement the background loop in `src/inference/engine.py` that manages:
   - Frame intake from OpenCV/RTSP.
   - Dual-model inference cadence (SAHI vs. Lightweight).
   - Async queuing of results for the WebSocket broadcaster.
 2. [x] **Concurrency Management**: Use `asyncio.to_thread` for the heavy inference loop to ensure the FastAPI server remains responsive.
 3. [x] **Data Fusion Logic**: Finalize the merging of Hand and Weapon detection lists before they reach the ThreatScorer.

## Learning Resources
- [Optimizing YOLO with TensorRT](https://docs.ultralytics.com/integrations/tensorrt/)
- [Asyncio in Python: A Guide](https://realpython.com/async-io-python/)
- [How to write a Great README](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
