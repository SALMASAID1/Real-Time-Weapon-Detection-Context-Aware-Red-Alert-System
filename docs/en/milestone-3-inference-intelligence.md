# Milestone 3: Inference Intelligence & Threat Logic

## Overview
Milestone 3 focused on the "Intelligence" layer that sits on top of the raw model detections. It ensures the system can handle 4K footage and distinguish between a detection and a genuine "Red Alert" threat.

## Key Components

### 1. SAHI Pipeline (Slicing Aided Hyper Inference)
Standard YOLO models fail on small objects in 4K frames because the objects shrink too much when the frame is resized.
*   **The Logic**: We partition the 4K frame into overlapping 640x640 tiles.
*   **Benefits**: Preserves high-resolution details, allowing the model to detect small knives or pistols even when the subject is far from the camera.
*   **Merging**: Uses NMS (Non-Maximum Suppression) to unify detections across overlapping tiles.

### 2. Context-Aware Threat Scoring
We use a composite scoring formula ($S$) to determine the risk level of a detection:
$$S = (0.3 \cdot \text{Confidence}) + (0.5 \cdot \text{Proximity}) + (0.2 \cdot \text{Persistence})$$

#### **A. Proximity (GIoU)**
We calculate the **Generalized Intersection over Union (GIoU)** between Weapon boxes and Hand boxes.
*   **GIoU > 0**: The weapon is physically in a hand (High Threat).
*   **Near Miss**: GIoU provides a score even if the boxes are very close but not yet touching, allowing for early warning.

#### **B. Temporal Persistence**
A detection must persist across multiple frames (default 10) to reach a "HIGH" threat level. This effectively filters out "flickering" false positives from the final alert stream.

### 3. XAI (Explainable AI with Grad-CAM)
When a "HIGH" threat is triggered, the system generates a Grad-CAM heatmap.
*   **Purpose**: Shows which visual features (e.g., the trigger, the barrel) caused the model to fire.
*   **Benefit**: Builds trust with human operators and allows for manual verification of alerts.

## Pipeline Scripts
*   `src/inference/sahi_pipeline.py`: The tiling and merging logic.
*   `src/threat_logic/iou_calculator.py`: The GIoU proximity math.
*   `src/threat_logic/threat_scorer.py`: The final decision-making engine.
*   `src/xai/gradcam.py`: Heatmap generation.
