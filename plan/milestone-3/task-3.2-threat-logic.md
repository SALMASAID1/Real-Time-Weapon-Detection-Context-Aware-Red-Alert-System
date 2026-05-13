# Task 3.2: Context-Aware Threat Logic (Proximity IoU)

## Description
Develop context-aware logic in `src/threat_logic/threat_scorer.py` to filter detections and trigger "Red Alerts" based on proximity and confidence.

## Details
 1. [x] **Two-Stream Data Fusion**: Integrate detection results from both the custom Weapon model and the pre-trained Hand model.
 2. [x] **Proximity IoU**: Implement logic to detect when a detected weapon bounding box overlaps with a detected hand bounding box ($B_h \cap B_w$).
 3. [x] **Threat Scoring**: Assign a threat level based on:
   - Detection confidence of both models.
   - Spatial overlap (IoU) between hand and weapon.
   - Persistence of detection across multiple frames.
 4. [x] **Alert Trigger**: Define the threshold at which a detection becomes a "Red Alert" (Active Handling).

## Learning Resources
- [Understanding Intersection over Union (IoU)](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [Object Tracking and Proximity Analysis](https://towardsdatascience.com/object-tracking-using-opencv-python-71906744f441)
- [Context-Aware Object Detection Concepts](https://arxiv.org/abs/1609.03605)
