# Task 3.2: Context-Aware Threat Logic (Proximity IoU)

## Description
Develop context-aware logic in `utils/threat_logic.py` to filter detections and trigger "Red Alerts" based on proximity and confidence.

## Details
1. **Proximity IoU**: Implement logic to detect when a weapon is in close proximity to a person or sensitive area using spatial distance or IoU-based overlap.
2. **Threat Scoring**: Assign a threat level based on:
   - Detection confidence.
   - Proximity to vulnerable subjects.
   - Persistence of detection across multiple frames.
3. **Alert Trigger**: Define the threshold at which a detection becomes a "Red Alert".

## Learning Resources
- [Understanding Intersection over Union (IoU)](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [Object Tracking and Proximity Analysis](https://towardsdatascience.com/object-tracking-using-opencv-python-71906744f441)
- [Context-Aware Object Detection Concepts](https://arxiv.org/abs/1609.03605)
